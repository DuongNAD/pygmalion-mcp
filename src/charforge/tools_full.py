"""Level 3 tools: auto_rig, animate, pipeline_run.

Registered into FastMCP server when CHARFORGE_LEVEL >= 3.
"""
from __future__ import annotations

import base64
import inspect
import json
import os
import struct
import time
from pathlib import Path
from typing import Annotated, Literal

try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from . import blender, config, providers, state
from .envelope import CFError, action, ok, tool_entry
from .pipeline_run import pipeline_run

LOCAL_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

ProjectId = Annotated[str, Field(description="Id returned by project_create, e.g. 'aria'", pattern=r"^[a-z0-9][a-z0-9-]{0,47}$")]
MeshPath = Annotated[str, Field(description="Project-relative path from a previous result (e.g. 'export/aria.glb') or an absolute path")]
RigMethod = Literal["auto", "blender", "provider"]
ActionName = Literal["idle", "walk", "wave"]
AnimateMethod = Literal["auto", "blender", "provider"]


def _find_provider_task_id(p: dict, rpath: str) -> str | None:
    """Find upstream provider generation task ID by walking the derived_from lineage chain."""
    return state.find_provider_task_id(p, rpath)


def _poll_provider(prov: providers.Provider, job_id: str, kind: str | None = None) -> dict:
    sig = inspect.signature(prov.poll)
    if "kind" in sig.parameters:
        return prov.poll(job_id, kind=kind)
    return prov.poll(job_id)


def inspect_glb_bones(glb_input: Path | str | bytes) -> int:
    """Count skeletal joints from glTF binary chunk 0 JSON `skins` (or nodes if skins absent)."""
    try:
        if isinstance(glb_input, (str, Path)):
            glb_bytes = Path(glb_input).read_bytes()
        else:
            glb_bytes = glb_input

        if len(glb_bytes) < 20:
            return 0
        magic, version, _ = struct.unpack("<4sII", glb_bytes[:12])
        if magic != b"glTF" or version != 2:
            return 0

        chunk_length, chunk_type = struct.unpack("<I4s", glb_bytes[12:20])
        if chunk_type != b"JSON":
            return 0

        json_data = glb_bytes[20 : 20 + chunk_length].decode("utf-8")
        gltf_obj = json.loads(json_data)
        skins = gltf_obj.get("skins") or []
        if skins:
            joints: set[int] = set()
            for skin in skins:
                for j in skin.get("joints") or []:
                    joints.add(int(j))
            if joints:
                return len(joints)
        nodes = gltf_obj.get("nodes") or []
        if nodes:
            return len(nodes)
    except Exception:
        pass
    return 0


def inspect_glb_animations(glb_input: Path | str | bytes) -> list[str]:
    """Parse glTF binary chunk 0 JSON header to extract animation names."""
    if isinstance(glb_input, (str, Path)):
        glb_bytes = Path(glb_input).read_bytes()
    else:
        glb_bytes = glb_input

    if len(glb_bytes) < 20:
        raise ValueError("Invalid GLB: file too small")
    magic, version, total_length = struct.unpack("<4sII", glb_bytes[:12])
    if magic != b"glTF" or version != 2:
        raise ValueError(f"Invalid GLB header: magic={magic!r}, version={version}")

    chunk_length, chunk_type = struct.unpack("<I4s", glb_bytes[12:20])
    if chunk_type != b"JSON":
        raise ValueError(f"Expected JSON chunk 0, got {chunk_type!r}")

    json_data = glb_bytes[20 : 20 + chunk_length].decode("utf-8")
    gltf_obj = json.loads(json_data)
    animations = gltf_obj.get("animations", [])
    return [a.get("name", f"action_{i}") for i, a in enumerate(animations)]


def rename_glb_animations(glb_path: Path, action_names: list[str], out_path: Path | None = None) -> Path:
    """Losslessly rename animation tracks inside a GLB's JSON chunk 0 to match `action_names`."""
    dst = Path(out_path) if out_path else Path(glb_path)
    glb_bytes = Path(glb_path).read_bytes()
    if len(glb_bytes) < 20:
        raise ValueError("Invalid GLB: file too small")
    magic, version, _ = struct.unpack("<4sII", glb_bytes[:12])
    if magic != b"glTF" or version != 2:
        raise ValueError(f"Invalid GLB header: magic={magic!r}, version={version}")

    chunk0_len, chunk0_type = struct.unpack("<I4s", glb_bytes[12:20])
    if chunk0_type != b"JSON":
        raise ValueError(f"Expected JSON chunk 0, got {chunk0_type!r}")

    json_str = glb_bytes[20 : 20 + chunk0_len].decode("utf-8")
    gltf_obj = json.loads(json_str)
    anims = gltf_obj.get("animations") or []
    if not anims:
        raise ValueError("GLB contains no animation tracks to rename.")

    new_anims = []
    for i, target_name in enumerate(action_names):
        src_anim = anims[i] if i < len(anims) else anims[-1]
        anim_copy = dict(src_anim)
        anim_copy["name"] = target_name
        new_anims.append(anim_copy)
    gltf_obj["animations"] = new_anims

    new_json_bytes = json.dumps(gltf_obj, ensure_ascii=False).encode("utf-8")
    padding = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b" " * padding

    rest = glb_bytes[20 + chunk0_len :]
    total_len = 12 + 8 + len(new_json_bytes) + len(rest)
    new_glb = (
        struct.pack("<4sII", b"glTF", 2, total_len)
        + struct.pack("<I4s", len(new_json_bytes), b"JSON")
        + new_json_bytes
        + rest
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(new_glb)
    return dst


def _finish_provider_rig(
    p: dict,
    pdir: Path,
    project_id: str,
    rpath: str,
    rig_task_id: str,
    poll_res: dict,
    rig_source: str,
    input_task_id: str | None,
) -> dict:
    url = poll_res.get("url")
    if not url:
        raise CFError(
            "PROVIDER_FAILED",
            "Meshy rigging succeeded but returned no rigged_character_glb_url.",
            "Use method='blender'",
        )

    out_rpath = f"export/{project_id}-rigged.glb"
    out_path = pdir / out_rpath
    prov = providers.get_provider("meshy")
    prov.download(url, out_path)

    bones = inspect_glb_bones(out_path)
    state.register_mesh(
        p,
        out_rpath,
        source="rigged",
        derived_from=rpath,
        task_id=input_task_id,
        provider="meshy",
        rig_task_id=rig_task_id,
    )

    if "outputs" not in p:
        p["outputs"] = {}
    p["outputs"]["rigged"] = out_rpath
    p["outputs"]["rig_task_id"] = rig_task_id
    p["stage"] = "rigged"
    p["rig_task_id"] = rig_task_id
    basic_anims = poll_res.get("basic_animations") or {}
    if basic_anims:
        p["basic_animations"] = basic_anims
        p["outputs"]["basic_animations"] = basic_anims

    if rig_task_id in p.get("jobs", {}):
        p["jobs"][rig_task_id]["status"] = "done"
        p["jobs"][rig_task_id]["mesh_path"] = out_rpath
        if poll_res.get("consumed_credits") is not None:
            p["jobs"][rig_task_id]["consumed_credits"] = poll_res["consumed_credits"]
    state.save(p)

    return ok(
        {
            "status": "done",
            "job_id": rig_task_id,
            "mesh_path": out_rpath,
            "bones": bones,
            "unweighted_ratio": 0.0,
            "rig_task_id": rig_task_id,
            "method": "provider",
            "rig_source": "provider",
            "rig_input_mode": rig_source,
            **({"basic_animations": basic_anims} if basic_anims else {}),
            **({"consumed_credits": poll_res["consumed_credits"]} if poll_res.get("consumed_credits") is not None else {}),
        },
        project_id,
        next_action=state.suggest_next(p),
    )


@tool_entry
def auto_rig(
    project_id: ProjectId,
    mesh_path: MeshPath,
    method: Annotated[RigMethod, Field(description="Rigging method: auto, blender, or provider")] = "auto",
    job_id: Annotated[str | None, Field(description="Previous rigging job_id if status was running")] = None,
) -> dict:
    """Add a humanoid armature and skin weights to a character mesh via Meshy API or local Blender."""
    has_meshy_key = bool(os.environ.get("MESHY_API_KEY"))
    if method == "provider" and not has_meshy_key:
        raise CFError(
            "NO_PROVIDER",
            "No provider API key configured for rigging.",
            "Use method='blender' or method='auto'",
        )

    p = state.load(project_id)
    pdir = state.project_dir(project_id)
    mesh, rpath = state.mesh_path(p, mesh_path)

    if "jobs" not in p:
        p["jobs"] = {}

    # 1. Polling existing provider rigging job (non-blocking)
    if job_id:
        job_info = p["jobs"].get(job_id)
        if not job_info:
            raise CFError(
                "PROVIDER_FAILED",
                f"Rigging job {job_id!r} not found in project jobs.",
                "Call auto_rig without job_id to start a new job.",
            )
        submitted_at = job_info.get("submitted_at", time.time())
        if time.time() - submitted_at > 1200:
            raise CFError(
                "PROVIDER_TIMEOUT",
                f"Rigging job {job_id} exceeded 20 minutes.",
                "Use method='blender'",
            )
        prov = providers.get_provider(job_info.get("provider", "meshy"))
        poll_res = _poll_provider(prov, job_id, kind=job_info.get("kind", "rigging"))
        status = poll_res.get("status")
        rig_source = job_info.get("rig_source", "input_task_id")

        if status == "running":
            return ok(
                {
                    "status": "running",
                    "job_id": job_id,
                    "rig_task_id": job_id,
                    "method": "provider",
                    "rig_source": "provider",
                    "rig_input_mode": rig_source,
                    "mesh_path": None,
                },
                project_id,
                action(
                    "auto_rig",
                    "Rigging job in progress; poll again.",
                    project_id=project_id,
                    mesh_path=rpath,
                    method="provider",
                    job_id=job_id,
                ),
            )
        elif status == "done":
            return _finish_provider_rig(
                p=p,
                pdir=pdir,
                project_id=project_id,
                rpath=rpath,
                rig_task_id=job_id,
                poll_res=poll_res,
                rig_source=rig_source,
                input_task_id=job_info.get("input_task_id"),
            )
        else:
            job_info["status"] = "failed"
            state.save(p)
            raise CFError(
                "PROVIDER_FAILED",
                f"Meshy rigging failed: {poll_res.get('error', 'unknown error')}",
                "Use method='blender'",
            )

    task_id = _find_provider_task_id(p, rpath)

    chosen_method = method
    if chosen_method == "auto":
        if has_meshy_key and task_id:
            chosen_method = "provider"
        else:
            chosen_method = "blender"

    if chosen_method == "provider":
        if not has_meshy_key:
            raise CFError(
                "NO_PROVIDER",
                "No provider API key configured for rigging.",
                "Use method='blender' or method='auto'",
            )
        if not task_id:
            raise CFError(
                "NO_PROVIDER",
                f"No upstream provider task ID found for mesh {rpath}.",
                "Use method='blender' or method='auto'",
            )

        height_cm = float(p.get("height_cm", 165.0))
        height_m = max(0.5, min(3.0, round(height_cm / 100.0, 2)))

        prov = providers.get_provider("meshy")
        rig_source = "input_task_id"
        try:
            rig_task_id = prov.submit_rig(input_task_id=task_id, height_meters=height_m)
        except CFError as e:
            # Fallback to model_url data URI if input_task_id is rejected (HTTP 400: unsupported task / >300k faces)
            if (
                "400" in e.message
                and mesh.suffix.lower() == ".glb"
                and mesh.is_file()
                and mesh.stat().st_size <= 100 * 1024 * 1024
                and "model_url" in inspect.signature(prov.submit_rig).parameters
            ):
                b64_glb = base64.b64encode(mesh.read_bytes()).decode("ascii")
                model_data_uri = f"data:model/gltf-binary;base64,{b64_glb}"
                tex_uri = None
                front_rel = (p.get("views") or {}).get("front", {}).get("image")
                if front_rel and (pdir / front_rel).is_file():
                    b64_tex = base64.b64encode((pdir / front_rel).read_bytes()).decode("ascii")
                    tex_uri = f"data:image/png;base64,{b64_tex}"
                rig_task_id = prov.submit_rig(
                    model_url=model_data_uri,
                    height_meters=height_m,
                    texture_image_url=tex_uri,
                )
                rig_source = "model_url"
            else:
                raise

        p["jobs"][rig_task_id] = {
            "provider": "meshy",
            "kind": "rigging",
            "submitted_at": time.time(),
            "status": "running",
            "input_task_id": task_id,
            "rig_source": rig_source,
            "source_mesh": rpath,
        }
        state.save(p)

        poll_res = _poll_provider(prov, rig_task_id, kind="rigging")
        status = poll_res.get("status")
        if status == "done":
            return _finish_provider_rig(
                p=p,
                pdir=pdir,
                project_id=project_id,
                rpath=rpath,
                rig_task_id=rig_task_id,
                poll_res=poll_res,
                rig_source=rig_source,
                input_task_id=task_id,
            )
        elif status == "failed":
            p["jobs"][rig_task_id]["status"] = "failed"
            state.save(p)
            raise CFError(
                "PROVIDER_FAILED",
                f"Meshy rigging failed: {poll_res.get('error', 'unknown error')}",
                "Use method='blender'",
            )
        else:
            return ok(
                {
                    "status": "running",
                    "job_id": rig_task_id,
                    "rig_task_id": rig_task_id,
                    "method": "provider",
                    "rig_source": "provider",
                    "rig_input_mode": rig_source,
                    "mesh_path": None,
                },
                project_id,
                action(
                    "auto_rig",
                    "Rigging job in progress; poll again.",
                    project_id=project_id,
                    mesh_path=rpath,
                    method="provider",
                    job_id=rig_task_id,
                ),
            )

    # Use Blender headless rigging (offline fallback)
    out_rpath = f"export/{project_id}-rigged.glb"
    out_path = pdir / out_rpath

    ref_path = pdir / "views" / "reference.json"
    neck_ratio = 0.15
    if ref_path.is_file():
        try:
            rdata = json.loads(ref_path.read_text())
            neck_ratio = float(rdata.get("neck_ratio", 0.15))
        except Exception:
            pass

    args: dict[str, str | float | Path] = {
        "mesh": mesh,
        "out": out_path,
        "neck_ratio": neck_ratio,
    }
    if ref_path.is_file():
        args["ref_json"] = ref_path

    r = blender.run("rig.py", args)

    state.register_mesh(p, out_rpath, source="rigged", derived_from=rpath)
    if "outputs" not in p:
        p["outputs"] = {}
    p["outputs"]["rigged"] = out_rpath
    p["stage"] = "rigged"
    state.save(p)

    return ok(
        {
            "status": "done",
            "mesh_path": out_rpath,
            "bones": r["bones"],
            "unweighted_ratio": r["unweighted_ratio"],
            "method": "blender",
            "rig_source": "blender",
        },
        project_id,
        next_action=state.suggest_next(p),
    )


def _finish_provider_animation(
    p: dict,
    pdir: Path,
    project_id: str,
    rpath: str,
    anim_task_id: str,
    poll_res: dict,
    deduped_actions: list[str],
    action_ids: list[int],
) -> dict:
    url = poll_res.get("url")
    if not url:
        raise CFError(
            "PROVIDER_FAILED",
            "Meshy animation succeeded but returned no animation_glb_url.",
            "Use method='blender'",
        )

    out_rpath = f"export/{project_id}-animated.glb"
    out_path = pdir / out_rpath
    prov = providers.get_provider("meshy")
    prov.download(url, out_path)

    # Rename animation tracks in the merged GLB to match requested action names
    try:
        rename_glb_animations(out_path, deduped_actions, out_path)
    except Exception as e:
        raise CFError(
            "ANIM_FAILED",
            f"Failed to rename animation tracks in downloaded GLB: {e}",
            "Use method='blender'",
        )

    # Programmatic verification: chunk 0 JSON header
    try:
        exported_clips = inspect_glb_animations(out_path)
    except Exception as e:
        raise CFError(
            "ANIM_FAILED",
            f"Failed to inspect exported GLB animations: {e}",
            "Check Blender animation exporter settings.",
        )

    if len(exported_clips) != len(deduped_actions):
        raise CFError(
            "ANIM_FAILED",
            f"Animation clip count mismatch: expected {len(deduped_actions)}, got {len(exported_clips)} ({exported_clips})",
            "Check Blender animation exporter settings.",
        )

    state.register_mesh(p, out_rpath, source="animated", derived_from=rpath, provider="meshy")
    if "outputs" not in p:
        p["outputs"] = {}
    p["outputs"]["animated"] = out_rpath
    p["stage"] = "animated"
    p["animations"] = [
        {"name": a, "action_id": action_ids[min(i, len(action_ids) - 1)] if action_ids else None}
        for i, a in enumerate(deduped_actions)
    ]
    if anim_task_id in p.get("jobs", {}):
        p["jobs"][anim_task_id]["status"] = "done"
        p["jobs"][anim_task_id]["mesh_path"] = out_rpath
        if poll_res.get("consumed_credits") is not None:
            p["jobs"][anim_task_id]["consumed_credits"] = poll_res["consumed_credits"]
    state.save(p)

    return ok(
        {
            "status": "done",
            "job_id": anim_task_id,
            "mesh_path": out_rpath,
            "actions": deduped_actions,
            "action_ids": action_ids,
            "clip_count": len(deduped_actions),
            "method": "provider",
            **({"consumed_credits": poll_res["consumed_credits"]} if poll_res.get("consumed_credits") is not None else {}),
        },
        project_id,
        next_action=state.suggest_next(p),
    )


@tool_entry
def animate(
    project_id: ProjectId,
    mesh_path: MeshPath | None = None,
    actions: list[str] | None = None,
    method: Annotated[AnimateMethod, Field(description="Animation method: auto, blender, or provider")] = "auto",
    job_id: Annotated[str | None, Field(description="Previous animation job_id if status was running")] = None,
) -> dict:
    """Generate multi-action skeletal animation clips into a single character GLB."""
    p = state.load(project_id)
    pdir = state.project_dir(project_id)

    if "jobs" not in p:
        p["jobs"] = {}

    if mesh_path is None:
        mesh_arg = p.get("outputs", {}).get("rigged")
        if not mesh_arg and job_id and job_id in p["jobs"]:
            mesh_arg = p["jobs"][job_id].get("source_mesh")
        if not mesh_arg:
            raise CFError(
                "NO_RIGGED_MESH",
                "No rigged mesh found in project outputs.",
                "Run auto_rig first, or provide mesh_path.",
            )
    else:
        mesh_arg = mesh_path

    mesh, rpath = state.mesh_path(p, mesh_arg)

    # Default actions: ["idle", "walk", "wave"]
    raw_actions = actions if actions is not None else ["idle", "walk", "wave"]
    if not raw_actions:
        raw_actions = ["idle", "walk", "wave"]

    # Deduplicate requested action names if duplicates are provided (e.g. ["walk", "walk"] -> ["walk", "walk_01"])
    deduped_actions: list[str] = []
    seen: dict[str, int] = {}
    for act in raw_actions:
        count = seen.get(act, 0)
        seen[act] = count + 1
        deduped_actions.append(f"{act}_{count:02d}" if count > 0 else act)

    # 1. Polling existing provider animation job (non-blocking)
    if job_id:
        job_info = p["jobs"].get(job_id)
        if not job_info:
            raise CFError(
                "PROVIDER_FAILED",
                f"Animation job {job_id!r} not found in project jobs.",
                "Call animate without job_id to start a new job.",
            )
        submitted_at = job_info.get("submitted_at", time.time())
        if time.time() - submitted_at > 1200:
            raise CFError(
                "PROVIDER_TIMEOUT",
                f"Animation job {job_id} exceeded 20 minutes.",
                "Use method='blender'",
            )
        prov = providers.get_provider(job_info.get("provider", "meshy"))
        poll_res = _poll_provider(prov, job_id, kind=job_info.get("kind", "animations"))
        status = poll_res.get("status")
        saved_actions = job_info.get("actions") or deduped_actions
        saved_action_ids = job_info.get("action_ids") or []

        if status == "running":
            return ok(
                {
                    "status": "running",
                    "job_id": job_id,
                    "method": "provider",
                    "actions": saved_actions,
                    "action_ids": saved_action_ids,
                    "mesh_path": None,
                },
                project_id,
                action(
                    "animate",
                    "Animation job in progress; poll again.",
                    project_id=project_id,
                    mesh_path=rpath,
                    actions=saved_actions,
                    method="provider",
                    job_id=job_id,
                ),
            )
        elif status == "done":
            return _finish_provider_animation(
                p=p,
                pdir=pdir,
                project_id=project_id,
                rpath=rpath,
                anim_task_id=job_id,
                poll_res=poll_res,
                deduped_actions=saved_actions,
                action_ids=saved_action_ids,
            )
        else:
            job_info["status"] = "failed"
            state.save(p)
            raise CFError(
                "PROVIDER_FAILED",
                f"Meshy animation failed: {poll_res.get('error', 'unknown error')}",
                "Use method='blender'",
            )

    has_meshy_key = bool(os.environ.get("MESHY_API_KEY"))
    rig_task_id = p.get("rig_task_id") or p.get("outputs", {}).get("rig_task_id")

    chosen_method = method
    if chosen_method == "auto":
        if has_meshy_key and rig_task_id:
            chosen_method = "provider"
        else:
            chosen_method = "blender"

    out_rpath = f"export/{project_id}-animated.glb"
    out_path = pdir / out_rpath

    if chosen_method == "provider":
        if not has_meshy_key:
            raise CFError(
                "NO_PROVIDER",
                "No provider API key configured for animation.",
                "Use method='blender' or method='auto'",
            )
        if not rig_task_id:
            raise CFError(
                "NO_PROVIDER",
                "No provider rig task ID found for animation.",
                "Run auto_rig(method='provider') first, or use method='blender'",
            )

        prov = providers.get_provider("meshy")
        library = prov.fetch_animation_library()
        resolved_ids = prov.resolve_action_ids(deduped_actions, library=library)
        unique_ids = list(dict.fromkeys(resolved_ids))
        anim_task_id = prov.submit_animation(rig_task_id=rig_task_id, action_ids=unique_ids)

        p["jobs"][anim_task_id] = {
            "provider": "meshy",
            "kind": "animations",
            "submitted_at": time.time(),
            "status": "running",
            "rig_task_id": rig_task_id,
            "actions": deduped_actions,
            "action_ids": unique_ids,
            "source_mesh": rpath,
        }
        state.save(p)

        poll_res = _poll_provider(prov, anim_task_id, kind="animations")
        status = poll_res.get("status")
        if status == "done":
            return _finish_provider_animation(
                p=p,
                pdir=pdir,
                project_id=project_id,
                rpath=rpath,
                anim_task_id=anim_task_id,
                poll_res=poll_res,
                deduped_actions=deduped_actions,
                action_ids=unique_ids,
            )
        elif status == "failed":
            p["jobs"][anim_task_id]["status"] = "failed"
            state.save(p)
            raise CFError(
                "PROVIDER_FAILED",
                f"Meshy animation failed: {poll_res.get('error', 'unknown error')}",
                "Use method='blender'",
            )
        else:
            return ok(
                {
                    "status": "running",
                    "job_id": anim_task_id,
                    "method": "provider",
                    "actions": deduped_actions,
                    "action_ids": unique_ids,
                    "mesh_path": None,
                },
                project_id,
                action(
                    "animate",
                    "Animation job in progress; poll again.",
                    project_id=project_id,
                    mesh_path=rpath,
                    actions=deduped_actions,
                    method="provider",
                    job_id=anim_task_id,
                ),
            )

    # Blender procedural fallback
    actions_arg = ",".join(deduped_actions)
    blender.run(
        "merge_animations.py",
        {
            "mesh": mesh,
            "out": out_path,
            "actions": actions_arg,
            "mode": "procedural",
        },
    )

    # Programmatic verification: chunk 0 JSON header
    try:
        exported_clips = inspect_glb_animations(out_path)
    except Exception as e:
        raise CFError(
            "ANIM_FAILED",
            f"Failed to inspect exported GLB animations: {e}",
            "Check Blender animation exporter settings.",
        )

    if len(exported_clips) != len(deduped_actions):
        raise CFError(
            "ANIM_FAILED",
            f"Animation clip count mismatch: expected {len(deduped_actions)}, got {len(exported_clips)} ({exported_clips})",
            "Check Blender animation exporter settings.",
        )

    state.register_mesh(p, out_rpath, source="animated", derived_from=rpath)
    if "outputs" not in p:
        p["outputs"] = {}
    p["outputs"]["animated"] = out_rpath
    p["stage"] = "animated"
    p["animations"] = [{"name": a} for a in deduped_actions]
    state.save(p)

    return ok(
        {
            "status": "done",
            "mesh_path": out_rpath,
            "actions": deduped_actions,
            "clip_count": len(deduped_actions),
            "method": "blender",
        },
        project_id,
        next_action=state.suggest_next(p),
    )


def register(mcp: FastMCP) -> None:
    """Register Level 3 tools onto FastMCP instance."""
    mcp.tool(annotations=LOCAL_WRITE)(auto_rig)
    mcp.tool(annotations=LOCAL_WRITE)(animate)
    mcp.tool(annotations=LOCAL_WRITE)(pipeline_run)
