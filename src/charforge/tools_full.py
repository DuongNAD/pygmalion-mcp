"""Level 3 tools: auto_rig, pipeline_run.

Registered into FastMCP server when CHARFORGE_LEVEL >= 3.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Literal

try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

import os
import struct

from . import blender, config, providers, state
from .envelope import CFError, action, ok, tool_entry
from .pipeline_run import pipeline_run

LOCAL_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

ProjectId = Annotated[str, Field(description="Id returned by project_create, e.g. 'aria'", pattern=r"^[a-z0-9][a-z0-9-]{0,47}$")]
MeshPath = Annotated[str, Field(description="Project-relative path from a previous result (e.g. 'export/aria.glb') or an absolute path")]
RigMethod = Literal["auto", "blender", "provider"]


def _find_provider_task_id(p: dict, rpath: str) -> str | None:
    """Find provider generation task ID associated with candidate mesh."""
    # 1. Direct mesh metadata check
    for m in p.get("meshes", []):
        if m.get("path") == rpath:
            if m.get("source") in ("blockout", "external", "fit", "optimized"):
                return None
            if m.get("task_id"):
                return str(m["task_id"])
            if m.get("job_id"):
                return str(m["job_id"])

    # 2. Check jobs recorded in project state with matching mesh path
    for jid, jinfo in p.get("jobs", {}).items():
        if isinstance(jinfo, dict):
            if jinfo.get("mesh_path") == rpath or jinfo.get("mesh") == rpath:
                return str(jid)

    # 3. If rpath filename contains the job ID
    for jid, jinfo in p.get("jobs", {}).items():
        if isinstance(jinfo, dict) and jinfo.get("provider") == "meshy":
            if jid in rpath:
                return str(jid)

    # 4. If this is the only mesh in the project and a meshy job exists
    meshy_jobs = [jid for jid, jinfo in p.get("jobs", {}).items() if isinstance(jinfo, dict) and jinfo.get("provider") == "meshy"]
    meshes = p.get("meshes", [])
    if len(meshes) == 1 and meshes[0].get("path") == rpath and len(meshy_jobs) == 1:
        if meshes[0].get("source") not in ("blockout", "external", "fit", "optimized") and meshes[0].get("source") == "meshy":
            return str(meshy_jobs[0])

    return None


@tool_entry
def auto_rig(
    project_id: ProjectId,
    mesh_path: MeshPath,
    method: Annotated[RigMethod, Field(description="Rigging method: auto, blender, or provider")] = "auto",
) -> dict:
    """Add a humanoid Mixamo armature (22 bones) and skin weights to a character mesh."""
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
        rig_task_id = prov.submit_rig(input_task_id=task_id, height_meters=height_m)

        poll_res = prov.poll(rig_task_id)
        t_start = time.time()
        while poll_res.get("status") == "running" and (time.time() - t_start) < 60:
            time.sleep(0.5)
            poll_res = prov.poll(rig_task_id)

        if poll_res.get("status") != "done" or not poll_res.get("url"):
            raise CFError(
                "PROVIDER_FAILED",
                f"Meshy rigging failed: {poll_res.get('error', 'unknown error')}",
                "Use method='blender'",
            )

        out_rpath = f"export/{project_id}-rigged.glb"
        out_path = pdir / out_rpath
        prov.download(poll_res["url"], out_path)

        if "outputs" not in p:
            p["outputs"] = {}
        p["outputs"]["rigged"] = out_rpath
        p["stage"] = "rigged"
        p["rig_task_id"] = rig_task_id
        state.save(p)

        return ok(
            {
                "mesh_path": out_rpath,
                "bones": 22,
                "unweighted_ratio": 0.0,
                "rig_task_id": rig_task_id,
                "method": "provider",
            },
            project_id,
            next_action=state.suggest_next(p),
        )

    # Use Blender headless rigging
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

    if "outputs" not in p:
        p["outputs"] = {}
    p["outputs"]["rigged"] = out_rpath
    p["stage"] = "rigged"
    state.save(p)

    return ok(
        {
            "mesh_path": out_rpath,
            "bones": r["bones"],
            "unweighted_ratio": r["unweighted_ratio"],
        },
        project_id,
        next_action=state.suggest_next(p),
    )


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


ActionName = Literal["idle", "walk", "wave"]
AnimateMethod = Literal["auto", "blender", "provider"]


@tool_entry
def animate(
    project_id: ProjectId,
    mesh_path: MeshPath | None = None,
    actions: list[str] | None = None,
    method: Annotated[AnimateMethod, Field(description="Animation method: auto, blender, or provider")] = "auto",
) -> dict:
    """Generate and merge multi-action skeletal animation clips into a single character GLB."""
    p = state.load(project_id)
    pdir = state.project_dir(project_id)

    # Resolves mesh_path: if None, use p["outputs"].get("rigged")
    if mesh_path is None:
        mesh_arg = p.get("outputs", {}).get("rigged")
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

    has_meshy_key = bool(os.environ.get("MESHY_API_KEY"))
    rig_task_id = p.get("rig_task_id")

    # Prioritization:
    # If method == "auto": if MESHY_API_KEY is present and project has rig_task_id, use method = "provider", else blender
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
        action_id_map = {"idle": 1, "walk": 2, "wave": 3}
        clip_paths: list[Path] = []
        temp_clips_dir = pdir / "meshes" / "anim_clips"
        temp_clips_dir.mkdir(parents=True, exist_ok=True)

        for idx, act in enumerate(deduped_actions):
            base_act = act.split("_")[0]
            act_id = action_id_map.get(base_act, idx + 1)
            anim_task_id = prov.submit_animation(rig_task_id=rig_task_id, action_id=act_id)

            poll_res = prov.poll(anim_task_id)
            t_start = time.time()
            while poll_res.get("status") == "running" and (time.time() - t_start) < 60:
                time.sleep(0.5)
                poll_res = prov.poll(anim_task_id)

            if poll_res.get("status") != "done" or not poll_res.get("url"):
                raise CFError(
                    "PROVIDER_FAILED",
                    f"Meshy animation failed for action '{act}': {poll_res.get('error', 'unknown error')}",
                    "Use method='blender'",
                )

            clip_file = temp_clips_dir / f"clip_{idx}_{act}.glb"
            prov.download(poll_res["url"], clip_file)
            clip_paths.append(clip_file)

        clips_arg = ",".join(str(c) for c in clip_paths)
        actions_arg = ",".join(deduped_actions)
        blender.run(
            "merge_animations.py",
            {
                "mesh": mesh,
                "out": out_path,
                "actions": actions_arg,
                "mode": "merge",
                "clips": clips_arg,
            },
        )
    else:
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

    # Update state
    if "outputs" not in p:
        p["outputs"] = {}
    p["outputs"]["animated"] = out_rpath
    p["stage"] = "animated"
    p["animations"] = [{"name": a} for a in deduped_actions]
    state.save(p)

    return ok(
        {
            "mesh_path": out_rpath,
            "actions": deduped_actions,
            "clip_count": len(deduped_actions),
            "method": chosen_method,
        },
        project_id,
        next_action=state.suggest_next(p),
    )


def register(mcp: FastMCP) -> None:
    """Register Level 3 tools onto FastMCP instance."""
    mcp.tool(annotations=LOCAL_WRITE)(auto_rig)
    mcp.tool(annotations=LOCAL_WRITE)(animate)
    mcp.tool(annotations=LOCAL_WRITE)(pipeline_run)
