"""Level 3 tools: auto_rig, animate, pipeline_run.

Registered into FastMCP server when CHARFORGE_LEVEL >= 3.

Quality gates (thresholds in config.py, per-project overrides in project.json):
- auto_rig runs a pose deformation check on every rig (provider or local): swing each upper arm and thigh 40 deg
  and measure whether parts that must stay still are dragged along. A failing rig gets its weights repaired and
  re-checked; the repair is kept only when it helps. A humanoid bone map is written for retargeting.
- animate renders a few frames of every clip from the front and side (renders/anim/<clip>.png) and measures
  surface stretch and floor penetration; it can add secondary motion (spring chains for hair and skirts)
  and Text to Motion clips (Meshy) to the preset clips.
"""
from __future__ import annotations

import base64
import inspect
import json
import os
import re
import shutil
import struct
import time
import uuid
from pathlib import Path
from typing import Annotated, Literal

try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from . import blender, config, critic, providers, score, state
from .envelope import CFError, action, ok, tool_entry
from .pipeline_run import pipeline_run

LOCAL_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

ProjectId = Annotated[str, Field(description="Id returned by project_create, e.g. 'aria'", pattern=r"^[a-z0-9][a-z0-9-]{0,47}$")]
MeshPath = Annotated[str, Field(description="Project-relative path from a previous result (e.g. 'export/aria.glb') or an absolute path")]
RigMethod = Literal["auto", "blender", "provider"]
ActionName = Literal["idle", "walk", "wave"]
AnimateMethod = Literal["auto", "blender", "provider"]
Secondary = Literal["auto", "off"]
DEFAULT_ACTIONS = ["idle", "walk", "wave"]


class MotionSpec(BaseModel):
    """A custom clip made by Meshy Text to Motion (needs a Meshy rig)."""

    prompt: Annotated[str, Field(min_length=1, max_length=400, description="What the character does, e.g. 'spins once with arms out, then bows'")]
    name: Annotated[str | None, Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,40}$", description="Clip name; default motion_1, motion_2, ...")] = None
    duration: Annotated[float, Field(default=4.0, ge=2.0, le=10.0, description="Seconds, 2-10 in 0.5 steps")] = 4.0


def _find_provider_task_id(p: dict, rpath: str) -> str | None:
    """Find upstream provider generation task ID by walking the derived_from lineage chain."""
    return state.find_provider_task_id(p, rpath)


def _poll_provider(prov: providers.Provider, job_id: str, kind: str | None = None) -> dict:
    sig = inspect.signature(prov.poll)
    if "kind" in sig.parameters:
        return prov.poll(job_id, kind=kind)
    return prov.poll(job_id)


# ---------------------------------------------------------------- GLB helpers (no Blender)

def _glb_json(glb_input: Path | str | bytes) -> dict:
    """Parse the glTF JSON chunk 0 of a GLB; raises ValueError on malformed input."""
    glb_bytes = Path(glb_input).read_bytes() if isinstance(glb_input, (str, Path)) else glb_input
    if len(glb_bytes) < 20:
        raise ValueError("Invalid GLB: file too small")
    magic, version, _ = struct.unpack("<4sII", glb_bytes[:12])
    if magic != b"glTF" or version != 2:
        raise ValueError(f"Invalid GLB header: magic={magic!r}, version={version}")
    chunk_length, chunk_type = struct.unpack("<I4s", glb_bytes[12:20])
    if chunk_type != b"JSON":
        raise ValueError(f"Expected JSON chunk 0, got {chunk_type!r}")
    return json.loads(glb_bytes[20 : 20 + chunk_length].decode("utf-8"))


def inspect_glb_bones(glb_input: Path | str | bytes) -> int:
    """Count skeletal joints from glTF binary chunk 0 JSON `skins` (or nodes if skins absent)."""
    try:
        gltf_obj = _glb_json(glb_input)
        joints: set[int] = set()
        for skin in gltf_obj.get("skins") or []:
            joints.update(int(j) for j in skin.get("joints") or [])
        if joints:
            return len(joints)
        return len(gltf_obj.get("nodes") or [])
    except Exception:
        return 0


def inspect_glb_animations(glb_input: Path | str | bytes) -> list[str]:
    """Parse glTF binary chunk 0 JSON header to extract animation names."""
    animations = _glb_json(glb_input).get("animations", [])
    return [a.get("name", f"action_{i}") for i, a in enumerate(animations)]


def glb_has_skinned_mesh(path: Path) -> bool:
    """True when the GLB holds a mesh bound to a skin (something Blender QA can deform and render)."""
    try:
        g = _glb_json(path)
    except Exception:
        return False
    return bool(g.get("skins")) and any("skin" in n and "mesh" in n for n in g.get("nodes") or [])


def _write_glb_json(glb_path: Path, gltf_obj: dict, out_path: Path) -> Path:
    """Replace the JSON chunk of a GLB, keeping the binary chunk byte for byte."""
    glb_bytes = Path(glb_path).read_bytes()
    chunk0_len = struct.unpack("<I", glb_bytes[12:16])[0]
    new_json_bytes = json.dumps(gltf_obj, ensure_ascii=False).encode("utf-8")
    new_json_bytes += b" " * ((4 - (len(new_json_bytes) % 4)) % 4)
    rest = glb_bytes[20 + chunk0_len :]
    total_len = 12 + 8 + len(new_json_bytes) + len(rest)
    new_glb = struct.pack("<4sII", b"glTF", 2, total_len) + struct.pack("<I4s", len(new_json_bytes), b"JSON") + new_json_bytes + rest
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(new_glb)
    return out_path


def rename_glb_animations(
    glb_path: Path, action_names: list[str], out_path: Path | None = None, sources: list[int] | None = None
) -> Path:
    """Losslessly rename animation tracks inside a GLB's JSON chunk 0 to match `action_names`.
    `sources[i]` is the index of the file animation that becomes action_names[i] (default: same position);
    out-of-range indices reuse the last animation."""
    gltf_obj = _glb_json(glb_path)
    anims = gltf_obj.get("animations") or []
    if not anims:
        raise ValueError("GLB contains no animation tracks to rename.")
    new_anims = []
    for i, target_name in enumerate(action_names):
        src = sources[i] if sources and i < len(sources) else i
        anim_copy = dict(anims[min(src, len(anims) - 1)])
        anim_copy["name"] = target_name
        new_anims.append(anim_copy)
    gltf_obj["animations"] = new_anims
    return _write_glb_json(glb_path, gltf_obj, Path(out_path) if out_path else Path(glb_path))


def reorder_glb_animations(glb_path: Path, names: list[str]) -> None:
    """Put the GLB's animations in the order of `names` (the glTF exporter writes them alphabetically).
    Leaves the file untouched when the names do not match one to one."""
    gltf_obj = _glb_json(glb_path)
    anims = gltf_obj.get("animations") or []
    if any(not isinstance(a.get("name"), str) for a in anims):
        return
    by_name = {a["name"]: a for a in anims}
    if len(by_name) != len(anims) or sorted(by_name) != sorted(names) or [a["name"] for a in anims] == names:
        return
    gltf_obj["animations"] = [by_name[n] for n in names]
    _write_glb_json(glb_path, gltf_obj, Path(glb_path))


# ---------------------------------------------------------------- rig QA

def _summarize_rig_qa(r: dict) -> dict:
    """Compact rig QA for tool results and project.json; the full report is in renders/qa/rig_check.json."""
    final = (r.get("after") if r.get("repaired") else r.get("before")) or {}
    tests = {}
    for t in final.get("tests") or []:
        if "skipped" in t:
            tests[t["name"]] = {"skipped": t["skipped"]}
            continue
        regions = t.get("regions") or {}
        worst = max(regions, key=lambda k: regions[k]["max_m"]) if regions else None
        item = {"passed": t["passed"], "follow": t.get("follow")}
        if worst:
            item.update(worst_region=worst, max_cm=round(100 * regions[worst]["max_m"], 1), leak_pct=round(100 * regions[worst]["leak_ratio"], 1))
        tests[t["name"]] = item
    contacts = {
        k: {
            "gap_cm": None if v.get("gap_m") is None else round(100 * v["gap_m"], 1),
            "touching": v.get("separated") is False,
            **({"joined_by_cloth": True} if v.get("bridged_by_cloth") else {}),
        }
        for k, v in (r.get("contacts") or {}).items()
    }
    mapping = r.get("mapping") or {}
    out = {
        "passed": r.get("passed"),
        "repaired": bool(r.get("repaired")),
        "issues": r.get("issues") or [],
        "tests": tests,
        "contacts": contacts,
        "skeleton_mapped": mapping.get("ok"),
        "missing_roles": mapping.get("missing") or [],
    }
    if r.get("repaired"):
        out["issues_before_repair"] = (r.get("before") or {}).get("issues") or []
    return out


def _rig_qa_warnings(qa: dict) -> list[str]:
    if qa.get("passed") is None:
        return ["Rig QA did not run: " + "; ".join(qa.get("issues") or ["unknown reason"])]
    out = []
    if qa.get("repaired"):
        out.append(f"Rig QA found weight leaks and repaired them ({len(qa.get('issues_before_repair') or [])} issue(s) before repair).")
    if not qa.get("passed"):
        out.append("Rig QA failed after repair: " + "; ".join((qa.get("issues") or [])[:3]))
    elif qa.get("issues"):
        out.append("Rig QA passed with notes: " + "; ".join(qa["issues"][:2]))
    return out


def _store_rig_qa(p: dict, pdir: Path, r: dict) -> dict:
    qa_dir = pdir / "renders" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "rig_check.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    qa = _summarize_rig_qa(r)
    qa["report"] = state.rel(p, qa_dir / "rig_check.json")
    p.setdefault("qa", {})["rig"] = qa
    return qa


def _provider_rig_qa(p: dict, pdir: Path, project_id: str, rig_rpath: str) -> tuple[dict, list[str]]:
    """rig_check.py on a downloaded provider rig; a repaired rig replaces it and the original is kept in meshes/."""
    rig_path = pdir / rig_rpath
    if not glb_has_skinned_mesh(rig_path):
        qa = {"passed": None, "issues": ["the rigged GLB has no skinned mesh to deform"]}
        p.setdefault("qa", {})["rig"] = qa
        return qa, _rig_qa_warnings(qa)
    fixed = pdir / "meshes" / f"{project_id}-rigged-fixed.glb"
    humanoid_rpath = f"export/{project_id}-humanoid.json"
    args = {"mesh": rig_path, "repair": True, "out": fixed, "humanoid": pdir / humanoid_rpath, **state.tunables(p, "rig_check_thresholds", config.RIG_CHECK)}
    try:
        r = blender.run("rig_check.py", args)
    except CFError as e:
        qa = {"passed": None, "issues": [f"rig check could not run: {e.message}"]}
        p.setdefault("qa", {})["rig"] = qa
        return qa, _rig_qa_warnings(qa)
    if r.get("repaired") and fixed.is_file():
        raw_rpath = f"meshes/{project_id}-rigged-provider.glb"
        os.replace(rig_path, pdir / raw_rpath)
        os.replace(fixed, rig_path)
        p["outputs"]["rigged_provider"] = raw_rpath
        p["outputs"]["rig_repaired"] = True
    if (pdir / humanoid_rpath).is_file():
        p["outputs"]["humanoid"] = humanoid_rpath
    qa = _store_rig_qa(p, pdir, r)
    return qa, _rig_qa_warnings(qa)


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
    p["outputs"].pop("rig_repaired", None)
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
    qa, warnings = _provider_rig_qa(p, pdir, project_id, out_rpath)
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
            "rig_check": qa,
            "humanoid_map": p["outputs"].get("humanoid"),
            **({"basic_animations": basic_anims} if basic_anims else {}),
            **({"consumed_credits": poll_res["consumed_credits"]} if poll_res.get("consumed_credits") is not None else {}),
        },
        project_id,
        next_action=state.suggest_next(p),
        warnings=warnings,
    )


@tool_entry
def auto_rig(
    project_id: ProjectId,
    mesh_path: MeshPath,
    method: Annotated[RigMethod, Field(description="Rigging method: auto, blender, or provider")] = "auto",
    job_id: Annotated[str | None, Field(description="Previous rigging job_id if status was running")] = None,
) -> dict:
    """Add a humanoid armature and skin weights (Meshy API or local Blender), then check how it deforms:
    data.rig_check reports weight leaks found by swinging arms and legs (repaired automatically when possible)."""
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
    humanoid_rpath = f"export/{project_id}-humanoid.json"

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
        "humanoid": pdir / humanoid_rpath,
        **state.tunables(p, "rig_check_thresholds", config.RIG_CHECK),
    }
    if ref_path.is_file():
        args["ref_json"] = ref_path

    r = blender.run("rig.py", args)

    state.register_mesh(p, out_rpath, source="rigged", derived_from=rpath)
    if "outputs" not in p:
        p["outputs"] = {}
    p["outputs"]["rigged"] = out_rpath
    p["outputs"].pop("rig_repaired", None)
    if (pdir / humanoid_rpath).is_file():
        p["outputs"]["humanoid"] = humanoid_rpath
    p["stage"] = "rigged"
    if r.get("rig_check"):
        qa = _store_rig_qa(p, pdir, r["rig_check"])
    else:
        qa = {"passed": None, "issues": ["rig.py returned no rig_check"]}
    state.save(p)

    return ok(
        {
            "status": "done",
            "mesh_path": out_rpath,
            "bones": r["bones"],
            "unweighted_ratio": r["unweighted_ratio"],
            "method": "blender",
            "rig_source": "blender",
            "rig_check": qa,
            "humanoid_map": p["outputs"].get("humanoid"),
        },
        project_id,
        next_action=state.suggest_next(p),
        warnings=_rig_qa_warnings(qa) if r.get("rig_check") else [],
    )


# ---------------------------------------------------------------- animation post-processing and QA

def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)[:60] or "clip"


def _post_animation(
    p: dict,
    pdir: Path,
    project_id: str,
    base: Path,
    names: list[str],
    clips: list[Path],
    secondary: str,
) -> tuple[str, dict, list[str]]:
    """Make export/<id>-animated.glb from a provider GLB: merge extra clips, copy repaired weights, add secondary
    motion. Blender is skipped when there is nothing to do; any failure keeps the provider file."""
    out_rpath = f"export/{project_id}-animated.glb"
    out = pdir / out_rpath
    warnings: list[str] = []
    info: dict = {}
    outputs = p.get("outputs") or {}
    weights_from = pdir / outputs["rigged"] if outputs.get("rig_repaired") and outputs.get("rigged") else None
    if weights_from is not None and not weights_from.is_file():
        weights_from = None
    changed = False
    if (clips or weights_from or secondary == "auto") and glb_has_skinned_mesh(base):
        args = {
            "mesh": base,
            "out": out,
            "actions": ",".join(names),
            "mode": "post",
            "clips": ",".join(str(c) for c in clips),
            "fps": 30,
            "secondary": secondary,
            "weights_from": weights_from,
            **_secondary_args(p),
        }
        try:
            r = blender.run("merge_animations.py", args)
            changed = bool(r.get("changed"))
            info = {k: r[k] for k in ("secondary", "weights") if r.get(k) is not None}
            if (info.get("weights") or {}).get("applied") is False:
                warnings.append(f"Repaired rig weights were not copied to the animations: {info['weights'].get('reason')}")
        except CFError as e:
            if clips:
                raise CFError("ANIM_FAILED", f"Could not merge Text to Motion clips: {e.message}", "Retry animate once; if it fails again drop motions.")
            warnings.append(f"Animation post-processing failed ({e.message}); kept the provider file as is.")
    elif clips:
        raise CFError("ANIM_FAILED", "Text to Motion clips need a skinned character GLB to merge into.", "Drop motions or rerun auto_rig.")
    if not changed:
        shutil.copyfile(base, out)
    return out_rpath, info, warnings


def _secondary_args(p: dict) -> dict:
    s = state.tunables(p, "secondary_params", config.SECONDARY)
    return {k: s[k] for k in ("stiffness", "drag", "gravity", "skirt_chains", "max_hair_chains")}


def _clip_qa(p: dict, pdir: Path, anim_rpath: str) -> tuple[dict, dict, list[str]]:
    """Render front/side frames of every clip into renders/anim/<clip>.png and grade the motion."""
    anim = pdir / anim_rpath
    if not glb_has_skinned_mesh(anim):
        mc = {"passed": None, "clips": {}, "issues": ["the animated GLB has no skinned mesh to render"]}
        p.setdefault("qa", {})["motion"] = mc
        return {}, mc, []
    out_dir = pdir / "renders" / "anim"
    frames_dir = out_dir / "frames"
    cfg = state.tunables(p, "anim_preview", config.ANIM_PREVIEW)
    try:
        r = blender.run("preview_clips.py", {"mesh": anim, "out": frames_dir, "frames": cfg["frames"], "res": cfg["res"], "views": cfg["views"]})
    except CFError as e:
        mc = {"passed": None, "clips": {}, "issues": [f"previews failed: {e.message}"]}
        p.setdefault("qa", {})["motion"] = mc
        return {}, mc, [f"Clip previews failed: {e.message}"]
    th = state.tunables(p, "anim_check_thresholds", config.ANIM_CHECK)
    previews, clips, issues = {}, {}, []
    for c in r.get("clips") or []:
        m = c["metrics"]
        title = f"{c['name']}  {c.get('seconds')} s  stretch p99 {m['stretch_p99']}  floor {m['ground_cm']} cm"
        sheet = score.clip_sheet(c["files"], c["frames"], title, out_dir / f"{_safe_name(c['name'])}.png")
        previews[c["name"]] = state.rel(p, sheet)
        clips[c["name"]] = m
        if m["stretch_p99"] > th["stretch_p99"]:
            issues.append(f"{c['name']}: surface stretches {m['stretch_p99']}x (p99): weights tear or balloon the mesh")
        if m["ground_cm"] > th["ground_cm"]:
            issues.append(f"{c['name']}: feet sink {m['ground_cm']} cm below the floor")
    shutil.rmtree(frames_dir, ignore_errors=True)
    mc = {"passed": not issues, "clips": clips, "issues": issues}
    warnings = ["Motion QA: " + "; ".join(issues[:3])] if issues else []
    if previews:
        mc["critic"], w = _motion_critic(p, pdir, previews)
        warnings += w
    p.setdefault("qa", {})["motion"] = mc
    p.setdefault("outputs", {})["previews"] = previews
    return previews, mc, warnings


def _motion_critic(p: dict, pdir: Path, previews: dict[str, str]) -> tuple[dict, list[str]]:
    """Independent visual critic on the clip sheets (advisory: never changes motion_check.passed)."""
    try:
        res = critic.critique_motion(p, {name: pdir / path for name, path in previews.items()})
    except critic.CriticUnavailable as e:
        return {"skipped": str(e)}, ([f"Visual critic skipped: {e}"] if critic.settings(p)["mode"] == "on" else [])
    except critic.CriticFailed as e:
        return {"error": str(e)}, [f"Visual critic failed: {e}"]
    low = critic.settings(p)["warn_below"]
    warnings = [
        f"Visual critic: {name} {c['overall']}/10; " + "; ".join(c["issues"][:2])
        for name, c in res["clips"].items() if c["overall"] < low
    ]
    return res, warnings


def _finalize_animation(
    p: dict,
    pdir: Path,
    project_id: str,
    rpath: str,
    out_rpath: str,
    names: list[str],
    method: str,
    extra: dict,
    warnings: list[str],
    provider: str | None = None,
) -> dict:
    out_path = pdir / out_rpath
    try:
        reorder_glb_animations(out_path, names)  # Blender writes clips alphabetically; keep the requested order
        exported_clips = inspect_glb_animations(out_path)
    except Exception as e:
        raise CFError("ANIM_FAILED", f"Failed to inspect exported GLB animations: {e}", "Check Blender animation exporter settings.")
    if len(exported_clips) != len(names):
        raise CFError(
            "ANIM_FAILED",
            f"Animation clip count mismatch: expected {len(names)}, got {len(exported_clips)} ({exported_clips})",
            "Check Blender animation exporter settings.",
        )
    if exported_clips != names:
        raise CFError(
            "ANIM_FAILED",
            f"Animation clip names mismatch: expected {names}, got {exported_clips}",
            "Retry animate once; if it fails again report this error.",
        )
    state.register_mesh(p, out_rpath, source="animated", derived_from=rpath, **({"provider": provider} if provider else {}))
    p.setdefault("outputs", {})["animated"] = out_rpath
    p["stage"] = "animated"
    previews, motion_check, w = _clip_qa(p, pdir, out_rpath)
    state.save(p)
    secondary = extra.pop("secondary_info", None)
    return ok(
        {
            "status": "done",
            "mesh_path": out_rpath,
            "actions": names,
            "clip_count": len(names),
            "method": method,
            "previews": previews,
            "motion_check": motion_check,
            "secondary": _secondary_summary(secondary),
            **extra,
        },
        project_id,
        next_action=state.suggest_next(p),
        warnings=warnings + w,
    )


def _secondary_summary(info: dict | None) -> dict:
    if not info:
        return {"applied": False}
    kinds: dict[str, int] = {}
    for c in info.get("chains") or []:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    out = {"applied": bool(info.get("applied")), "chains": kinds}
    for k in ("reason", "stretch_before", "stretch_after", "skirt_in_legs_before", "skirt_in_legs_after"):
        if info.get(k) is not None:
            out[k] = info[k]
    return out


def _finish_provider_animation(
    p: dict,
    pdir: Path,
    project_id: str,
    rpath: str,
    anim_task_id: str,
    poll_res: dict,
    deduped_actions: list[str],
    action_ids: list[int],
    secondary: str = "auto",
    name_action_ids: list[int] | None = None,
) -> dict:
    url = poll_res.get("url")
    if not url:
        raise CFError(
            "PROVIDER_FAILED",
            "Meshy animation succeeded but returned no animation_glb_url.",
            "Use method='blender'",
        )

    raw = pdir / "meshes" / f"{project_id}-anim-provider.glb"
    prov = providers.get_provider("meshy")
    prov.download(url, raw)

    # Rename animation tracks in the merged GLB to match requested action names
    _rename_presets(raw, deduped_actions, action_ids, name_action_ids)

    out_rpath, info, warnings = _post_animation(p, pdir, project_id, raw, deduped_actions, [], secondary)
    p["animations"] = _preset_records(deduped_actions, action_ids, name_action_ids)
    if anim_task_id in p.get("jobs", {}):
        p["jobs"][anim_task_id]["status"] = "done"
        p["jobs"][anim_task_id]["mesh_path"] = out_rpath
        if poll_res.get("consumed_credits") is not None:
            p["jobs"][anim_task_id]["consumed_credits"] = poll_res["consumed_credits"]
    extra = {
        "job_id": anim_task_id,
        "action_ids": action_ids,
        "secondary_info": info.get("secondary"),
        **({"consumed_credits": poll_res["consumed_credits"]} if poll_res.get("consumed_credits") is not None else {}),
    }
    return _finalize_animation(p, pdir, project_id, rpath, out_rpath, deduped_actions, "provider", extra, warnings, provider="meshy")


def _rename_presets(path: Path, names: list[str], unique_ids: list[int], name_ids: list[int] | None) -> None:
    """Name the clips of a Meshy multi-action GLB. Meshy returns one clip per unique action_id, in request order;
    two names that resolved to the same id share its clip (['walk', 'walk_01', 'idle'] -> clips W, W, I)."""
    sources = None
    if name_ids and unique_ids and len(name_ids) == len(names):
        sources = [unique_ids.index(x) if x in unique_ids else i for i, x in enumerate(name_ids)]
    try:
        rename_glb_animations(path, names, path, sources=sources)
    except Exception as e:
        raise CFError("ANIM_FAILED", f"Failed to rename animation tracks in downloaded GLB: {e}", "Use method='blender'")


def _preset_records(names: list[str], unique_ids: list[int], name_ids: list[int] | None) -> list[dict]:
    if name_ids and len(name_ids) == len(names):
        return [{"name": a, "action_id": name_ids[i]} for i, a in enumerate(names)]
    return [{"name": a, "action_id": unique_ids[min(i, len(unique_ids) - 1)] if unique_ids else None} for i, a in enumerate(names)]


def _reusable_motion(p: dict, rig_task_id: str, spec: dict) -> dict | None:
    """A Text to Motion task already paid for in an earlier group (same rig, prompt and duration, < 3 days old,
    not failed): resume it instead of buying it again. Animation results are re-polled for fresh URLs."""
    for job in p.get("jobs", {}).values():
        if job.get("kind") != "animate-group" or job.get("rig_task_id") != rig_task_id:
            continue
        if time.time() - job.get("submitted_at", 0) > 3 * 24 * 3600:
            continue
        for m in (job.get("parts") or {}).get("motions") or []:
            if m.get("prompt") != spec["prompt"] or m.get("duration") != spec["duration"] or not m.get("motion_task_id"):
                continue
            if m.get("status") == "failed":
                continue
            if m.get("anim_task_id"):
                return {"motion_task_id": m["motion_task_id"], "anim_task_id": m["anim_task_id"], "status": "anim"}
            return {"motion_task_id": m["motion_task_id"], "status": "motion"}
    return None


def _dedupe(names: list[str]) -> list[str]:
    """['walk', 'walk', 'idle'] -> ['walk', 'walk_01', 'idle'] (unique clip names)."""
    out: list[str] = []
    seen: dict[str, int] = {}
    for act in names:
        count = seen.get(act, 0)
        seen[act] = count + 1
        out.append(f"{act}_{count:02d}" if count > 0 else act)
    return out


def _motion_specs(motions: list | None, taken: list[str]) -> list[dict]:
    specs = []
    used = set(taken)
    for i, m in enumerate(motions or []):
        d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
        prompt = str(d.get("prompt") or "").strip()
        if not prompt:
            raise CFError("ANIM_FAILED", f"motions[{i}] has an empty prompt.", "Give every motion a short prompt, e.g. 'jumps with joy'.")
        name = d.get("name") or f"motion_{i + 1}"
        base, k = name, 1
        while name in used:
            name, k = f"{base}_{k:02d}", k + 1
        used.add(name)
        dur = float(d.get("duration") or config.MESHY_MOTION_DURATION)
        specs.append({"name": name, "prompt": prompt[:400], "duration": max(2.0, min(10.0, round(dur * 2) / 2))})
    return specs


def _advance_group(p: dict, pdir: Path, project_id: str, rpath: str, gid: str, prov: providers.Provider) -> dict | None:
    """Poll every part of a Text to Motion job group once; start motion animations when their motion is ready.
    Returns the finished tool result, or None while parts are still running."""
    g = p["jobs"][gid]
    parts = g["parts"]
    pre = parts.get("presets")

    def credit(r: dict) -> None:
        g["consumed_credits"] = (g.get("consumed_credits") or 0) + (r.get("consumed_credits") or 0)

    def need_url(r: dict, what: str) -> str:
        if not r.get("url"):
            raise CFError("PROVIDER_FAILED", f"Meshy {what} succeeded but returned no download URL.", "Retry animate with the same job_id once; if it repeats drop motions.")
        return r["url"]

    try:  # every state change is saved, even when a later poll raises: paid tasks are never lost or re-bought
        if pre and pre["status"] == "running":
            r = _poll_provider(prov, pre["id"], kind="animations")
            if r.get("status") == "done":
                pre.update(status="done", url=need_url(r, "preset animation"))
                credit(r)
            elif r.get("status") == "failed":
                g["status"] = pre["status"] = "failed"
                raise CFError("PROVIDER_FAILED", f"Meshy animation failed: {r.get('error', 'unknown error')}", "Use method='blender'")
        for m in parts["motions"]:
            if m["status"] == "motion":
                r = _poll_provider(prov, m["motion_task_id"], kind="text-to-motion")
                if r.get("status") == "done":
                    credit(r)
                    m["anim_task_id"] = prov.submit_animation(rig_task_id=g["rig_task_id"], motion_task_id=m["motion_task_id"])
                    m["status"] = "anim"
                    state.save(p)
                elif r.get("status") == "failed":
                    g["status"] = m["status"] = "failed"
                    raise CFError(
                        "PROVIDER_FAILED",
                        f"Meshy Text to Motion failed for {m['name']!r}: {r.get('error', 'unknown error')}",
                        "Rephrase that motion prompt or drop it from motions.",
                    )
            if m["status"] == "anim":
                r = _poll_provider(prov, m["anim_task_id"], kind="animations")
                if r.get("status") == "done":
                    m.update(status="done", url=need_url(r, f"animation for motion {m['name']!r}"))
                    credit(r)
                elif r.get("status") == "failed":
                    g["status"] = m["status"] = "failed"
                    raise CFError("PROVIDER_FAILED", f"Meshy animation failed for motion {m['name']!r}: {r.get('error', 'unknown error')}", "Drop that motion and retry.")
    finally:
        state.save(p)
    if (pre and pre["status"] != "done") or any(m["status"] != "done" for m in parts["motions"]):
        return None

    # all parts ready: download, rename, merge into one GLB
    files: list[Path] = []
    names: list[str] = []
    if pre:
        base = pdir / "meshes" / f"{project_id}-anim-provider.glb"
        prov.download(pre["url"], base)
        _rename_presets(base, g["actions"], g.get("action_ids") or [], g.get("name_action_ids"))
        files.append(base)
        names += g["actions"]
    for m in parts["motions"]:
        path = pdir / "meshes" / f"{project_id}-motion-{_safe_name(m['name'])}.glb"
        prov.download(m["url"], path)
        try:
            rename_glb_animations(path, [m["name"]], path)
        except ValueError:
            raise CFError("PROVIDER_FAILED", f"Motion clip {m['name']!r} came back without animation.", "Drop that motion and retry.")
        files.append(path)
        names.append(m["name"])
    out_rpath, info, warnings = _post_animation(p, pdir, project_id, files[0], names, files[1:], g.get("secondary", "auto"))
    p["animations"] = _preset_records(g["actions"], g.get("action_ids") or [], g.get("name_action_ids")) if pre else []
    p["animations"] += [{"name": m["name"], "prompt": m["prompt"], "motion_task_id": m["motion_task_id"]} for m in parts["motions"]]
    g["status"] = "done"
    g["mesh_path"] = out_rpath
    extra = {
        "job_id": gid,
        "action_ids": g.get("action_ids") or [],
        "motion_task_ids": {m["name"]: m["motion_task_id"] for m in parts["motions"]},
        "consumed_credits": g.get("consumed_credits") or 0,
        "secondary_info": info.get("secondary"),
    }
    return _finalize_animation(p, pdir, project_id, rpath, out_rpath, names, "provider", extra, warnings, provider="meshy")


def _group_running(project_id: str, rpath: str, gid: str, g: dict) -> dict:
    parts = g["parts"]
    return ok(
        {
            "status": "running",
            "job_id": gid,
            "method": "provider",
            "actions": g["actions"] + [m["name"] for m in parts["motions"]],
            "parts": {
                **({"presets": parts["presets"]["status"]} if parts.get("presets") else {}),
                **{m["name"]: m["status"] for m in parts["motions"]},
            },
            "mesh_path": None,
        },
        project_id,
        action("animate", "Animation and Text to Motion jobs in progress; poll again.", project_id=project_id, mesh_path=rpath, method="provider", job_id=gid),
    )


@tool_entry
def animate(
    project_id: ProjectId,
    mesh_path: MeshPath | None = None,
    actions: list[str] | None = None,
    motions: Annotated[
        list[MotionSpec] | None,
        Field(description="Custom clips from a text prompt (Meshy Text to Motion, needs a Meshy rig), merged after the preset actions"),
    ] = None,
    method: Annotated[AnimateMethod, Field(description="Animation method: auto, blender, or provider")] = "auto",
    secondary: Annotated[Secondary, Field(description="auto = add spring motion to hanging hair and skirts when found; off = keep the rig as is")] = "auto",
    job_id: Annotated[str | None, Field(description="Previous animation job_id if status was running")] = None,
) -> dict:
    """Make skeletal animation clips (Meshy presets / Text to Motion, or Blender procedural) in one GLB, then
    render front+side frames of every clip (data.previews) and grade the motion (data.motion_check)."""
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

    # Preset actions: default idle/walk/wave unless only custom motions were asked for
    raw_actions = actions if actions is not None else ([] if motions else list(DEFAULT_ACTIONS))
    if not raw_actions and not motions:
        raw_actions = list(DEFAULT_ACTIONS)
    deduped_actions = _dedupe(raw_actions)
    motion_specs = _motion_specs(motions, deduped_actions)

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
        if job_info.get("kind") == "animate-group":
            done = _advance_group(p, pdir, project_id, rpath, job_id, prov)
            return done if done is not None else _group_running(project_id, rpath, job_id, job_info)
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
                secondary=job_info.get("secondary", secondary),
                name_action_ids=job_info.get("name_action_ids"),
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

    if motion_specs and chosen_method != "provider":
        raise CFError(
            "NO_PROVIDER",
            "Text to Motion clips need Meshy: a MESHY_API_KEY and a rig made by auto_rig(method='provider').",
            "Call animate without motions (preset actions only), or rig with Meshy first.",
        )

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
        unique_ids: list[int] = []
        resolved_ids: list[int] = []
        anim_task_id = None
        if deduped_actions:
            library = prov.fetch_animation_library()
            resolved_ids = prov.resolve_action_ids(deduped_actions, library=library)
            unique_ids = list(dict.fromkeys(resolved_ids))

        if motion_specs:
            gid = f"group-{uuid.uuid4().hex[:12]}"
            g = {
                "provider": "meshy",
                "kind": "animate-group",
                "submitted_at": time.time(),
                "status": "submitting",
                "rig_task_id": rig_task_id,
                "actions": deduped_actions,
                "action_ids": unique_ids,
                "name_action_ids": resolved_ids,
                "parts": {"presets": None, "motions": []},
                "source_mesh": rpath,
                "secondary": secondary,
            }
            p["jobs"][gid] = g
            state.save(p)
            try:  # motions first (a refused prompt then costs no preset credits); every paid task is saved at once
                for m in motion_specs:
                    reused = _reusable_motion(p, rig_task_id, m)
                    if reused is None:
                        reused = {"motion_task_id": prov.submit_text_to_motion(m["prompt"], duration=m["duration"]), "status": "motion"}
                    g["parts"]["motions"].append({**m, **reused})
                    state.save(p)
                if unique_ids:
                    g["parts"]["presets"] = {"id": prov.submit_animation(rig_task_id=rig_task_id, action_ids=unique_ids), "status": "running"}
                    state.save(p)
            except CFError:
                g["status"] = "failed"
                state.save(p)
                raise
            g["status"] = "running"
            state.save(p)
            done = _advance_group(p, pdir, project_id, rpath, gid, prov)
            return done if done is not None else _group_running(project_id, rpath, gid, g)

        anim_task_id = prov.submit_animation(rig_task_id=rig_task_id, action_ids=unique_ids)
        p["jobs"][anim_task_id] = {
            "provider": "meshy",
            "kind": "animations",
            "submitted_at": time.time(),
            "status": "running",
            "rig_task_id": rig_task_id,
            "actions": deduped_actions,
            "action_ids": unique_ids,
            "name_action_ids": resolved_ids,
            "source_mesh": rpath,
            "secondary": secondary,
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
                secondary=secondary,
                name_action_ids=resolved_ids,
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

    # Blender procedural fallback (any humanoid rig: bones are found by role)
    r = blender.run(
        "merge_animations.py",
        {
            "mesh": mesh,
            "out": out_path,
            "actions": ",".join(deduped_actions),
            "mode": "procedural",
            "fps": 30,
            "secondary": secondary,
            **_secondary_args(p),
        },
    )
    p["animations"] = [{"name": a} for a in deduped_actions]
    return _finalize_animation(
        p, pdir, project_id, rpath, out_rpath, deduped_actions, "blender", {"secondary_info": (r or {}).get("secondary")}, []
    )


def register(mcp: FastMCP) -> None:
    """Register Level 3 tools onto FastMCP instance."""
    mcp.tool(annotations=LOCAL_WRITE)(auto_rig)
    mcp.tool(annotations=LOCAL_WRITE)(animate)
    mcp.tool(annotations=LOCAL_WRITE)(pipeline_run)
