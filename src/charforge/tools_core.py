"""Level 2 core tools: generate_mesh, mesh_fit, mesh_optimize, mesh_export.

Registered into FastMCP server when CHARFORGE_LEVEL >= 2.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated, Literal

logger = logging.getLogger("charforge.fit")

import numpy as np
try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from . import blender, blockout, config, providers, score, state
from .envelope import CFError, action, ok, tool_entry

LOCAL_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

ProjectId = Annotated[str, Field(description="Id returned by project_create, e.g. 'aria'", pattern=r"^[a-z0-9][a-z0-9-]{0,47}$")]
MeshPath = Annotated[str, Field(description="Project-relative path from a previous result (e.g. 'meshes/cand-1.glb') or an absolute path to .glb/.fbx/.obj/.ply/.stl")]
ProviderChoice = Literal["auto", "tripo", "meshy", "rodin", "hunyuan", "blockout"]
ExportFormat = Literal["glb", "fbx", "both"]


@tool_entry
def generate_mesh(
    project_id: ProjectId,
    provider: ProviderChoice = "auto",
    job_id: Annotated[str | None, Field(description="Previous job_id if status was running")] = None,
    seed: Annotated[int | None, Field(description="Optional random seed for generation")] = None,
) -> dict:
    """Generate a 3D candidate mesh via cloud provider (Tripo, Meshy, Rodin, Hunyuan) or local blockout."""
    p = state.load(project_id)
    pdir = state.project_dir(project_id)
    t0 = time.time()

    if not p.get("views"):
        raise CFError("VIEWS_MISSING", "Reference views are not extracted yet.", "Call sheet_split first.", action("sheet_split", "Needed before mesh generation.", project_id=project_id))

    if "jobs" not in p:
        p["jobs"] = {}

    # 1. Polling existing job
    if job_id:
        job_info = p["jobs"].get(job_id)
        if not job_info:
            raise CFError("PROVIDER_FAILED", f"Job {job_id!r} not found in project jobs.", "Call generate_mesh without job_id to start a new job.")

        prov_name = job_info.get("provider", provider)
        submitted_at = job_info.get("submitted_at", time.time())
        if time.time() - submitted_at > 1200:  # 20 minutes timeout
            raise CFError("PROVIDER_TIMEOUT", f"Job {job_id} exceeded 20 minutes.", "Retry with another provider or use provider='blockout'")

        prov = providers.get_provider(prov_name)
        res = prov.poll(job_id)
        status = res.get("status")

        if status == "running":
            return ok(
                {"status": "running", "job_id": job_id, "provider": prov_name, "mesh_path": None, "seconds": round(time.time() - t0, 2)},
                project_id,
                action("generate_mesh", "Job in progress; poll again.", project_id=project_id, job_id=job_id, provider=prov_name),
            )
        elif status == "done":
            n_cands = sum(1 for m in p["meshes"] if m.get("source") in ("blockout", "tripo", "meshy", "rodin", "hunyuan")) + 1
            rpath = f"meshes/cand-{n_cands}-{prov_name}.glb"
            dst = pdir / rpath
            prov.download(res["url"], dst)
            state.register_mesh(p, rpath, source=prov_name, seed=job_info.get("seed", seed), task_id=job_id)
            job_info["status"] = "done"
            job_info["mesh_path"] = rpath
            state.save(p)
            return ok(
                {"status": "done", "job_id": job_id, "provider": prov_name, "mesh_path": rpath, "seconds": round(time.time() - t0, 2)},
                project_id,
                action("score_mesh", "Score the generated candidate mesh.", project_id=project_id, mesh_path=rpath),
            )
        else:
            raise CFError("PROVIDER_FAILED", f"Provider {prov_name} job failed: {res.get('error')}", "Call generate_mesh with provider='blockout'")

    # 2. Starting new generation task
    chosen = provider
    if chosen == "auto":
        avail = config.providers_available()
        used = {m.get("source") for m in p.get("meshes", [])} | {j.get("provider") for j in p.get("jobs", {}).values()}
        untried = [name for name in avail if name not in used]
        if untried:
            chosen = untried[0]
        elif avail:
            chosen = avail[0]
        else:
            chosen = "blockout"

    # Local synchronous visual hull blockout
    if chosen == "blockout":
        n_cands = sum(1 for m in p["meshes"] if m.get("source") in ("blockout", "tripo", "meshy", "rodin", "hunyuan")) + 1
        rpath = f"meshes/cand-{n_cands}-blockout.ply"
        dst = pdir / rpath
        blockout.blockout(p["views"], p["reference"], p["height_cm"], dst)
        state.register_mesh(p, rpath, source="blockout", seed=seed)
        state.save(p)
        return ok(
            {"status": "done", "job_id": None, "provider": "blockout", "mesh_path": rpath, "seconds": round(time.time() - t0, 2)},
            project_id,
            action("score_mesh", "Score the generated candidate mesh.", project_id=project_id, mesh_path=rpath),
        )

    # Cloud provider submission
    prov = providers.get_provider(chosen)
    view_paths = {v: pdir / p["views"][v]["image"] for v in ("front", "side", "back") if v in p["views"]}
    new_job_id = prov.submit(view_paths, seed=seed)
    p["jobs"][new_job_id] = {"provider": chosen, "submitted_at": time.time(), "seed": seed, "status": "running"}
    state.save(p)

    poll_res = prov.poll(new_job_id)
    if poll_res.get("status") == "done":
        n_cands = sum(1 for m in p["meshes"] if m.get("source") in ("blockout", "tripo", "meshy", "rodin", "hunyuan")) + 1
        rpath = f"meshes/cand-{n_cands}-{chosen}.glb"
        dst = pdir / rpath
        prov.download(poll_res["url"], dst)
        state.register_mesh(p, rpath, source=chosen, seed=seed, task_id=new_job_id)
        p["jobs"][new_job_id]["status"] = "done"
        p["jobs"][new_job_id]["mesh_path"] = rpath
        state.save(p)
        return ok(
            {"status": "done", "job_id": new_job_id, "provider": chosen, "mesh_path": rpath, "seconds": round(time.time() - t0, 2)},
            project_id,
            action("score_mesh", "Score the generated candidate mesh.", project_id=project_id, mesh_path=rpath),
        )
    elif poll_res.get("status") == "failed":
        raise CFError("PROVIDER_FAILED", f"Provider {chosen} failed: {poll_res.get('error')}", "Call generate_mesh with provider='blockout'")
    else:
        return ok(
            {"status": "running", "job_id": new_job_id, "provider": chosen, "mesh_path": None, "seconds": round(time.time() - t0, 2)},
            project_id,
            action("generate_mesh", "Job in progress; poll again.", project_id=project_id, job_id=new_job_id, provider=chosen),
        )


@tool_entry
def mesh_fit(project_id: ProjectId, mesh_path: MeshPath) -> dict:
    """Fit mesh proportions to reference silhouettes using profile-guided 2x2x16 lattice deformation."""
    p = state.load(project_id)
    pdir = state.project_dir(project_id)
    mesh, rpath = state.mesh_path(p, mesh_path)

    # 1. Retrieve last score for this mesh
    last_score = next((s for s in reversed(p["scores"]) if s.get("mesh") == rpath), None)
    if not last_score:
        raise CFError("FIT_FAILED", f"Mesh {rpath} has not been scored yet.", "Call score_mesh first before fitting.")

    # 2. Render mask if not already on disk
    render_dir = state.render_dir(p, rpath)
    front_ren_mask = render_dir / "front_mask.png"
    side_ren_mask = render_dir / "side_mask.png"
    if not front_ren_mask.is_file() or not side_ren_mask.is_file():
        blender.run("render_views.py", {"mesh": mesh, "out": render_dir, "mode": "mask", "rotate_z": last_score.get("rotate_z", 0)})

    front_ref_mask = pdir / p["views"]["front"]["mask"]
    side_ref_mask = pdir / p["views"]["side"]["mask"]

    c_ref_f = score.canonical_mask(score.load_mask(front_ref_mask))
    c_ren_f = score.canonical_mask(score.load_mask(front_ren_mask))

    # Check if reference and candidate mesh arm poses differ
    span_ref = score.arm_span_ratio(c_ref_f)
    span_ren = score.arm_span_ratio(c_ren_f)
    arm_poses_differ = bool(
        abs(span_ref - span_ren) > 0.15 and (span_ref > 0.50 or span_ren > 0.50)
    )

    if arm_poses_differ:
        prof_ref_f = score.width_profile(c_ref_f, torso_only=True)
        prof_ren_f = score.width_profile(c_ren_f, torso_only=True)
    else:
        prof_ref_f = score.width_profile(c_ref_f)
        prof_ren_f = score.width_profile(c_ren_f)

    c_ref_s = score.canonical_mask(score.load_mask(side_ref_mask))
    side_m = score.load_mask(side_ren_mask)
    if last_score.get("side_mirrored"):
        side_m = np.fliplr(side_m)
    c_ren_s = score.canonical_mask(side_m)
    prof_ref_s = score.width_profile(c_ref_s)
    prof_ren_s = score.width_profile(c_ren_s)

    # 3. Calculate 64-band ratios, then downsample to 16 lattice layers
    # Note: width_profile has bin 0 at top (head) and bin 63 at bottom (feet).
    # Blender lattice has layer w=0 at bottom (feet) and w=15 at top (head).
    ratio_x = [
        float(np.clip(prof_ref_f[i] / prof_ren_f[i], 0.8, 1.25))
        if prof_ren_f[i] > 0.01 and prof_ref_f[i] > 0.01 else 1.0
        for i in range(config.PROFILE_BINS)
    ]
    ratio_y = [
        float(np.clip(prof_ref_s[i] / prof_ren_s[i], 0.8, 1.25))
        if prof_ren_s[i] > 0.01 and prof_ref_s[i] > 0.01 else 1.0
        for i in range(config.PROFILE_BINS)
    ]

    scale_x = []
    scale_y = []
    for w in range(16):
        b = 15 - w  # map lattice bottom->top to profile top->bottom
        sx = float(np.mean(ratio_x[b * 4:(b + 1) * 4]))
        sy = float(np.mean(ratio_y[b * 4:(b + 1) * 4]))
        scale_x.append(sx)
        scale_y.append(sy)

    # 4. Smooth scale factors
    def _smooth(arr: list[float]) -> list[float]:
        res = []
        for i in range(len(arr)):
            l = arr[max(0, i - 1)]
            m = arr[i]
            r = arr[min(len(arr) - 1, i + 1)]
            res.append(float(np.clip(0.25 * l + 0.5 * m + 0.25 * r, 0.8, 1.25)))
        return res

    scale_x = _smooth(scale_x)
    scale_y = _smooth(scale_y)

    if arm_poses_differ:
        # Damp horizontal lattice scaling on arm height bands (w in 7..13)
        for w in range(7, 14):
            scale_x[w] = 1.0 + (scale_x[w] - 1.0) * 0.20

    # 5. Output path and fit execution
    stem = Path(rpath).stem
    fit_count = sum(1 for m in p["meshes"] if f"{stem}-fit" in m["path"]) + 1
    out_rpath = f"meshes/{stem}-fit{fit_count}.glb"
    out_path = pdir / out_rpath

    args = {
        "mesh": mesh,
        "out": out_path,
        "height_m": p["height_cm"] / 100.0,
        "rotate_z": last_score.get("rotate_z", 0),
        "scale_x": ",".join(f"{x:.4f}" for x in scale_x),
        "scale_y": ",".join(f"{y:.4f}" for y in scale_y),
        "damp_arms": arm_poses_differ,
    }
    blender.run("fit.py", args)

    # 6. Monotonicity guard: verify fit improved silhouette score
    out_render = state.render_dir(p, out_rpath)
    r_ren = blender.run("render_views.py", {"mesh": out_path, "out": out_render, "mode": "mask", "rotate_z": last_score.get("rotate_z", 0)})
    ref_masks = {v: score.load_mask(pdir / p["views"][v]["mask"]) for v in state.VIEWS}
    ren_masks = {v: score.load_mask(r_ren["files"][v]) for v in state.VIEWS}
    res_fit = score.compare(ref_masks, ren_masks, p["weights"], p["thresholds"], p["reference"].get("neck_ratio", 0.15))

    score_before = float(last_score.get("total", 0.0))
    score_after = float(res_fit.get("total", 0.0))
    warnings = []

    if score_after < score_before:
        warn_msg = f"Lattice fit degraded silhouette IoU from {score_before:.3f} to {score_after:.3f}; retaining original candidate."
        logger.warning(warn_msg)
        warnings.append(warn_msg)
        chosen_rpath = rpath
        state.record_score(p, {**last_score, "mesh": rpath})
        state.save(p)
        nxt = state.suggest_next(p)
    else:
        chosen_rpath = out_rpath
        state.register_mesh(p, out_rpath, source="fit")
        state.record_score(p, {"mesh": out_rpath, "rotate_z": last_score.get("rotate_z", 0), "tris": r_ren["tris"], **res_fit})
        state.save(p)
        nxt = action("score_mesh", "Score the fitted mesh.", project_id=project_id, mesh_path=out_rpath)

    return ok(
        {
            "mesh_path": chosen_rpath,
            "scale_x": [round(x, 4) for x in scale_x],
            "scale_y": [round(y, 4) for y in scale_y],
        },
        project_id,
        nxt,
        warnings=warnings,
    )


@tool_entry
def mesh_optimize(
    project_id: ProjectId,
    mesh_path: MeshPath,
    target_tris: Annotated[int, Field(description="Target triangle count", ge=2000, le=200000)] = 30000,
) -> dict:
    """Clean up mesh (merge vertices, remove loose, fix normals), decimate to target tris, and project smart UVs."""
    p = state.load(project_id)
    pdir = state.project_dir(project_id)
    mesh, rpath = state.mesh_path(p, mesh_path)

    stem = Path(rpath).stem
    out_rpath = f"meshes/{stem}-opt.glb"
    out_path = pdir / out_rpath

    # 1. First optimization pass
    r = blender.run("optimize.py", {"mesh": mesh, "out": out_path, "target_tris": target_tris})
    state.register_mesh(p, out_rpath, source="optimized")

    # 2. Self-check re-scoring
    out_render = state.render_dir(p, out_rpath)
    r_render = blender.run("render_views.py", {"mesh": out_path, "out": out_render, "mode": "mask"})
    ref_f = score.canonical_mask(score.load_mask(pdir / p["views"]["front"]["mask"]))
    ren_f = score.canonical_mask(score.load_mask(r_render["files"]["front"]))
    front_after = score.iou(ref_f, ren_f)

    prev_score = next((s for s in reversed(p["scores"]) if s.get("mesh") == rpath), None)
    front_before = prev_score["iou"]["front"] if prev_score else front_after
    iou_drop = max(0.0, front_before - front_after)

    # If front IoU dropped by > 0.02, retry with target_tris * 1.5
    if iou_drop > 0.02:
        new_target = int(target_tris * 1.5)
        r = blender.run("optimize.py", {"mesh": mesh, "out": out_path, "target_tris": new_target})
        r_render = blender.run("render_views.py", {"mesh": out_path, "out": out_render, "mode": "mask"})
        ren_f = score.canonical_mask(score.load_mask(r_render["files"]["front"]))
        front_after = score.iou(ref_f, ren_f)
        iou_drop = max(0.0, front_before - front_after)

    p["outputs"]["optimized"] = out_rpath
    p["stage"] = "optimized"
    state.save(p)

    return ok(
        {
            "mesh_path": out_rpath,
            "tris_before": r["tris_before"],
            "tris_after": r["tris_after"],
            "iou_drop": round(iou_drop, 4),
        },
        project_id,
        state.suggest_next(p),
    )


@tool_entry
def mesh_export(
    project_id: ProjectId,
    mesh_path: MeshPath,
    format: Annotated[ExportFormat, Field(description="Export format: glb, fbx, or both")] = "glb",
) -> dict:
    """Export final model with applied transforms and metric units to export/ directory."""
    p = state.load(project_id)
    pdir = state.project_dir(project_id)
    mesh, rpath = state.mesh_path(p, mesh_path)

    out_dir = pdir / "export"
    r = blender.run("export.py", {"mesh": mesh, "out_dir": out_dir, "stem": project_id, "format": format})

    rel_files = {fmt: state.rel(p, path) for fmt, path in r["files"].items()}
    if "glb" in rel_files:
        p["outputs"]["glb"] = rel_files["glb"]
    if "fbx" in rel_files:
        p["outputs"]["fbx"] = rel_files["fbx"]
    p["stage"] = "exported"
    state.save(p)

    return ok(
        {
            "files": rel_files,
            "size_mb": r["size_mb"],
            "tris": r["tris"],
        },
        project_id,
        state.suggest_next(p),
    )


def register(mcp: FastMCP) -> None:
    """Register Level 2 core tools onto FastMCP instance."""
    mcp.tool(annotations=LOCAL_WRITE)(generate_mesh)
    mcp.tool(annotations=LOCAL_WRITE)(mesh_fit)
    mcp.tool(annotations=LOCAL_WRITE)(mesh_optimize)
    mcp.tool(annotations=LOCAL_WRITE)(mesh_export)
