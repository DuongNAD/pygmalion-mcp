"""CharForge MCP server (stdio). Tools stay thin: validate -> domain module -> update state -> envelope.

Level gating (CHARFORGE_LEVEL): 1 = the tools below; 2 adds tools_core.py; 3 adds tools_full.py.
Tool contracts: docs/TOOLS.md. Changing a name, parameter or data key there is a breaking change.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal

try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from . import blender, config, score, sheet, state
from .envelope import CFError, action, ok, tool_entry

PLAYBOOK = Path(__file__).with_name("playbook.md").read_text()
INSTRUCTIONS = """CharForge turns a character turnaround sheet into a 3D model, fully automatically.
1. Do what `next_action` in the previous result says; it is the plan.
2. Every tool takes project_id. Lost track? Call project_status.
3. Trust scores, not your eyes: a mesh is good only when score_mesh returns passed=true.
4. ok=false: do exactly error.fix. The same error twice: stop and report it.
5. Finished when next_action is null. Full procedure: prompt `charforge_playbook`."""

mcp = FastMCP("charforge", instructions=INSTRUCTIONS)

READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
LOCAL_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

ProjectId = Annotated[str, Field(description="Id returned by project_create, e.g. 'aria'", pattern=r"^[a-z0-9][a-z0-9-]{0,47}$")]
MeshPath = Annotated[str, Field(description="Project-relative path from a previous result (e.g. 'meshes/cand-1.glb') or an absolute path to .glb/.fbx/.obj/.ply/.stl")]
RotateZ = Annotated[Literal[0, 90, 180, 270], Field(description="Degrees to turn the mesh around Z before rendering; use the value a warning tells you")]
View = Literal["front", "side", "back"]


@mcp.tool(annotations=LOCAL_WRITE)
@tool_entry
def project_create(
    sheet_path: Annotated[str, Field(description="Absolute path to the turnaround sheet image (.png/.jpg/.webp)")],
    name: Annotated[str, Field(description="Character name; becomes the project_id", min_length=1, max_length=40)],
    height_cm: Annotated[float, Field(description="Character height in cm (from the sheet if stated)", ge=50, le=300)] = 165.0,
) -> dict:
    """Create a project from a character turnaround sheet. Returns project_id. Next: sheet_split."""
    p = state.create(sheet_path, name, height_cm)
    return ok({"project_dir": str(state.project_dir(p["id"])), "stage": p["stage"]}, p["id"], state.suggest_next(p))


@mcp.tool(annotations=READ_ONLY)
@tool_entry
def project_status(project_id: ProjectId) -> dict:
    """Project state: stage, views, meshes, last scores, best mesh, outputs, call stats. Use it to resume."""
    p = state.load(project_id)
    data = {k: p[k] for k in ("stage", "name", "height_cm", "views", "meshes", "best", "outputs", "stop_reason")}
    data["scores"] = [{k: s.get(k) for k in ("mesh", "total", "passed", "iou")} for s in p["scores"][-5:]]
    data["stats"] = state.stats(project_id)
    return ok(data, project_id, state.suggest_next(p))


@mcp.tool(annotations=LOCAL_WRITE)
@tool_entry
def sheet_split(
    project_id: ProjectId,
    assign: Annotated[list[int] | None, Field(description="Candidate indices for [front, side, back], read from the numbered preview; only after low confidence", min_length=3, max_length=3)] = None,
    boxes: Annotated[dict[View, Annotated[list[int], Field(min_length=4, max_length=4)]] | None, Field(description="Last resort: {view: [x, y, w, h]} in sheet pixels")] = None,
) -> dict:
    """Cut the sheet into front/side/back reference views, masks and metrics. If confidence < 0.8 look at data.preview."""
    p = state.load(project_id)
    pdir = state.project_dir(project_id)
    res = sheet.split_sheet(pdir / p["sheet"], pdir / "views", assign, {k: tuple(v) for k, v in boxes.items()} if boxes else None)
    p["views"], p["reference"], p["stage"] = res["files"], res["reference"], "views_ready"
    state.save(p)
    warnings, nxt = [], state.suggest_next(p)
    if res["confidence"] < 0.8:
        warnings.append("Low split confidence: open data.preview and check the numbered candidates.")
        nxt = action("sheet_split", "Confirm views from the preview.", project_id=project_id, assign=["<front>", "<side>", "<back>"])
    data = {k: res[k] for k in ("candidates", "boxes", "confidence", "files", "preview")}
    return ok(data, project_id, nxt, warnings)


@mcp.tool(annotations=LOCAL_WRITE)
@tool_entry
def render_views(
    project_id: ProjectId,
    mesh_path: MeshPath,
    mode: Annotated[Literal["color", "mask"], Field(description="color = flat albedo preview, mask = silhouettes")] = "color",
    rotate_z: RotateZ = 0,
) -> dict:
    """Render orthographic front/side/back images with Blender + compare.png (reference | render). For looking only; decide with score_mesh."""
    p = state.load(project_id)
    mesh, rpath = state.mesh_path(p, mesh_path)
    out = state.render_dir(p, rpath)
    r = blender.run("render_views.py", {"mesh": mesh, "out": out, "mode": mode, "rotate_z": rotate_z})
    pdir = state.project_dir(project_id)
    data = {"files": {v: state.rel(p, f) for v, f in r["files"].items()}, "dims_m": r["dims_m"], "tris": r["tris"]}
    if p["views"]:
        rows = [[pdir / p["views"][v]["image"], r["files"][v]] for v in state.VIEWS]
        data["compare"] = state.rel(p, score.contact_sheet(rows, out / f"compare_{mode}.png"))
    state.save(p)
    return ok(data, project_id, state.suggest_next(p))


@mcp.tool(annotations=LOCAL_WRITE)
@tool_entry
def score_mesh(project_id: ProjectId, mesh_path: MeshPath, rotate_z: RotateZ = 0) -> dict:
    """Score a mesh against the reference: silhouette IoU per view, total, passed, proportion hints, diff images."""
    p = state.load(project_id)
    if not p["views"]:
        raise CFError("VIEWS_MISSING", "Reference views are not extracted yet.", "Call sheet_split first.", action("sheet_split", "Needed before scoring.", project_id=project_id))
    mesh, rpath = state.mesh_path(p, mesh_path)
    is_prov = state.is_provider_mesh(p, rpath)
    out = state.render_dir(p, rpath)
    r = blender.run("render_views.py", {"mesh": mesh, "out": out, "mode": "mask", "rotate_z": rotate_z})
    pdir = state.project_dir(project_id)
    ref = {v: score.load_mask(pdir / p["views"][v]["mask"]) for v in state.VIEWS}
    ren = {v: score.load_mask(r["files"][v]) for v in state.VIEWS}
    res = score.compare(ref, ren, p["weights"], p["thresholds"], p["reference"].get("neck_ratio", 0.15), provider_mesh=is_prov)
    warnings: list[str] = []
    max_tris = p.get("max_provider_tris", config.MAX_PROVIDER_TRIS)
    if is_prov and r["tris"] > max_tris:
        res["passed"] = False
        res["over_poly_budget"] = True
        warnings.append(f"Provider mesh has {r['tris']} triangles (exceeds max {max_tris}); regenerate with smaller target_polycount.")
    if res.get("rotated_90") and rotate_z == 0:
        res["passed"] = False
    diffs = {v: state.rel(p, f) for v, f in score.write_diffs(ref, ren, out).items()}
    next(m for m in p["meshes"] if m["path"] == rpath)["rotate_z"] = rotate_z
    state.record_score(p, {"mesh": rpath, "rotate_z": rotate_z, "tris": r["tris"], **res})
    state.save(p)
    nxt = state.suggest_next(p)
    if res.get("rotated_90") and rotate_z == 0:
        warnings.append("Mesh seems turned 90 degrees.")
        nxt = action("score_mesh", "Re-score with the mesh turned 90 degrees.", project_id=project_id, mesh_path=rpath, rotate_z=90)
    data = {**res, "diffs": diffs, "iteration": len(p["scores"]), "best": p["best"], "stage": p["stage"], "stop_reason": p["stop_reason"]}
    return ok(data, project_id, nxt, warnings)


@mcp.prompt(name="charforge_playbook", description="Step-by-step procedure for driving CharForge to a finished model")
def charforge_playbook() -> str:
    return PLAYBOOK


def _register_levels() -> None:
    for lvl, module in ((2, "tools_core"), (3, "tools_full")):
        if config.level() < lvl:
            return
        try:
            __import__(f"{__package__}.{module}", fromlist=["register"]).register(mcp)
        except ImportError as e:
            print(f"charforge: CHARFORGE_LEVEL={config.level()} but {module}.py is missing ({e}); running Level {lvl - 1}.", file=sys.stderr)
            return


_register_levels()
