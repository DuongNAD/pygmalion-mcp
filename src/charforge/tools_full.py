"""Level 3 tools: auto_rig, pipeline_run.

Registered into FastMCP server when CHARFORGE_LEVEL >= 3.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from . import blender, config, state
from .envelope import CFError, action, ok, tool_entry

LOCAL_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

ProjectId = Annotated[str, Field(description="Id returned by project_create, e.g. 'aria'", pattern=r"^[a-z0-9][a-z0-9-]{0,47}$")]
MeshPath = Annotated[str, Field(description="Project-relative path from a previous result (e.g. 'export/aria.glb') or an absolute path")]
RigMethod = Literal["auto", "blender", "provider"]


@tool_entry
def auto_rig(
    project_id: ProjectId,
    mesh_path: MeshPath,
    method: Annotated[RigMethod, Field(description="Rigging method: auto, blender, or provider")] = "auto",
) -> dict:
    """Add a humanoid Mixamo armature (22 bones) and skin weights to a character mesh."""
    p = state.load(project_id)
    pdir = state.project_dir(project_id)

    if method == "provider":
        avail = config.providers_available()
        rig_provs = [name for name in ("tripo", "meshy") if name in avail]
        if not rig_provs:
            raise CFError(
                "NO_PROVIDER",
                "No provider API key configured for rigging.",
                "Use method='blender' or method='auto'",
            )

    mesh, rpath = state.mesh_path(p, mesh_path)

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
        next_action=None,
    )


def _dispatch_tool(tool_name: str, args: dict) -> dict:
    """Dispatch pipeline step to the corresponding underlying MCP tool function."""
    from . import server, tools_core

    tool_map = {
        "sheet_split": server.sheet_split,
        "score_mesh": server.score_mesh,
        "render_views": server.render_views,
        "generate_mesh": tools_core.generate_mesh,
        "mesh_fit": tools_core.mesh_fit,
        "mesh_optimize": tools_core.mesh_optimize,
        "mesh_export": tools_core.mesh_export,
        "auto_rig": auto_rig,
    }
    fn = tool_map.get(tool_name)
    if fn is None:
        raise CFError("INTERNAL", f"Unknown tool in pipeline dispatch: {tool_name}", "Check state policy.")
    return fn(**args)


@tool_entry
def pipeline_run(
    project_id: ProjectId,
    max_seconds: Annotated[int, Field(description="Max runtime in seconds", ge=60, le=540)] = 240,
    rig: Annotated[bool, Field(description="Whether to run auto_rig at the end")] = True,
) -> dict:
    """Execute time-boxed autonomous pipeline driving the project to completion without human intervention."""
    t0 = time.time()
    p = state.load(project_id)
    steps: list[dict] = []
    warnings: list[str] = []

    status = "running"
    next_act = state.suggest_next(p)

    while next_act is not None:
        # Check time budget
        elapsed = time.time() - t0
        if elapsed >= max_seconds:
            status = "running"
            next_act = action(
                "pipeline_run",
                "Time budget elapsed; continue autonomous pipeline run.",
                project_id=project_id,
                max_seconds=max_seconds,
                rig=rig,
            )
            break

        # Check if rig tool suggested but rig is disabled
        if p.get("stage") == "exported" and not rig:
            status = "done"
            next_act = None
            break

        tool_name = next_act["tool"]
        tool_args = dict(next_act.get("args", {}))

        # Check if rig tool suggested but rig is disabled
        if tool_name == "auto_rig" and not rig:
            status = "done"
            next_act = None
            break

        # Execute underlying tool function
        res = _dispatch_tool(tool_name, tool_args)

        if not res.get("ok"):
            err = res.get("error") or {}
            steps.append({
                "tool": tool_name,
                "ok": False,
                "summary": f"Failed: {err.get('message', 'unknown error')}",
            })
            status = "blocked"
            next_act = res.get("next_action")
            if err.get("message"):
                warnings.append(err["message"])
            break

        # Step succeeded: format concise summary
        data = res.get("data") or {}
        if tool_name == "sheet_split":
            conf = data.get("confidence", 1.0)
            summary = f"Split turnaround sheet (confidence: {conf:.2f})"
            if conf < 0.8:
                steps.append({"tool": tool_name, "ok": True, "summary": summary})
                status = "blocked"
                next_act = res.get("next_action")
                warnings.append(f"Sheet split confidence {conf:.2f} < 0.8 requires user confirmation.")
                break
        elif tool_name == "generate_mesh":
            status_val = data.get("status")
            prov = data.get("provider", "unknown")
            m_path = data.get("mesh_path")
            summary = f"Generated mesh via {prov}: {m_path}" if status_val == "done" else f"Generation in progress via {prov}"
            if status_val == "running":
                steps.append({"tool": tool_name, "ok": True, "summary": summary})
                # If cloud job running, reload state and poll on next iteration
                next_act = res.get("next_action")
                p = state.load(project_id)
                time.sleep(1.0)
                continue
        elif tool_name == "score_mesh":
            tot = data.get("total", 0.0)
            passed = data.get("passed", False)
            summary = f"Scored mesh: total={tot:.3f}, passed={passed}"
        elif tool_name == "mesh_fit":
            summary = f"Fitted mesh proportions: {data.get('mesh_path')}"
        elif tool_name == "mesh_optimize":
            tb = data.get("tris_before", 0)
            ta = data.get("tris_after", 0)
            summary = f"Optimized mesh: {tb} -> {ta} tris"
        elif tool_name == "mesh_export":
            summary = f"Exported files: {list(data.get('files', {}).keys())}"
        elif tool_name == "auto_rig":
            summary = f"Rigged model with {data.get('bones')} bones: {data.get('mesh_path')}"
        else:
            summary = f"Completed {tool_name}"

        steps.append({"tool": tool_name, "ok": True, "summary": summary})

        # Reload state
        p = state.load(project_id)
        if p.get("stage") == "exported" and not rig:
            status = "done"
            next_act = None
            break

        # Advance: use returned next_action if available, else suggest_next(p)
        next_act = res.get("next_action")
        if next_act is None:
            next_act = state.suggest_next(p)

    if next_act is None and status != "blocked":
        status = "done"

    return ok(
        {
            "status": status,
            "stage": p.get("stage"),
            "steps": steps,
            "best": p.get("best"),
            "outputs": p.get("outputs", {}),
        },
        project_id,
        next_action=next_act,
        warnings=warnings,
    )


def register(mcp: FastMCP) -> None:
    """Register Level 3 tools onto FastMCP instance."""
    mcp.tool(annotations=LOCAL_WRITE)(auto_rig)
    mcp.tool(annotations=LOCAL_WRITE)(pipeline_run)
