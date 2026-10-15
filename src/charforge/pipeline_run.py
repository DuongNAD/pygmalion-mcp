"""Autonomous pipeline runner for CharForge projects."""
from __future__ import annotations

import time
from typing import Annotated

from pydantic import Field

from . import state
from .envelope import CFError, action, ok, tool_entry

ProjectId = Annotated[str, Field(description="Id returned by project_create, e.g. 'aria'", pattern=r"^[a-z0-9][a-z0-9-]{0,47}$")]


def _dispatch_tool(tool_name: str, args: dict) -> dict:
    """Dispatch pipeline step to the corresponding underlying MCP tool function."""
    from . import server, tools_core, tools_full

    tool_map = {
        "sheet_split": server.sheet_split,
        "score_mesh": server.score_mesh,
        "render_views": server.render_views,
        "generate_mesh": tools_core.generate_mesh,
        "mesh_fit": tools_core.mesh_fit,
        "mesh_optimize": tools_core.mesh_optimize,
        "mesh_export": tools_core.mesh_export,
        "auto_rig": tools_full.auto_rig,
        "animate": tools_full.animate,
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
    animate: Annotated[bool, Field(description="Whether to run animate at the end")] = False,
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
                animate=animate,
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

        # Check if animate tool suggested but animate is disabled
        if p.get("stage") == "rigged" and not animate:
            status = "done"
            next_act = None
            break

        if tool_name == "animate" and not animate:
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
        status_val = data.get("status")
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
            prov = data.get("provider", "unknown")
            m_path = data.get("mesh_path")
            summary = f"Generated mesh via {prov}: {m_path}" if status_val == "done" else f"Generation in progress via {prov}"
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
            summary = (
                f"Rigging in progress ({data.get('job_id')})"
                if status_val == "running"
                else f"Rigged model with {data.get('bones')} bones: {data.get('mesh_path')}"
            )
        elif tool_name == "animate":
            summary = (
                f"Animation in progress ({data.get('job_id')})"
                if status_val == "running"
                else f"Animated model ({data.get('clip_count')} clips): {data.get('mesh_path')}"
            )
        else:
            summary = f"Completed {tool_name}"

        steps.append({"tool": tool_name, "ok": True, "summary": summary})

        if status_val == "running":
            # Cloud job in progress: reload state and poll on next iteration
            next_act = res.get("next_action")
            p = state.load(project_id)
            rem = max(0.0, max_seconds - (time.time() - t0))
            time.sleep(min(0.2, rem))
            continue

        # Reload state
        p = state.load(project_id)
        if p.get("stage") == "exported" and not rig:
            status = "done"
            next_act = None
            break

        if p.get("stage") == "rigged" and not animate:
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
