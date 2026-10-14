from pathlib import Path

import pytest

from charforge import config, server, state, tools_core

pytestmark = [pytest.mark.blender, pytest.mark.skipif(config.blender_exe() is None, reason="Blender not installed")]

TOOL_MAP = {
    "project_create": server.project_create,
    "project_status": server.project_status,
    "sheet_split": server.sheet_split,
    "render_views": server.render_views,
    "score_mesh": server.score_mesh,
    "generate_mesh": tools_core.generate_mesh,
    "mesh_fit": tools_core.mesh_fit,
    "mesh_optimize": tools_core.mesh_optimize,
    "mesh_export": tools_core.mesh_export,
}


def test_e2e_l2_blockout_pipeline(sheet_path, monkeypatch):
    """L2-T5: End-to-End Level 2 verification.

    Sets CHARFORGE_LEVEL=2, runs project_create, and then purely follows next_action
    until next_action is None.
    Asserts:
    - Entire workflow finishes in <= 12 tool calls.
    - Final stage is 'exported'.
    - outputs.glb exists and is a valid file.
    """
    monkeypatch.setenv("CHARFORGE_LEVEL", "2")
    server._register_levels()

    # Step 1: Initialize project
    res = server.project_create(sheet_path=str(sheet_path), name="E2E L2", height_cm=165.0)
    assert res["ok"] is True
    pid = res["project_id"]

    call_count = 1
    call_log = [("project_create", res)]
    next_act = res["next_action"]

    # Step 2..N: Follow next_action purely
    while next_act is not None and call_count < 15:
        tool_name = next_act["tool"]
        tool_args = next_act["args"]
        print(f"Call #{call_count + 1}: {tool_name} with args {tool_args}")

        assert tool_name in TOOL_MAP, f"Unknown tool in next_action: {tool_name}"
        fn = TOOL_MAP[tool_name]
        res = fn(**tool_args)

        assert res["ok"] is True, f"Tool {tool_name} failed: {res.get('error')}"
        call_count += 1
        call_log.append((tool_name, res))
        next_act = res["next_action"]

    # Verify completion criteria
    print(f"Pipeline finished in {call_count} calls: {[t for t, _ in call_log]}")
    assert call_count <= 12, f"Expected <= 12 calls, took {call_count}"
    assert next_act is None, f"Expected pipeline to finish with null next_action, got {next_act}"

    p = state.load(pid)
    assert p["stage"] == "exported", f"Expected stage 'exported', got {p['stage']}"
    assert "glb" in p["outputs"], "Expected outputs.glb to be populated"

    pdir = state.project_dir(pid)
    glb_file = pdir / p["outputs"]["glb"]
    assert glb_file.is_file(), f"Exported GLB not found at {glb_file}"
    assert glb_file.stat().st_size > 1000, "Exported GLB file is empty"

    # Status check
    status = server.project_status(project_id=pid)
    assert status["ok"] is True
    assert status["data"]["stats"]["errors"] == 0
    assert status["data"]["stats"]["error_rate"] == 0.0
