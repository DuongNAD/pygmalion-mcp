import time
import pytest
from pathlib import Path

from charforge import config, server, state, tools_full


@pytest.fixture
def l3_setup(monkeypatch, tmp_home):
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")


@pytest.mark.blender
def test_pipeline_run_autonomous(l3_setup, sheet_path):
    t0 = time.time()
    cr = server.project_create(sheet_path=sheet_path, name="AutonomousGirl", height_cm=165.0)
    assert cr["ok"]
    pid = cr["project_id"]

    # Run the autonomous pipeline
    res = tools_full.pipeline_run(project_id=pid, max_seconds=240, rig=True)
    elapsed = time.time() - t0

    assert res["ok"]
    assert res["data"]["status"] == "done"
    assert res["next_action"] is None
    assert elapsed <= 240 + 30

    # Verify outputs
    outputs = res["data"]["outputs"]
    assert "glb" in outputs
    assert "rigged" in outputs
    assert outputs["glb"].endswith(".glb")
    assert outputs["rigged"].endswith("-rigged.glb")

    p = state.load(pid)
    assert p["stage"] == "rigged"
    assert p["outputs"]["glb"] == outputs["glb"]
    assert p["outputs"]["rigged"] == outputs["rigged"]

    pdir = state.project_dir(pid)
    assert (pdir / outputs["glb"]).is_file()
    assert (pdir / outputs["rigged"]).is_file()

    # Verify steps logged
    steps = res["data"]["steps"]
    step_tools = [s["tool"] for s in steps]
    assert "sheet_split" in step_tools
    assert "generate_mesh" in step_tools
    assert "score_mesh" in step_tools
    assert "mesh_export" in step_tools
    assert "auto_rig" in step_tools
    assert all(s["ok"] for s in steps)


@pytest.mark.blender
def test_pipeline_run_no_rig(l3_setup, sheet_path):
    cr = server.project_create(sheet_path=sheet_path, name="NoRigRun", height_cm=160.0)
    pid = cr["project_id"]

    res = tools_full.pipeline_run(project_id=pid, max_seconds=240, rig=False)
    assert res["ok"]
    assert res["data"]["status"] == "done"
    assert res["next_action"] is None

    p = state.load(pid)
    assert p["stage"] == "exported"
    assert "glb" in p["outputs"]
    assert "rigged" not in p["outputs"]


@pytest.mark.blender
def test_pipeline_run_time_budget(l3_setup, sheet_path, monkeypatch):
    cr = server.project_create(sheet_path=sheet_path, name="TimeBudgetRun", height_cm=165.0)
    pid = cr["project_id"]

    # Simulate elapsed time reaching max_seconds quickly
    # By passing max_seconds=60 and having sleep or monkeypatching time
    real_time = time.time
    call_count = 0

    def mock_time():
        nonlocal call_count
        call_count += 1
        # After initial check, jump time by 300s
        if call_count > 3:
            return real_time() + 300.0
        return real_time()

    monkeypatch.setattr(time, "time", mock_time)

    res = tools_full.pipeline_run(project_id=pid, max_seconds=60, rig=True)
    assert res["ok"]
    assert res["data"]["status"] == "running"
    assert res["next_action"] is not None
    assert res["next_action"]["tool"] == "pipeline_run"
