"""Adversarial stress test suite for Level 3: pipeline_run and driver_report.py.

Designed by Challenger 2 (teamwork_preview_challenger_m3_2).
Covers:
1. pipeline_run time-boxing with tight limits (max_seconds=1, max_seconds=5):
   - Confirms incremental execution, status="running", next_action pointing to pipeline_run.
   - Confirms strict execution boundary: elapsed <= max_seconds + 30.
   - Confirms chained re-invocations progress to completion (stage="rigged", status="done").
2. pipeline_run with rig=False:
   - Confirms termination at stage="exported" with status="done" and next_action=None.
   - Confirms outputs["glb"] is created and outputs["rigged"] is absent.
3. pipeline_run idempotency and state safety:
   - Confirms calling pipeline_run on already completed/exported projects.
   - Confirms status="blocked" handling on tool failure or ambiguity.
4. driver_report.py empirical resilience:
   - Empty directory handling (0 projects, zero divide-by-zero crashes).
   - Non-project subdirectories, corrupted project.json, missing calls.jsonl.
   - Zero-call projects, 100% error projects, passed projects.
   - Non-existent parent directory auto-creation for --out.
   - Environment variable CHARFORGE_HOME integration.
   - Empirical vulnerability probe: float conversion crash on non-numeric or null 'sec' values in calls.jsonl.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from charforge import config, server, state, tools_full
from scripts import driver_report


@pytest.fixture
def l3_setup(monkeypatch, tmp_home):
    """Enable Level 3 tools and isolated project home."""
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")


# ==============================================================================
# SECTION 1: pipeline_run Adversarial Stress Tests
# ==============================================================================

@pytest.mark.blender
def test_pipeline_run_tight_time_limit_and_chaining(l3_setup, sheet_path):
    """Test pipeline_run with tight budget (max_seconds=1):

    Must run initial steps, pause with status='running', stay within max_seconds + 30,
    and then complete when resumed iteratively.
    """
    cr = server.project_create(sheet_path=sheet_path, name="TightTimeRun", height_cm=165.0)
    assert cr["ok"]
    pid = cr["project_id"]

    # Initial call with tight time limit
    t0 = time.time()
    res1 = tools_full.pipeline_run(project_id=pid, max_seconds=1, rig=True)
    el1 = time.time() - t0

    assert res1["ok"]
    assert el1 <= 1 + 30, f"Call exceeded max_seconds + 30: {el1:.2f}s"
    assert res1["data"]["status"] == "running"
    assert res1["next_action"] is not None
    assert res1["next_action"]["tool"] == "pipeline_run"
    assert res1["next_action"]["args"]["project_id"] == pid
    assert res1["next_action"]["args"]["max_seconds"] == 1
    assert res1["next_action"]["args"]["rig"] is True
    assert len(res1["data"]["steps"]) >= 1

    # Resume until completion, testing chaining contract
    max_passes = 10
    passes = 1
    curr_res = res1
    while curr_res["data"]["status"] == "running" and passes < max_passes:
        passes += 1
        next_act = curr_res["next_action"]
        assert next_act["tool"] == "pipeline_run"

        t_pass = time.time()
        curr_res = tools_full.pipeline_run(**next_act["args"])
        el_pass = time.time() - t_pass

        assert curr_res["ok"]
        assert el_pass <= 1 + 30, f"Pass {passes} exceeded budget: {el_pass:.2f}s"

    assert curr_res["data"]["status"] == "done"
    assert curr_res["next_action"] is None

    p = state.load(pid)
    assert p["stage"] == "rigged"
    assert "glb" in p["outputs"]
    assert "rigged" in p["outputs"]
    pdir = state.project_dir(pid)
    assert (pdir / p["outputs"]["glb"]).is_file()
    assert (pdir / p["outputs"]["rigged"]).is_file()


@pytest.mark.blender
def test_pipeline_run_rig_false(l3_setup, sheet_path):
    """Test pipeline_run with rig=False:

    Must terminate at stage 'exported', produce outputs.glb, and NOT produce outputs.rigged.
    """
    cr = server.project_create(sheet_path=sheet_path, name="NoRigPipeline", height_cm=160.0)
    assert cr["ok"]
    pid = cr["project_id"]

    t0 = time.time()
    res = tools_full.pipeline_run(project_id=pid, max_seconds=240, rig=False)
    el = time.time() - t0

    assert res["ok"]
    assert el <= 240 + 30
    assert res["data"]["status"] == "done"
    assert res["next_action"] is None

    p = state.load(pid)
    assert p["stage"] == "exported"
    assert "glb" in p["outputs"]
    assert "rigged" not in p["outputs"]

    pdir = state.project_dir(pid)
    assert (pdir / p["outputs"]["glb"]).is_file()
    rigged_path = pdir / f"export/{pid}-rigged.glb"
    assert not rigged_path.exists()


@pytest.mark.blender
def test_pipeline_run_idempotency_already_finished(l3_setup, sheet_path):
    """Test calling pipeline_run on an already-finished project:

    Should immediately return status='done' with no steps and next_action=None.
    """
    cr = server.project_create(sheet_path=sheet_path, name="IdempotentRun", height_cm=165.0)
    pid = cr["project_id"]

    # Run to completion with rig=False
    res1 = tools_full.pipeline_run(project_id=pid, max_seconds=240, rig=False)
    assert res1["data"]["status"] == "done"

    # Call again with rig=False: should be immediately done
    res2 = tools_full.pipeline_run(project_id=pid, max_seconds=240, rig=False)
    assert res2["ok"]
    assert res2["data"]["status"] == "done"
    assert res2["next_action"] is None
    assert len(res2["data"]["steps"]) == 0

    # Call with rig=True: should execute only auto_rig and finish
    res3 = tools_full.pipeline_run(project_id=pid, max_seconds=240, rig=True)
    assert res3["ok"]
    assert res3["data"]["status"] == "done"
    assert len(res3["data"]["steps"]) == 1
    assert res3["data"]["steps"][0]["tool"] == "auto_rig"

    # Call again after rigged: should be immediately done
    res4 = tools_full.pipeline_run(project_id=pid, max_seconds=240, rig=True)
    assert res4["ok"]
    assert res4["data"]["status"] == "done"
    assert res4["next_action"] is None
    assert len(res4["data"]["steps"]) == 0


def test_pipeline_run_blocked_on_failure(l3_setup, tmp_path):
    """Test that pipeline_run returns status='blocked' when a tool step fails."""
    # Create invalid sheet (10x10 blank)
    bad_sheet = tmp_path / "bad_sheet.png"
    cv2.imwrite(str(bad_sheet), np.full((10, 10, 3), 255, np.uint8))

    cr = server.project_create(sheet_path=bad_sheet, name="BadSheetRun", height_cm=165.0)
    assert cr["ok"]
    pid = cr["project_id"]

    res = tools_full.pipeline_run(project_id=pid, max_seconds=60, rig=True)
    assert res["ok"]
    assert res["data"]["status"] == "blocked"
    assert len(res["data"]["steps"]) == 1
    assert res["data"]["steps"][0]["ok"] is False
    assert "SPLIT_FAILED" in res["data"]["steps"][0]["summary"] or "Failed" in res["data"]["steps"][0]["summary"]


# ==============================================================================
# SECTION 2: scripts/driver_report.py Empirical Stress Tests
# ==============================================================================

def test_driver_report_empty_directory(tmp_path):
    """driver_report on an empty directory must produce a valid report with zero divide-by-zero crashes."""
    empty_dir = tmp_path / "empty_projects"
    empty_dir.mkdir()
    out_file = tmp_path / "report_empty.md"

    cmd = [
        sys.executable,
        "scripts/driver_report.py",
        "--projects-dir",
        str(empty_dir),
        "--model",
        "mock-challenger",
        "--out",
        str(out_file),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

    assert out_file.is_file()
    content = out_file.read_text()

    assert "**Projects Analyzed**: 0" in content
    assert "Total Projects | 0" in content
    assert "Total Tool Calls | 0" in content
    assert "Overall Error Rate | 0.0%" in content
    assert "Pass Rate | 0.0% (0/0)" in content
    assert "Avg Iterations to Pass | 0.00" in content
    assert "No errors recorded across all evaluated projects." in content


def test_driver_report_heterogeneous_projects(tmp_path):
    """driver_report must withstand non-project subdirectories, corrupted project.json, empty projects, and mixed states."""
    projs_dir = tmp_path / "projects"
    projs_dir.mkdir()

    # 1. Non-project directory (e.g. random scratch directory)
    (projs_dir / "random_scratch").mkdir()
    (projs_dir / "random_scratch" / "readme.txt").write_text("not a project")

    # 2. Corrupted project.json syntax
    p_corrupt = projs_dir / "corrupted_proj"
    p_corrupt.mkdir()
    (p_corrupt / "project.json").write_text("{invalid json syntax:;;;")

    # 3. Project with no calls.jsonl and no scores
    p_empty = projs_dir / "empty_proj"
    p_empty.mkdir()
    (p_empty / "project.json").write_text(json.dumps({
        "id": "empty_proj",
        "name": "Empty Project",
        "stage": "created",
        "scores": [],
        "best": None,
    }))

    # 4. Project with calls and errors
    p_err_calls = projs_dir / "err_calls_proj"
    p_err_calls.mkdir()
    (p_err_calls / "project.json").write_text(json.dumps({
        "id": "err_calls_proj",
        "name": "Error Calls Proj",
        "stage": "scored",
        "scores": [{"total": 0.5, "passed": False}],
        "best": {"total": 0.5, "passed": False},
    }))
    (p_err_calls / "calls.jsonl").write_text(
        '{"t": 1.0, "tool": "sheet_split", "ok": true, "sec": 0.5}\n'
        '\n'
        '{"t": 2.0, "tool": "score_mesh", "ok": false, "code": "VIEWS_MISSING", "sec": 0.8}\n'
        '{"t": 3.0, "tool": "score_mesh", "ok": false, "sec": 0.2}\n'
    )

    # 5. Project that passed with multiple iterations
    p_passed = projs_dir / "passed_proj"
    p_passed.mkdir()
    (p_passed / "project.json").write_text(json.dumps({
        "id": "passed_proj",
        "name": "Passed Proj",
        "stage": "passed",
        "scores": [
            {"total": 0.7, "passed": False},
            {"total": 0.95, "passed": True},
        ],
        "best": {"total": 0.95, "passed": True},
    }))
    (p_passed / "calls.jsonl").write_text(
        '{"t": 1.0, "tool": "sheet_split", "ok": true, "sec": 1.0}\n'
        '{"t": 2.0, "tool": "generate_mesh", "ok": true, "sec": 2.5}\n'
        '{"t": 3.0, "tool": "score_mesh", "ok": true, "sec": 1.2}\n'
    )

    out_file = tmp_path / "deep" / "nested" / "report.md"

    # Run CLI
    cmd = [
        sys.executable,
        "scripts/driver_report.py",
        "--projects-dir",
        str(projs_dir),
        "--model",
        "adversarial-model",
        "--out",
        str(out_file),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

    assert out_file.is_file(), "Report file was not created in nested directory."
    content = out_file.read_text()

    # Verify metrics:
    # Analyzed projects: empty_proj, err_calls_proj, passed_proj -> 3 projects
    assert "**Projects Analyzed**: 3" in content
    assert "Total Projects | 3" in content

    # Total calls:
    # empty_proj: 0
    # err_calls_proj: 3 valid lines (sheet_split ok, score_mesh fail VIEWS_MISSING, score_mesh fail UNKNOWN)
    # passed_proj: 3 valid lines
    # Total calls = 6
    assert "Total Tool Calls | 6" in content
    # Total errors: 2 (from err_calls_proj)
    assert "Total Errors | 2" in content
    # Error rate: 2/6 = 33.3%
    assert "Overall Error Rate | 33.3%" in content

    # Passed projects: passed_proj (1/3 = 33.3%)
    assert "Pass Rate | 33.3% (1/3)" in content
    # Avg iterations for passed projects: passed_proj has 2 scores -> 2.00
    assert "Avg Iterations to Pass | 2.00" in content

    # Error code distribution
    assert "| `VIEWS_MISSING` | 1 | 50.0% |" in content
    assert "| `UNKNOWN` | 1 | 50.0% |" in content

    # Project breakdown table
    assert "`empty_proj`" in content
    assert "`err_calls_proj`" in content
    assert "`passed_proj`" in content


def test_driver_report_env_var_fallback(tmp_path, monkeypatch):
    """driver_report must read CHARFORGE_HOME when --projects-dir is omitted."""
    projs_dir = tmp_path / "env_projects"
    projs_dir.mkdir()
    monkeypatch.setenv("CHARFORGE_HOME", str(projs_dir))

    p = projs_dir / "p1"
    p.mkdir()
    (p / "project.json").write_text(json.dumps({"id": "p1", "stage": "created"}))

    out_file = tmp_path / "env_report.md"
    cmd = [
        sys.executable,
        "scripts/driver_report.py",
        "--model",
        "env-model",
        "--out",
        str(out_file),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "**Projects Analyzed**: 1" in out_file.read_text()


def test_driver_report_vulnerability_on_null_or_invalid_sec(tmp_path):
    """Empirically document that driver_report crashes if calls.jsonl contains null or invalid string in 'sec' field."""
    projs_dir = tmp_path / "projects"
    projs_dir.mkdir()

    p_bad = projs_dir / "p_bad"
    p_bad.mkdir()
    (p_bad / "project.json").write_text(json.dumps({"id": "p_bad", "stage": "created"}))
    (p_bad / "calls.jsonl").write_text('{"t": 1.0, "tool": "test", "ok": true, "sec": null}\n')

    # Calling collect_project_metrics directly reproduces TypeError: float(None)
    with pytest.raises(TypeError, match="float.*NoneType"):
        driver_report.collect_project_metrics(p_bad)
