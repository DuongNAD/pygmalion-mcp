import json

import pytest

from charforge import server, state
from charforge.envelope import CFError


def test_create_slug_and_next(sheet_path):
    p = state.create(str(sheet_path), "Aria Nikki", 165)
    assert p["id"] == "aria-nikki" and p["stage"] == "created"
    assert state.create(str(sheet_path), "Aria Nikki", 165)["id"] == "aria-nikki-2"
    assert state.suggest_next(p)["tool"] == "sheet_split"


def test_bad_and_missing_ids():
    with pytest.raises(CFError) as e:
        state.project_dir("../etc")
    assert e.value.code == "BAD_PROJECT_ID"
    with pytest.raises(CFError) as e:
        state.load("nope")
    assert e.value.code == "PROJECT_NOT_FOUND"


def test_external_mesh_is_copied(sheet_path, tmp_path):
    p = state.create(str(sheet_path), "Aria", 165)
    ext = tmp_path / "model.obj"
    ext.write_text("v 0 0 0\n")
    path, rpath = state.mesh_path(p, str(ext))
    assert rpath == "meshes/model.obj" and path.is_file()
    assert p["meshes"][0]["source"] == "external"
    with pytest.raises(CFError) as e:
        state.mesh_path(p, "meshes/missing.glb")
    assert e.value.code == "MESH_NOT_FOUND"


def _score(p, total, passed=False, mesh="meshes/a.glb"):
    state.record_score(p, {"mesh": mesh, "total": total, "passed": passed, "iou": {}})


def test_policy_pass(sheet_path):
    p = state.create(str(sheet_path), "Aria", 165)
    p["stage"] = "views_ready"
    _score(p, 0.80)
    assert p["stage"] == "scored"
    _score(p, 0.93, True, "meshes/b.glb")
    assert p["stage"] == "passed" and p["best"]["mesh"] == "meshes/b.glb"


def test_policy_plateau_and_budget(sheet_path):
    p = state.create(str(sheet_path), "Aria", 165)
    p["stage"] = "views_ready"
    for t in (0.70, 0.71, 0.712, 0.713, 0.714):
        _score(p, t)
    assert (p["stage"], p["stop_reason"]) == ("stopped", "plateau")
    q = state.create(str(sheet_path), "Bea", 165)
    q["stage"] = "views_ready"
    for i in range(8):
        _score(q, 0.5 + 0.02 * i)
    assert (q["stage"], q["stop_reason"]) == ("stopped", "budget")


def test_tool_envelope_and_call_log(sheet_path):
    bad = server.project_create(sheet_path="/no/such.png", name="X")
    assert bad["ok"] is False and bad["error"]["code"] == "SHEET_NOT_FOUND" and bad["error"]["fix"]
    res = server.project_create(sheet_path=str(sheet_path), name="Aria")
    assert res["ok"] and res["next_action"]["tool"] == "sheet_split"
    st = server.project_status(project_id=res["project_id"])
    assert st["ok"] and st["data"]["stats"]["calls"] == 1  # project_create logged; status logs after it returns
    log = (state.project_dir(res["project_id"]) / "calls.jsonl").read_text().splitlines()
    assert [json.loads(line)["tool"] for line in log] == ["project_create", "project_status"]
    assert server.project_status(project_id="ghost")["error"]["code"] == "PROJECT_NOT_FOUND"
