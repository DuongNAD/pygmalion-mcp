"""Pose-gap analysis (A-pose readiness) and the policy that reacts to fused arms/legs. No Blender needed."""
import math

import cv2
import numpy as np
import pytest

from charforge import config, score, server, state


def silhouette(arm_deg=35.0, legs_apart=True, long_dress=False, arm_len=330):
    """Front silhouette: head, torso, two legs, two straight arms at arm_deg from the body (90 = T-pose).
    The crotch is at 400 px below the top; arms start 125 px below the top."""
    img = np.zeros((900, 600), np.uint8)
    cx, top = 300, 60
    cv2.circle(img, (cx, top + 60), 55, 255, -1)
    cv2.rectangle(img, (cx - 70, top + 110), (cx + 70, top + 420), 255, -1)
    if long_dress:
        cv2.fillPoly(img, [np.array([[cx - 70, top + 380], [cx + 70, top + 380], [cx + 110, top + 760], [cx - 110, top + 760]])], 255)
    gap = 18 if legs_apart else 0
    for s in (-1, 1):
        x = cx + s * (gap + 32)
        cv2.rectangle(img, (x - 32, top + 400), (x + 32, top + 780), 255, -1)
        a = math.radians(arm_deg)
        sx, sy = cx + s * 75, top + 125
        cv2.line(img, (sx, sy), (int(sx + s * math.sin(a) * arm_len), int(sy + math.cos(a) * arm_len)), 255, 34)
    return img > 0


def test_pose_gaps_apose_is_clean():
    r = score.pose_gaps(silhouette(35))
    assert r["arms"]["left"]["separated"] and r["arms"]["right"]["separated"]
    assert r["legs"]["separated"]
    assert r["ok"] is True
    assert r["arms"]["left"]["gap"] > config.POSE_GAP_MIN


def test_pose_gaps_arms_down_touching():
    r = score.pose_gaps(silhouette(4))
    assert r["arms"]["left"]["separated"] is False and r["arms"]["right"]["separated"] is False
    assert r["ok"] is False
    warnings = score.pose_warnings(r, "sheet")
    assert any("left and right arm touches the body" in w for w in warnings)


@pytest.mark.parametrize("deg", [0, 2, 4])
def test_pose_gaps_thighs_are_not_mistaken_for_arms(deg):
    # short arms fused to the torso, ending above the crotch: below it the two thighs (thin runs with the
    # inner-thigh gap between them) must not count as separated arms
    r = score.pose_gaps(silhouette(deg, arm_len=240))
    assert r["arms"]["left"]["separated"] is False and r["arms"]["right"]["separated"] is False
    assert score.pose_mismatch(r, score.pose_gaps(silhouette(4))) == []


def test_pose_gaps_hands_apart_below_the_crotch_still_count():
    # long arms hanging past the crotch with a gap to the thighs: hand rows sit outside the two leg runs
    r = score.pose_gaps(silhouette(14, arm_len=400))
    assert r["arms"]["left"]["separated"] and r["arms"]["right"]["separated"]


def test_pose_gaps_t_pose_counts_as_separated():
    r = score.pose_gaps(silhouette(88))
    assert r["t_pose"] is True
    assert r["arms"]["left"]["separated"] and r["arms"]["right"]["separated"]


@pytest.mark.parametrize("kw", [{"long_dress": True}, {"legs_apart": False}])
def test_pose_gaps_legs_not_apart(kw):
    r = score.pose_gaps(silhouette(35, **kw))
    assert r["legs"]["separated"] is False
    assert any("legs are not apart" in w for w in score.pose_warnings(r))


def test_pose_gaps_empty_mask():
    r = score.pose_gaps(np.zeros((50, 50), bool))
    assert r["ok"] is False


def test_pose_mismatch_only_reports_parts_the_sheet_keeps_apart():
    clean, fused = score.pose_gaps(silhouette(35)), score.pose_gaps(silhouette(4))
    assert score.pose_mismatch(clean, fused) == ["left arm", "right arm"]
    assert score.pose_mismatch(fused, fused) == []  # the sheet itself is fused: regenerating cannot help
    assert score.pose_mismatch(clean, clean) == []
    assert score.pose_mismatch(None, fused) == []


def test_sheet_split_reports_pose(sheet_path):
    pid = server.project_create(sheet_path=str(sheet_path), name="PoseSheet", height_cm=165.0)["project_id"]
    res = server.sheet_split(project_id=pid)
    assert res["ok"]
    assert set(res["data"]["pose"]) >= {"arms", "legs", "t_pose", "ok"}
    p = state.load(pid)
    assert p["reference"]["pose"] == res["data"]["pose"]
    # the synthetic figure has no arms: the agent is told so, the plan is unchanged
    assert any("arm touches the body" in w for w in res["warnings"])
    assert res["next_action"]["tool"] == "generate_mesh" or res["next_action"]["tool"] == "score_mesh"


def _provider_project(sheet_path, monkeypatch, name):
    monkeypatch.setenv("CHARFORGE_LEVEL", "2")
    pid = server.project_create(sheet_path=str(sheet_path), name=name, height_cm=165.0)["project_id"]
    p = state.load(pid)
    p["stage"] = "views_ready"
    p["reference"] = {"pose": score.pose_gaps(silhouette(35))}
    state.register_mesh(p, "meshes/cand-1-meshy.glb", source="meshy", task_id="t1")
    return p


def _rec(mesh, total, mismatch):
    return {"mesh": mesh, "total": total, "passed": True, "iou": {}, "pose_mismatch": mismatch}


def test_policy_one_extra_candidate_when_mesh_fuses_clean_sheet(sheet_path, monkeypatch):
    p = _provider_project(sheet_path, monkeypatch, "PoseRetry")
    state.record_score(p, _rec("meshes/cand-1-meshy.glb", 0.9, ["left arm"]))
    assert p["stage"] == "scored" and p["pose_retries"] == 1
    nxt = state.suggest_next(p)
    assert nxt["tool"] == "generate_mesh" and "left arm" in nxt["why"]

    # second candidate is clean but scores lower: it still becomes best and the search ends
    state.register_mesh(p, "meshes/cand-2-meshy.glb", source="meshy", task_id="t2")
    state.record_score(p, _rec("meshes/cand-2-meshy.glb", 0.85, []))
    assert p["best"]["mesh"] == "meshes/cand-2-meshy.glb"
    assert p["stage"] == "passed"
    assert state.suggest_next(p)["tool"] == "mesh_export"


def test_policy_retry_budget_is_one(sheet_path, monkeypatch):
    p = _provider_project(sheet_path, monkeypatch, "PoseRetryBudget")
    state.record_score(p, _rec("meshes/cand-1-meshy.glb", 0.9, ["legs"]))
    state.register_mesh(p, "meshes/cand-2-meshy.glb", source="meshy", task_id="t2")
    state.record_score(p, _rec("meshes/cand-2-meshy.glb", 0.88, ["legs"]))
    assert p["stage"] == "passed"  # still fused, but only one retry is allowed
    assert p["best"]["mesh"] == "meshes/cand-1-meshy.glb"  # both fused: higher total wins


def test_policy_no_retry_for_blockout(sheet_path, monkeypatch):
    p = _provider_project(sheet_path, monkeypatch, "PoseBlockout")
    state.register_mesh(p, "meshes/cand-1-blockout.ply", source="blockout")
    state.record_score(p, _rec("meshes/cand-1-blockout.ply", 0.95, ["left arm"]))
    assert p["stage"] == "passed"


def test_policy_ignores_pose_for_failed_meshes(sheet_path, monkeypatch):
    p = _provider_project(sheet_path, monkeypatch, "PoseFailed")
    rec = {**_rec("meshes/cand-1-meshy.glb", 0.7, ["left arm"]), "passed": False}
    state.record_score(p, rec)
    assert "pose_retries" not in p
