from pathlib import Path

import pytest

from charforge import config, server, state

pytestmark = [pytest.mark.blender, pytest.mark.skipif(config.blender_exe() is None, reason="Blender not installed")]

# 0.4 (x) x 1.6 (up) x 0.2 (depth) m box; OBJ is Y-up, Blender imports it Z-up.
V = [(x, y, z) for x in (-0.2, 0.2) for y in (0, 1.6) for z in (-0.1, 0.1)]
F = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]


def test_e2e_l1_pipeline(sheet_path, tmp_path):
    """End-to-end Level 1 MCP tool verification.
    Follows: project_create -> sheet_split -> score_mesh -> render_views -> project_status.
    """
    # 1. Create box mesh
    obj = tmp_path / "char_box.obj"
    obj.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in V) + "".join("f " + " ".join(map(str, f)) + "\n" for f in F))

    # 2. project_create
    res_create = server.project_create(sheet_path=str(sheet_path), name="E2E Char", height_cm=160.0)
    assert res_create["ok"] is True
    pid = res_create["project_id"]
    assert pid == "e2e-char"
    pdir = state.project_dir(pid)
    assert pdir.is_dir()
    assert (pdir / "project.json").is_file()
    assert res_create["next_action"]["tool"] == "sheet_split"
    assert res_create["next_action"]["args"]["project_id"] == pid

    # 3. sheet_split
    res_split = server.sheet_split(project_id=pid)
    assert res_split["ok"] is True
    assert res_split["data"]["confidence"] >= 0.8
    assert len(res_split["data"]["candidates"]) == 3
    for v in ("front", "side", "back"):
        img_rel = res_split["data"]["files"][v]["image"]
        mask_rel = res_split["data"]["files"][v]["mask"]
        assert (pdir / img_rel).is_file(), f"Missing view image: {img_rel}"
        assert (pdir / mask_rel).is_file(), f"Missing view mask: {mask_rel}"
    assert (pdir / res_split["data"]["preview"]).is_file()
    assert (pdir / "views" / "reference.json").is_file()
    assert res_split["next_action"]["tool"] == "score_mesh"
    assert res_split["next_action"]["args"]["project_id"] == pid

    # 4. score_mesh
    res_score = server.score_mesh(project_id=pid, mesh_path=str(obj))
    assert res_score["ok"] is True
    assert res_score["data"]["stage"] in ("scored", "passed")
    assert res_score["data"]["total"] > 0
    assert set(res_score["data"]["iou"]) == {"front", "side", "back"}
    for v in ("front", "side", "back"):
        diff_rel = res_score["data"]["diffs"][v]
        assert (pdir / diff_rel).is_file(), f"Missing diff file: {diff_rel}"
    assert res_score["next_action"] is not None

    # 5. render_views (color preview)
    res_render = server.render_views(project_id=pid, mesh_path=str(obj), mode="color")
    assert res_render["ok"] is True
    assert set(res_render["data"]["files"]) == {"front", "side", "back"}
    for v in ("front", "side", "back"):
        render_rel = res_render["data"]["files"][v]
        assert (pdir / render_rel).is_file(), f"Missing render file: {render_rel}"
    assert "compare" in res_render["data"]
    assert (pdir / res_render["data"]["compare"]).is_file()

    # 6. project_status
    status = server.project_status(project_id=pid)
    assert status["ok"] is True
    stats = status["data"]["stats"]
    assert stats["errors"] == 0
    assert stats["error_rate"] == 0.0
    assert stats["calls"] >= 4
    assert len(status["data"]["scores"]) >= 1
    assert status["data"]["best"] is not None


def test_e2e_l1_score_before_split_error(sheet_path, tmp_path):
    """Calling score_mesh before sheet_split should fail with VIEWS_MISSING and suggest sheet_split."""
    obj = tmp_path / "char_box.obj"
    obj.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in V) + "".join("f " + " ".join(map(str, f)) + "\n" for f in F))

    res_create = server.project_create(sheet_path=str(sheet_path), name="Unsplit Char", height_cm=160.0)
    pid = res_create["project_id"]

    res_score = server.score_mesh(project_id=pid, mesh_path=str(obj))
    assert res_score["ok"] is False
    assert res_score["error"]["code"] == "VIEWS_MISSING"
    assert res_score["next_action"]["tool"] == "sheet_split"
