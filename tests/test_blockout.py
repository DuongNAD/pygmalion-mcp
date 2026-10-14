from pathlib import Path

import cv2
import numpy as np
import pytest

from charforge import blockout, config, server, state

pytestmark = [pytest.mark.blender, pytest.mark.skipif(config.blender_exe() is None, reason="Blender not installed")]


def test_blockout_generation_and_score(sheet_path, tmp_path):
    """L2-T2: Build blockout from synthetic sheet, score with score_mesh, assert front >= 0.90 and side >= 0.85."""
    # 1. Create project & split sheet
    res_create = server.project_create(sheet_path=str(sheet_path), name="Blockout Test", height_cm=165.0)
    assert res_create["ok"] is True
    pid = res_create["project_id"]

    res_split = server.sheet_split(project_id=pid)
    assert res_split["ok"] is True

    p = state.load(pid)
    pdir = state.project_dir(pid)

    # 2. Run blockout generator
    ply_dst = pdir / "meshes" / "cand-1-blockout.ply"
    res_ply = blockout.blockout(
        views=p["views"],
        reference=p["reference"],
        height_cm=p["height_cm"],
        dst=ply_dst,
    )
    assert res_ply.is_file()
    assert res_ply.stat().st_size > 1000

    # 3. Score the blockout mesh using score_mesh
    res_score = server.score_mesh(project_id=pid, mesh_path=str(ply_dst))
    assert res_score["ok"] is True
    ious = res_score["data"]["iou"]
    print(f"DEBUG Blockout IoUs: {ious}, Total: {res_score['data']['total']}")

    assert ious["front"] >= 0.90, f"Front IoU too low: {ious['front']}"
    assert ious["side"] >= 0.85, f"Side IoU too low: {ious['side']}"


def test_blockout_missing_color_images_fallback(tmp_path):
    """Verify blockout with only mask paths succeeds without ZeroDivisionError."""
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(mask, (200, 100), (312, 450), 255, -1)
    for v in ("front", "side", "back"):
        cv2.imwrite(str(tmp_path / f"{v}_mask.png"), mask)
    views = {v: str(tmp_path / f"{v}_mask.png") for v in ("front", "side", "back")}
    out_ply = tmp_path / "out.ply"
    res = blockout.blockout(views, {}, 165.0, out_ply)
    assert res.is_file()
    assert res.stat().st_size > 0

