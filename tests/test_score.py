import cv2
import numpy as np
import pytest

from charforge import config, score

from .conftest import figure_mask


def test_canonical_removes_scale_and_translation():
    a = figure_mask(h=500, shape=(700, 500))
    b = np.zeros((1400, 1400), bool)
    big = cv2.resize(a.astype(np.uint8), None, fx=1.7, fy=1.7, interpolation=cv2.INTER_NEAREST).astype(bool)
    b[100:100 + big.shape[0], 300:300 + big.shape[1]] = big
    assert score.iou(score.canonical_mask(a), score.canonical_mask(b)) > 0.95


def test_iou_and_profile_basics():
    m = figure_mask()
    c = score.canonical_mask(m)
    assert score.iou(c, c) == 1.0 and score.iou(c, np.zeros_like(c)) == 0.0
    prof = score.width_profile(c)
    assert len(prof) == config.PROFILE_BINS and all(0 <= w <= 1 for w in prof)
    assert prof[2] < prof[40]  # head band narrower than skirt band


def test_compare_mirrored_side_passes():
    ref = {"front": figure_mask(), "side": figure_mask(ws=0.55, nose=18), "back": figure_mask()}
    ren = {**ref, "side": np.fliplr(ref["side"])}
    res = score.compare(ref, ren, config.VIEW_WEIGHTS, config.THRESHOLDS)
    assert res["side_mirrored"] and res["iou"]["side"] > 0.97 and res["passed"] and not res["rotated_90"]


def test_compare_hints_wide_torso():
    ref = {"front": figure_mask(), "side": figure_mask(ws=0.55), "back": figure_mask()}
    wide = figure_mask().copy()
    wide[260:400, 100:500] = True  # torso band ~26-45% of height, ~2x wider
    res = score.compare(ref, {**ref, "front": wide}, config.VIEW_WEIGHTS, config.THRESHOLDS)
    assert not res["passed"] and res["iou"]["front"] < 0.9
    band = res["worst_bands"][0]
    assert band["view"] == "front" and band["delta"] > 0 and band["region"] == "torso"
    assert any("front" in h and "wider" in h for h in res["hints"])


def test_write_diffs(tmp_path):
    ref = {"front": figure_mask(), "side": figure_mask(ws=0.55), "back": figure_mask()}
    paths = score.write_diffs(ref, ref, tmp_path)
    assert set(paths) == {"front", "side", "back"} and all(p.is_file() for p in paths.values())


def test_extract_torso_core_and_profile():
    """Verify extract_torso_core drops wide lateral arm extensions while preserving torso."""
    size = 512
    mask = np.zeros((size, size), bool)
    mask[100:400, 220:292] = True
    mask[150:190, 50:462] = True

    c = score.canonical_mask(mask)
    prof_full = score.width_profile(c, torso_only=False)
    prof_torso = score.width_profile(c, torso_only=True)

    arm_band_idx = 12
    assert prof_full[arm_band_idx] > prof_torso[arm_band_idx] * 1.5
    assert prof_full[25] == pytest.approx(prof_torso[25], abs=0.01)

    core = score.extract_torso_core(c)
    assert core.any()
    ys, xs = np.nonzero(c)
    h = ys.max() - ys.min() + 1
    max_dx = 0.22 * h
    core_ys, core_xs = np.nonzero(core)
    assert np.all(np.abs(core_xs - 256) <= max_dx + 1)

