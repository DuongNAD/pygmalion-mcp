"""Empirical adversarial test suite for src/charforge/score.py.

Designed by Challenger 2 (teamwork_preview_challenger_m1_2).
Covers:
1. Identical masks (IoU=1.0, passed=True, total=1.0, side_mirrored=False, rotated_90=False)
2. Mirrored side mask (side_mirrored=True, side IoU > 0.97, passed=True)
3. 90° rotated render mask (rotated_90=True)
4. Extreme proportion mismatch (worst_bands classification, regions, directional hints)
5. write_diffs: dimensions (512, 512, 3), BGR channels (white, red, blue, black)
6. Empty or all-zero masks: graceful handling without unhandled exceptions
7. Boundary cases: single pixel, full canvas True, extreme aspect ratios, disconnected masks
"""
import shutil
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

from charforge import config, score
from .conftest import figure_mask


def test_empirical_identical_masks():
    """Identical masks must yield IoU=1.0 across all views, total=1.0, passed=True."""
    ref = {
        "front": figure_mask(),
        "side": figure_mask(ws=0.55, nose=18),
        "back": figure_mask(),
    }
    ren = {
        "front": figure_mask(),
        "side": figure_mask(ws=0.55, nose=18),
        "back": figure_mask(),
    }
    res = score.compare(ref, ren, config.VIEW_WEIGHTS, config.THRESHOLDS)

    assert res["iou"]["front"] == 1.0, f"Front IoU was {res['iou']['front']}, expected 1.0"
    assert res["iou"]["side"] == 1.0, f"Side IoU was {res['iou']['side']}, expected 1.0"
    assert res["iou"]["back"] == 1.0, f"Back IoU was {res['iou']['back']}, expected 1.0"
    assert res["total"] == 1.0, f"Total was {res['total']}, expected 1.0"
    assert res["passed"] is True, "passed should be True for identical masks"
    assert res["side_mirrored"] is False, "side_mirrored should be False when not mirrored"
    assert res["rotated_90"] is False, "rotated_90 should be False for matching views"
    assert res["aspect_delta"] == {"front": 0.0, "side": 0.0, "back": 0.0}
    assert res["worst_bands"] == [], f"worst_bands should be empty, got {res['worst_bands']}"
    assert res["hints"] == [], f"hints should be empty, got {res['hints']}"


def test_empirical_mirrored_side_mask():
    """Mirrored side mask must flag side_mirrored=True and compute IoU > 0.97."""
    ref = {
        "front": figure_mask(),
        "side": figure_mask(ws=0.55, nose=18),
        "back": figure_mask(),
    }
    # Flip the side view horizontally
    ren = {
        "front": figure_mask(),
        "side": np.fliplr(ref["side"]),
        "back": figure_mask(),
    }
    res = score.compare(ref, ren, config.VIEW_WEIGHTS, config.THRESHOLDS)

    assert res["side_mirrored"] is True, "side_mirrored must be True"
    assert res["iou"]["side"] > 0.97, f"Side IoU was {res['iou']['side']}, expected > 0.97"
    assert res["passed"] is True, "Overall passed must be True"
    assert res["rotated_90"] is False, "rotated_90 must be False"
    # Aspect delta should be 0.0 because horizontal flip does not alter bbox aspect ratio
    assert res["aspect_delta"]["side"] == 0.0


def test_empirical_rotated_90():
    """90° rotated render mask must detect rotated_90=True."""
    ref = {
        "front": figure_mask(ws=1.0),
        "side": figure_mask(ws=0.55, nose=18),
        "back": figure_mask(ws=1.0),
    }
    # Mesh rotated 90 degrees: front render is actually the side view of the character!
    ren = {
        "front": figure_mask(ws=0.55, nose=18),  # Render front camera sees character's side
        "side": figure_mask(ws=1.0),            # Render side camera sees character's front/back
        "back": figure_mask(ws=0.55, nose=18),
    }
    res = score.compare(ref, ren, config.VIEW_WEIGHTS, config.THRESHOLDS)

    assert res["rotated_90"] is True, "rotated_90 must be detected when front render matches ref side"


def test_empirical_rotated_90_mirrored_side():
    """Check rotated_90 behavior when the mesh side facing front camera is flipped."""
    ref = {
        "front": figure_mask(ws=1.0),
        "side": figure_mask(ws=0.55, nose=18),
        "back": figure_mask(ws=1.0),
    }
    # Mesh rotated 90 degrees in opposite direction: front camera sees horizontally flipped side
    flipped_side = np.fliplr(figure_mask(ws=0.55, nose=18))
    c_ref_side = score.canonical_mask(ref["side"])
    c_ren_front = score.canonical_mask(flipped_side)
    c_ref_front = score.canonical_mask(ref["front"])

    iou_side_match = score.iou(c_ref_side, c_ren_front)
    iou_front_match = score.iou(c_ref_front, c_ren_front)
    print(f"DEBUG: rotated_90_mirrored: iou_side={iou_side_match}, iou_front={iou_front_match}")

    ren = {
        "front": flipped_side,
        "side": figure_mask(ws=1.0),
        "back": flipped_side,
    }
    res = score.compare(ref, ren, config.VIEW_WEIGHTS, config.THRESHOLDS)
    # Check whether score.py detects rotated_90 in this scenario
    print(f"DEBUG: res['rotated_90']={res['rotated_90']}")


def test_empirical_extreme_proportion_mismatch():
    """Extreme proportion mismatches across multiple regions (head, torso, hips, legs)."""
    ref = {
        "front": figure_mask(h=760),
        "side": figure_mask(h=760, ws=0.55),
        "back": figure_mask(h=760),
    }

    # 1. Torso 3x too wide
    wide_torso = figure_mask(h=760).copy()
    wide_torso[260:400, 50:550] = True
    res_torso = score.compare(ref, {**ref, "front": wide_torso}, config.VIEW_WEIGHTS, config.THRESHOLDS)
    assert not res_torso["passed"]
    assert len(res_torso["worst_bands"]) >= 1
    band_torso = res_torso["worst_bands"][0]
    assert band_torso["view"] == "front"
    assert band_torso["region"] == "torso"
    assert band_torso["delta"] > 0
    assert any("front" in h and "torso" in h and "wider" in h and "make it narrower" in h for h in res_torso["hints"])

    # 2. Head too narrow (mesh head squished to 20% width)
    narrow_head = figure_mask(h=760).copy()
    for y in range(60, 180):
        for x in range(600):
            if narrow_head[y, x] and (x < 290 or x > 310):
                narrow_head[y, x] = False
    res_head = score.compare(ref, {**ref, "front": narrow_head}, config.VIEW_WEIGHTS, config.THRESHOLDS)
    head_bands = [b for b in res_head["worst_bands"] if b["region"] == "head"]
    assert len(head_bands) >= 1, f"Expected head in worst_bands, got {res_head['worst_bands']}"
    assert head_bands[0]["delta"] < 0, f"Expected delta < 0, got {head_bands[0]['delta']}"
    assert any("head" in h and "narrower" in h and "make it wider" in h for h in res_head["hints"])

    # 3. Legs too wide
    wide_legs = figure_mask(h=760).copy()
    # Bottom 30% made wider
    wide_legs[600:820, 150:450] = True
    res_legs = score.compare(ref, {**ref, "front": wide_legs}, config.VIEW_WEIGHTS, config.THRESHOLDS)
    assert not res_legs["passed"]
    legs_bands = [b for b in res_legs["worst_bands"] if b["region"] == "legs"]
    assert len(legs_bands) >= 1
    assert legs_bands[0]["delta"] > 0
    assert any("legs" in h and "wider" in h and "make it narrower" in h for h in res_legs["hints"])


def test_empirical_write_diffs(tmp_path):
    """write_diffs must output 512x512x3 images with precise BGR channels."""
    ref = {
        "front": figure_mask(),
        "side": figure_mask(ws=0.55, nose=18),
        "back": figure_mask(),
    }
    # Mesh has wider torso in front, missing nose in side
    ren = {
        "front": figure_mask().copy(),
        "side": figure_mask(ws=0.55, nose=0),
        "back": figure_mask(),
    }
    ren["front"][260:400, 100:500] = True  # extra mesh pixels

    paths = score.write_diffs(ref, ren, tmp_path)
    assert set(paths.keys()) == {"front", "side", "back"}

    for view, path in paths.items():
        assert path.is_file(), f"File {path} does not exist"
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert img is not None, f"Failed to read {path}"
        assert img.shape == (512, 512, 3), f"Expected (512, 512, 3), got {img.shape}"
        assert img.dtype == np.uint8, f"Expected uint8, got {img.dtype}"

        # Distinct colors present in diff image:
        # Expected colors:
        # Black (0, 0, 0) - background
        # White (255, 255, 255) - both
        # Red (0, 0, 255) - mesh only
        # Blue (255, 0, 0) - reference only
        pixels = img.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        # All colors must be one of the 4 valid diff colors
        valid_colors = {
            (0, 0, 0),        # black: background
            (255, 255, 255),  # white: both
            (0, 0, 255),      # red (BGR): mesh only
            (255, 0, 0),      # blue (BGR): ref only
        }
        for color in unique_colors:
            assert tuple(color) in valid_colors, f"Invalid color found in diff {view}: {color}"

    # Specifically for front view: must have white, red, and black
    front_img = cv2.imread(str(paths["front"]))
    has_white = np.any(np.all(front_img == [255, 255, 255], axis=-1))
    has_red = np.any(np.all(front_img == [0, 0, 255], axis=-1))
    has_black = np.any(np.all(front_img == [0, 0, 0], axis=-1))
    assert has_white, "Front diff missing white (both)"
    assert has_red, "Front diff missing red (mesh only)"
    assert has_black, "Front diff missing black (background)"

    # Specifically for side view: ref has nose, ren does not -> must have blue (ref only)
    side_img = cv2.imread(str(paths["side"]))
    has_blue = np.any(np.all(side_img == [255, 0, 0], axis=-1))
    assert has_blue, "Side diff missing blue (ref only - nose)"


def test_empirical_empty_and_zero_masks(tmp_path):
    """Empty or all-zero masks: must handle gracefully without unhandled exceptions."""
    empty_mask = np.zeros((900, 600), bool)
    valid_mask = figure_mask()

    # 1. canonical_mask on empty
    c_empty = score.canonical_mask(empty_mask)
    assert c_empty.shape == (config.CANON_SIZE, config.CANON_SIZE)
    assert not np.any(c_empty), "canonical_mask of empty should be all zeros"

    # 2. iou on empty
    assert score.iou(c_empty, c_empty) == 0.0
    assert score.iou(c_empty, score.canonical_mask(valid_mask)) == 0.0

    # 3. width_profile on empty
    prof_empty = score.width_profile(c_empty)
    assert len(prof_empty) == config.PROFILE_BINS
    assert all(p == 0.0 for p in prof_empty)

    # 4. compare where ren is completely empty
    ref = {"front": valid_mask, "side": valid_mask, "back": valid_mask}
    ren_empty = {"front": empty_mask, "side": empty_mask, "back": empty_mask}
    res_ren_empty = score.compare(ref, ren_empty, config.VIEW_WEIGHTS, config.THRESHOLDS)
    assert res_ren_empty["passed"] is False
    assert res_ren_empty["total"] == 0.0
    assert res_ren_empty["iou"] == {"front": 0.0, "side": 0.0, "back": 0.0}
    assert res_ren_empty["side_mirrored"] is False
    assert res_ren_empty["rotated_90"] is False

    # 5. compare where ref is completely empty
    res_ref_empty = score.compare(ren_empty, ref, config.VIEW_WEIGHTS, config.THRESHOLDS)
    assert res_ref_empty["passed"] is False
    assert res_ref_empty["total"] == 0.0

    # 6. compare where both ref and ren are completely empty
    res_both_empty = score.compare(ren_empty, ren_empty, config.VIEW_WEIGHTS, config.THRESHOLDS)
    assert res_both_empty["passed"] is False
    assert res_both_empty["total"] == 0.0

    # 7. write_diffs with empty masks
    paths = score.write_diffs(ref, ren_empty, tmp_path / "diffs_empty")
    for view, p in paths.items():
        assert p.is_file()
        img = cv2.imread(str(p))
        assert img.shape == (512, 512, 3)
        # Empty render vs non-empty ref: should only have blue (ref only) and black (background)
        assert np.any(np.all(img == [255, 0, 0], axis=-1))
        assert not np.any(np.all(img == [0, 0, 255], axis=-1))


def test_empirical_boundary_and_stress_cases(tmp_path):
    """Stress test boundary cases: single pixel, extreme aspect ratios, full True canvas."""
    # Single-pixel mask
    single_px = np.zeros((500, 500), bool)
    single_px[250, 250] = True
    c_single = score.canonical_mask(single_px)
    assert c_single.shape == (config.CANON_SIZE, config.CANON_SIZE)
    prof_single = score.width_profile(c_single)
    assert len(prof_single) == config.PROFILE_BINS

    # Extreme aspect ratio: 1x800 vertical slit
    vert_slit = np.zeros((800, 800), bool)
    vert_slit[:, 400] = True
    c_vert = score.canonical_mask(vert_slit)
    assert c_vert.shape == (config.CANON_SIZE, config.CANON_SIZE)
    assert score.iou(c_vert, c_vert) == 1.0

    # Extreme aspect ratio: 800x1 horizontal line
    horiz_line = np.zeros((800, 800), bool)
    horiz_line[400, :] = True
    c_horiz = score.canonical_mask(horiz_line)
    assert c_horiz.shape == (config.CANON_SIZE, config.CANON_SIZE)
    assert score.iou(c_horiz, c_horiz) == 1.0

    # All-True mask
    full_mask = np.ones((600, 600), bool)
    c_full = score.canonical_mask(full_mask)
    assert c_full.shape == (config.CANON_SIZE, config.CANON_SIZE)
    assert score.iou(c_full, c_full) == 1.0

    # Compare single pixel vs figure
    ref = {"front": figure_mask(), "side": figure_mask(ws=0.55), "back": figure_mask()}
    ren_single = {"front": single_px, "side": single_px, "back": single_px}
    res = score.compare(ref, ren_single, config.VIEW_WEIGHTS, config.THRESHOLDS)
    assert res["passed"] is False
    assert isinstance(res["total"], float)
    assert isinstance(res["hints"], list)
