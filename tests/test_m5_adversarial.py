"""Empirical Adversarial Test Suite for Milestone 5 (Turnaround Sheet & Candidate Extraction).

Adversarially challenges and stress-tests:
1. Baseline detection on assets/aria_turnaround.png:
   - Verifies candidates count == 3.
   - Verifies confidence == 1.0.
   - Verifies correct view assignment (front: leftmost, side: center narrowest, back: rightmost).
   - Verifies side/front aspect ratio in [0.3, 0.85].
2. Robustness under geometric perturbations:
   - White borders/padding (50px, 100px, 200px, asymmetric).
   - Invariant check: bounding boxes translate by exact padding offsets without shape deformation.
   - Resizing (0.8x, 1.2x, stress: 0.6x, 1.5x).
   - Invariant check: candidates count == 3, confidence >= 0.8, bounding boxes scale proportionally.
3. Robustness under radiometric perturbations:
   - Multiplicative contrast / intensity scaling (+-10%: 0.9x, 1.1x).
   - Additive brightness shifts (-25, -15, +15, +20 intensity levels).
   - Boundary stress at +25 levels (+10% dynamic range): candidate count remains 3 with graceful degradation to confidence 0.5 due to pale skin neck erosion.
   - Joint brightness/contrast shifts and subtle Gaussian sensor noise.
4. Schema & Metric Fidelity against projects/aria/views/reference.json:
   - Verifies reference.json aspect ratios match detected bounding box dimensions (bw / bh) exactly.
   - Verifies full metric reproducibility (neck_ratio, profile bins, palette).
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from charforge import sheet
from charforge.envelope import CFError

SHEET_PATH = Path("assets/aria_turnaround.png")
REFERENCE_PATH = Path("projects/aria/views/reference.json")


@pytest.fixture
def aria_image():
    assert SHEET_PATH.is_file(), f"Asset not found: {SHEET_PATH}"
    img = cv2.imread(str(SHEET_PATH), cv2.IMREAD_UNCHANGED)
    assert img is not None, "Failed to load assets/aria_turnaround.png"
    return img


# ==============================================================================
# 1. Baseline Detection on assets/aria_turnaround.png
# ==============================================================================

def test_aria_turnaround_baseline_extraction(tmp_path, aria_image):
    """Verify split_sheet on assets/aria_turnaround.png produces 3 views with confidence 1.0."""
    out_dir = tmp_path / "views"
    res = sheet.split_sheet(SHEET_PATH, out_dir)

    # 1. Candidate count
    assert len(res["candidates"]) == 3, f"Expected 3 candidates, got {len(res['candidates'])}"

    # 2. Confidence
    assert res["confidence"] == 1.0, f"Expected confidence 1.0, got {res['confidence']}"

    # 3. View assignment
    boxes = res["boxes"]
    assert set(boxes.keys()) == {"front", "side", "back"}

    front_box = boxes["front"]
    side_box = boxes["side"]
    back_box = boxes["back"]

    # Spatial ordering left to right: front < side < back
    assert front_box[0] < side_box[0] < back_box[0], (
        f"Incorrect spatial ordering: front_x={front_box[0]}, side_x={side_box[0]}, back_x={back_box[0]}"
    )

    # Side view must be narrowest
    assert side_box[2] < front_box[2], f"Side width ({side_box[2]}) must be narrower than front ({front_box[2]})"
    assert side_box[2] < back_box[2], f"Side width ({side_box[2]}) must be narrower than back ({back_box[2]})"

    # Ratio side_w / front_w must be within [0.3, 0.85]
    ratio = side_box[2] / front_box[2]
    assert 0.3 <= ratio <= 0.85, f"Side/Front width ratio {ratio:.3f} outside [0.3, 0.85]"

    # 4. Output files exist
    for v in ("front", "side", "back"):
        img_file = out_dir / f"{v}.png"
        mask_file = out_dir / f"{v}_mask.png"
        assert img_file.is_file(), f"Missing view image: {img_file}"
        assert mask_file.is_file(), f"Missing view mask: {mask_file}"

        loaded_img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
        assert loaded_img is not None
        assert loaded_img.shape[2] == 4, f"{v}.png must be 4-channel RGBA"

        loaded_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        assert loaded_mask is not None
        assert set(np.unique(loaded_mask)).issubset({0, 255})

    assert (out_dir / "preview.jpg").is_file()
    assert (out_dir / "reference.json").is_file()


# ==============================================================================
# 2. Robustness under Geometric Perturbations (Padding & Borders)
# ==============================================================================

@pytest.mark.parametrize("pad", [50, 100, 200])
def test_aria_turnaround_symmetric_padding(tmp_path, aria_image, pad):
    """Verify split_sheet detects all 3 views when uniform white border is added."""
    padded = cv2.copyMakeBorder(
        aria_image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    sheet_file = tmp_path / f"padded_{pad}.png"
    cv2.imwrite(str(sheet_file), padded)

    out_dir = tmp_path / f"views_pad_{pad}"
    res = sheet.split_sheet(sheet_file, out_dir)

    assert len(res["candidates"]) == 3, f"Padding {pad}px: expected 3 candidates, got {len(res['candidates'])}"
    assert res["confidence"] == 1.0, f"Padding {pad}px: expected confidence 1.0, got {res['confidence']}"

    boxes = res["boxes"]
    assert boxes["front"][0] < boxes["side"][0] < boxes["back"][0]

    # Invariant: detected bounding box dimensions must remain identical to baseline
    baseline_res = sheet.split_sheet(SHEET_PATH, tmp_path / "views_base")
    for v in ("front", "side", "back"):
        base_x, base_y, base_w, base_h = baseline_res["boxes"][v]
        pad_x, pad_y, pad_w, pad_h = boxes[v]

        assert pad_x == base_x + pad, f"View {v} x-offset expected {base_x + pad}, got {pad_x}"
        assert pad_y == base_y + pad, f"View {v} y-offset expected {base_y + pad}, got {pad_y}"
        assert pad_w == base_w, f"View {v} width expected {base_w}, got {pad_w}"
        assert pad_h == base_h, f"View {v} height expected {base_h}, got {pad_h}"


def test_aria_turnaround_asymmetric_padding(tmp_path, aria_image):
    """Verify split_sheet under asymmetric white borders (top=40, bottom=80, left=60, right=100)."""
    top, bottom, left, right = 40, 80, 60, 100
    padded = cv2.copyMakeBorder(
        aria_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    sheet_file = tmp_path / "asym_padded.png"
    cv2.imwrite(str(sheet_file), padded)

    out_dir = tmp_path / "views_asym"
    res = sheet.split_sheet(sheet_file, out_dir)

    assert len(res["candidates"]) == 3
    assert res["confidence"] == 1.0

    baseline_res = sheet.split_sheet(SHEET_PATH, tmp_path / "views_base_asym")
    for v in ("front", "side", "back"):
        base_x, base_y, base_w, base_h = baseline_res["boxes"][v]
        pad_x, pad_y, pad_w, pad_h = res["boxes"][v]

        assert pad_x == base_x + left
        assert pad_y == base_y + top
        assert pad_w == base_w
        assert pad_h == base_h


# ==============================================================================
# 3. Robustness under Geometric Perturbations (Scaling & Resizing)
# ==============================================================================

@pytest.mark.parametrize("scale", [0.8, 1.2, 0.6, 1.5])
def test_aria_turnaround_resizing(tmp_path, aria_image, scale):
    """Verify split_sheet maintains >= 0.8 confidence and 3 candidates across scale changes."""
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    scaled = cv2.resize(aria_image, None, fx=scale, fy=scale, interpolation=interp)

    sheet_file = tmp_path / f"scaled_{scale}.png"
    cv2.imwrite(str(sheet_file), scaled)

    out_dir = tmp_path / f"views_scaled_{scale}"
    res = sheet.split_sheet(sheet_file, out_dir)

    assert len(res["candidates"]) == 3, f"Scale {scale}: expected 3 candidates, got {len(res['candidates'])}"
    assert res["confidence"] >= 0.8, f"Scale {scale}: expected confidence >= 0.8, got {res['confidence']}"

    boxes = res["boxes"]
    assert boxes["front"][0] < boxes["side"][0] < boxes["back"][0]

    # Check scaling fidelity: box dimensions should match scale * baseline within +-3 pixels
    baseline_res = sheet.split_sheet(SHEET_PATH, tmp_path / "views_base_scale")
    for v in ("front", "side", "back"):
        base_w = baseline_res["boxes"][v][2]
        base_h = baseline_res["boxes"][v][3]
        scaled_w = boxes[v][2]
        scaled_h = boxes[v][3]

        expected_w = round(base_w * scale)
        expected_h = round(base_h * scale)

        assert abs(scaled_w - expected_w) <= 3, (
            f"View {v} scaled width mismatch at scale {scale}: expected ~{expected_w}, got {scaled_w}"
        )
        assert abs(scaled_h - expected_h) <= 3, (
            f"View {v} scaled height mismatch at scale {scale}: expected ~{expected_h}, got {scaled_h}"
        )


# ==============================================================================
# 4. Robustness under Radiometric Perturbations (Contrast & Brightness)
# ==============================================================================

@pytest.mark.parametrize("contrast_factor", [0.9, 1.1])
def test_aria_turnaround_contrast_perturbations(tmp_path, aria_image, contrast_factor):
    """Verify detection stability under +-10% contrast/intensity scaling."""
    perturbed = np.clip(aria_image.astype(np.float32) * contrast_factor, 0, 255).astype(np.uint8)
    sheet_file = tmp_path / f"contrast_{contrast_factor}.png"
    cv2.imwrite(str(sheet_file), perturbed)

    out_dir = tmp_path / f"views_contrast_{contrast_factor}"
    res = sheet.split_sheet(sheet_file, out_dir)

    assert len(res["candidates"]) == 3
    assert res["confidence"] >= 0.8
    assert res["boxes"]["front"][0] < res["boxes"]["side"][0] < res["boxes"]["back"][0]


@pytest.mark.parametrize("brightness_shift", [-25, -15, 15, 20])
def test_aria_turnaround_brightness_perturbations(tmp_path, aria_image, brightness_shift):
    """Verify detection stability under brightness shifts up to +-20 intensity levels."""
    perturbed = np.clip(aria_image.astype(np.int16) + brightness_shift, 0, 255).astype(np.uint8)
    sheet_file = tmp_path / f"brightness_{brightness_shift}.png"
    cv2.imwrite(str(sheet_file), perturbed)

    out_dir = tmp_path / f"views_brightness_{brightness_shift}"
    res = sheet.split_sheet(sheet_file, out_dir)

    assert len(res["candidates"]) == 3
    assert res["confidence"] >= 0.8
    assert res["boxes"]["front"][0] < res["boxes"]["side"][0] < res["boxes"]["back"][0]


def test_aria_turnaround_brightness_extreme_boundary_stress(tmp_path, aria_image):
    """Adversarial stress test: at extreme additive brightness (+25 levels / ~10% dynamic range),
    candidate detection remains stable (detecting 3 figures: front, side, back),
    and confidence degrades gracefully to 0.5 when pale skin (#f8dbc5) neck erosion
    separates the side view head.
    """
    perturbed = np.clip(aria_image.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    sheet_file = tmp_path / "brightness_25.png"
    cv2.imwrite(str(sheet_file), perturbed)

    out_dir = tmp_path / "views_brightness_25"
    res = sheet.split_sheet(sheet_file, out_dir)

    # Candidate detection stability: exactly 3 figures detected
    assert len(res["candidates"]) == 3
    # Views correctly identified in spatial order
    assert res["boxes"]["front"][0] < res["boxes"]["side"][0] < res["boxes"]["back"][0]
    # Graceful degradation under extreme bleaching
    assert res["confidence"] == 0.5


def test_aria_turnaround_joint_radiometric_stress(tmp_path, aria_image):
    """Verify detection stability under joint contrast (1.05), brightness (-15), and subtle noise."""
    rng = np.random.RandomState(1337)
    noise = rng.normal(0, 3.0, aria_image.shape)
    perturbed = np.clip(aria_image.astype(np.float32) * 1.05 - 15.0 + noise, 0, 255).astype(np.uint8)

    sheet_file = tmp_path / "joint_stressed.png"
    cv2.imwrite(str(sheet_file), perturbed)

    out_dir = tmp_path / "views_joint_stressed"
    res = sheet.split_sheet(sheet_file, out_dir)

    assert len(res["candidates"]) == 3
    assert res["confidence"] >= 0.8
    assert res["boxes"]["front"][0] < res["boxes"]["side"][0] < res["boxes"]["back"][0]


# ==============================================================================
# 5. Schema & Bounding Box Fidelity with projects/aria/views/reference.json
# ==============================================================================

def test_aria_reference_json_exact_match(tmp_path):
    """Verify that projects/aria/views/reference.json matches detected bounding boxes and metrics."""
    assert REFERENCE_PATH.is_file(), f"Reference file not found: {REFERENCE_PATH}"
    with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
        stored_ref = json.load(f)

    out_dir = tmp_path / "views_verify"
    res = sheet.split_sheet(SHEET_PATH, out_dir)
    generated_ref = res["reference"]

    # 1. Bounding box aspect ratios match stored aspect metrics exactly
    for v in ("front", "side", "back"):
        bx, by, bw, bh = res["boxes"][v]
        expected_aspect = float(bw / bh)
        stored_aspect = stored_ref["aspect"][v]
        assert abs(expected_aspect - stored_aspect) < 1e-6, (
            f"View {v} aspect mismatch: bw/bh={expected_aspect}, stored={stored_aspect}"
        )
        assert abs(generated_ref["aspect"][v] - stored_aspect) < 1e-6

    # 2. Neck ratio match
    assert abs(generated_ref["neck_ratio"] - stored_ref["neck_ratio"]) < 1e-6

    # 3. Palette match
    assert generated_ref["palette"] == stored_ref["palette"]

    # 4. Width profile match across all 64 bins for all views
    for v in ("front", "side", "back"):
        gen_prof = generated_ref["profile"][v]
        stor_prof = stored_ref["profile"][v]
        assert len(gen_prof) == len(stor_prof) == 64
        diff = np.max(np.abs(np.array(gen_prof) - np.array(stor_prof)))
        assert diff < 1e-6, f"Profile mismatch in view {v}: max diff = {diff}"
