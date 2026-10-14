import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from charforge import config, sheet
from charforge.envelope import CFError
from tests.conftest import draw_figure


def make_base_sheet(bg_color=(255, 255, 255), shape=(900, 1600), channels=3, alpha=None):
    """Generate a standard 3-figure turnaround sheet on a custom background."""
    if channels == 4:
        img = np.zeros((*shape, 4), dtype=np.uint8)
        img[..., :3] = bg_color[:3]
        img[..., 3] = 0 if alpha is None else alpha
    else:
        img = np.full((*shape, 3), bg_color, dtype=np.uint8)

    # Draw 3 figures: front (cx=300, ws=1.0), side (cx=750, ws=0.55, nose=18), back (cx=1200, ws=1.0)
    for cx, ws, nose in ((300, 1.0, 0), (750, 0.55, 18), (1200, 1.0, 0)):
        if channels == 4 and alpha is None:
            # For RGBA transparent background, draw figures with alpha=255
            fig_img = np.zeros((*shape, 3), dtype=np.uint8)
            draw_figure(fig_img, cx, 60, 760, ws, nose)
            fig_mask = (fig_img != 0).any(axis=2)
            img[fig_mask, :3] = fig_img[fig_mask]
            img[fig_mask, 3] = 255
        else:
            draw_figure(img, cx, 60, 760, ws, nose)
            if channels == 4:
                # Set figure alpha
                fig_mask = (img[..., :3] != bg_color[:3]).any(axis=2)
                img[fig_mask, 3] = 255

    return img


# ==============================================================================
# Category 1: Non-standard sheets
# ==============================================================================

def test_rgba_transparent_background(tmp_path):
    """Test sheet with RGBA transparent background (alpha == 0 for bg, 255 for figures)."""
    img = make_base_sheet(channels=4, alpha=None)
    sheet_file = tmp_path / "sheet_rgba.png"
    cv2.imwrite(str(sheet_file), img)

    out_dir = tmp_path / "views"
    res = sheet.split_sheet(sheet_file, out_dir)
    assert len(res["candidates"]) == 3
    assert res["confidence"] >= 0.8
    assert res["boxes"]["front"][0] < res["boxes"]["side"][0] < res["boxes"]["back"][0]
    for v in ("front", "side", "back"):
        assert (out_dir / f"{v}.png").is_file()
        assert (out_dir / f"{v}_mask.png").is_file()
    assert (out_dir / "preview.jpg").is_file()
    assert (out_dir / "reference.json").is_file()


def test_rgba_opaque_background(tmp_path):
    """Test sheet with RGBA format but alpha channel is all 255 (should fall back to Lab color diff)."""
    img = make_base_sheet(bg_color=(255, 255, 255), channels=4, alpha=255)
    sheet_file = tmp_path / "sheet_rgba_opaque.png"
    cv2.imwrite(str(sheet_file), img)

    out_dir = tmp_path / "views"
    res = sheet.split_sheet(sheet_file, out_dir)
    assert len(res["candidates"]) == 3
    assert res["confidence"] >= 0.8


def test_rgb_dark_background(tmp_path):
    """Test sheet with solid dark background (e.g. dark gray / black)."""
    img = make_base_sheet(bg_color=(20, 20, 20), channels=3)
    sheet_file = tmp_path / "sheet_dark.png"
    cv2.imwrite(str(sheet_file), img)

    out_dir = tmp_path / "views"
    res = sheet.split_sheet(sheet_file, out_dir)
    assert len(res["candidates"]) == 3
    assert res["confidence"] >= 0.8


def test_rgb_colored_background(tmp_path):
    """Test sheet with solid colored background (e.g. beige / tan)."""
    img = make_base_sheet(bg_color=(200, 215, 225), channels=3)  # BGR
    sheet_file = tmp_path / "sheet_beige.png"
    cv2.imwrite(str(sheet_file), img)

    out_dir = tmp_path / "views"
    res = sheet.split_sheet(sheet_file, out_dir)
    assert len(res["candidates"]) == 3
    assert res["confidence"] >= 0.8


def test_rgb_noisy_border(tmp_path):
    """Test sheet with subtle texture/noise in border ring."""
    img = make_base_sheet(bg_color=(240, 240, 240), channels=3)
    # Add random noise of +-5 in border (well within bg_tol=18)
    rng = np.random.RandomState(42)
    noise = rng.randint(-5, 6, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    sheet_file = tmp_path / "sheet_noisy.png"
    cv2.imwrite(str(sheet_file), noisy_img)

    out_dir = tmp_path / "views"
    res = sheet.split_sheet(sheet_file, out_dir)
    assert len(res["candidates"]) == 3
    assert res["confidence"] >= 0.8


# ==============================================================================
# Category 2: Tiny swatches, text annotations, noisy borders, floating parts
# ==============================================================================

def test_swatches_and_annotations_and_floating_parts(tmp_path):
    """Test sheet with color swatches, text annotations, and floating accessories (staff, hair)."""
    img = make_base_sheet(bg_color=(255, 255, 255), channels=3)

    # 1. Color swatches (small squares)
    for i in range(8):
        x, y = 50 + i * 40, 50
        cv2.rectangle(img, (x, y), (x + 30, y + 30), (30 * i, 100, 200), -1)

    # 2. Text annotations: "FRONT", "SIDE", "BACK", copyright
    cv2.putText(img, "FRONT VIEW", (240, 860), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "SIDE VIEW", (710, 860), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "BACK VIEW", (1150, 860), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Model Sheet v1.0 (c) Studio", (50, 880), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    # 3. Floating detached accessories: floating hair ribbon 10px above head of front figure
    # Note: 25x3 vertical dilation kernel bridges vertical gaps up to ~12-24 px
    cv2.circle(img, (300, 48), 6, (90, 60, 160), -1)

    # 4. Tiny noise specs (<0.2% area) scattered around
    for px, py in [(100, 200), (500, 300), (950, 400), (1450, 600)]:
        cv2.rectangle(img, (px, py), (px + 5, py + 5), (0, 0, 0), -1)

    sheet_file = tmp_path / "sheet_cluttered.png"
    cv2.imwrite(str(sheet_file), img)

    out_dir = tmp_path / "views"
    res = sheet.split_sheet(sheet_file, out_dir)
    # The 3 main figures must be detected; swatches, text, and tiny noise must be rejected
    assert len(res["candidates"]) == 3
    assert res["confidence"] >= 0.8
    assert res["boxes"]["front"][0] < res["boxes"]["side"][0] < res["boxes"]["back"][0]


def test_tall_isolated_object_yields_candidates(tmp_path):
    """When a 4th tall object (e.g., full-height spear) is present, confidence drops to 0.5."""
    img = make_base_sheet(bg_color=(255, 255, 255), channels=3)
    # Draw a tall spear on the right (cx=1500, h=700, w=15) -> h/w > 1.6 and h >= 0.6*tallest
    cv2.rectangle(img, (1495, 80), (1505, 780), (40, 40, 180), -1)

    sheet_file = tmp_path / "sheet_spear.png"
    cv2.imwrite(str(sheet_file), img)

    out_dir = tmp_path / "views"
    res = sheet.split_sheet(sheet_file, out_dir)
    assert len(res["candidates"]) == 4
    # With 4 candidates, confidence must be 0.5 prompting manual confirmation
    assert res["confidence"] == 0.5


# ==============================================================================
# Category 3: Manual assign and manual boxes overrides
# ==============================================================================

def test_manual_assign_valid_and_invalid(sheet_path, tmp_path):
    out_dir = tmp_path / "views"
    # Valid manual assign
    res = sheet.split_sheet(sheet_path, out_dir, assign=[0, 1, 2])
    assert res["confidence"] == 1.0

    # Inverted manual assign
    res_inv = sheet.split_sheet(sheet_path, out_dir, assign=[2, 1, 0])
    assert res_inv["confidence"] == 1.0
    assert res_inv["boxes"]["front"][0] > res_inv["boxes"]["back"][0]

    # Invalid assign: wrong length (2 indices instead of 3)
    with pytest.raises(CFError) as exc_info:
        sheet.split_sheet(sheet_path, out_dir, assign=[0, 1])
    assert exc_info.value.code == "SPLIT_FAILED"

    # Invalid assign: out-of-bounds candidate index
    with pytest.raises(CFError) as exc_info:
        sheet.split_sheet(sheet_path, out_dir, assign=[0, 1, 99])
    assert exc_info.value.code == "SPLIT_FAILED"

    # Invalid assign: negative index
    with pytest.raises(CFError) as exc_info:
        sheet.split_sheet(sheet_path, out_dir, assign=[-1, 0, 1])
    assert exc_info.value.code == "SPLIT_FAILED"


def test_manual_boxes_override(sheet_path, tmp_path):
    out_dir = tmp_path / "views"
    custom_boxes = {
        "front": (200, 50, 220, 780),
        "side": (680, 50, 140, 780),
        "back": (1100, 50, 220, 780),
    }
    res = sheet.split_sheet(sheet_path, out_dir, boxes=custom_boxes)
    assert res["confidence"] == 1.0
    assert res["boxes"]["front"] == (200, 50, 220, 780)
    assert res["boxes"]["side"] == (680, 50, 140, 780)
    assert res["boxes"]["back"] == (1100, 50, 220, 780)

    for v in ("front", "side", "back"):
        assert (out_dir / f"{v}.png").is_file()
        assert (out_dir / f"{v}_mask.png").is_file()


def test_manual_boxes_boundary_clamping(sheet_path, tmp_path):
    """Boxes touching sheet boundaries (0, 0) should be clamped safely without crashing."""
    out_dir = tmp_path / "views"
    edge_boxes = {
        "front": (0, 0, 300, 800),
        "side": (600, 0, 200, 800),
        "back": (1300, 0, 300, 900),
    }
    res = sheet.split_sheet(sheet_path, out_dir, boxes=edge_boxes)
    assert res["confidence"] == 1.0
    for v in ("front", "side", "back"):
        assert (out_dir / f"{v}.png").is_file()


# ==============================================================================
# Category 4: Sheets with only 1 or 2 figures (must raise CFError SPLIT_FAILED)
# ==============================================================================

def test_insufficient_figures_zero(tmp_path):
    """Blank sheet with zero figures must raise CFError('SPLIT_FAILED')."""
    img = np.full((900, 1600, 3), 255, np.uint8)
    sheet_file = tmp_path / "sheet_blank.png"
    cv2.imwrite(str(sheet_file), img)

    with pytest.raises(CFError) as exc_info:
        sheet.split_sheet(sheet_file, tmp_path / "views")
    assert exc_info.value.code == "SPLIT_FAILED"
    assert "boxes={front,side,back" in exc_info.value.fix


def test_insufficient_figures_one(tmp_path):
    """Sheet with only 1 figure must raise CFError('SPLIT_FAILED')."""
    img = np.full((900, 1600, 3), 255, np.uint8)
    draw_figure(img, 800, 60, 760, 1.0, 0)
    sheet_file = tmp_path / "sheet_one.png"
    cv2.imwrite(str(sheet_file), img)

    with pytest.raises(CFError) as exc_info:
        sheet.split_sheet(sheet_file, tmp_path / "views")
    assert exc_info.value.code == "SPLIT_FAILED"
    assert "Detected 1 figures, need at least 3" in exc_info.value.message


def test_insufficient_figures_two(tmp_path):
    """Sheet with only 2 figures must raise CFError('SPLIT_FAILED')."""
    img = np.full((900, 1600, 3), 255, np.uint8)
    draw_figure(img, 500, 60, 760, 1.0, 0)
    draw_figure(img, 1100, 60, 760, 0.55, 18)
    sheet_file = tmp_path / "sheet_two.png"
    cv2.imwrite(str(sheet_file), img)

    with pytest.raises(CFError) as exc_info:
        sheet.split_sheet(sheet_file, tmp_path / "views")
    assert exc_info.value.code == "SPLIT_FAILED"
    assert "Detected 2 figures, need at least 3" in exc_info.value.message


# ==============================================================================
# Category 5: reference_metrics with single-pixel or extreme aspect ratios
# ==============================================================================

def test_reference_metrics_empty_masks():
    """All-zero masks should produce zero aspect/profiles without zero-division."""
    masks = {v: np.zeros((200, 200), bool) for v in ("front", "side", "back")}
    rgb = np.zeros((200, 200, 3), np.uint8)
    ref = sheet.reference_metrics(masks, rgb)

    assert ref["aspect"] == {"front": 0.0, "side": 0.0, "back": 0.0}
    assert len(ref["profile"]["front"]) == config.PROFILE_BINS
    assert all(x == 0.0 for x in ref["profile"]["front"])
    assert len(ref["palette"]) == 8


def test_reference_metrics_single_pixel_mask():
    """Single-pixel foreground mask should produce aspect=1.0 and valid profile/palette."""
    masks = {v: np.zeros((100, 100), bool) for v in ("front", "side", "back")}
    for v in ("front", "side", "back"):
        masks[v][50, 50] = True
    rgb = np.full((100, 100, 3), 128, np.uint8)

    ref = sheet.reference_metrics(masks, rgb)
    assert ref["aspect"]["front"] == 1.0
    assert len(ref["profile"]["front"]) == config.PROFILE_BINS
    assert 0.0 <= ref["neck_ratio"] <= 1.0
    assert len(ref["palette"]) == 8


def test_reference_metrics_extreme_tall_aspect():
    """Extreme tall/narrow mask (1x800) should compute aspect=1/800 without crash."""
    masks = {v: np.zeros((800, 50), bool) for v in ("front", "side", "back")}
    for v in ("front", "side", "back"):
        masks[v][:, 25] = True
    rgb = np.full((800, 50, 3), 200, np.uint8)

    ref = sheet.reference_metrics(masks, rgb)
    assert pytest.approx(ref["aspect"]["front"], 1e-3) == 1.0 / 800.0
    assert len(ref["profile"]["front"]) == config.PROFILE_BINS


def test_reference_metrics_extreme_wide_aspect():
    """Extreme flat/wide mask (800x1) should compute aspect=800 without crash."""
    masks = {v: np.zeros((50, 800), bool) for v in ("front", "side", "back")}
    for v in ("front", "side", "back"):
        masks[v][25, :] = True
    rgb = np.full((50, 800, 3), 200, np.uint8)

    ref = sheet.reference_metrics(masks, rgb)
    assert pytest.approx(ref["aspect"]["front"], 1e-3) == 800.0
    assert len(ref["profile"]["front"]) == config.PROFILE_BINS


def test_reference_metrics_palette_boundary_pixel_counts():
    """Palette extraction when fg_pixels is exactly 7 (below threshold 8) vs 8 (threshold)."""
    # Exactly 7 pixels -> fallback to 8 black colors
    masks7 = {v: np.zeros((50, 50), bool) for v in ("front", "side", "back")}
    masks7["front"][10, :7] = True
    for v in ("side", "back"):
        masks7[v][10, :10] = True
    rgb = np.zeros((50, 50, 3), np.uint8)
    rgb[10, :7] = [255, 0, 0]

    ref7 = sheet.reference_metrics(masks7, rgb)
    assert ref7["palette"] == ["#000000"] * 8

    # Exactly 8 pixels -> cv2.kmeans run with K=8
    masks8 = {v: np.zeros((50, 50), bool) for v in ("front", "side", "back")}
    masks8["front"][10, :8] = True
    for v in ("side", "back"):
        masks8[v][10, :10] = True
    rgb[10, :8] = [[i * 30, 100, 200] for i in range(8)]

    ref8 = sheet.reference_metrics(masks8, rgb)
    assert len(ref8["palette"]) == 8
    assert all(c.startswith("#") and len(c) == 7 for c in ref8["palette"])


# ==============================================================================
# Category 6: Empirical Findings / Edge Case Vulnerabilities
# ==============================================================================

def test_grayscale_sheet_input(tmp_path):
    """Grayscale sheet image should be converted/handled without raising ValueError."""
    img_color = np.full((900, 1600, 3), 255, dtype=np.uint8)
    for cx, ws, nose in ((300, 1.0, 0), (750, 0.55, 18), (1200, 1.0, 0)):
        draw_figure(img_color, cx, 60, 760, ws, nose)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    gray_file = tmp_path / "sheet_gray.png"
    cv2.imwrite(str(gray_file), img_gray)

    out_dir = tmp_path / "views"
    res = sheet.split_sheet(gray_file, out_dir)
    assert len(res["candidates"]) == 3


def test_manual_boxes_missing_view_key(sheet_path, tmp_path):
    """Passing boxes without all 3 views should raise CFError('SPLIT_FAILED'), not KeyError."""
    out_dir = tmp_path / "views"
    with pytest.raises(CFError) as exc_info:
        sheet.split_sheet(sheet_path, out_dir, boxes={"front": (200, 50, 220, 780)})
    assert exc_info.value.code == "SPLIT_FAILED"


def test_manual_boxes_zero_dimension(sheet_path, tmp_path):
    """Degenerate zero-size box should raise CFError('SPLIT_FAILED'), not cv2.error."""
    out_dir = tmp_path / "views"
    degenerate_boxes = {
        "front": (100, 50, 0, 0),
        "side": (600, 50, 100, 500),
        "back": (1100, 50, 100, 500),
    }
    with pytest.raises(CFError) as exc_info:
        sheet.split_sheet(sheet_path, out_dir, boxes=degenerate_boxes)
    assert exc_info.value.code == "SPLIT_FAILED"

