import cv2
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def tmp_home(tmp_path, monkeypatch):
    """Every test gets its own project root and Level 1."""
    monkeypatch.setenv("CHARFORGE_HOME", str(tmp_path / "projects"))
    monkeypatch.setenv("CHARFORGE_LEVEL", "1")
    return tmp_path / "projects"


def draw_figure(img, cx, top, h, ws, nose=0):
    """Toy character: head, torso, skirt, two legs. ws < 1 -> narrower (side view); nose > 0 adds a bump on the right."""
    c = (90, 60, 160)
    r = int(h * 0.08)
    cv2.circle(img, (cx, top + r), r, c, -1)
    if nose:
        cv2.circle(img, (cx + r, top + r), nose, c, -1)
    tw, sw = int(h * 0.14 * ws), int(h * 0.2 * ws)
    y1, y2 = top + int(h * 0.45), top + int(h * 0.6)
    cv2.rectangle(img, (cx - tw, top + 2 * r - 4), (cx + tw, y1), c, -1)
    cv2.fillPoly(img, [np.array([[cx - tw, y1], [cx + tw, y1], [cx + sw, y2], [cx - sw, y2]])], c)
    lw, gap = int(h * 0.04 * ws) + 2, int(h * 0.03 * ws)
    for s in (-1, 1):
        x = cx + s * (gap + lw)
        cv2.rectangle(img, (x - lw, y2), (x + lw, top + h), c, -1)


def figure_mask(h=760, ws=1.0, nose=0, shape=(900, 600)):
    img = np.full((*shape, 3), 255, np.uint8)
    draw_figure(img, shape[1] // 2, 60, h, ws, nose)
    return (img != 255).any(axis=2)


@pytest.fixture
def sheet_path(tmp_path):
    """1600x900 white sheet: front, side, back figures + palette swatches that must not be detected."""
    img = np.full((900, 1600, 3), 255, np.uint8)
    for cx, ws, nose in ((300, 1.0, 0), (750, 0.55, 18), (1200, 1.0, 0)):
        draw_figure(img, cx, 60, 760, ws, nose)
    for i in range(6):
        x, y = 1400 + (i % 2) * 70, 100 + (i // 2) * 70
        cv2.rectangle(img, (x, y), (x + 50, y + 50), (40 * i, 120, 200), -1)
    path = tmp_path / "sheet.png"
    cv2.imwrite(str(path), img)
    return path
