"""Turnaround sheet -> per-view reference images, masks and metrics.   [TODO L1-T1, L1-T2]

Pure numpy/OpenCV, no Blender. Writes into <project>/views/. Returned paths are project-relative.
Spec and acceptance tests: docs/ROADMAP.md (L1-T1, L1-T2), tests/test_sheet.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import config, score
from .envelope import CFError

Box = tuple[int, int, int, int]  # x, y, w, h in sheet pixels


def foreground_mask(img: np.ndarray, bg_tol: float = 18.0) -> np.ndarray:
    """[L1-T1] RGB(A) uint8 image -> bool mask of non-background pixels.
    1. If an alpha channel exists and is not all 255: return alpha > 127.
    2. bg = median color of an 8 px border ring; fg = ||Lab(px) - Lab(bg)|| > bg_tol.
    3. Morphological close (7x7 ellipse), then open (3x3); fill holes (flood fill from the border on the inverse).
    """
    if img.ndim == 3 and img.shape[2] == 4 and not np.all(img[..., 3] == 255):
        return img[..., 3] > 127

    bgr = img[..., :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    border_px = np.concatenate([
        bgr[:8, :].reshape(-1, 3),
        bgr[-8:, :].reshape(-1, 3),
        bgr[:, :8].reshape(-1, 3),
        bgr[:, -8:].reshape(-1, 3),
    ], axis=0)
    bg_color = np.median(border_px, axis=0).astype(np.uint8)

    lab_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_bg = cv2.cvtColor(bg_color.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32).reshape(3)
    dist = np.sqrt(np.sum((lab_img - lab_bg) ** 2, axis=2))
    fg = (dist > bg_tol).astype(np.uint8) * 255

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # Hole filling: flood fill from border on the inverse
    padded = np.pad(opened, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 128)
    filled = (flood[1:-1, 1:-1] != 128)
    return filled


def detect_candidates(mask: np.ndarray) -> list[Box]:
    """[L1-T1] Full-body figure boxes, sorted left -> right.
    1. Label the mask dilated with a tall kernel (3x25) so floating hair/accessories join their figure (8-connectivity).
    2. Keep boxes with h >= 0.6 * tallest h and h / w >= 1.6; drop components < 0.2% of the image area.
       (Palette swatches, text and close-ups fail these rules.)
    """
    H, W = mask.shape[:2]
    min_area = 0.002 * (H * W)
    kernel = np.ones((25, 3), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    candidates = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        ys, xs = np.nonzero(mask & (labels == lbl))
        if xs.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        w = x1 - x0 + 1
        h = y1 - y0 + 1
        candidates.append((x0, y0, w, h))

    if not candidates:
        return []

    tallest_h = max(c[3] for c in candidates)
    filtered = [
        c for c in candidates
        if c[3] >= 0.6 * tallest_h and (c[3] / c[2]) >= 1.6
    ]
    filtered.sort(key=lambda b: b[0])
    return filtered


def assign_views(cands: list[Box]) -> tuple[dict[str, int], float]:
    """[L1-T1] Map candidates to views -> ({view: index}, confidence 0..1).
    side = narrowest box; front = leftmost of the rest; back = rightmost of the rest.
    confidence = 1.0 if exactly 3 candidates and side.w / front.w in [0.3, 0.85]; 0.5 if more than 3 or the ratio is off.
    Fewer than 3 candidates -> raise CFError("SPLIT_FAILED", ..., fix="Call sheet_split with boxes={front,side,back: [x,y,w,h]}").
    """
    if len(cands) < 3:
        raise CFError(
            "SPLIT_FAILED",
            f"Detected {len(cands)} figures, need at least 3.",
            "Call sheet_split with boxes={front,side,back: [x,y,w,h]}",
        )
    i_side = min(range(len(cands)), key=lambda i: cands[i][2])
    rest = [i for i in range(len(cands)) if i != i_side]
    i_front = min(rest, key=lambda i: cands[i][0])
    i_back = max(rest, key=lambda i: cands[i][0])

    side_w = cands[i_side][2]
    front_w = cands[i_front][2]
    ratio = side_w / front_w if front_w > 0 else 0.0

    if len(cands) == 3 and (0.3 <= ratio <= 0.85):
        confidence = 1.0
    else:
        confidence = 0.5

    return {"front": i_front, "side": i_side, "back": i_back}, confidence


def split_sheet(sheet: Path, out_dir: Path, assign: list[int] | None = None, boxes: dict[str, Box] | None = None) -> dict:
    """[L1-T1] Main entry, called by the sheet_split tool.
    Box source priority: boxes (manual) > assign (indices into candidates for [front, side, back]) > assign_views().
    Manual boxes or assign -> confidence 1.0.
    Per view: crop box + 2% padding -> views/<view>.png (RGBA, alpha = mask) and views/<view>_mask.png (0/255).
    views/preview.jpg: sheet scaled to 1024 px wide, every candidate drawn with its index, assigned views labeled.
    Returns {"candidates": [Box], "boxes": {view: Box}, "confidence": float,
             "files": {view: {"image": "views/front.png", "mask": "views/front_mask.png"}},
             "preview": "views/preview.jpg", "reference": reference_metrics(...)}
    and writes views/reference.json with the same "reference" dict.
    """
    sheet = Path(sheet)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(sheet), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Sheet image not found: {sheet}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    mask = foreground_mask(img)
    candidates = detect_candidates(mask)

    if boxes is not None:
        missing = {"front", "side", "back"} - set(boxes.keys())
        if missing:
            raise CFError("SPLIT_FAILED", f"Missing boxes for views: {sorted(missing)}", "Call sheet_split with boxes={front,side,back: [x,y,w,h]}")
        for v, b in boxes.items():
            if len(b) != 4 or b[2] <= 0 or b[3] <= 0:
                raise CFError("SPLIT_FAILED", f"Invalid box for view '{v}': {b}", "Box format must be [x, y, w, h] with w > 0 and h > 0.")
        view_boxes = {k: tuple(boxes[k]) for k in ("front", "side", "back")}
        confidence = 1.0
    elif assign is not None:
        if len(assign) != 3 or any(i < 0 or i >= len(candidates) for i in assign):
            raise CFError("SPLIT_FAILED", f"Invalid assign indices: {assign}", "Specify 3 valid candidate indices.")
        view_boxes = {
            "front": candidates[assign[0]],
            "side": candidates[assign[1]],
            "back": candidates[assign[2]],
        }
        confidence = 1.0
    else:
        assigned_map, confidence = assign_views(candidates)
        view_boxes = {v: candidates[idx] for v, idx in assigned_map.items()}

    H, W = img.shape[:2]
    cropped_masks = {}
    cropped_rgbs = {}

    for v in ("front", "side", "back"):
        bx, by, bw, bh = view_boxes[v]
        pad_x = round(0.02 * bw)
        pad_y = round(0.02 * bh)
        x0 = max(0, bx - pad_x)
        y0 = max(0, by - pad_y)
        x1 = min(W, bx + bw + pad_x)
        y1 = min(H, by + bh + pad_y)

        c_img = img[y0:y1, x0:x1]
        c_mask = mask[y0:y1, x0:x1]
        cropped_masks[v] = c_mask

        # Mask uint8 (0 / 255)
        mask_uint8 = (c_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(out_dir / f"{v}_mask.png"), mask_uint8)

        # RGBA with alpha = mask
        bgr = c_img[..., :3]
        rgba = np.dstack([bgr, mask_uint8])
        cv2.imwrite(str(out_dir / f"{v}.png"), rgba)

        if v == "front":
            cropped_rgbs["front"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Preview image (1024px width)
    scale = 1024.0 / W
    new_w = 1024
    new_h = max(1, round(H * scale))
    preview_img = cv2.resize(img[..., :3].copy(), (new_w, new_h), interpolation=cv2.INTER_AREA)

    box_to_view = {tuple(b): v for v, b in view_boxes.items()}
    for i, cand in enumerate(candidates):
        cx, cy, cw, ch = cand
        sx, sy = round(cx * scale), round(cy * scale)
        sw, sh = round(cw * scale), round(ch * scale)
        v_name = box_to_view.get(tuple(cand))

        color = (0, 220, 0) if v_name else (0, 140, 255)
        cv2.rectangle(preview_img, (sx, sy), (sx + sw, sy + sh), color, 2)

        label = f"[{i}] {v_name.upper()}" if v_name else f"[{i}]"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = max(th + 6, sy - 6)
        cv2.rectangle(preview_img, (sx, ty - th - 4), (sx + tw + 4, ty + baseline), (0, 0, 0), -1)
        cv2.putText(preview_img, label, (sx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / "preview.jpg"), preview_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    try:
        ref_data = reference_metrics(cropped_masks, cropped_rgbs.get("front", np.zeros((1, 1, 3), np.uint8)))
        with open(out_dir / "reference.json", "w", encoding="utf-8") as f:
            json.dump(ref_data, f, indent=2)
    except NotImplementedError:
        ref_data = {}

    return {
        "candidates": [tuple(b) for b in candidates],
        "boxes": {v: tuple(view_boxes[v]) for v in ("front", "side", "back")},
        "confidence": float(confidence),
        "files": {v: {"image": f"views/{v}.png", "mask": f"views/{v}_mask.png"} for v in ("front", "side", "back")},
        "preview": "views/preview.jpg",
        "reference": ref_data,
    }


def reference_metrics(masks: dict[str, np.ndarray], front_rgb: np.ndarray) -> dict:
    """[L1-T2] Metrics consumed by score_mesh, mesh_fit and the blockout.
    aspect[view]  = bbox w / h of the mask
    profile[view] = score.width_profile(score.canonical_mask(mask))  (PROFILE_BINS floats)
    neck_ratio    = (argmin of the front profile within 8-30% height from the top + 0.5) / PROFILE_BINS
    palette       = 8 hex colors "#rrggbb": cv2.kmeans over front foreground pixels, sorted by cluster size
    pose          = score.pose_gaps(front mask): are forearms/hands and shins visibly apart from the body
    """
    aspect = {}
    profile = {}
    for v in ("front", "side", "back"):
        m = masks[v]
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            aspect[v] = 0.0
        else:
            aspect[v] = float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))
        canon = score.canonical_mask(m)
        profile[v] = score.width_profile(canon, bins=config.PROFILE_BINS)

    bins = config.PROFILE_BINS
    i_start = int(round(0.08 * bins))
    i_end = int(round(0.30 * bins))
    front_prof = profile["front"]
    sub_prof = front_prof[i_start:i_end]
    min_idx = i_start + int(np.argmin(sub_prof))
    neck_ratio = float((min_idx + 0.5) / bins)

    fg_pixels = front_rgb[masks["front"]]
    if len(fg_pixels) < 8:
        palette = ["#000000"] * 8
    else:
        data = fg_pixels.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        K = 8
        _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()
        counts = np.bincount(labels, minlength=K)
        order = np.argsort(-counts)
        palette = [
            f"#{int(np.clip(round(centers[i][0]), 0, 255)):02x}{int(np.clip(round(centers[i][1]), 0, 255)):02x}{int(np.clip(round(centers[i][2]), 0, 255)):02x}"
            for i in order
        ]

    return {
        "aspect": aspect,
        "profile": profile,
        "neck_ratio": neck_ratio,
        "palette": palette,
        "pose": score.pose_gaps(masks["front"]),
    }
