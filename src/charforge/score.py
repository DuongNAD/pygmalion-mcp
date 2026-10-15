"""Silhouette and proportion scoring. Deterministic: the agent must trust these numbers over its own eyes.

canonical_mask() removes scale and translation, so hand-drawn reference views and Blender renders
compare directly without matching cameras. Spec of the TODO parts: docs/ROADMAP.md L1-T3.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config


def load_mask(path: Path | str) -> np.ndarray:
    """PNG -> bool mask: alpha channel if present (renders), else gray > 127 (reference *_mask.png)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:
        return img[..., 3] > 127
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img > 127


def canonical_mask(mask: np.ndarray, size: int = config.CANON_SIZE) -> np.ndarray:
    """Crop to bbox, scale to 94% of `size` tall (shrink more if too wide), 3% bottom margin, centroid x centered."""
    ys, xs = np.nonzero(mask)
    out = np.zeros((size, size), bool)
    if ys.size == 0:
        return out
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    h, w = crop.shape
    s = min(0.94 * size / h, 0.98 * size / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA) > 127
    cx = int(round(np.nonzero(crop)[1].mean())) if crop.any() else nw // 2
    top, left = size - int(0.03 * size) - nh, size // 2 - cx
    x0, x1 = max(0, left), min(size, left + nw)
    out[top:top + nh, x0:x1] = crop[:, x0 - left:x1 - left]
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def extract_torso_core(canon: np.ndarray) -> np.ndarray:
    """Extract torso core silhouette (|x - 256| <= 0.22 * height) so arm extensions do not corrupt torso width profile."""
    ys = np.nonzero(canon.any(axis=1))[0]
    if ys.size == 0:
        return np.zeros_like(canon, dtype=bool)
    top, bottom = ys.min(), ys.max() + 1
    height = max(1, bottom - top)
    max_dx = 0.22 * height
    x_indices = np.arange(canon.shape[1])
    core_cols = np.abs(x_indices - (canon.shape[1] // 2)) <= max_dx
    core = np.zeros_like(canon, dtype=bool)
    core[:, core_cols] = canon[:, core_cols]
    return core


torso_core_mask = extract_torso_core


def arm_span_ratio(canon: np.ndarray) -> float:
    """Maximum horizontal width ratio in the arm band (18% - 65% from top)."""
    ys, xs = np.nonzero(canon)
    if ys.size == 0:
        return 0.0
    h = max(1, ys.max() - ys.min() + 1)
    y_top = ys.min()
    y_start = int(y_top + 0.18 * h)
    y_end = int(y_top + 0.65 * h)
    arm_band = canon[y_start:y_end, :]
    ys_a, xs_a = np.nonzero(arm_band)
    if xs_a.size == 0:
        return 0.0
    return float((xs_a.max() - xs_a.min() + 1) / h)


def width_profile(canon: np.ndarray, bins: int = config.PROFILE_BINS, torso_only: bool = False) -> list[float]:
    """Filled width per horizontal band (top -> bottom) as a fraction of the figure height."""
    ys = np.nonzero(canon.any(axis=1))[0]
    if ys.size == 0:
        return [0.0] * bins
    top, bottom = ys.min(), ys.max() + 1
    edges = np.linspace(top, bottom, bins + 1).astype(int)
    height = bottom - top
    mask = extract_torso_core(canon) if torso_only else canon
    return [float(mask[a:max(b, a + 1)].any(axis=0).sum() / height) for a, b in zip(edges[:-1], edges[1:])]


def contact_sheet(rows: list[list[Path | str]], out: Path, height: int = 384) -> Path:
    """Grid of images (each row = one view: reference, render, diff...) on white, for visual review."""
    def tile(path):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return np.full((height, height // 2, 3), 255, np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:  # composite on white
            a = img[..., 3:] / 255.0
            img = (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
        return cv2.resize(img, (max(1, round(img.shape[1] * height / img.shape[0])), height))

    strips = [np.hstack([tile(p) for p in row]) for row in rows]
    width = max(s.shape[1] for s in strips)
    grid = np.vstack([np.pad(s, ((0, 0), (0, width - s.shape[1]), (0, 0)), constant_values=255) for s in strips])
    cv2.imwrite(str(out), grid)
    return out


def compare(
    ref: dict[str, np.ndarray],
    ren: dict[str, np.ndarray],
    weights: dict,
    thresholds: dict,
    neck_ratio: float = 0.15,
    provider_mesh: bool = False,
) -> dict:
    """[L1-T3] ref/ren: {view: raw bool mask}. Returns
    {"iou": {view: float}, "total": float, "passed": bool, "side_mirrored": bool, "rotated_90": bool,
     "aspect_delta": {view: float}, "worst_bands": [{"view", "from", "to", "region", "delta"}], "hints": [str]}
    - canonicalize both; side IoU = max(IoU(ref, ren), IoU(ref, fliplr(ren))), side_mirrored = flipped won.
    - total = sum(weights[v] * iou[v]); passed = (total >= PROVIDER_ACCEPT_TOTAL) for provider meshes,
      else all(iou[v] >= thresholds[v]); round floats to 4 places.
    - worst_bands: width_profile delta (ren - ref) per bin; merge adjacent bins with the same sign and |delta| > 0.02;
      keep the top 3 by summed |delta|; from/to = % of height from the top; region by band center:
      < neck_ratio "head", < 0.45 "torso", < 0.60 "hips/skirt", else "legs"; delta = mean delta.
    - hints: one line per worst band, e.g. "front 38-46% (torso): mesh 11% wider than reference -> make it narrower".
    - rotated_90 = IoU(canon ref side, canon ren front) > IoU(canon ref front, canon ren front) + 0.05.
    """
    c_ref = {v: canonical_mask(ref[v]) for v in ("front", "side", "back")}
    c_ren = {v: canonical_mask(ren[v]) for v in ("front", "side", "back")}

    c_ren_side_normal = c_ren["side"]
    c_ren_side_flipped = canonical_mask(np.fliplr(ren["side"]))
    iou_normal = iou(c_ref["side"], c_ren_side_normal)
    iou_flipped = iou(c_ref["side"], c_ren_side_flipped)

    side_mirrored = bool(iou_flipped > iou_normal)
    side_iou = iou_flipped if side_mirrored else iou_normal
    c_ren_side_best = c_ren_side_flipped if side_mirrored else c_ren_side_normal

    ious = {
        "front": round(iou(c_ref["front"], c_ren["front"]), 4),
        "side": round(side_iou, 4),
        "back": round(iou(c_ref["back"], c_ren["back"]), 4),
    }

    total = round(sum(weights[v] * ious[v] for v in ("front", "side", "back")), 4)
    if provider_mesh:
        passed = bool(total >= config.PROVIDER_ACCEPT_TOTAL)
    else:
        passed = bool(all(ious[v] >= thresholds[v] for v in ("front", "side", "back")))
    rotated_90 = bool(iou(c_ref["side"], c_ren["front"]) > iou(c_ref["front"], c_ren["front"]) + 0.05)

    def _aspect(m: np.ndarray) -> float:
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            return 0.0
        return float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))

    aspect_delta = {
        v: round(_aspect(ren[v]) - _aspect(ref[v]), 4)
        for v in ("front", "side", "back")
    }

    candidate_bands = []
    for v in ("front", "side", "back"):
        ref_prof = width_profile(c_ref[v])
        ren_prof = width_profile(c_ren_side_best if v == "side" else c_ren[v])
        deltas = [ren_prof[i] - ref_prof[i] for i in range(len(ref_prof))]

        i = 0
        n = len(deltas)
        while i < n:
            d = deltas[i]
            if abs(d) > 0.02:
                sign = 1 if d > 0 else -1
                start_idx = i
                band_deltas = [d]
                i += 1
                while i < n and abs(deltas[i]) > 0.02 and ((deltas[i] > 0) == (sign > 0)):
                    band_deltas.append(deltas[i])
                    i += 1
                end_idx = i - 1

                sum_abs = sum(abs(x) for x in band_deltas)
                mean_delta = round(float(np.mean(band_deltas)), 4)
                from_pct = int(round(start_idx / n * 100))
                to_pct = int(round((end_idx + 1) / n * 100))
                center = ((start_idx + end_idx + 1) / 2) / n

                if center < neck_ratio:
                    region = "head"
                elif center < 0.45:
                    region = "torso"
                elif center < 0.60:
                    region = "hips/skirt"
                else:
                    region = "legs"

                candidate_bands.append({
                    "view": v,
                    "from": from_pct,
                    "to": to_pct,
                    "region": region,
                    "delta": mean_delta,
                    "_sum_abs": sum_abs,
                })
            else:
                i += 1

    candidate_bands.sort(key=lambda b: b["_sum_abs"], reverse=True)
    top_bands = candidate_bands[:3]
    worst_bands = [
        {"view": b["view"], "from": b["from"], "to": b["to"], "region": b["region"], "delta": b["delta"]}
        for b in top_bands
    ]

    hints = []
    for b in worst_bands:
        pct = int(round(abs(b["delta"]) * 100))
        direction = "narrower" if b["delta"] > 0 else "wider"
        adjective = "wider" if b["delta"] > 0 else "narrower"
        hints.append(f"{b['view']} {b['from']}-{b['to']}% ({b['region']}): mesh {pct}% {adjective} than reference -> make it {direction}")

    return {
        "iou": ious,
        "total": total,
        "passed": passed,
        "side_mirrored": side_mirrored,
        "rotated_90": rotated_90,
        "aspect_delta": aspect_delta,
        "worst_bands": worst_bands,
        "hints": hints,
    }


def write_diffs(ref: dict[str, np.ndarray], ren: dict[str, np.ndarray], out_dir: Path) -> dict[str, Path]:
    """[L1-T3] Per view save <view>_diff.png from canonical masks (side: the better-matching flip):
    white = both, red = mesh only, blue = reference only. Returns {view: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for v in ("front", "side", "back"):
        c_ref = canonical_mask(ref[v])
        if v == "side":
            c_ren_normal = canonical_mask(ren["side"])
            c_ren_flipped = canonical_mask(np.fliplr(ren["side"]))
            c_ren = c_ren_flipped if iou(c_ref, c_ren_flipped) > iou(c_ref, c_ren_normal) else c_ren_normal
        else:
            c_ren = canonical_mask(ren[v])

        canvas = np.zeros((config.CANON_SIZE, config.CANON_SIZE, 3), dtype=np.uint8)
        canvas[c_ref & c_ren] = (255, 255, 255)       # white: both
        canvas[c_ren & ~c_ref] = (0, 0, 255)         # red (BGR): mesh only
        canvas[c_ref & ~c_ren] = (255, 0, 0)         # blue (BGR): reference only

        diff_path = out_dir / f"{v}_diff.png"
        cv2.imwrite(str(diff_path), canvas)
        paths[v] = diff_path

    return paths
