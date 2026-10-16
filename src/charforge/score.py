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


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True segments of a 1-D bool array as (start, end_exclusive)."""
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def pose_gaps(mask: np.ndarray, gap_min: float | None = None) -> dict:
    """A-pose readiness of a FRONT silhouette: is there background between each forearm/hand and the body,
    and between the shins? Touching parts make auto-rigging bleed weights ("sticking" when animated).
    Returns {"arms": {"left": part, "right": part}, "legs": part, "t_pose": bool, "ok": bool} with
    part = {"separated": bool, "share": rows showing the gap / rows checked, "gap": median gap / figure height}.
    left/right are the character's: the front view shows the character's left on the image right.
    Arms: rows at 30-62% of the height; the outermost run must be arm-thin (< 12% height) with a gap after it.
    Legs: rows at 72-92% of the height must show two runs around the centre with a gap."""
    gap_min = config.POSE_GAP_MIN if gap_min is None else gap_min
    c = canonical_mask(mask)
    ys = np.nonzero(c.any(axis=1))[0]
    if ys.size == 0:
        none = {"separated": False, "share": 0.0, "gap": None}
        return {"arms": {"left": dict(none), "right": dict(none)}, "legs": dict(none), "t_pose": False, "ok": False}
    top, bottom = int(ys.min()), int(ys.max()) + 1
    h = bottom - top
    min_px = max(2.0, gap_min * h)
    t_pose = arm_span_ratio(c) > 0.75  # arms held out sideways cannot touch the hips

    arm_rows = range(top + int(0.30 * h), top + int(0.62 * h))
    cx = c.shape[1] / 2.0
    found = {"left": [], "right": []}
    for y in arm_rows:
        rs = _runs(c[y])
        if len(rs) < 2:
            continue
        # body span: the run over the centre (torso), or the two legs around a background centre (below the crotch);
        # only runs outside it can be arms, so thighs are never counted as arms
        mid = next((k for k, (a, b) in enumerate(rs) if a <= cx < b), None)
        if mid is not None:
            lo_k = hi_k = mid
        else:
            hi_k = next((k for k, (a, _) in enumerate(rs) if a >= cx), len(rs))
            lo_k = hi_k - 1
            if lo_k < 0 or hi_k >= len(rs):
                continue
        if lo_k >= 1:  # image left of the body = character right
            (a0, a1), (b0, _) = rs[lo_k - 1], rs[lo_k]
            if a1 - a0 < 0.12 * h and b0 - a1 >= min_px:
                found["right"].append((b0 - a1) / h)
        if hi_k + 1 < len(rs):
            (_, p1), (l0, l1) = rs[hi_k], rs[hi_k + 1]
            if l1 - l0 < 0.12 * h and l0 - p1 >= min_px:
                found["left"].append((l0 - p1) / h)
    arms = {}
    for side in ("left", "right"):
        share = len(found[side]) / max(1, len(arm_rows))
        arms[side] = {
            "separated": bool(t_pose or share >= config.POSE_ARM_BAND_MIN),
            "share": round(share, 3),
            "gap": round(float(np.median(found[side])), 4) if found[side] else None,
        }

    leg_rows = range(top + int(0.72 * h), top + int(0.92 * h))
    cx = c.shape[1] / 2.0
    leg_gaps = []
    for y in leg_rows:
        rs = [r for r in _runs(c[y]) if r[1] - r[0] >= 2]
        left_of = [r for r in rs if (r[0] + r[1]) / 2 < cx]
        right_of = [r for r in rs if (r[0] + r[1]) / 2 >= cx]
        if left_of and right_of:
            g = right_of[0][0] - left_of[-1][1]
            if g >= min_px:
                leg_gaps.append(g / h)
    share = len(leg_gaps) / max(1, len(leg_rows))
    legs = {
        "separated": bool(share >= config.POSE_LEG_BAND_MIN),
        "share": round(share, 3),
        "gap": round(float(np.median(leg_gaps)), 4) if leg_gaps else None,
    }
    return {"arms": arms, "legs": legs, "t_pose": t_pose, "ok": bool(arms["left"]["separated"] and arms["right"]["separated"] and legs["separated"])}


def pose_mismatch(reference_pose: dict | None, mesh_pose: dict | None) -> list[str]:
    """Parts apart in the sheet but touching in the mesh render, e.g. ["left arm"]: the generator fused them."""
    if not reference_pose or not mesh_pose:
        return []
    out = []
    for name, get in (
        ("left arm", lambda d: ((d.get("arms") or {}).get("left") or {}).get("separated")),
        ("right arm", lambda d: ((d.get("arms") or {}).get("right") or {}).get("separated")),
        ("legs", lambda d: (d.get("legs") or {}).get("separated")),
    ):
        if get(reference_pose) is True and get(mesh_pose) is False:
            out.append(name)
    return out


def integrity_report(raw: dict | None, cfg: dict | None = None) -> dict | None:
    """Turn the raw geometry counts from render_views --integrity into {..., ok, issues}.
    ok = no detached parts, few non-manifold edges and few self-intersecting faces (rig-breaking defects);
    debris and holes are reported but do not fail it."""
    if not raw:
        return None
    cfg = cfg or config.INTEGRITY
    if raw.get("issues"):  # e.g. no geometry
        return {**raw, "ok": False}
    edges, faces = max(1, raw.get("edges", 0)), max(1, raw.get("faces", 0))
    nm_ratio = raw.get("non_manifold_edges", 0) / edges
    si_ratio = raw.get("self_intersecting_faces", 0) / faces
    issues = []
    if raw.get("detached_parts"):
        issues.append(
            f"{raw['detached_parts']} detached part(s) ({raw['detached_pct']}% of the mesh) float away from the body: "
            "they will not follow the rig"
        )
    if raw.get("debris_parts"):
        issues.append(f"{raw['debris_parts']} floating fragment(s) of debris ({raw['debris_pct']}% of the mesh)")
    if nm_ratio > cfg["non_manifold_max"]:
        issues.append(f"{100 * nm_ratio:.1f}% of edges are non-manifold (3+ faces or wire edges)")
    if si_ratio > cfg["self_intersect_max"]:
        issues.append(f"{100 * si_ratio:.1f}% of faces cut through other faces")
    if raw.get("holes", 0) > cfg["holes_max"]:
        issues.append(f"{raw['holes']} open holes in the surface")
    ok = not raw.get("detached_parts") and nm_ratio <= cfg["non_manifold_max"] and si_ratio <= cfg["self_intersect_max"]
    return {
        **raw,
        "non_manifold_ratio": round(nm_ratio, 5),
        "self_intersection_ratio": round(si_ratio, 5),
        "ok": bool(ok),
        "issues": issues,
    }


def pose_warnings(pose: dict | None, subject: str = "sheet") -> list[str]:
    """Agent-facing warnings for touching parts in a front silhouette."""
    if not pose:
        return []
    out = []
    touching = [s for s in ("left", "right") if not ((pose.get("arms") or {}).get(s) or {}).get("separated", True)]
    if touching:
        out.append(
            f"{subject} front view: {' and '.join(touching)} arm touches the body; rig weights will bleed there "
            "(sticking when animated). Best fix: a sheet in A-pose with a clear gap under the arms."
        )
    if not (pose.get("legs") or {}).get("separated", True):
        out.append(f"{subject} front view: legs are not apart (long skirt or legs together); thighs may stick when walking.")
    return out


def clip_sheet(files: dict[str, list[str]], frames: list[float], title: str, out: Path, height: int = 300) -> Path:
    """Contact sheet of one animation clip: a labeled row per view, a column per sampled frame, on white."""
    views = [v for v in files if files[v]]
    pad, head = 8, 34
    tiles: dict[str, list[np.ndarray]] = {}
    for v in views:
        row = []
        for path in files[v]:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                img = np.full((height, height, 3), 255, np.uint8)
            elif img.ndim == 3 and img.shape[2] == 4:
                a = img[..., 3:] / 255.0
                img = (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
            row.append(cv2.resize(img, (max(1, round(img.shape[1] * height / img.shape[0])), height)))
        tiles[v] = row
    ncol = max((len(r) for r in tiles.values()), default=0)
    tile_w = max((t.shape[1] for r in tiles.values() for t in r), default=height)
    label_w = 70
    W = label_w + ncol * (tile_w + pad) + pad
    H = head + len(views) * (height + pad + 18) + pad
    canvas = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(canvas, title, (pad, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    for r, v in enumerate(views):
        y0 = head + r * (height + pad + 18)
        cv2.putText(canvas, v, (pad, y0 + height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
        for k, t in enumerate(tiles[v]):
            x0 = label_w + pad + k * (tile_w + pad)
            canvas[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
            cv2.rectangle(canvas, (x0, y0), (x0 + t.shape[1] - 1, y0 + t.shape[0] - 1), (225, 225, 225), 1)
            if r == len(views) - 1 and k < len(frames):
                cv2.putText(canvas, f"f{frames[k]:g}", (x0 + 4, y0 + height + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out


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
