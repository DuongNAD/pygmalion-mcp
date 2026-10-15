"""Visual hull blockout generator. Creates a watertight, vertex-colored binary PLY mesh.

Conventions: Z-up, meters, character faces -Y.
Visual hull: intersection of canonical front, side (facing -Y) and mirrored back masks.
Smoothing: 10 Laplacian smoothing passes (lambda=0.5).
Colors: sampled from front/side/back images along the dominant normal axis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.measure import marching_cubes

from . import config, score
from .envelope import CFError


def _canonical_rgb(img: np.ndarray, mask: np.ndarray, size: int = config.CANON_SIZE) -> np.ndarray:
    """Project RGB image into canonical canvas using the same transform as canonical_mask."""
    ys, xs = np.nonzero(mask)
    out = np.full((size, size, 3), 255, dtype=np.uint8)
    if ys.size == 0:
        return out
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return out
    s = min(0.94 * size / h, 0.98 * size / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    crop_resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    mask_crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    mask_resized = cv2.resize(mask_crop, (nw, nh), interpolation=cv2.INTER_AREA) > 127
    cx = int(round(np.nonzero(mask_resized)[1].mean())) if mask_resized.any() else nw // 2
    top, left = size - int(0.03 * size) - nh, size // 2 - cx
    x0, x1 = max(0, left), min(size, left + nw)
    out[top:top + nh, x0:x1] = crop_resized[:, x0 - left:x1 - left]

    # Dilate interior foreground colors outward into background so 3D boundary vertices
    # never sample white background halos.
    fg = np.zeros((size, size), dtype=bool)
    fg[top:top + nh, x0:x1] = mask_resized[:, x0 - left:x1 - left]
    if fg.any() and not fg.all():
        fg_inner = cv2.erode(fg.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        cur_fg = fg_inner if fg_inner.any() else fg
        kernel = np.ones((5, 5), np.uint8)
        for _ in range(8):
            dil_rgb = cv2.dilate(np.where(cur_fg[:, :, None], out, 0), kernel, iterations=1)
            dil_mask = cv2.dilate(cur_fg.astype(np.uint8), kernel, iterations=1).astype(bool)
            new_px = dil_mask & ~cur_fg
            out[new_px] = dil_rgb[new_px]
            cur_fg = dil_mask
    return out



def _resolve_path(base_dir: Path, p: Any) -> Path:
    if isinstance(p, dict):
        p = p.get("mask") or p.get("image")
    path = Path(p)
    if not path.is_absolute():
        if (base_dir / path).exists():
            return base_dir / path
        if (base_dir.parent / path).exists():
            return base_dir.parent / path
    return path


def blockout(views: dict[str, Any], reference: dict, height_cm: float, dst: Path | str) -> Path:
    """Generate visual hull blockout mesh and save as binary PLY.

    views: {view: {"image": ..., "mask": ...}} or {view: mask_path}
    reference: reference dict from sheet_split (aspect, profile, etc.)
    height_cm: target height in centimeters
    dst: destination PLY path
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pdir = dst.parent.parent

    # 1. Resolve mask and image paths
    mask_paths = {}
    img_paths = {}
    for v in ("front", "side", "back"):
        entry = views.get(v, {})
        if isinstance(entry, dict):
            m_path = _resolve_path(pdir, entry.get("mask", f"views/{v}_mask.png"))
            i_path = _resolve_path(pdir, entry.get("image", f"views/{v}.png"))
        else:
            m_path = _resolve_path(pdir, entry)
            i_path = _resolve_path(pdir, str(entry).replace("_mask.png", ".png"))
        mask_paths[v] = m_path
        img_paths[v] = i_path

    # 2. Load masks and canonicalize
    raw_masks = {v: score.load_mask(mask_paths[v]) for v in ("front", "side", "back")}
    canon_masks = {v: score.canonical_mask(raw_masks[v]) for v in ("front", "side", "back")}

    # 3. Detect side facing direction
    # Side camera in Blender looks from +X to -X; character faces -Y (image-left).
    # If the reference side view faces right, we flip it horizontally.
    side_canon = canon_masks["side"]
    ys_s, xs_s = np.nonzero(side_canon)
    if ys_s.size == 0:
        raise CFError("BLOCKOUT_FAILED", "Side mask is empty.", "Verify sheet_split output.")
    
    side_faces = reference.get("side_faces")
    if side_faces == "right":
        flip_side = True
    elif side_faces == "left":
        flip_side = False
    else:
        y_min_s, y_max_s = ys_s.min(), ys_s.max()
        h_s = y_max_s - y_min_s + 1
        shin_xs = np.nonzero(side_canon[y_min_s + int(round(0.82 * h_s)):y_min_s + int(round(0.90 * h_s)), :])[1]
        foot_xs = np.nonzero(side_canon[y_min_s + int(round(0.92 * h_s)):y_max_s + 1, :])[1]
        if shin_xs.size > 0 and foot_xs.size > 0:
            shin_c = float(np.median(shin_xs))
            left_toe = shin_c - float(foot_xs.min())
            right_toe = float(foot_xs.max()) - shin_c
        else:
            left_toe = right_toe = 0.0

        if abs(right_toe - left_toe) > 4.0:
            flip_side = bool(right_toe > left_toe)
        else:
            # Fallback: check protrusion in the head band (top 8-22%)
            head_mask = side_canon[y_min_s:y_min_s + int(round(0.22 * h_s)), :]
            h_ys, h_xs = np.nonzero(head_mask)
            if h_xs.size > 0:
                left_ext = 256 - h_xs.min()
                right_ext = h_xs.max() - 256
                flip_side = bool((right_ext - left_ext) > 4)
            else:
                flip_side = False

    side_facing_left = np.fliplr(side_canon) if flip_side else side_canon
    back_mirrored = np.fliplr(canon_masks["back"])
    front_back = canon_masks["front"] & back_mirrored

    # 4. Vertical and horizontal bounds in canonical space
    ys_fb, xs_fb = np.nonzero(front_back)
    if ys_fb.size == 0:
        front_back = canon_masks["front"]
        ys_fb, xs_fb = np.nonzero(front_back)
        if ys_fb.size == 0:
            raise CFError("BLOCKOUT_FAILED", "Front mask is empty.", "Verify sheet_split output.")

    y_top = int(ys_fb.min())
    y_bottom = int(ys_fb.max())
    h_canon = max(1, y_bottom - y_top + 1)

    w_front = int(xs_fb.max() - xs_fb.min() + 1)
    w_side = int(xs_s.max() - xs_s.min() + 1)

    D_max = max(h_canon, w_front, w_side)
    s_v = max(D_max / 187.0, 1.0)

    Nx = min(224, max(48, int(round(w_front / s_v)) + 8))
    Ny = min(224, max(48, int(round(w_side / s_v)) + 8))
    Nz = min(224, max(48, int(round(h_canon / s_v)) + 8))
    if Nx % 2 != 0:
        Nx += 1
    if Ny % 2 != 0:
        Ny += 1
    if Nz % 2 != 0:
        Nz += 1

    k_min, k_max = 2, Nz - 3

    rows = np.clip(np.round(y_bottom - (np.arange(Nz) - k_min) * s_v).astype(int), 0, config.CANON_SIZE - 1)
    cols_x = np.clip(np.round(256 + (np.arange(Nx) - (Nx - 1) / 2.0) * s_v).astype(int), 0, config.CANON_SIZE - 1)
    cols_y = np.clip(np.round(256 + (np.arange(Ny) - (Ny - 1) / 2.0) * s_v).astype(int), 0, config.CANON_SIZE - 1)

    valid_z = (np.arange(Nz) >= k_min) & (np.arange(Nz) <= k_max)

    mask_xz = front_back[rows[:, None], cols_x[None, :]]  # (Nz, Nx)
    mask_xz[~valid_z, :] = False

    mask_yz = side_facing_left[rows[:, None], cols_y[None, :]]  # (Nz, Ny)
    mask_yz[~valid_z, :] = False

    # 3D visual hull voxels: shape (Nx, Ny, Nz) with elliptical cross-sections,
    # round lateral arm depth bounding, and sagittal centering for detached side features.
    def _runs_1d(arr: np.ndarray) -> list[tuple[int, int]]:
        padded = np.empty(len(arr) + 2, dtype=bool)
        padded[0] = False
        padded[1:-1] = arr
        padded[-1] = False
        diffs = np.flatnonzero(padded[1:] != padded[:-1])
        return list(zip(diffs[0::2].tolist(), (diffs[1::2] - 1).tolist()))

    voxels = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
    torso_x_bounds = np.zeros((Nz, 2), dtype=np.float32)
    torso_x_bounds[:, 1] = float(Nx - 1)
    mid_x = (Nx - 1) / 2.0
    mid_y = (Ny - 1) / 2.0
    xs_grid = np.arange(Nx, dtype=np.float32)[:, None]
    ys_grid = np.arange(Ny, dtype=np.float32)[None, :]

    # Reference sagittal torso center and upper-cranium back bound (to separate attached ponytails)
    torso_ys = np.nonzero(mask_yz[int(0.20 * Nz):int(0.72 * Nz), :])[1]
    body_y_center = float(np.median(torso_ys)) if torso_ys.size > 0 else mid_y

    cranium_ys = np.nonzero(mask_yz[int(0.91 * Nz):int(0.97 * Nz), :])[1]
    head_y_back = int(np.percentile(cranium_ys, 95)) if cranium_ys.size > 0 else Ny - 1
    has_detached_back = any(
        len(_runs_1d(mask_yz[kk])) > 1 for kk in range(int(0.74 * Nz), int(0.90 * Nz))
    )

    for k in range(k_min, k_max + 1):
        x_runs = _runs_1d(mask_xz[k])
        y_runs = _runs_1d(mask_yz[k])
        if not x_runs or not y_runs:
            continue

        # If a posterior hair/ponytail bridge merges with the cranium in upper-head slices, split at head_y_back
        if has_detached_back and k >= int(0.80 * Nz) and len(y_runs) == 1:
            y0_m, y1_m = y_runs[0]
            if y0_m < head_y_back < y1_m - 3:
                y_runs = [(y0_m, head_y_back), (head_y_back, y1_m)]

        main_y_run = min(y_runs, key=lambda r: abs(0.5 * (r[0] + r[1]) - body_y_center))
        main_x_run = min(x_runs, key=lambda r: abs(0.5 * (r[0] + r[1]) - mid_x))

        if len(x_runs) >= 3:
            core_x_runs = set(x_runs[1:-1])
            torso_x_bounds[k, 0] = float(x_runs[1][0])
            torso_x_bounds[k, 1] = float(x_runs[-2][1])
        elif len(x_runs) == 2 and k > int(0.48 * Nz) and main_x_run[0] <= mid_x <= main_x_run[1]:
            core_x_runs = {main_x_run}
            torso_x_bounds[k, 0] = float(main_x_run[0])
            torso_x_bounds[k, 1] = float(main_x_run[1])
        else:
            core_x_runs = set(x_runs)

        for x0, x1 in x_runs:
            is_lat_arm = (x0, x1) not in core_x_runs
            for y0, y1 in y_runs:
                is_sec_y = len(y_runs) > 1 and (y0, y1) != main_y_run
                if is_sec_y and is_lat_arm:
                    continue

                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
                rx = 0.5 * (x1 - x0)
                ry = 0.5 * (y1 - y0)

                if is_sec_y:
                    rx = min(rx, 1.25 * ry + 1.5)
                if is_lat_arm:
                    ry = min(ry, 1.15 * rx + 1.5)

                ix0 = max(x0, int(np.floor(cx - rx)))
                ix1 = min(x1, int(np.ceil(cx + rx)))
                iy0 = max(y0, int(np.floor(cy - ry)))
                iy1 = min(y1, int(np.ceil(cy + ry)))

                if rx < 1.5 or ry < 1.5:
                    voxels[ix0:ix1 + 1, iy0:iy1 + 1, k] = 1
                else:
                    dx = (xs_grid[ix0:ix1 + 1] - cx) / (rx + 0.65)
                    dy = (ys_grid[:, iy0:iy1 + 1] - cy) / (ry + 0.65)
                    voxels[ix0:ix1 + 1, iy0:iy1 + 1, k] |= (dx * dx + dy * dy <= 1.05).astype(np.uint8)

    if not np.any(voxels):
        voxels = (mask_xz.T[:, None, :] & mask_yz.T[None, :, :]).astype(np.uint8)
    if not np.any(voxels):
        raise CFError("BLOCKOUT_FAILED", "Visual hull intersection is empty.", "Check view alignments.")

    # 5. Marching Cubes surface extraction (orient faces outward so normals point out of the volume)
    verts, faces, normals, _ = marching_cubes(voxels, level=0.5)
    faces = faces[:, [0, 2, 1]]

    # 6. Taubin volume-preserving smoothing (10 alternating passes: lambda=0.50, mu=-0.52)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    deg = np.zeros(len(verts), dtype=int)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    deg[deg == 0] = 1

    for step in range(10):
        weight = 0.50 if (step % 2 == 0) else -0.52
        sum_neighbors = np.zeros_like(verts)
        np.add.at(sum_neighbors, edges[:, 0], verts[edges[:, 1]])
        np.add.at(sum_neighbors, edges[:, 1], verts[edges[:, 0]])
        verts += weight * (sum_neighbors / deg[:, None] - verts)

    # 7. Recompute outward vertex normals after smoothing
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    v_normals = np.zeros_like(verts)
    np.add.at(v_normals, faces[:, 0], face_normals)
    np.add.at(v_normals, faces[:, 1], face_normals)
    np.add.at(v_normals, faces[:, 2], face_normals)
    norm = np.linalg.norm(v_normals, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    v_normals /= norm

    # 8. Vertex color sampling
    # Prepare canonical RGB images
    raw_rgbs = {}
    for v in ("front", "side", "back"):
        img = cv2.imread(str(img_paths[v]), cv2.IMREAD_UNCHANGED)
        if img is None:
            raw_rgbs[v] = np.full((*raw_masks[v].shape, 3), 160, dtype=np.uint8)
        else:
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[..., :3]
            elif img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            raw_rgbs[v] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    side_rgb = np.fliplr(raw_rgbs["side"]) if flip_side else raw_rgbs["side"]
    side_m = np.fliplr(raw_masks["side"]) if flip_side else raw_masks["side"]

    front_canon_rgb = _canonical_rgb(raw_rgbs["front"], raw_masks["front"])
    side_canon_rgb = _canonical_rgb(side_rgb, side_m)
    back_canon_rgb = _canonical_rgb(raw_rgbs["back"], raw_masks["back"])

    # Sample colors for each vertex based on dominant normal
    # Coordinate system: character faces -Y in world space.
    # Therefore, -Y is FRONT, +Y is BACK, +-X is SIDE.
    vx = verts[:, 0]
    vy = verts[:, 1]
    vz = verts[:, 2]

    v_rows = np.clip(np.round(y_bottom - (vz - k_min) * s_v).astype(int), 0, config.CANON_SIZE - 1)
    v_cols_f = np.clip(np.round(256 + (vx - (Nx - 1) / 2.0) * s_v).astype(int), 0, config.CANON_SIZE - 1)
    v_cols_b = np.clip(np.round(256 - (vx - (Nx - 1) / 2.0) * s_v).astype(int), 0, config.CANON_SIZE - 1)
    v_cols_s = np.clip(np.round(256 + (vy - (Ny - 1) / 2.0) * s_v).astype(int), 0, config.CANON_SIZE - 1)

    colors = np.zeros((len(verts), 3), dtype=np.uint8)
    nx = v_normals[:, 0]
    ny = v_normals[:, 1]
    nz = v_normals[:, 2]

    abs_nx = np.abs(nx)
    abs_ny = np.abs(ny)
    abs_nz = np.abs(nz)

    # Dominant Y:
    is_front = (abs_ny >= abs_nx) & (abs_ny >= abs_nz) & (ny <= 0)
    is_back = (abs_ny >= abs_nx) & (abs_ny >= abs_nz) & (ny > 0)
    # Dominant X:
    is_side = (abs_nx > abs_ny) & (abs_nx >= abs_nz)

    # Lateral arm vertices (and thigh slices overlapped by side-view hands in A-pose) sample front/back colors
    vk = np.clip(np.round(vz).astype(int), 0, Nz - 1)
    on_lat_arm = (vx < torso_x_bounds[vk, 0] - 1.5) | (vx > torso_x_bounds[vk, 1] + 1.5)
    has_apose_arms = bool(np.any(torso_x_bounds[:, 0] > 0.5))
    hand_overlap_zone = has_apose_arms & (vk >= int(0.35 * Nz)) & (vk <= int(0.53 * Nz))
    no_side_proj = on_lat_arm | hand_overlap_zone
    is_side = is_side & ~no_side_proj
    is_front = is_front | (no_side_proj & (ny <= 0))
    is_back = is_back | (no_side_proj & (ny > 0))

    # Top/Bottom:
    is_top_bottom = ~is_front & ~is_back & ~is_side
    is_top_front = is_top_bottom & (ny <= 0)
    is_top_back = is_top_bottom & (ny > 0)

    colors[is_front | is_top_front] = front_canon_rgb[v_rows[is_front | is_top_front], v_cols_f[is_front | is_top_front]]
    colors[is_back | is_top_back] = back_canon_rgb[v_rows[is_back | is_top_back], v_cols_b[is_back | is_top_back]]
    colors[is_side] = side_canon_rgb[v_rows[is_side], v_cols_s[is_side]]

    # 9. Grounding feet at Z=0, scaling to height_cm, centering XY at (0, 0)
    z_min = verts[:, 2].min()
    verts[:, 2] -= z_min
    cur_h = verts[:, 2].max() - verts[:, 2].min()
    scale_m = (height_cm / 100.0) / (cur_h if cur_h > 0 else 1.0)
    verts *= scale_m

    x_mid = (verts[:, 0].min() + verts[:, 0].max()) / 2.0
    y_mid = (verts[:, 1].min() + verts[:, 1].max()) / 2.0
    verts[:, 0] -= x_mid
    verts[:, 1] -= y_mid

    # 10. Write binary little-endian PLY
    _write_binary_ply(dst, verts.astype(np.float32), colors.astype(np.uint8), faces.astype(np.int32))
    return dst


def _write_binary_ply(path: Path, verts: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> None:
    """Write binary little-endian PLY with vertex positions, uchar RGB colors, and triangle faces."""
    n_verts = len(verts)
    n_faces = len(faces)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {n_faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    # Vertex struct: x, y, z (float32, 12 bytes) + r, g, b (uint8, 3 bytes)
    v_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    v_data = np.empty(n_verts, dtype=v_dtype)
    v_data["x"] = verts[:, 0]
    v_data["y"] = verts[:, 1]
    v_data["z"] = verts[:, 2]
    v_data["r"] = colors[:, 0]
    v_data["g"] = colors[:, 1]
    v_data["b"] = colors[:, 2]

    # Face struct: count (uchar=3, 1 byte) + i0, i1, i2 (int32, 12 bytes)
    f_dtype = np.dtype([
        ("count", "u1"),
        ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4"),
    ])
    f_data = np.empty(n_faces, dtype=f_dtype)
    f_data["count"] = 3
    f_data["i0"] = faces[:, 0]
    f_data["i1"] = faces[:, 1]
    f_data["i2"] = faces[:, 2]

    with open(path, "wb") as f:
        f.write(header)
        v_data.tofile(f)
        f_data.tofile(f)
