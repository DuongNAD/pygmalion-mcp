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
        # Check protrusion in the head band (top 8-22%)
        y_min_s, y_max_s = ys_s.min(), ys_s.max()
        h_s = y_max_s - y_min_s + 1
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

    Nz = 192
    k_min, k_max = 2, Nz - 3
    s_v = h_canon / (k_max - k_min)

    w_front = int(xs_fb.max() - xs_fb.min() + 1)
    w_side = int(xs_s.max() - xs_s.min() + 1)

    Nx = max(96, int(round(w_front / s_v)) + 8)
    Ny = max(64, int(round(w_side / s_v)) + 8)
    if Nx % 2 != 0:
        Nx += 1
    if Ny % 2 != 0:
        Ny += 1

    rows = np.clip(np.round(y_bottom - (np.arange(Nz) - k_min) * s_v).astype(int), 0, config.CANON_SIZE - 1)
    cols_x = np.clip(np.round(256 + (np.arange(Nx) - (Nx - 1) / 2.0) * s_v).astype(int), 0, config.CANON_SIZE - 1)
    cols_y = np.clip(np.round(256 + (np.arange(Ny) - (Ny - 1) / 2.0) * s_v).astype(int), 0, config.CANON_SIZE - 1)

    valid_z = (np.arange(Nz) >= k_min) & (np.arange(Nz) <= k_max)

    mask_xz = front_back[rows[:, None], cols_x[None, :]]  # (Nz, Nx)
    mask_xz[~valid_z, :] = False

    mask_yz = side_facing_left[rows[:, None], cols_y[None, :]]  # (Nz, Ny)
    mask_yz[~valid_z, :] = False

    # 3D visual hull voxels: shape (Nx, Ny, Nz)
    voxels = mask_xz.T[:, None, :] & mask_yz.T[None, :, :]
    if not np.any(voxels):
        raise CFError("BLOCKOUT_FAILED", "Visual hull intersection is empty.", "Check view alignments.")

    # 5. Marching Cubes surface extraction
    verts, faces, normals, _ = marching_cubes(voxels.astype(float), level=0.5)

    # 6. Laplacian smoothing (10 passes, lambda=0.5)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    deg = np.zeros(len(verts), dtype=int)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    deg[deg == 0] = 1

    for _ in range(10):
        sum_neighbors = np.zeros_like(verts)
        np.add.at(sum_neighbors, edges[:, 0], verts[edges[:, 1]])
        np.add.at(sum_neighbors, edges[:, 1], verts[edges[:, 0]])
        verts += 0.5 * (sum_neighbors / deg[:, None] - verts)

    # 7. Recompute vertex normals after smoothing
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
