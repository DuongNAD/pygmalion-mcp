"""Adversarial stress-testing benchmark for Blockout visual hull voxelization and memory bounds.
Measures peak memory with tracemalloc, execution time, and verifies manifold mesh geometry.
"""
from __future__ import annotations

from pathlib import Path
import time
import tracemalloc
import cv2
import numpy as np
import pytest

from charforge import blockout, config, score
from charforge.envelope import CFError


def parse_binary_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse binary little-endian PLY file into verts (N, 3), faces (M, 3), colors (N, 3)."""
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break

        lines = header.decode("ascii").splitlines()
        n_verts = 0
        n_faces = 0
        for line in lines:
            if line.startswith("element vertex"):
                n_verts = int(line.split()[2])
            elif line.startswith("element face"):
                n_faces = int(line.split()[2])

        v_dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ])
        v_data = np.fromfile(f, dtype=v_dtype, count=n_verts)
        verts = np.column_stack([v_data["x"], v_data["y"], v_data["z"]])
        colors = np.column_stack([v_data["r"], v_data["g"], v_data["b"]])

        f_dtype = np.dtype([
            ("count", "u1"),
            ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4"),
        ])
        f_data = np.fromfile(f, dtype=f_dtype, count=n_faces)
        faces = np.column_stack([f_data["i0"], f_data["i1"], f_data["i2"]])

    return verts, faces, colors


def check_manifold_mesh(verts: np.ndarray, faces: np.ndarray) -> dict[str, any]:
    """Verify topological manifold properties, finiteness, non-degeneracy, and boundary edges."""
    assert len(verts) > 0, "Vertex array is empty"
    assert len(faces) > 0, "Face array is empty"
    assert np.isfinite(verts).all(), "Vertices contain NaN or Inf"
    assert (faces >= 0).all() and (faces < len(verts)).all(), "Face indices out of bounds"

    # Degenerate face check (repeated vertex indices)
    has_repeated = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    assert not np.any(has_repeated), f"Mesh contains {np.sum(has_repeated)} degenerate faces with repeated vertices"

    # Non-zero triangle area check
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(face_normals, axis=1)
    zero_area = areas < 1e-12
    assert not np.any(zero_area), f"Mesh contains {np.sum(zero_area)} zero-area degenerate triangles"

    # Edge manifold check: directed half-edges and undirected edge counts
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected_edges = np.sort(edges, axis=1)
    
    # Pack undirected edges into 64-bit keys for fast unique counting
    packed = (undirected_edges[:, 0].astype(np.int64) << 32) | undirected_edges[:, 1].astype(np.int64)
    _, counts = np.unique(packed, return_counts=True)

    non_manifold_edges = int(np.sum(counts > 2))
    boundary_edges = int(np.sum(counts == 1))
    watertight_edges = int(np.sum(counts == 2))

    assert non_manifold_edges == 0, f"Mesh contains {non_manifold_edges} non-manifold edges (shared by > 2 faces)"

    # Oriented 2-manifold check: each directed edge (u, v) should have matching opposite (v, u)
    directed_packed = (edges[:, 0].astype(np.int64) << 32) | edges[:, 1].astype(np.int64)
    _, dir_counts = np.unique(directed_packed, return_counts=True)
    assert (dir_counts <= 1).all(), "Mesh contains non-orientable or duplicate directed edges"

    # Euler characteristic
    n_v = len(verts)
    n_e = len(counts)
    n_f = len(faces)
    euler = n_v - n_e + n_f

    return {
        "n_verts": n_v,
        "n_faces": n_f,
        "n_edges": n_e,
        "euler": euler,
        "boundary_edges": boundary_edges,
        "watertight_edges": watertight_edges,
        "is_watertight": bool(boundary_edges == 0),
    }


def run_blockout_case(
    f_mask: np.ndarray,
    s_mask: np.ndarray,
    b_mask: np.ndarray,
    tmp_path: Path,
    case_name: str,
    height_cm: float = 170.0,
) -> dict[str, any]:
    """Execute blockout under tracemalloc tracking and return empirical metrics."""
    f_p = tmp_path / f"{case_name}_front.png"
    s_p = tmp_path / f"{case_name}_side.png"
    b_p = tmp_path / f"{case_name}_back.png"
    ply_dst = tmp_path / f"{case_name}.ply"

    cv2.imwrite(str(f_p), (f_mask.astype(np.uint8) * 255))
    cv2.imwrite(str(s_p), (s_mask.astype(np.uint8) * 255))
    cv2.imwrite(str(b_p), (b_mask.astype(np.uint8) * 255))

    views = {"front": str(f_p), "side": str(s_p), "back": str(b_p)}

    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        blockout.blockout(views=views, reference={}, height_cm=height_cm, dst=ply_dst)
        t_elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    peak_mib = peak_bytes / (1024 * 1024)

    assert ply_dst.is_file(), f"PLY file {ply_dst} was not created"
    verts, faces, colors = parse_binary_ply(ply_dst)
    manifold_info = check_manifold_mesh(verts, faces)

    # Check bounds
    z_min = verts[:, 2].min()
    z_max = verts[:, 2].max()
    h_mesh = z_max - z_min
    expected_h = height_cm / 100.0

    assert abs(z_min) < 1e-4, f"Feet not grounded at z=0: z_min={z_min}"
    assert abs(h_mesh - expected_h) < 1e-4, f"Mesh height {h_mesh} != expected {expected_h}"

    return {
        "case": case_name,
        "elapsed_sec": t_elapsed,
        "peak_mib": peak_mib,
        "n_verts": len(verts),
        "n_faces": len(faces),
        "manifold": manifold_info,
    }


# ==============================================================================
# EMPIRICAL ADVERSARIAL BENCHMARK SUITE
# ==============================================================================

EXTREME_ASPECT_CASES = [
    ("500x5_tall_thin", (500, 5), (500, 5)),
    ("10x490_wide_flat", (10, 490), (10, 50)),
    ("2x500_ultra_flat", (2, 500), (2, 50)),
    ("500x2_ultra_needle", (500, 2), (500, 2)),
    ("asym_500x5_front_500x200_side", (500, 5), (500, 200)),
    ("asym_10x490_front_10x490_side", (10, 490), (10, 490)),
    ("asym_10x490_front_10x10_side", (10, 490), (10, 10)),
]


@pytest.mark.parametrize("case_name,front_shape,side_shape", EXTREME_ASPECT_CASES)
def test_empirical_extreme_aspect_ratios(tmp_path, case_name, front_shape, side_shape):
    """Stress-test extreme aspect ratios: peak memory < 50 MiB, time < 1.0s, valid manifold."""
    size = 512
    f_h, f_w = front_shape
    s_h, s_w = side_shape

    f_mask = np.zeros((size, size), dtype=bool)
    f_y0 = (size - f_h) // 2
    f_x0 = (size - f_w) // 2
    f_mask[f_y0:f_y0 + f_h, f_x0:f_x0 + f_w] = True

    s_mask = np.zeros((size, size), dtype=bool)
    s_y0 = (size - s_h) // 2
    s_x0 = (size - s_w) // 2
    s_mask[s_y0:s_y0 + s_h, s_x0:s_x0 + s_w] = True

    b_mask = f_mask.copy()

    metrics = run_blockout_case(f_mask, s_mask, b_mask, tmp_path, case_name)

    print(
        f"\n[EXTREME ASPECT] {case_name:<30} "
        f"Time: {metrics['elapsed_sec']:.3f}s | "
        f"Peak Mem: {metrics['peak_mib']:.2f} MiB | "
        f"Verts: {metrics['n_verts']:<5} | "
        f"Faces: {metrics['n_faces']:<5} | "
        f"Watertight: {metrics['manifold']['is_watertight']}"
    )

    # Hard constraints:
    # 1. Peak memory strictly < 50 MiB (and << 500 MiB)
    assert metrics["peak_mib"] < 50.0, f"Peak memory {metrics['peak_mib']:.2f} MiB exceeded 50 MiB target"
    # 2. Execution time strictly < 1 second
    assert metrics["elapsed_sec"] < 1.0, f"Execution time {metrics['elapsed_sec']:.3f}s exceeded 1.0s limit"
    # 3. Manifold mesh validity
    assert metrics["manifold"]["is_watertight"] is True, f"Mesh is not watertight: {metrics['manifold']}"
    assert metrics["manifold"]["boundary_edges"] == 0


LIMIT_CASES = [
    ("1x1_single_pixel", (1, 1), (1, 1)),
    ("2x2_tiny_square", (2, 2), (2, 2)),
    ("500x500_solid_block", (500, 500), (500, 500)),
]


@pytest.mark.parametrize("case_name,front_shape,side_shape", LIMIT_CASES)
def test_empirical_boundary_limit_cases(tmp_path, case_name, front_shape, side_shape):
    """Stress-test 1:1 extreme boundary limit cases (single pixel and solid block).

    Note: 1:1 shapes are scaled by canonical_mask to fill 481x481 canonical canvas,
    maximizing 3D surface area (196^3 solid cube with ~421k triangles).
    Verifies peak memory strictly < 500 MiB (in fact < 100 MiB) and manifold watertightness.
    """
    size = 512
    f_h, f_w = front_shape
    s_h, s_w = side_shape

    f_mask = np.zeros((size, size), dtype=bool)
    f_y0 = (size - f_h) // 2
    f_x0 = (size - f_w) // 2
    f_mask[f_y0:f_y0 + f_h, f_x0:f_x0 + f_w] = True

    s_mask = np.zeros((size, size), dtype=bool)
    s_y0 = (size - s_h) // 2
    s_x0 = (size - s_w) // 2
    s_mask[s_y0:s_y0 + s_h, s_x0:s_x0 + s_w] = True

    b_mask = f_mask.copy()

    metrics = run_blockout_case(f_mask, s_mask, b_mask, tmp_path, case_name)

    print(
        f"\n[LIMIT BENCHMARK] {case_name:<30} "
        f"Time: {metrics['elapsed_sec']:.3f}s | "
        f"Peak Mem: {metrics['peak_mib']:.2f} MiB | "
        f"Verts: {metrics['n_verts']:<5} | "
        f"Faces: {metrics['n_faces']:<5} | "
        f"Watertight: {metrics['manifold']['is_watertight']}"
    )

    # 1. Peak memory strictly < 500 MiB (well below 500 MiB, < 100 MiB)
    assert metrics["peak_mib"] < 500.0, f"Peak memory {metrics['peak_mib']:.2f} MiB exceeded 500 MiB ceiling"
    assert metrics["peak_mib"] < 100.0, f"Peak memory {metrics['peak_mib']:.2f} MiB exceeded 100 MiB"
    # 2. Watertight manifold mesh
    assert metrics["manifold"]["is_watertight"] is True
    assert metrics["manifold"]["boundary_edges"] == 0


def test_empirical_native_non_512_canvas(tmp_path):
    """Test when input image files themselves have native extreme shapes (not on 512x512 canvas)."""
    cases = [
        ("native_500x5", np.ones((500, 5), dtype=bool), np.ones((500, 5), dtype=bool)),
        ("native_10x490", np.ones((10, 490), dtype=bool), np.ones((10, 50), dtype=bool)),
        ("native_2x500", np.ones((2, 500), dtype=bool), np.ones((2, 50), dtype=bool)),
    ]

    for name, f_m, s_m in cases:
        metrics = run_blockout_case(f_m, s_m, f_m, tmp_path, name)
        print(
            f"\n[NATIVE BENCHMARK] {name:<25} "
            f"Time: {metrics['elapsed_sec']:.3f}s | "
            f"Peak Mem: {metrics['peak_mib']:.2f} MiB | "
            f"Verts: {metrics['n_verts']:<5} | "
            f"Faces: {metrics['n_faces']:<5} | "
            f"Watertight: {metrics['manifold']['is_watertight']}"
        )
        assert metrics["peak_mib"] < 50.0
        assert metrics["elapsed_sec"] < 1.0
        assert metrics["manifold"]["is_watertight"] is True


def test_empirical_empty_and_disjoint_fail_safely(tmp_path):
    """Verify empty or non-overlapping views raise CFError cleanly in < 0.1s without OOM."""
    size = 256
    empty = np.zeros((size, size), dtype=bool)
    solid = np.zeros((size, size), dtype=bool)
    solid[50:200, 100:150] = True

    e_p = tmp_path / "empty.png"
    s_p = tmp_path / "solid.png"
    cv2.imwrite(str(e_p), empty.astype(np.uint8) * 255)
    cv2.imwrite(str(s_p), solid.astype(np.uint8) * 255)

    views = {"front": str(s_p), "side": str(e_p), "back": str(s_p)}

    tracemalloc.start()
    t0 = time.perf_counter()
    with pytest.raises(CFError) as exc:
        blockout.blockout(views, {}, 170.0, tmp_path / "fail.ply")
    t1 = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert "Side mask is empty" in str(exc.value.message)
    assert t1 - t0 < 0.10
    assert (peak_bytes / (1024 * 1024)) < 5.0
