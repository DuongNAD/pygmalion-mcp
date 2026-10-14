from pathlib import Path

import numpy as np
import pytest

from charforge import config, server, state, tools_core

pytestmark = [pytest.mark.blender, pytest.mark.skipif(config.blender_exe() is None, reason="Blender not installed")]


def _generate_dense_mesh(path: Path, target_tris: int = 100800) -> Path:
    """Generate a dense ~100k triangle cylinder mesh."""
    # 225 height slices, 225 perimeter segments -> 224 * 225 * 2 = 100,800 tris
    n_h = 225
    n_theta = 225
    ys = np.linspace(0.0, 1.65, n_h)
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    verts = []
    for y in ys:
        for th in thetas:
            verts.append((0.25 * np.cos(th), y, 0.15 * np.sin(th)))

    faces = []
    for i in range(n_h - 1):
        for j in range(n_theta):
            j_next = (j + 1) % n_theta
            v0 = i * n_theta + j + 1
            v1 = i * n_theta + j_next + 1
            v2 = (i + 1) * n_theta + j_next + 1
            v3 = (i + 1) * n_theta + j + 1
            faces.append((v0, v1, v2))
            faces.append((v0, v2, v3))

    with open(path, "w") as f:
        for x, y, z in verts:
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for v0, v1, v2 in faces:
            f.write(f"f {v0} {v1} {v2}\n")
    return path


def test_mesh_optimize_and_export(sheet_path, tmp_path):
    """L2-T4: Test dense mesh (~100k tris) decimated to target +-5% with iou_drop <= 0.02,

    and verify exported GLB re-imports with triangle count within +-1%.
    """
    res_create = server.project_create(sheet_path=str(sheet_path), name="Dense Opt", height_cm=165.0)
    pid = res_create["project_id"]
    server.sheet_split(project_id=pid)

    # 1. Create dense mesh (~100k tris)
    dense_path = _generate_dense_mesh(tmp_path / "dense.obj")

    # 2. Score initial dense mesh
    res_score = server.score_mesh(project_id=pid, mesh_path=str(dense_path))
    assert res_score["ok"] is True
    scored_mesh = res_score["data"]["best"]["mesh"]

    # 3. Optimize mesh to target_tris = 30000
    target_tris = 30000
    res_opt = tools_core.mesh_optimize(project_id=pid, mesh_path=scored_mesh, target_tris=target_tris)
    assert res_opt["ok"] is True
    data_opt = res_opt["data"]
    print(f"DEBUG Optimize: before={data_opt['tris_before']}, after={data_opt['tris_after']}, iou_drop={data_opt['iou_drop']}")

    assert data_opt["tris_before"] >= 80000, f"Expected ~100k tris before, got {data_opt['tris_before']}"
    assert data_opt["tris_after"] <= target_tris * 1.05, f"Expected tris_after <= {target_tris * 1.05}, got {data_opt['tris_after']}"
    assert data_opt["iou_drop"] <= 0.02, f"Expected iou_drop <= 0.02, got {data_opt['iou_drop']}"

    p = state.load(pid)
    assert p["stage"] == "optimized"
    assert p["outputs"]["optimized"] == data_opt["mesh_path"]

    # 4. Export mesh to GLB
    res_exp = tools_core.mesh_export(project_id=pid, mesh_path=data_opt["mesh_path"], format="glb")
    assert res_exp["ok"] is True
    data_exp = res_exp["data"]
    assert "glb" in data_exp["files"]
    glb_rel = data_exp["files"]["glb"]
    pdir = state.project_dir(pid)
    assert (pdir / glb_rel).is_file()
    assert data_exp["size_mb"] > 0

    p = state.load(pid)
    assert p["stage"] == "exported"
    assert p["outputs"]["glb"] == glb_rel

    # 5. Verify exported GLB re-imports with triangle count within +-1%
    res_render = server.render_views(project_id=pid, mesh_path=glb_rel, mode="mask")
    assert res_render["ok"] is True
    reimported_tris = res_render["data"]["tris"]
    print(f"DEBUG Re-imported tris: {reimported_tris} vs expected: {data_opt['tris_after']}")
    assert abs(reimported_tris - data_opt["tris_after"]) <= data_opt["tris_after"] * 0.01
