from pathlib import Path

import pytest

from charforge import blender, config, server, state, tools_core

pytestmark = [pytest.mark.blender, pytest.mark.skipif(config.blender_exe() is None, reason="Blender not installed")]


def _create_bad_box_mesh(path: Path) -> Path:
    """Create an improperly proportioned box mesh with inverted body proportions."""
    n_slices = 16
    ys = [1.65 * i / n_slices for i in range(n_slices + 1)]
    verts = []
    for i, y in enumerate(ys):
        f = i / n_slices
        if f >= 0.85:  # head
            wx, wz = 0.22, 0.14
        elif f >= 0.55:  # torso (too wide)
            wx, wz = 0.38, 0.24
        elif f >= 0.35:  # skirt (too narrow)
            wx, wz = 0.22, 0.11
        else:  # legs (too wide)
            wx, wz = 0.24, 0.14
        for x in (-wx, wx):
            for z in (-wz, wz):
                verts.append((x, y, z))

    faces = []
    for s in range(n_slices):
        base = s * 4
        faces.append((base + 1, base + 2, base + 6, base + 5))
        faces.append((base + 2, base + 4, base + 8, base + 6))
        faces.append((base + 4, base + 3, base + 7, base + 8))
        faces.append((base + 3, base + 1, base + 5, base + 7))
    faces.append((1, 3, 4, 2))
    top = n_slices * 4
    faces.append((top + 1, top + 2, top + 4, top + 3))

    obj_lines = [f"v {x} {y} {z}\n" for x, y, z in verts]
    obj_lines += ["f " + " ".join(map(str, f)) + "\n" for f in faces]
    path.write_text("".join(obj_lines))
    return path


def test_mesh_fit_proportion_correction(sheet_path, tmp_path):
    """L2-T3: Test that mesh_fit scales and deforms a poorly proportioned box to increase score by >= 0.10,

    and scales height to height_cm within +-1%.
    """
    height_cm = 165.0
    res_create = server.project_create(sheet_path=str(sheet_path), name="Fit Test", height_cm=height_cm)
    pid = res_create["project_id"]
    server.sheet_split(project_id=pid)

    # 1. Create poorly proportioned box mesh
    box = _create_bad_box_mesh(tmp_path / "bad_box.obj")

    # 2. Initial score
    res_score1 = server.score_mesh(project_id=pid, mesh_path=str(box))
    assert res_score1["ok"] is True
    total_before = res_score1["data"]["total"]
    mesh_path1 = res_score1["data"]["best"]["mesh"]

    # 3. Apply profile-guided mesh_fit
    res_fit = tools_core.mesh_fit(project_id=pid, mesh_path=mesh_path1)
    assert res_fit["ok"] is True
    fit_mesh_path = res_fit["data"]["mesh_path"]
    assert "fit" in fit_mesh_path

    # 4. Re-score the fitted mesh
    res_score2 = server.score_mesh(project_id=pid, mesh_path=fit_mesh_path)
    assert res_score2["ok"] is True
    total_after = res_score2["data"]["total"]

    score_diff = total_after - total_before
    print(f"DEBUG Fit scores: before={total_before}, after={total_after}, diff={score_diff}")
    assert score_diff >= 0.10, f"Expected total score to increase by >= 0.10, got {score_diff}"

    # 5. Check height within +-1% of height_cm
    res_render = server.render_views(project_id=pid, mesh_path=fit_mesh_path, mode="mask")
    assert res_render["ok"] is True
    h_m = res_render["data"]["dims_m"][2]
    expected_h = height_cm / 100.0
    assert abs(h_m - expected_h) <= expected_h * 0.01, f"Height {h_m} not within +-1% of {expected_h}"


def test_fit_with_rotate_z_bakes_rotation(tmp_path):
    """Verify rotation=90 does not crash and bakes rotation cleanly."""
    obj = tmp_path / "test.obj"
    obj.write_text("v -0.2 0 0\nv 0.2 0 0\nv 0 1.65 0\nf 1 2 3\n")
    out = tmp_path / "out.glb"
    res = blender.run("fit.py", {
        "mesh": obj,
        "out": out,
        "height_m": 1.65,
        "rotate_z": 90,
        "scale_x": ",".join(["1.0"] * 16),
        "scale_y": ",".join(["1.0"] * 16),
    })
    assert res["ok"] is True
    assert out.is_file()
    assert out.stat().st_size > 0

    # Verify rotation was baked: original X span is now on Y axis
    res_render = blender.run("render_views.py", {
        "mesh": out,
        "out": tmp_path / "views",
        "mode": "mask",
    })
    assert res_render["ok"] is True
    dims = res_render["dims_m"]
    assert dims[2] == pytest.approx(1.65, rel=0.01)
    assert dims[1] == pytest.approx(0.40, rel=0.01)
    assert dims[0] < 0.05

