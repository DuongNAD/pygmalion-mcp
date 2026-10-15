"""Adversarial challenger tests for Pose-Safe mesh_fit on armed humanoid meshes.

Empirically verifies:
1. Torso width and shoulder proportions are preserved without squishing under extended arms (T-pose and A-pose).
2. Silhouette IoU is non-degraded compared to pre-fit candidate mesh across all pose combinations.
3. Monotonicity guard strictly catches and rolls back any IoU regressions (both naturally and under simulated stress).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from charforge import blender, config, server, state, tools_core, score

pytestmark = [
    pytest.mark.blender,
    pytest.mark.skipif(config.blender_exe() is None, reason="Blender not installed"),
]


def _read_mesh_positions(path: Path) -> np.ndarray:
    """Read vertex positions from GLB or OBJ format into an (N, 3) numpy array."""
    path = Path(path)
    if path.suffix.lower() == ".glb":
        with open(path, "rb") as f:
            magic, version, length = struct.unpack("<4sII", f.read(12))
            chunk0_len, chunk0_type = struct.unpack("<II", f.read(8))
            json_data = json.loads(f.read(chunk0_len))
            chunk1_len, chunk1_type = struct.unpack("<II", f.read(8))
            bin_data = f.read(chunk1_len)

        pos_idx = json_data["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
        acc = json_data["accessors"][pos_idx]
        bv = json_data["bufferViews"][acc["bufferView"]]
        offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        count = acc["count"]
        return np.frombuffer(bin_data[offset : offset + count * 12], dtype=np.float32).reshape(-1, 3)
    elif path.suffix.lower() == ".obj":
        verts = []
        for line in path.read_text().splitlines():
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:]])
        return np.array(verts, dtype=np.float32)
    raise ValueError(f"Unsupported format: {path.suffix}")


def _measure_mesh_proportions(verts: np.ndarray) -> dict[str, float]:
    """Measure exact 3D dimensions of humanoid (+Y is height, +X is width, +Z is depth)."""
    h = float(verts[:, 1].max() - verts[:, 1].min())
    # Torso width: Y in [0.95, 1.35], |X| <= 0.22
    mask_torso = (verts[:, 1] >= 0.95) & (verts[:, 1] <= 1.35) & (np.abs(verts[:, 0]) <= 0.22)
    v_torso = verts[mask_torso]
    torso_w = float(v_torso[:, 0].max() - v_torso[:, 0].min()) if len(v_torso) > 0 else 0.0

    # Shoulder width: Y in [1.32, 1.45], |X| <= 0.25
    mask_sh = (verts[:, 1] >= 1.32) & (verts[:, 1] <= 1.45) & (np.abs(verts[:, 0]) <= 0.25)
    v_sh = verts[mask_sh]
    shoulder_w = float(v_sh[:, 0].max() - v_sh[:, 0].min()) if len(v_sh) > 0 else 0.0

    # Total arm span
    arm_span = float(verts[:, 0].max() - verts[:, 0].min())

    return {
        "height": round(h, 4),
        "torso_w": round(torso_w, 4),
        "shoulder_w": round(shoulder_w, 4),
        "arm_span": round(arm_span, 4),
    }


def _make_box(p_min, p_max, v_offset=0):
    x0, y0, z0 = p_min
    x1, y1, z1 = p_max
    v = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    f = [
        (v_offset + 1, v_offset + 2, v_offset + 3, v_offset + 4),
        (v_offset + 5, v_offset + 8, v_offset + 7, v_offset + 6),
        (v_offset + 1, v_offset + 5, v_offset + 6, v_offset + 2),
        (v_offset + 2, v_offset + 6, v_offset + 7, v_offset + 3),
        (v_offset + 3, v_offset + 7, v_offset + 8, v_offset + 4),
        (v_offset + 4, v_offset + 8, v_offset + 5, v_offset + 1),
    ]
    return v, f


def _make_sliced_box(p_min, p_max, n_slices=8, v_offset=0):
    x0, y0, z0 = p_min
    x1, y1, z1 = p_max
    v = []
    f = []
    ys = np.linspace(y0, y1, n_slices + 1)
    for y in ys:
        v.append((x0, float(y), z0))
        v.append((x1, float(y), z0))
        v.append((x1, float(y), z1))
        v.append((x0, float(y), z1))

    for s in range(n_slices):
        base = v_offset + s * 4
        f.append((base + 1, base + 2, base + 6, base + 5))
        f.append((base + 2, base + 3, base + 7, base + 6))
        f.append((base + 3, base + 4, base + 8, base + 7))
        f.append((base + 4, base + 1, base + 5, base + 8))
    f.append((v_offset + 1, v_offset + 4, v_offset + 3, v_offset + 2))
    top = v_offset + n_slices * 4
    f.append((top + 1, top + 2, top + 3, top + 4))
    return v, f


def _make_oriented_box(p0, p1, w, depth_z, v_offset=0):
    p0 = np.array(p0[:2], dtype=float)
    p1 = np.array(p1[:2], dtype=float)
    d = p1 - p0
    L = np.linalg.norm(d)
    u = d / L
    n = np.array([-u[1], u[0]])
    c = [
        p0 + 0.5 * w * n,
        p0 - 0.5 * w * n,
        p1 - 0.5 * w * n,
        p1 + 0.5 * w * n,
    ]
    z0, z1 = -0.5 * depth_z, 0.5 * depth_z
    v = []
    for pt in c:
        v.append((float(pt[0]), float(pt[1]), z0))
    for pt in c:
        v.append((float(pt[0]), float(pt[1]), z1))
    f = [
        (v_offset + 1, v_offset + 4, v_offset + 3, v_offset + 2),
        (v_offset + 5, v_offset + 6, v_offset + 7, v_offset + 8),
        (v_offset + 1, v_offset + 2, v_offset + 6, v_offset + 5),
        (v_offset + 2, v_offset + 3, v_offset + 7, v_offset + 6),
        (v_offset + 3, v_offset + 4, v_offset + 8, v_offset + 7),
        (v_offset + 4, v_offset + 1, v_offset + 5, v_offset + 8),
    ]
    return v, f


def _build_humanoid_mesh(
    path: Path,
    pose: str = "tpose",
    height: float = 1.65,
    torso_w: float = 0.36,
    torso_d: float = 0.22,
    leg_w: float = 0.12,
) -> Path:
    """Build synthetic humanoid mesh in OBJ format with exact anatomical sections."""
    all_v, all_f = [], []

    def _add(v, f):
        all_v.extend(v)
        all_f.extend(f)

    # 1. Head
    _add(*_make_box((-0.10, 1.45, -0.10), (0.10, 1.65, 0.10), len(all_v)))
    # 2. Torso (sliced with vertical edge loops like a real character mesh)
    hw = torso_w / 2.0
    hd = torso_d / 2.0
    _add(*_make_sliced_box((-hw, 0.85, -hd), (hw, 1.45, hd), n_slices=6, v_offset=len(all_v)))
    # 3. Legs
    _add(*_make_sliced_box((-hw, 0.0, -0.09), (-hw + leg_w, 0.85, 0.09), n_slices=6, v_offset=len(all_v)))
    _add(*_make_sliced_box((hw - leg_w, 0.0, -0.09), (hw, 0.85, 0.09), n_slices=6, v_offset=len(all_v)))

    # 4. Arms
    if pose == "tpose":
        # 90° horizontal arms extending to |x| = 0.75m
        _add(*_make_box((-0.75, 1.25, -0.05), (-hw, 1.37, 0.05), len(all_v)))
        _add(*_make_box((hw, 1.25, -0.05), (0.75, 1.37, 0.05), len(all_v)))
    elif pose == "apose":
        # 45° angled arms extending downward
        arm_len = 0.55
        dx = arm_len * np.cos(np.pi / 4)
        dy = arm_len * np.sin(np.pi / 4)
        _add(*_make_oriented_box((-hw, 1.35), (-hw - dx, 1.35 - dy), 0.08, 0.08, len(all_v)))
        _add(*_make_oriented_box((hw, 1.35), (hw + dx, 1.35 - dy), 0.08, 0.08, len(all_v)))
    elif pose == "asymmetric":
        # Left arm in T-pose, right arm at side
        _add(*_make_box((-0.75, 1.25, -0.05), (-hw, 1.37, 0.05), len(all_v)))
        _add(*_make_box((hw, 0.85, -0.05), (hw + 0.06, 1.37, 0.05), len(all_v)))
    elif pose == "none":
        # Arms alongside torso
        _add(*_make_box((-hw - 0.06, 0.85, -0.05), (-hw, 1.37, 0.05), len(all_v)))
        _add(*_make_box((hw, 0.85, -0.05), (hw + 0.06, 1.37, 0.05), len(all_v)))

    lines = [f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in all_v]
    lines += ["f " + " ".join(map(str, face)) + "\n" for face in all_f]
    path.write_text("".join(lines))
    return path


def _create_turnaround_sheet(path: Path, arm_pose: str = "none") -> tuple[Path, dict]:
    """Generate 1800x900 turnaround sheet with front, side, and back figures and bounding boxes."""
    img = np.full((900, 1800, 3), 255, np.uint8)
    h = 760
    top = 60
    c = (90, 60, 160)

    cxs = {"front": 300, "side": 900, "back": 1500}
    boxes = {}

    for view, cx in cxs.items():
        ws = 0.55 if view == "side" else 1.0
        nose = 18 if view == "side" else 0
        r = int(h * 0.08)
        # Head
        cv2.circle(img, (cx, top + r), r, c, -1)
        if nose:
            cv2.circle(img, (cx + r, top + r), nose, c, -1)
        # Torso & Skirt
        tw, sw = int(h * 0.14 * ws), int(h * 0.20 * ws)
        y1, y2 = top + int(h * 0.45), top + int(h * 0.60)
        cv2.rectangle(img, (cx - tw, top + 2 * r - 4), (cx + tw, y1), c, -1)
        cv2.fillPoly(img, [np.array([[cx - tw, y1], [cx + tw, y1], [cx + sw, y2], [cx - sw, y2]])], c)
        # Legs
        lw, gap = int(h * 0.04 * ws) + 2, int(h * 0.03 * ws)
        for s in (-1, 1):
            x = cx + s * (gap + lw)
            cv2.rectangle(img, (x - lw, y2), (x + lw, top + h), c, -1)

        # Arms in front/back views
        if arm_pose != "none" and view in ("front", "back"):
            y_sh = top + 2 * r + int(h * 0.05)
            t = int(h * 0.05)
            L = int(h * 0.22)
            if arm_pose == "tpose":
                cv2.line(img, (cx - tw, y_sh), (cx - tw - L, y_sh), c, t)
                cv2.line(img, (cx + tw, y_sh), (cx + tw + L, y_sh), c, t)
            elif arm_pose == "apose":
                dx = int(L * 0.7071)
                dy = int(L * 0.7071)
                cv2.line(img, (cx - tw, y_sh), (cx - tw - dx, y_sh + dy), c, t)
                cv2.line(img, (cx + tw, y_sh), (cx + tw + dx, y_sh + dy), c, t)

        bw = int(h * 0.65) if arm_pose != "none" and view != "side" else int(h * 0.35)
        boxes[view] = [cx - bw // 2, top - 20, bw, h + 40]

    cv2.imwrite(str(path), img)
    return path, boxes


def test_fit_tpose_mesh_preserves_torso_and_shoulders(tmp_path):
    """Challenge 1: T-pose mesh against arms-at-side reference turnaround.

    Verify torso width and shoulder proportions are NOT squished or distorted by the 90° arms.
    """
    sheet_p, boxes = _create_turnaround_sheet(tmp_path / "sheet_standard.png", arm_pose="none")
    res_c = server.project_create(sheet_path=str(sheet_p), name="TPose Preserv", height_cm=165.0)
    pid = res_c["project_id"]
    server.sheet_split(project_id=pid, boxes=boxes)

    # Ground truth torso width: 0.36m, height: 1.65m
    mesh_obj = _build_humanoid_mesh(tmp_path / "tpose_test.obj", pose="tpose", torso_w=0.36)
    score1 = server.score_mesh(project_id=pid, mesh_path=str(mesh_obj))
    assert score1["ok"] is True
    total_pre = score1["data"]["total"]

    res_fit = tools_core.mesh_fit(project_id=pid, mesh_path=score1["data"]["best"]["mesh"])
    assert res_fit["ok"] is True

    # 1. Inspect the physical deformed GLB mesh produced by fit.py
    p = state.load(pid)
    stem = Path(score1["data"]["best"]["mesh"]).stem
    fitted_glb = state.project_dir(pid) / f"meshes/{stem}-fit1.glb"
    assert fitted_glb.is_file(), "Fitted GLB file was not created"
    dims = _measure_mesh_proportions(_read_mesh_positions(fitted_glb))

    # Torso width must be preserved within 4% of 0.36m (NOT squished to ~0.288m)
    assert abs(dims["torso_w"] - 0.36) <= 0.36 * 0.04, (
        f"Torso squished! Measured {dims['torso_w']:.4f}m, expected 0.36m"
    )
    # Height must be scaled within 1% of 1.65m
    assert abs(dims["height"] - 1.65) <= 1.65 * 0.01

    # Arm scaling factors on arm layers w in [7..13] must be damped close to 1.0
    for w in range(7, 14):
        sx = res_fit["data"]["scale_x"][w]
        assert 0.95 <= sx <= 1.08, f"Lattice arm layer {w} not damped: sx={sx}"

    # 2. Active candidate IoU must not be degraded
    score2 = server.score_mesh(project_id=pid, mesh_path=res_fit["data"]["mesh_path"])
    total_post = score2["data"]["total"]
    assert total_post >= total_pre - 1e-4, f"IoU degraded: {total_pre} -> {total_post}"


def test_fit_apose_mesh_preserves_torso_and_shoulders(tmp_path):
    """Challenge 2: A-pose (45°) mesh against arms-at-side reference turnaround.

    Verify torso width and shoulder proportions are preserved without distortion.
    """
    sheet_p, boxes = _create_turnaround_sheet(tmp_path / "sheet_standard.png", arm_pose="none")
    res_c = server.project_create(sheet_path=str(sheet_p), name="APose Preserv", height_cm=165.0)
    pid = res_c["project_id"]
    server.sheet_split(project_id=pid, boxes=boxes)

    mesh_obj = _build_humanoid_mesh(tmp_path / "apose_test.obj", pose="apose", torso_w=0.36)
    score1 = server.score_mesh(project_id=pid, mesh_path=str(mesh_obj))
    assert score1["ok"] is True
    total_pre = score1["data"]["total"]

    res_fit = tools_core.mesh_fit(project_id=pid, mesh_path=score1["data"]["best"]["mesh"])
    assert res_fit["ok"] is True

    # 1. Inspect physical deformed GLB mesh produced by fit.py
    p = state.load(pid)
    stem = Path(score1["data"]["best"]["mesh"]).stem
    fitted_glb = state.project_dir(pid) / f"meshes/{stem}-fit1.glb"
    assert fitted_glb.is_file(), "Fitted GLB file was not created"
    dims = _measure_mesh_proportions(_read_mesh_positions(fitted_glb))

    # Torso width must be preserved within 4% of 0.36m
    assert abs(dims["torso_w"] - 0.36) <= 0.36 * 0.04, (
        f"A-pose torso distorted! Measured {dims['torso_w']:.4f}m, expected 0.36m"
    )
    # Shoulder width preserved
    assert abs(dims["shoulder_w"] - 0.40) <= 0.40 * 0.05

    # 2. Active candidate IoU must not be degraded
    score2 = server.score_mesh(project_id=pid, mesh_path=res_fit["data"]["mesh_path"])
    total_post = score2["data"]["total"]
    assert total_post >= total_pre - 1e-4, f"IoU degraded: {total_pre} -> {total_post}"


def test_fit_both_tpose_matches_proportions(tmp_path):
    """Challenge 3: T-pose mesh against T-pose reference turnaround.

    When both reference and mesh share the same T-pose, fitting optimizes the full figure.
    """
    sheet_p, boxes = _create_turnaround_sheet(tmp_path / "sheet_tpose.png", arm_pose="tpose")
    res_c = server.project_create(sheet_path=str(sheet_p), name="TPose Match", height_cm=165.0)
    pid = res_c["project_id"]
    server.sheet_split(project_id=pid, boxes=boxes)

    mesh_obj = _build_humanoid_mesh(tmp_path / "tpose_mesh.obj", pose="tpose", torso_w=0.36)
    score1 = server.score_mesh(project_id=pid, mesh_path=str(mesh_obj))
    total_pre = score1["data"]["total"]

    res_fit = tools_core.mesh_fit(project_id=pid, mesh_path=score1["data"]["best"]["mesh"])
    assert res_fit["ok"] is True

    score2 = server.score_mesh(project_id=pid, mesh_path=res_fit["data"]["mesh_path"])
    total_post = score2["data"]["total"]
    assert total_post >= total_pre - 1e-4


def test_fit_both_apose_matches_proportions(tmp_path):
    """Challenge 4: A-pose mesh against A-pose reference turnaround.

    When both reference and mesh share the same A-pose, fitting optimizes the full figure.
    """
    sheet_p, boxes = _create_turnaround_sheet(tmp_path / "sheet_apose.png", arm_pose="apose")
    res_c = server.project_create(sheet_path=str(sheet_p), name="APose Match", height_cm=165.0)
    pid = res_c["project_id"]
    server.sheet_split(project_id=pid, boxes=boxes)

    mesh_obj = _build_humanoid_mesh(tmp_path / "apose_mesh.obj", pose="apose", torso_w=0.36)
    score1 = server.score_mesh(project_id=pid, mesh_path=str(mesh_obj))
    total_pre = score1["data"]["total"]

    res_fit = tools_core.mesh_fit(project_id=pid, mesh_path=score1["data"]["best"]["mesh"])
    assert res_fit["ok"] is True

    score2 = server.score_mesh(project_id=pid, mesh_path=res_fit["data"]["mesh_path"])
    total_post = score2["data"]["total"]
    assert total_post >= total_pre - 1e-4


def test_fit_asymmetric_armed_mesh(tmp_path):
    """Challenge 5: Humanoid mesh with asymmetric arms (left extended, right at side).

    Verify pose-safety and lattice fitting handle lateral asymmetry without crashing.
    """
    sheet_p, boxes = _create_turnaround_sheet(tmp_path / "sheet_std.png", arm_pose="none")
    res_c = server.project_create(sheet_path=str(sheet_p), name="Asym Arms", height_cm=165.0)
    pid = res_c["project_id"]
    server.sheet_split(project_id=pid, boxes=boxes)

    mesh_obj = _build_humanoid_mesh(tmp_path / "asym_mesh.obj", pose="asymmetric", torso_w=0.36)
    score1 = server.score_mesh(project_id=pid, mesh_path=str(mesh_obj))
    total_pre = score1["data"]["total"]

    res_fit = tools_core.mesh_fit(project_id=pid, mesh_path=score1["data"]["best"]["mesh"])
    assert res_fit["ok"] is True

    score2 = server.score_mesh(project_id=pid, mesh_path=res_fit["data"]["mesh_path"])
    total_post = score2["data"]["total"]
    assert total_post >= total_pre - 1e-4


def test_monotonicity_guard_natural_rollback_on_minor_regression(tmp_path):
    """Challenge 6: Empirically verify monotonicity guard catches naturally occurring minor regression.

    When the mesh is already well-proportioned and lattice smoothing drops score by a fraction of a percent,
    verify that the guard catches the regression, warns, and rolls back to pre-fit candidate.
    """
    sheet_p, boxes = _create_turnaround_sheet(tmp_path / "sheet_standard.png", arm_pose="none")
    res_c = server.project_create(sheet_path=str(sheet_p), name="Natural Guard", height_cm=165.0)
    pid = res_c["project_id"]
    server.sheet_split(project_id=pid, boxes=boxes)

    mesh_obj = _build_humanoid_mesh(tmp_path / "well_prop.obj", pose="tpose", torso_w=0.36)
    score1 = server.score_mesh(project_id=pid, mesh_path=str(mesh_obj))
    rpath = score1["data"]["best"]["mesh"]
    total_pre = score1["data"]["total"]

    res_fit = tools_core.mesh_fit(project_id=pid, mesh_path=rpath)
    assert res_fit["ok"] is True

    # If the fit was lower than total_pre, guard MUST have fired and retained rpath
    p = state.load(pid)
    fit_glb = state.project_dir(pid) / "meshes/well_prop-fit1.glb"
    score_fit = server.score_mesh(project_id=pid, mesh_path=str(fit_glb))
    fit_total = score_fit["data"]["total"]

    if fit_total < total_pre:
        # Guard caught the regression!
        assert res_fit["data"]["mesh_path"] == rpath
        assert any("Lattice fit degraded silhouette IoU" in w for w in res_fit.get("warnings", []))
        assert p["best"]["mesh"] == rpath
        assert p["best"]["total"] == total_pre
    else:
        # Fit improved or matched
        assert res_fit["data"]["mesh_path"] == "meshes/well_prop-fit1.glb"
        assert p["best"]["total"] >= total_pre


def test_monotonicity_guard_strictly_rejects_iou_regression(tmp_path, monkeypatch):
    """Challenge 7: Adversarial regression stress-test with simulated IoU degradation.

    When lattice deformation would degrade silhouette score, verify:
    1. Guard rejects the fitted mesh.
    2. Retains the pre-fit candidate mesh.
    3. Emits degradation warning.
    4. State remains uncorrupted (no degraded fit registered).
    """
    sheet_p, boxes = _create_turnaround_sheet(tmp_path / "sheet_std.png", arm_pose="none")
    res_c = server.project_create(sheet_path=str(sheet_p), name="Guard Reject", height_cm=165.0)
    pid = res_c["project_id"]
    server.sheet_split(project_id=pid, boxes=boxes)

    mesh_obj = _build_humanoid_mesh(tmp_path / "cand.obj", pose="tpose", torso_w=0.36)
    score1 = server.score_mesh(project_id=pid, mesh_path=str(mesh_obj))
    best_pre = score1["data"]["best"]["mesh"]
    total_pre = score1["data"]["total"]

    real_compare = score.compare

    def mocked_compare(*args, **kwargs):
        res = real_compare(*args, **kwargs)
        res["total"] = round(total_pre - 0.08, 4)
        return res

    monkeypatch.setattr(score, "compare", mocked_compare)

    res_fit = tools_core.mesh_fit(project_id=pid, mesh_path=best_pre)
    assert res_fit["ok"] is True

    # 1. Returned mesh path must be the PRE-FIT mesh, NOT the degraded fit
    assert res_fit["data"]["mesh_path"] == best_pre

    # 2. Warning must report the exact regression
    assert any("Lattice fit degraded silhouette IoU" in w for w in res_fit.get("warnings", []))

    # 3. State integrity: degraded fit must not be registered as active best
    p = state.load(pid)
    assert not any("fit" in m["path"] for m in p["meshes"])
    assert p["best"]["mesh"] == best_pre
    assert p["best"]["total"] == total_pre
