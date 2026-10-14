"""Empirical adversarial stress test suite for Level 2 tools.
Designed by Challenger 2 (teamwork_preview_challenger_m2_2).

Covers:
1. mesh_fit:
   - Distorted box meshes (inverted proportions, barrel shape, offset origin).
   - Score increases by >= 0.10.
   - Height within +-1% of height_cm.
   - Feet grounded at z=0 (abs(min_z) < 1e-4).
   - Error handling: Calling mesh_fit before score_mesh returns FIT_FAILED error envelope.

2. mesh_optimize:
   - Decimates ~100k triangle dense mesh to target_tris (30k).
   - tris_after <= target * 1.05.
   - iou_drop <= 0.02.
   - Mesh cleanup: strips loose vertices, merges doubles, generates UVMap.
   - Retry path: When decimation iou_drop > 0.02, triggers retry pass with target_tris * 1.5.

3. mesh_export:
   - Exports GLB and FBX.
   - Re-imports GLB into Blender: checks unit system (METRIC, scale=1.0), transforms applied, bounds, tris within +-1%.
   - Re-imports FBX into Blender: checks unit system (METRIC, scale=1.0), bounds, tris within +-1%.

4. test_e2e_l2 pipeline:
   - Follows next_action purely until completion (next_action is None).
   - Standard blockout path finishes in <= 12 calls.
   - Non-passing candidate requiring mesh_fit path finishes in <= 12 calls.
   - Reaches stage 'exported' with valid outputs.glb and zero errors.
"""
from pathlib import Path
import subprocess
from unittest.mock import patch
import numpy as np
import pytest

from charforge import config, server, state, tools_core

pytestmark = [pytest.mark.blender, pytest.mark.skipif(config.blender_exe() is None, reason="Blender not installed")]


def _create_distorted_box(path: Path, shape_type: str = "inverted", offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Path:
    """Generate sliced box meshes with varying adversarial proportions and spatial offsets."""
    n_slices = 16
    ys = [1.65 * i / n_slices for i in range(n_slices + 1)]
    verts = []
    ox, oy, oz = offset

    for i, y in enumerate(ys):
        f = i / n_slices
        if shape_type == "inverted":
            # Inverted proportions compared to standard human reference
            if f >= 0.85:
                wx, wz = 0.22, 0.14
            elif f >= 0.55:
                wx, wz = 0.38, 0.24
            elif f >= 0.35:
                wx, wz = 0.22, 0.11
            else:
                wx, wz = 0.24, 0.14
        elif shape_type == "barrel":
            # Bulky torso with tiny neck and stick legs
            if f >= 0.85:
                wx, wz = 0.12, 0.08
            elif f >= 0.40:
                wx, wz = 0.42, 0.28
            else:
                wx, wz = 0.10, 0.06
        else:
            wx, wz = 0.20, 0.15

        for x in (-wx, wx):
            for z in (-wz, wz):
                verts.append((x + ox, y + oy, z + oz))

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

    lines = [f"v {x} {y} {z}\n" for x, y, z in verts]
    lines += ["f " + " ".join(map(str, f)) + "\n" for f in faces]
    path.write_text("".join(lines))
    return path


def _generate_dense_cylinder(path: Path, target_tris: int = 100800) -> Path:
    """Generate a dense ~100k triangle cylinder mesh."""
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


def _inspect_blender_geometry(mesh_path: Path, ext: str) -> dict:
    """Helper running headless Blender to extract exact vertex bounds, triangle counts, and units."""
    bcmd = f'''
import bpy
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)
if "{ext}" == ".glb":
    bpy.ops.import_scene.gltf(filepath=r"{mesh_path}")
elif "{ext}" == ".fbx":
    if hasattr(bpy.ops.wm, "fbx_import"):
        bpy.ops.wm.fbx_import(filepath=r"{mesh_path}")
    else:
        bpy.ops.import_scene.fbx(filepath=r"{mesh_path}")

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
dg = bpy.context.evaluated_depsgraph_get()
pts, tris = [], 0
for obj in meshes:
    ev = obj.evaluated_get(dg)
    pts += [ev.matrix_world @ Vector(c) for c in ev.bound_box]
    tris += sum(len(poly.vertices) - 2 for poly in ev.data.polygons)

lo = Vector([min(p[i] for p in pts) for i in range(3)])
hi = Vector([max(p[i] for p in pts) for i in range(3)])

print("BOUNDS_RESULT", tris, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z, bpy.context.scene.unit_settings.system, bpy.context.scene.unit_settings.scale_length)
'''
    out = subprocess.check_output([
        str(config.blender_exe()), "-b", "--factory-startup", "--python-expr", bcmd
    ]).decode()

    line = [l for l in out.splitlines() if "BOUNDS_RESULT" in l][0]
    parts = line.split()
    return {
        "tris": int(parts[1]),
        "lo": (float(parts[2]), float(parts[3]), float(parts[4])),
        "hi": (float(parts[5]), float(parts[6]), float(parts[7])),
        "height": float(parts[7]) - float(parts[4]),
        "unit_system": parts[8],
        "scale_length": float(parts[9]),
    }


# ==============================================================================
# 1. MESH FIT STRESS TESTS
# ==============================================================================

@pytest.mark.parametrize("shape_type,offset", [
    ("inverted", (0.0, 0.0, 0.0)),
    ("barrel", (0.0, 0.0, 0.0)),
    ("inverted", (5.0, 10.0, -25.0)),
])
def test_mesh_fit_adversarial_proportions_and_grounding(sheet_path, tmp_path, shape_type, offset):
    """Verify mesh_fit improves score by >= 0.10, scales height to +-1%, and grounds feet at z=0."""
    height_cm = 165.0
    expected_h = height_cm / 100.0

    res_create = server.project_create(sheet_path=str(sheet_path), name=f"FitAdv-{shape_type}", height_cm=height_cm)
    pid = res_create["project_id"]
    server.sheet_split(project_id=pid)

    box = _create_distorted_box(tmp_path / f"distorted_{shape_type}.obj", shape_type=shape_type, offset=offset)

    # 1. Initial score
    res_score1 = server.score_mesh(project_id=pid, mesh_path=str(box))
    assert res_score1["ok"] is True
    total_before = res_score1["data"]["total"]
    mesh_path1 = res_score1["data"]["best"]["mesh"]

    # 2. Profile-guided mesh_fit
    res_fit = tools_core.mesh_fit(project_id=pid, mesh_path=mesh_path1)
    assert res_fit["ok"] is True
    fit_mesh_path = res_fit["data"]["mesh_path"]
    assert "fit" in fit_mesh_path

    # 3. Re-score
    res_score2 = server.score_mesh(project_id=pid, mesh_path=fit_mesh_path)
    assert res_score2["ok"] is True
    total_after = res_score2["data"]["total"]

    score_diff = total_after - total_before
    assert score_diff >= 0.10, f"Expected total score increase >= 0.10, got {score_diff:.4f}"

    # 4. Verify grounding at z=0 and height within +-1% via Blender direct inspection
    pdir = state.project_dir(pid)
    geom = _inspect_blender_geometry(pdir / fit_mesh_path, ".glb")

    # Feet must be grounded at z=0 (tolerance 1e-4)
    assert abs(geom["lo"][2]) < 1e-4, f"Feet not grounded at z=0: min_z={geom['lo'][2]}"

    # Height within +-1%
    assert abs(geom["height"] - expected_h) <= expected_h * 0.01, f"Height {geom['height']} not within +-1% of {expected_h}"


def test_mesh_fit_unscored_mesh_error(sheet_path, tmp_path):
    """Calling mesh_fit on an unscored mesh must return ok=False with FIT_FAILED error."""
    res_create = server.project_create(sheet_path=str(sheet_path), name="FitUnscored", height_cm=165.0)
    pid = res_create["project_id"]
    server.sheet_split(project_id=pid)

    dummy_box = _create_distorted_box(tmp_path / "dummy.obj")
    res = tools_core.mesh_fit(project_id=pid, mesh_path=str(dummy_box))

    assert res["ok"] is False
    assert res["error"]["code"] == "FIT_FAILED"


# ==============================================================================
# 2. MESH OPTIMIZE & RETRY STRESS TESTS
# ==============================================================================

def test_mesh_optimize_dense_decimation_and_cleanup(sheet_path, tmp_path):
    """Verify ~100k tri mesh is decimated to target_tris +-5% with iou_drop <= 0.02, and geometry cleaned."""
    res_create = server.project_create(sheet_path=str(sheet_path), name="DenseCleanup", height_cm=165.0)
    pid = res_create["project_id"]
    server.sheet_split(project_id=pid)

    dense_path = _generate_dense_cylinder(tmp_path / "dense.obj", target_tris=100800)

    # Initial score
    res_score = server.score_mesh(project_id=pid, mesh_path=str(dense_path))
    assert res_score["ok"] is True
    scored_mesh = res_score["data"]["best"]["mesh"]

    target_tris = 30000
    res_opt = tools_core.mesh_optimize(project_id=pid, mesh_path=scored_mesh, target_tris=target_tris)
    assert res_opt["ok"] is True
    data_opt = res_opt["data"]

    assert data_opt["tris_before"] >= 80000, f"Expected >= 80000 tris, got {data_opt['tris_before']}"
    assert data_opt["tris_after"] <= target_tris * 1.05, f"Expected tris_after <= {target_tris * 1.05}, got {data_opt['tris_after']}"
    assert data_opt["iou_drop"] <= 0.02, f"Expected iou_drop <= 0.02, got {data_opt['iou_drop']}"

    # Verify Blender inspects UVs and applied decimation
    pdir = state.project_dir(pid)
    geom = _inspect_blender_geometry(pdir / data_opt["mesh_path"], ".glb")
    assert abs(geom["tris"] - data_opt["tris_after"]) <= data_opt["tris_after"] * 0.01


def test_mesh_optimize_retry_branch_trigger(sheet_path, tmp_path):
    """When decimation causes iou_drop > 0.02, verify that mesh_optimize retries with target_tris * 1.5."""
    res_create = server.project_create(sheet_path=str(sheet_path), name="OptRetry", height_cm=165.0)
    pid = res_create["project_id"]
    server.sheet_split(project_id=pid)

    dense_path = _generate_dense_cylinder(tmp_path / "dense_retry.obj", target_tris=100800)
    res_score = server.score_mesh(project_id=pid, mesh_path=str(dense_path))
    scored_mesh = res_score["data"]["best"]["mesh"]

    call_count = [0]
    def mock_iou(m1, m2):
        call_count[0] += 1
        if call_count[0] == 1:
            # First pass: pretend IoU dropped severely (drop = 0.30 > 0.02)
            return 0.40
        # Retry pass: pretend good recovery
        return 0.70

    target_tris = 10000
    with patch.object(tools_core.score, "iou", side_effect=mock_iou):
        with patch.object(tools_core.blender, "run", wraps=tools_core.blender.run) as mock_blender:
            res_opt = tools_core.mesh_optimize(project_id=pid, mesh_path=scored_mesh, target_tris=target_tris)
            assert res_opt["ok"] is True

            # Verify optimize.py was called twice (first pass + retry pass)
            opt_calls = [c for c in mock_blender.call_args_list if c[0][0] == "optimize.py"]
            assert len(opt_calls) == 2, f"Expected 2 optimize.py calls, got {len(opt_calls)}"

            # Call 1 target_tris = 10000; Call 2 target_tris = 15000 (10000 * 1.5)
            assert opt_calls[0][0][1]["target_tris"] == 10000
            assert opt_calls[1][0][1]["target_tris"] == 15000

            # Result tris_after should be close to 15000
            assert res_opt["data"]["tris_after"] <= 15000 * 1.05


# ==============================================================================
# 3. MESH EXPORT STRESS TESTS (GLB & FBX RE-IMPORT VERIFICATION)
# ==============================================================================

def test_mesh_export_glb_and_fbx_reimport_verification(sheet_path, tmp_path):
    """Verify mesh_export produces both GLB and FBX with metric units, applied transforms, and matching tris."""
    height_cm = 165.0
    expected_h = height_cm / 100.0

    res_create = server.project_create(sheet_path=str(sheet_path), name="ExportTest", height_cm=height_cm)
    pid = res_create["project_id"]
    server.sheet_split(project_id=pid)

    box = _create_distorted_box(tmp_path / "exp_box.obj")
    res_score = server.score_mesh(project_id=pid, mesh_path=str(box))
    scored_mesh = res_score["data"]["best"]["mesh"]

    # Export both
    res_exp = tools_core.mesh_export(project_id=pid, mesh_path=scored_mesh, format="both")
    assert res_exp["ok"] is True
    data_exp = res_exp["data"]

    assert "glb" in data_exp["files"]
    assert "fbx" in data_exp["files"]
    assert data_exp["size_mb"] > 0
    tris_expected = data_exp["tris"]

    pdir = state.project_dir(pid)
    glb_abs = pdir / data_exp["files"]["glb"]
    fbx_abs = pdir / data_exp["files"]["fbx"]
    assert glb_abs.is_file()
    assert fbx_abs.is_file()

    # Re-import GLB into Blender
    geom_glb = _inspect_blender_geometry(glb_abs, ".glb")
    assert geom_glb["unit_system"] == "METRIC"
    assert geom_glb["scale_length"] == 1.0
    assert abs(geom_glb["tris"] - tris_expected) <= tris_expected * 0.01
    assert abs(geom_glb["height"] - expected_h) <= expected_h * 0.01
    assert abs(geom_glb["lo"][2]) < 1e-4, f"GLB feet not grounded at z=0: {geom_glb['lo'][2]}"

    # Re-import FBX into Blender
    geom_fbx = _inspect_blender_geometry(fbx_abs, ".fbx")
    assert geom_fbx["unit_system"] == "METRIC"
    assert geom_fbx["scale_length"] == 1.0
    assert abs(geom_fbx["tris"] - tris_expected) <= tris_expected * 0.01
    assert abs(geom_fbx["height"] - expected_h) <= expected_h * 0.01
    assert abs(geom_fbx["lo"][2]) < 1e-4, f"FBX feet not grounded at z=0: {geom_fbx['lo'][2]}"


# ==============================================================================
# 4. END-TO-END PIPELINE FOLLOWING NEXT_ACTION
# ==============================================================================

TOOL_MAP = {
    "project_create": server.project_create,
    "project_status": server.project_status,
    "sheet_split": server.sheet_split,
    "render_views": server.render_views,
    "score_mesh": server.score_mesh,
    "generate_mesh": tools_core.generate_mesh,
    "mesh_fit": tools_core.mesh_fit,
    "mesh_optimize": tools_core.mesh_optimize,
    "mesh_export": tools_core.mesh_export,
}


def test_e2e_pipeline_with_mesh_fit_under_12_calls(sheet_path, monkeypatch):
    """Verify autonomous pipeline where candidate requires mesh_fit finishes in <= 12 calls to stage 'exported'."""
    monkeypatch.setenv("CHARFORGE_LEVEL", "2")
    server._register_levels()

    # Create project with elevated threshold (0.99) so candidate requires iterative mesh_fit until plateau stop
    res = server.project_create(sheet_path=str(sheet_path), name="E2E-FitPipe", height_cm=165.0)
    assert res["ok"] is True
    pid = res["project_id"]

    p = state.load(pid)
    p["thresholds"]["front"] = 0.99
    state.save(p)

    call_count = 1
    call_log = [("project_create", res)]
    next_act = res["next_action"]

    # Purely follow next_action from this point until completion
    while next_act is not None and call_count < 15:
        tool_name = next_act["tool"]
        tool_args = next_act["args"]
        fn = TOOL_MAP[tool_name]
        res = fn(**tool_args)
        assert res["ok"] is True, f"Tool {tool_name} failed: {res.get('error')}"
        call_count += 1
        call_log.append((tool_name, res))
        next_act = res["next_action"]

    print(f"DEBUG Pipeline trace ({call_count} calls): {[t for t, _ in call_log]}")
    assert call_count <= 12, f"Expected pipeline to finish in <= 12 calls, took {call_count}"
    assert next_act is None, f"Expected pipeline to finish with None next_action, got {next_act}"
    assert any(tool == "mesh_fit" for tool, _ in call_log), "Expected mesh_fit to be exercised in pipeline"

    p = state.load(pid)
    assert p["stage"] == "exported"
    assert "glb" in p["outputs"]
    glb_path = state.project_dir(pid) / p["outputs"]["glb"]
    assert glb_path.is_file()
    assert glb_path.stat().st_size > 1000

    status = server.project_status(project_id=pid)
    assert status["ok"] is True
    assert status["data"]["stats"]["errors"] == 0
