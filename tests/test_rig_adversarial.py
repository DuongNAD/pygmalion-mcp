"""Empirical adversarial stress test suite for Mixamo humanoid auto-rigging.
Designed by Challenger 1 (teamwork_preview_challenger_m3_1).

Verification dimensions:
1. Re-import GLB verification:
   - Armature presence & single Armature modifier on mesh.
   - Exactly 22 bones matching Mixamo naming convention.
   - Parent hierarchy verification for all 22 bones.
   - Vertex group verification on mesh.
2. Dual-joint pose deformation:
   - Pose LeftArm by 45 deg: evaluate depsgraph, check max_disp > 0.01m, zero NaN/Inf.
   - Pose LeftLeg by 30 deg: evaluate depsgraph, check max_disp > 0.01m, zero NaN/Inf.
   - Combined pose (LeftArm 45 deg + LeftLeg 30 deg): evaluate depsgraph, zero NaN/Inf.
3. Unweighted vertex handling on challenging geometry:
   - Disconnected floating islands (e.g., floating hat/headpiece, distant orb).
   - Strict assertion: unweighted_ratio < 0.01 (enforced 0.00).
4. Morphological robustness:
   - Chibi / dwarf mesh (height 80cm) and tall stylized mesh (height 220cm).
5. Comprehensive error envelopes:
   - MESH_NOT_FOUND on non-existent file.
   - NO_PROVIDER when method="provider" without API keys.
   - MESH_FORMAT on unsupported extension (.txt).
   - RIG_FAILED / BLENDER_FAILED on corrupted binary GLB.
   - RIG_FAILED on empty mesh / scene without meshes.
   - PROJECT_NOT_FOUND on non-existent project ID.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import pytest

from charforge import config, server, state, tools_core, tools_full

pytestmark = [pytest.mark.blender, pytest.mark.skipif(config.blender_exe() is None, reason="Blender not installed")]

MIXAMO_22_HIERARCHY = {
    "Hips": None,
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Head": "Neck",
    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToeBase": "LeftFoot",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToeBase": "RightFoot",
}


@pytest.fixture
def l3_env(monkeypatch, tmp_home):
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")


def _run_blender_script(script_code: str) -> subprocess.CompletedProcess:
    exe = config.blender_exe()
    assert exe is not None
    return subprocess.run(
        [exe, "-b", "--factory-startup", "--python-exit-code", "1", "--python-expr", script_code],
        capture_output=True,
        text=True,
    )


def _setup_exported_character(sheet_path, name="TestRigChar", height_cm=165.0):
    cr = server.project_create(sheet_path=sheet_path, name=name, height_cm=height_cm)
    pid = cr["project_id"]
    sp = server.sheet_split(project_id=pid)
    assert sp["ok"]
    gm = tools_core.generate_mesh(project_id=pid, provider="blockout")
    assert gm["ok"]
    cand_path = gm["data"]["mesh_path"]
    sc = server.score_mesh(project_id=pid, mesh_path=cand_path)
    assert sc["ok"]
    opt = tools_core.mesh_optimize(project_id=pid, mesh_path=cand_path, target_tris=15000)
    assert opt["ok"]
    exp = tools_core.mesh_export(project_id=pid, mesh_path=opt["data"]["mesh_path"], format="glb")
    assert exp["ok"]
    return pid, exp["data"]["files"]["glb"]


# ==============================================================================
# SECTION 1: Re-import Rigged GLB, Armature, Bones, Hierarchy & Modifiers
# ==============================================================================

def test_rigged_glb_reimport_armature_hierarchy_skinning(l3_env, sheet_path):
    """Stress-test: re-import export/<id>-rigged.glb in Blender and verify exact Mixamo structure."""
    pid, glb_path = _setup_exported_character(sheet_path, name="MixamoVerify")
    rig_res = tools_full.auto_rig(project_id=pid, mesh_path=glb_path, method="blender")
    assert rig_res["ok"]
    assert rig_res["data"]["bones"] == 22
    assert rig_res["data"]["unweighted_ratio"] < 0.01

    pdir = state.project_dir(pid)
    rigged_glb = pdir / f"export/{pid}-rigged.glb"
    assert rigged_glb.is_file()

    verify_script = f"""
import json, sys
import bpy

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

bpy.ops.import_scene.gltf(filepath={str(rigged_glb)!r})

armatures = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']

assert len(armatures) == 1, f"Expected exactly 1 armature, got {{len(armatures)}}"
assert len(meshes) >= 1, f"Expected at least 1 mesh, got {{len(meshes)}}"

arm = armatures[0]
mesh = meshes[0]

# Verify bone count and names
bone_names = [b.name for b in arm.data.bones]
assert len(bone_names) == 22, f"Expected 22 bones, got {{len(bone_names)}}: {{bone_names}}"

expected_bones = {json.dumps(list(MIXAMO_22_HIERARCHY.keys()))}
for eb in expected_bones:
    assert eb in bone_names, f"Missing required bone {{eb}} in reimported armature"

# Verify parent hierarchy
expected_hierarchy = {repr(MIXAMO_22_HIERARCHY)}
for child_name, exp_parent in expected_hierarchy.items():
    b = arm.data.bones[child_name]
    actual_parent = b.parent.name if b.parent else None
    assert actual_parent == exp_parent, f"Bone {{child_name}} parent mismatch: expected {{exp_parent}}, got {{actual_parent}}"

# Verify Skinning modifier on mesh
arm_modifiers = [m for m in mesh.modifiers if m.type == 'ARMATURE']
assert len(arm_modifiers) >= 1, "Mesh lacks ARMATURE modifier"
assert arm_modifiers[0].object == arm, f"Armature modifier does not target armature: {{arm_modifiers[0].object}}"

# Verify vertex groups exist on mesh
vg_names = [vg.name for vg in mesh.vertex_groups]
for eb in expected_bones:
    assert eb in vg_names, f"Mesh missing vertex group for bone {{eb}}"

print("GLB_REIMPORT_OK")
"""
    proc = _run_blender_script(verify_script)
    assert proc.returncode == 0, f"Blender re-import verification failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "GLB_REIMPORT_OK" in proc.stdout


# ==============================================================================
# SECTION 2: Dual-Joint Pose Deformation & Coordinate Integrity (No NaN/Inf)
# ==============================================================================

def test_pose_deformation_leftarm_and_leftleg(l3_env, sheet_path):
    """Stress-test: rotate LeftArm by 45 deg and LeftLeg by 30 deg, evaluate depsgraph, verify non-zero displacement and no NaN/Inf."""
    pid, glb_path = _setup_exported_character(sheet_path, name="PoseDeform")
    rig_res = tools_full.auto_rig(project_id=pid, mesh_path=glb_path, method="blender")
    assert rig_res["ok"]

    pdir = state.project_dir(pid)
    rigged_glb = pdir / f"export/{pid}-rigged.glb"

    pose_script = f"""
import math, sys
import bpy
from mathutils import Euler

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

bpy.ops.import_scene.gltf(filepath={str(rigged_glb)!r})
arm = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE'][0]
mesh = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]

bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')

orig_coords = [v.co.copy() for v in mesh.data.vertices]

# 1. Test LeftArm rotated by 45 deg
pb_arm = arm.pose.bones.get("LeftArm")
assert pb_arm is not None, "LeftArm pose bone missing"
pb_arm.rotation_mode = 'XYZ'
pb_arm.rotation_euler = Euler((0, math.radians(45), 0), 'XYZ')
bpy.context.view_layer.update()

dg = bpy.context.evaluated_depsgraph_get()
eval_mesh = mesh.evaluated_get(dg)
eval_coords = [v.co.copy() for v in eval_mesh.data.vertices]

# Verify no NaN or Inf
for i, c in enumerate(eval_coords):
    assert not math.isnan(c.x) and not math.isnan(c.y) and not math.isnan(c.z), f"LeftArm 45 deg: NaN vertex co at {{i}}"
    assert not math.isinf(c.x) and not math.isinf(c.y) and not math.isinf(c.z), f"LeftArm 45 deg: Inf vertex co at {{i}}"

arm_disp = max((e - o).length for e, o in zip(eval_coords, orig_coords))
assert arm_disp > 0.02, f"LeftArm 45 deg produced insufficient displacement: {{arm_disp:.4f}}m <= 0.02m"

# Reset LeftArm
pb_arm.rotation_euler = Euler((0, 0, 0), 'XYZ')
bpy.context.view_layer.update()

# 2. Test LeftLeg rotated by 30 deg
pb_leg = arm.pose.bones.get("LeftLeg")
assert pb_leg is not None, "LeftLeg pose bone missing"
pb_leg.rotation_mode = 'XYZ'
pb_leg.rotation_euler = Euler((math.radians(30), 0, 0), 'XYZ')
bpy.context.view_layer.update()

dg = bpy.context.evaluated_depsgraph_get()
eval_mesh_leg = mesh.evaluated_get(dg)
eval_coords_leg = [v.co.copy() for v in eval_mesh_leg.data.vertices]

for i, c in enumerate(eval_coords_leg):
    assert not math.isnan(c.x) and not math.isnan(c.y) and not math.isnan(c.z), f"LeftLeg 30 deg: NaN vertex co at {{i}}"
    assert not math.isinf(c.x) and not math.isinf(c.y) and not math.isinf(c.z), f"LeftLeg 30 deg: Inf vertex co at {{i}}"

leg_disp = max((e - o).length for e, o in zip(eval_coords_leg, orig_coords))
assert leg_disp > 0.02, f"LeftLeg 30 deg produced insufficient displacement: {{leg_disp:.4f}}m <= 0.02m"

# 3. Test simultaneous pose: LeftArm 45 deg + LeftLeg 30 deg
pb_arm.rotation_euler = Euler((0, math.radians(45), 0), 'XYZ')
bpy.context.view_layer.update()

dg = bpy.context.evaluated_depsgraph_get()
eval_mesh_both = mesh.evaluated_get(dg)
eval_coords_both = [v.co.copy() for v in eval_mesh_both.data.vertices]

for i, c in enumerate(eval_coords_both):
    assert not math.isnan(c.x) and not math.isnan(c.y) and not math.isnan(c.z), f"Both posed: NaN vertex co at {{i}}"
    assert not math.isinf(c.x) and not math.isinf(c.y) and not math.isinf(c.z), f"Both posed: Inf vertex co at {{i}}"

print("POSE_DEFORMATION_OK")
"""
    proc = _run_blender_script(pose_script)
    assert proc.returncode == 0, f"Blender pose deformation failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "POSE_DEFORMATION_OK" in proc.stdout


# ==============================================================================
# SECTION 3: Unweighted Vertex Handling on Disconnected Geometries
# ==============================================================================

def test_unweighted_vertices_disconnected_islands(l3_env, sheet_path):
    """Stress-test: mesh with disconnected floating elements (crown, orb, floating hands).
    Verify fallback assigns weights to closest bone and unweighted_ratio < 0.01 holds strictly.
    """
    cr = server.project_create(sheet_path=sheet_path, name="FloatingIslands", height_cm=165.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    # Generate a humanoid mesh with floating isolated parts (floating crown above head, orb next to hand)
    gen_script = f"""
import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

# Main humanoid body approximation
bpy.ops.mesh.primitive_cylinder_add(radius=0.15, depth=1.65, location=(0, 0, 0.825))
body = bpy.context.active_object
body.name = "Body"

# Floating crown above head at z=1.80
bpy.ops.mesh.primitive_torus_add(major_radius=0.12, minor_radius=0.03, location=(0, 0, 1.80))
crown = bpy.context.active_object
crown.name = "Crown"

# Floating magic orb at (0.45, 0, 0.90)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.08, location=(0.45, 0, 0.90))
orb = bpy.context.active_object
orb.name = "Orb"

# Join into single mesh
bpy.ops.object.select_all(action='DESELECT')
body.select_set(True)
crown.select_set(True)
orb.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.object.join()

out_glb = {str(pdir / "meshes" / "floating_model.glb")!r}
bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB')
"""
    proc = _run_blender_script(gen_script)
    assert proc.returncode == 0, f"Generating floating test mesh failed:\n{proc.stderr}"

    state.register_mesh(state.load(pid), "meshes/floating_model.glb", "test")

    # Run auto_rig
    rig_res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/floating_model.glb", method="blender")
    assert rig_res["ok"], f"auto_rig failed on disconnected islands: {rig_res}"
    assert rig_res["data"]["unweighted_ratio"] < 0.01
    assert rig_res["data"]["unweighted_ratio"] == 0.0, "All disconnected vertices must be assigned to closest bone"


# ==============================================================================
# SECTION 4: Morphological Proportions (Extreme Heights)
# ==============================================================================

@pytest.mark.parametrize("height_cm", [80.0, 220.0])
def test_morphological_scaling(l3_env, sheet_path, height_cm):
    """Stress-test auto_rig with dwarf (80cm) and giant (220cm) humanoid proportions."""
    cr = server.project_create(sheet_path=sheet_path, name=f"Char_{int(height_cm)}", height_cm=height_cm)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    H = height_cm / 100.0
    gen_script = f"""
import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

# Torso
bpy.ops.mesh.primitive_cylinder_add(radius={0.12*H}, depth={0.5*H}, location=(0, 0, {0.55*H}))
torso = bpy.context.active_object

# Arms (T-pose)
bpy.ops.mesh.primitive_cylinder_add(radius={0.04*H}, depth={0.8*H}, location=(0, 0, {0.70*H}), rotation=(0, 1.5708, 0))

# Legs
bpy.ops.mesh.primitive_cylinder_add(radius={0.05*H}, depth={0.4*H}, location=({0.08*H}, 0, {0.2*H}))
bpy.ops.mesh.primitive_cylinder_add(radius={0.05*H}, depth={0.4*H}, location=({-0.08*H}, 0, {0.2*H}))

bpy.ops.object.select_all(action="SELECT")
bpy.context.view_layer.objects.active = torso
bpy.ops.object.join()

out_glb = {str(pdir / "meshes" / "humanoid.glb")!r}
bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB')
"""
    proc = _run_blender_script(gen_script)
    assert proc.returncode == 0, proc.stderr

    rig_res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/humanoid.glb", method="blender")
    assert rig_res["ok"]
    assert rig_res["data"]["bones"] == 22
    assert rig_res["data"]["unweighted_ratio"] < 0.01


def test_armless_mesh_triggers_pose_deformation_failure(l3_env, sheet_path):
    """Adversarial check: mesh lacking arm geometry fails pose self-check and returns RIG_FAILED."""
    cr = server.project_create(sheet_path=sheet_path, name="ArmlessPillar", height_cm=165.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    gen_script = f"""
import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

# Narrow vertical pillar with no arms
bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=1.65, location=(0, 0, 0.825))
out_glb = {str(pdir / "meshes" / "pillar.glb")!r}
bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB')
"""
    proc = _run_blender_script(gen_script)
    assert proc.returncode == 0

    res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/pillar.glb", method="blender")
    assert not res["ok"]
    assert res["error"]["code"] == "RIG_FAILED"
    assert "Pose deformation check failed" in res["error"]["message"]


# ==============================================================================
# SECTION 5: Comprehensive Error Handling
# ==============================================================================

def test_error_handling_mesh_not_found(l3_env, sheet_path):
    """Error check: MESH_NOT_FOUND when mesh path does not exist."""
    cr = server.project_create(sheet_path=sheet_path, name="ErrNotFound")
    pid = cr["project_id"]
    res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/non_existent.glb")
    assert not res["ok"]
    assert res["error"]["code"] == "MESH_NOT_FOUND"
    assert "not found" in res["error"]["message"].lower()


def test_error_handling_no_provider(l3_env, sheet_path, monkeypatch):
    """Error check: NO_PROVIDER when method='provider' without cloud credentials."""
    monkeypatch.delenv("TRIPO_API_KEY", raising=False)
    monkeypatch.delenv("MESHY_API_KEY", raising=False)
    cr = server.project_create(sheet_path=sheet_path, name="ErrNoProv")
    pid = cr["project_id"]
    res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/any.glb", method="provider")
    assert not res["ok"]
    assert res["error"]["code"] == "NO_PROVIDER"
    assert "provider" in res["error"]["message"].lower()


def test_error_handling_invalid_mesh_format(l3_env, sheet_path):
    """Error check: MESH_FORMAT when mesh has unsupported extension."""
    cr = server.project_create(sheet_path=sheet_path, name="ErrFormat")
    pid = cr["project_id"]
    pdir = state.project_dir(pid)
    bad_file = pdir / "meshes" / "bad.txt"
    bad_file.write_text("not a 3d model")
    res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/bad.txt")
    assert not res["ok"]
    assert res["error"]["code"] == "MESH_FORMAT"


def test_error_handling_corrupted_glb_binary(l3_env, sheet_path):
    """Error check: Corrupt GLB file produces RIG_FAILED or BLENDER_FAILED envelope."""
    cr = server.project_create(sheet_path=sheet_path, name="ErrCorrupt")
    pid = cr["project_id"]
    pdir = state.project_dir(pid)
    corrupt_glb = pdir / "meshes" / "corrupted.glb"
    corrupt_glb.write_bytes(b"glTF\x02\x00\x00\x00\xff\xff\xff\xffCORRUPTED_DATA_HEADER_PAYLOAD")

    res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/corrupted.glb")
    assert not res["ok"]
    assert res["error"]["code"] in ("RIG_FAILED", "BLENDER_FAILED")


def test_error_handling_empty_scene_no_mesh(l3_env, sheet_path):
    """Error check: GLB containing camera/light but 0 meshes produces RIG_FAILED envelope."""
    cr = server.project_create(sheet_path=sheet_path, name="ErrEmptyMesh")
    pid = cr["project_id"]
    pdir = state.project_dir(pid)
    empty_glb = pdir / "meshes" / "empty_scene.glb"

    create_empty_script = f"""
import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.light_add(type='POINT', location=(0, 0, 1))
bpy.ops.export_scene.gltf(filepath={str(empty_glb)!r}, export_format='GLB')
"""
    proc = _run_blender_script(create_empty_script)
    assert proc.returncode == 0

    res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/empty_scene.glb")
    assert not res["ok"]
    assert res["error"]["code"] == "RIG_FAILED"
    assert "contains no mesh" in res["error"]["message"].lower()


def test_error_handling_project_not_found(l3_env):
    """Error check: PROJECT_NOT_FOUND when non-existent project_id is provided."""
    res = tools_full.auto_rig(project_id="nonexistent-id", mesh_path="meshes/any.glb")
    assert not res["ok"]
    assert res["error"]["code"] == "PROJECT_NOT_FOUND"


# ==============================================================================
# SECTION 6: Advanced Structural Stress Tests
# ==============================================================================

def test_multi_mesh_objects_joined_and_rigged(l3_env, sheet_path):
    """Stress-test: GLB with multiple separate mesh objects (head, torso, limbs).
    Verify load_mesh joins them into a single mesh and rigs them without error.
    """
    cr = server.project_create(sheet_path=sheet_path, name="MultiMeshChar", height_cm=165.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    gen_script = f"""
import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

# Separate objects (do NOT join in blender before export)
bpy.ops.mesh.primitive_cylinder_add(radius=0.14, depth=0.6, location=(0, 0, 0.9))
bpy.context.active_object.name = "Torso"

bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, location=(0, 0, 1.45))
bpy.context.active_object.name = "Head"

bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.6, location=(0.3, 0, 1.1), rotation=(0, 1.5708, 0))
bpy.context.active_object.name = "LeftArmMesh"

bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.6, location=(-0.3, 0, 1.1), rotation=(0, 1.5708, 0))
bpy.context.active_object.name = "RightArmMesh"

bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=0.6, location=(0.1, 0, 0.3))
bpy.context.active_object.name = "LeftLegMesh"

bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=0.6, location=(-0.1, 0, 0.3))
bpy.context.active_object.name = "RightLegMesh"

out_glb = {str(pdir / "meshes" / "multi_part.glb")!r}
bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB')
"""
    proc = _run_blender_script(gen_script)
    assert proc.returncode == 0

    rig_res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/multi_part.glb", method="blender")
    assert rig_res["ok"], f"Multi-mesh auto_rig failed: {rig_res}"
    assert rig_res["data"]["bones"] == 22
    assert rig_res["data"]["unweighted_ratio"] < 0.01


def test_corrupted_reference_json_fallback(l3_env, sheet_path):
    """Stress-test: corrupt or malformed reference.json falls back safely to default neck_ratio=0.15."""
    cr = server.project_create(sheet_path=sheet_path, name="CorruptRef", height_cm=165.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    # Write corrupt reference.json
    ref_file = pdir / "views" / "reference.json"
    ref_file.write_text("{malformed_json: true, neck_ratio: 'invalid_float'")

    # Create basic humanoid mesh
    H = 1.65
    gen_script = f"""
import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.mesh.primitive_cylinder_add(radius=0.14, depth=0.6, location=(0, 0, 0.9))
torso = bpy.context.active_object
bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.7, location=(0, 0, 1.1), rotation=(0, 1.5708, 0))
bpy.ops.object.select_all(action="SELECT")
bpy.context.view_layer.objects.active = torso
bpy.ops.object.join()
out_glb = {str(pdir / "meshes" / "simple_human.glb")!r}
bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB')
"""
    proc = _run_blender_script(gen_script)
    assert proc.returncode == 0

    rig_res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/simple_human.glb", method="blender")
    assert rig_res["ok"], f"auto_rig should recover from corrupt reference.json: {rig_res}"


def test_unnormalized_transforms_applied(l3_env, sheet_path):
    """Stress-test: mesh with non-zero location, rotation, and non-uniform scale gets transforms applied cleanly."""
    cr = server.project_create(sheet_path=sheet_path, name="TransformedMesh", height_cm=165.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    gen_script = f"""
import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

# Cylinder with non-zero transforms
bpy.ops.mesh.primitive_cylinder_add(radius=0.14, depth=0.6, location=(0.5, -0.3, 1.2))
torso = bpy.context.active_object
bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.7, location=(0.5, -0.3, 1.4), rotation=(0, 1.5708, 0))
bpy.ops.object.select_all(action="SELECT")
bpy.context.view_layer.objects.active = torso
bpy.ops.object.join()

# Apply non-uniform scale and rotation without applying transforms
torso.scale = (1.2, 0.8, 1.5)
torso.rotation_euler = (0.2, 0.1, -0.3)

out_glb = {str(pdir / "meshes" / "unapplied_transforms.glb")!r}
bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB')
"""
    proc = _run_blender_script(gen_script)
    assert proc.returncode == 0

    rig_res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/unapplied_transforms.glb", method="blender")
    assert rig_res["ok"], f"auto_rig failed with unapplied transforms: {rig_res}"
    assert rig_res["data"]["bones"] == 22


def test_auto_method_defaults_to_blender(l3_env, sheet_path, monkeypatch):
    """Verify method='auto' defaults to Blender when no cloud API keys exist."""
    monkeypatch.delenv("TRIPO_API_KEY", raising=False)
    monkeypatch.delenv("MESHY_API_KEY", raising=False)
    pid, glb_path = _setup_exported_character(sheet_path, name="AutoMethodFallback")
    res = tools_full.auto_rig(project_id=pid, mesh_path=glb_path, method="auto")
    assert res["ok"]
    assert res["data"]["bones"] == 22
    assert res["data"]["unweighted_ratio"] < 0.01


def test_dense_mesh_rigging(l3_env, sheet_path):
    """Stress-test: mesh with ~25,000 vertices runs within timeout and verifies unweighted_ratio < 0.01."""
    cr = server.project_create(sheet_path=sheet_path, name="DenseRig", height_cm=165.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    gen_script = f"""
import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

# Subdivided dense humanoid mesh
bpy.ops.mesh.primitive_cylinder_add(radius=0.15, depth=0.8, location=(0, 0, 0.8), vertices=64)
torso = bpy.context.active_object

bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.8, location=(0, 0, 1.05), rotation=(0, 1.5708, 0), vertices=64)

bpy.ops.object.select_all(action="SELECT")
bpy.context.view_layer.objects.active = torso
bpy.ops.object.join()

# Add subdivision surface
sub = torso.modifiers.new("Subsurf", 'SUBSURF')
sub.levels = 2
sub.render_levels = 2
bpy.ops.object.modifier_apply(modifier="Subsurf")

out_glb = {str(pdir / "meshes" / "dense.glb")!r}
bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB')
"""
    proc = _run_blender_script(gen_script)
    assert proc.returncode == 0

    rig_res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/dense.glb", method="blender")
    assert rig_res["ok"], f"Dense mesh rigging failed: {rig_res}"
    assert rig_res["data"]["bones"] == 22
    assert rig_res["data"]["unweighted_ratio"] < 0.01

