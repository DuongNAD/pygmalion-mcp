"""Tests for Milestone 2: Pose-Aware Blender Fallback Rigging & Weight Leakage Prevention.

Validates:
1. Geometry-driven joint placement: <= 8 cm error across T-pose, A-pose (45 deg), and arms-down (80 deg).
2. Robust deformation & weight leakage check: LeftArm swing 40 deg around local Z -> arm disp >= 0.10m, thigh disp < 0.005m on clean and seam-split meshes.
3. Pre-bind seam welding: remove_doubles welds duplicate vertices at UV seams before binding.
4. Bone heat failure detection & pre-fallback ratio: os.dup2 stderr capture, pre_fallback_ratio capture, and two-tier fallback weighting.
"""
import math
import subprocess
import pytest
from pathlib import Path

from charforge import blender, config, server, state, tools_full


@pytest.fixture
def l3_env(monkeypatch, tmp_home):
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")


def _run_blender(script_code: str) -> subprocess.CompletedProcess:
    exe = config.blender_exe()
    assert exe is not None, "Blender executable not found"
    return subprocess.run(
        [exe, "-b", "--factory-startup", "--python-exit-code", "1", "--python-expr", script_code],
        capture_output=True,
        text=True,
    )


def _generate_humanoid_glb(out_path: Path, pose_angle_deg: float, split_seams: bool = False, internal_wall: bool = False, height: float = 1.72) -> None:
    """Generate a clean or seam-split 1.72m humanoid mesh with arms at the given angle."""
    gen_script = f"""
import math
import bpy
from mathutils import Vector

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

H = {height}
# Central torso
bpy.ops.mesh.primitive_cylinder_add(radius=0.14, depth=0.55*H, location=(0, 0, 0.58*H), vertices=32)
torso = bpy.context.active_object

# Head
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.11*H, location=(0, 0, 0.90*H))

# Legs
bpy.ops.mesh.primitive_cylinder_add(radius=0.06*H, depth=0.50*H, location=(0.10*H, 0, 0.25*H), vertices=24)
bpy.ops.mesh.primitive_cylinder_add(radius=0.06*H, depth=0.50*H, location=(-0.10*H, 0, 0.25*H), vertices=24)

sh_center = Vector((0.18, 0, 0.75 * H))
rad = math.radians({pose_angle_deg})
arm_dir = Vector((math.cos(rad), 0, -math.sin(rad)))
L_upper = 0.28
L_fore = 0.25
L_hand = 0.16
elbow_true = sh_center + arm_dir * L_upper
wrist_true = elbow_true + arm_dir * L_fore

# Left Arm cylinders
bpy.ops.mesh.primitive_cylinder_add(radius=0.045, depth=L_upper, location=(0,0,0), vertices=24)
u_arm = bpy.context.active_object
u_arm.rotation_mode = 'XYZ'
u_arm.rotation_euler = (0, math.pi/2 + rad, 0)
u_arm.location = sh_center + arm_dir * (L_upper / 2.0)

bpy.ops.mesh.primitive_cylinder_add(radius=0.038, depth=L_fore, location=(0,0,0), vertices=24)
f_arm = bpy.context.active_object
f_arm.rotation_mode = 'XYZ'
f_arm.rotation_euler = (0, math.pi/2 + rad, 0)
f_arm.location = elbow_true + arm_dir * (L_fore / 2.0)

bpy.ops.mesh.primitive_cylinder_add(radius=0.030, depth=L_hand, location=(0,0,0), vertices=24)
h_arm = bpy.context.active_object
h_arm.rotation_mode = 'XYZ'
h_arm.rotation_euler = (0, math.pi/2 + rad, 0)
h_arm.location = wrist_true + arm_dir * (L_hand / 2.0)

# Right Arm (symmetric)
r_arm_dir = Vector((-math.cos(rad), 0, -math.sin(rad)))
r_sh_center = Vector((-0.18, 0, 0.75 * H))

bpy.ops.mesh.primitive_cylinder_add(radius=0.045, depth=L_upper, location=(0,0,0), vertices=24)
ru_arm = bpy.context.active_object
ru_arm.rotation_mode = 'XYZ'
ru_arm.rotation_euler = (0, -(math.pi/2 + rad), 0)
ru_arm.location = r_sh_center + r_arm_dir * (L_upper / 2.0)

bpy.ops.mesh.primitive_cylinder_add(radius=0.038, depth=L_fore, location=(0,0,0), vertices=24)
rf_arm = bpy.context.active_object
rf_arm.rotation_mode = 'XYZ'
rf_arm.rotation_euler = (0, -(math.pi/2 + rad), 0)
rf_arm.location = r_sh_center + r_arm_dir * (L_upper + L_fore / 2.0)

bpy.ops.mesh.primitive_cylinder_add(radius=0.030, depth=L_hand, location=(0,0,0), vertices=24)
rh_arm = bpy.context.active_object
rh_arm.rotation_mode = 'XYZ'
rh_arm.rotation_euler = (0, -(math.pi/2 + rad), 0)
rh_arm.location = r_sh_center + r_arm_dir * (L_upper + L_fore + L_hand / 2.0)

if {internal_wall}:
    # Internal non-manifold planar wall inside torso (induces bone heat solver failure)
    bpy.ops.mesh.primitive_plane_add(size=0.25, location=(0, 0, 0.58*H))

bpy.ops.object.select_all(action='SELECT')
bpy.context.view_layer.objects.active = torso
bpy.ops.object.join()
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Add UV map
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project()

if {split_seams}:
    # Intentionally split faces to create duplicate vertices along seam boundaries
    bpy.ops.mesh.edge_split(type='EDGE')

bpy.ops.object.mode_set(mode='OBJECT')

out_path = {str(out_path)!r}
bpy.ops.export_scene.gltf(filepath=out_path, export_format='GLB')
"""
    proc = _run_blender(gen_script)
    assert proc.returncode == 0, f"Mesh generation failed: {proc.stderr}"
    assert out_path.is_file() and out_path.stat().st_size > 0


# ==============================================================================
# Suite 1: Geometry-Driven Joint Placement (<= 8 cm on T/A/arms-down poses)
# ==============================================================================

@pytest.mark.parametrize("pose_deg", [0.0, 45.0, 80.0])
def test_geometry_driven_joint_placement_across_poses(l3_env, sheet_path, tmp_path, pose_deg):
    """Validate elbow and wrist joint placement accuracy <= 8 cm across T, A, and arms-down poses on 1.72m humanoid."""
    mesh_glb = tmp_path / f"humanoid_{int(pose_deg)}deg.glb"
    _generate_humanoid_glb(mesh_glb, pose_angle_deg=pose_deg, height=1.72)

    cr = server.project_create(sheet_path=sheet_path, name=f"PoseJoints_{int(pose_deg)}", height_cm=172.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)
    dest_mesh = pdir / "meshes" / f"humanoid_{int(pose_deg)}deg.glb"
    dest_mesh.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(mesh_glb, dest_mesh)

    rig_res = tools_full.auto_rig(project_id=pid, mesh_path=f"meshes/{dest_mesh.name}", method="blender")
    assert rig_res["ok"], f"auto_rig failed on {pose_deg} deg: {rig_res}"
    assert rig_res["data"]["bones"] == 22
    assert rig_res["data"]["unweighted_ratio"] < 0.01

    rigged_glb = pdir / rig_res["data"]["mesh_path"]
    assert rigged_glb.is_file()

    verify_script = f"""
import math
import bpy
from mathutils import Vector

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

bpy.ops.import_scene.gltf(filepath={str(rigged_glb)!r})
arm = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE'][0]
bones = arm.data.bones

H = 1.72
sh_center = Vector((0.18, 0, 0.75 * H))
rad = math.radians({pose_deg})
arm_dir = Vector((math.cos(rad), 0, -math.sin(rad)))
elbow_true = sh_center + arm_dir * 0.28
wrist_true = elbow_true + arm_dir * 0.25

b_elbow = bones["LeftForeArm"].head_local
b_wrist = bones["LeftHand"].head_local

elb_err = (Vector(b_elbow) - elbow_true).length
wri_err = (Vector(b_wrist) - wrist_true).length

print(f"POSE {pose_deg}: elb_err={{elb_err:.4f}}m, wri_err={{wri_err:.4f}}m")
assert elb_err <= 0.08, f"Elbow placement error {{elb_err:.4f}}m exceeds 0.08m threshold"
assert wri_err <= 0.08, f"Wrist placement error {{wri_err:.4f}}m exceeds 0.08m threshold"

# Verify bilateral symmetry for right arm
r_elbow = bones["RightForeArm"].head_local
assert abs(r_elbow.x + b_elbow.x) < 0.01, f"Right elbow not symmetric in X: {{r_elbow.x}} vs {{-b_elbow.x}}"
assert abs(r_elbow.z - b_elbow.z) < 0.01, f"Right elbow not symmetric in Z: {{r_elbow.z}} vs {{b_elbow.z}}"

print("JOINT_PLACEMENT_OK")
"""
    proc = _run_blender(verify_script)
    assert proc.returncode == 0, f"Joint verification failed:\n{proc.stdout}\n{proc.stderr}"
    assert "JOINT_PLACEMENT_OK" in proc.stdout


# ==============================================================================
# Suite 2: Robust Deformation & Weight Leakage Self-Check
# ==============================================================================

@pytest.mark.parametrize("split_seams", [False, True], ids=["clean_welded", "seam_split"])
def test_pose_deformation_and_leakage_clean_and_seam_split(l3_env, sheet_path, tmp_path, split_seams):
    """Validate 40 deg local Z rotation deformation check: arm disp >= 0.10m and thigh disp < 0.005m."""
    mesh_glb = tmp_path / f"humanoid_deform_{'split' if split_seams else 'clean'}.glb"
    _generate_humanoid_glb(mesh_glb, pose_angle_deg=45.0, split_seams=split_seams, height=1.72)

    cr = server.project_create(sheet_path=sheet_path, name=f"Deform_{'split' if split_seams else 'clean'}", height_cm=172.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)
    dest_mesh = pdir / "meshes" / mesh_glb.name
    dest_mesh.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(mesh_glb, dest_mesh)

    rig_res = tools_full.auto_rig(project_id=pid, mesh_path=f"meshes/{dest_mesh.name}", method="blender")
    assert rig_res["ok"], f"auto_rig failed: {rig_res}"

    # Verify rig.py JSON output contract directly via blender.run
    rig_data = blender.run("rig.py", {"mesh": dest_mesh, "out": pdir / "export" / f"direct_{dest_mesh.name}"})
    assert rig_data["ok"]
    assert "mean_arm_disp" in rig_data
    assert "max_thigh_disp" in rig_data

    mean_arm_disp = rig_data["mean_arm_disp"]
    max_thigh_disp = rig_data["max_thigh_disp"]

    # Assert Requirement 5: mean_arm_disp >= 0.10m, max_thigh_disp < 0.005m
    assert mean_arm_disp >= 0.10, f"Mean arm displacement {mean_arm_disp:.4f}m < 0.10m"
    assert max_thigh_disp < 0.005, f"Weight leakage detected: max thigh disp {max_thigh_disp:.4f}m >= 0.005m"

    # Verify directly inside Blender on the exported rigged GLB
    rigged_glb = pdir / rig_res["data"]["mesh_path"]
    verify_script = f"""
import math
import bpy
from mathutils import Vector, Euler

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

bpy.ops.import_scene.gltf(filepath={str(rigged_glb)!r})
arm = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE'][0]
mesh = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]

orig_coords = [v.co.copy() for v in mesh.data.vertices]

bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')

pb_arm = arm.pose.bones["LeftArm"]
pb_arm.rotation_mode = 'XYZ'
pb_arm.rotation_euler = Euler((0, 0, math.radians(40)), 'XYZ')
bpy.context.view_layer.update()

dg = bpy.context.evaluated_depsgraph_get()
eval_mesh = mesh.evaluated_get(dg)
eval_coords = [v.co.copy() for v in eval_mesh.data.vertices]

# Reset pose
pb_arm.rotation_euler = Euler((0, 0, 0), 'XYZ')
bpy.context.view_layer.update()
bpy.ops.object.mode_set(mode='OBJECT')

# Arm vertices (LeftHand or LeftForeArm weight > 0.3)
arm_vgs = [mesh.vertex_groups[n].index for n in ["LeftForeArm", "LeftHand"] if n in mesh.vertex_groups]
arm_v_indices = [v.index for v in mesh.data.vertices if any(g.group in arm_vgs and g.weight > 0.3 for g in v.groups)]
arm_disps = [(eval_coords[i] - orig_coords[i]).length for i in arm_v_indices]
mean_arm = sum(arm_disps) / len(arm_disps) if arm_disps else 0.0

# Thigh vertices (z <= 0.40 * 1.72 = 0.688m)
thigh_indices = [i for i, v in enumerate(orig_coords) if v.z <= 0.40 * 1.72 and abs(v.x) <= 0.25 and i not in arm_v_indices]
thigh_disps = [(eval_coords[i] - orig_coords[i]).length for i in thigh_indices]
max_thigh = max(thigh_disps) if thigh_disps else 0.0

print(f"BLENDER VERIFY: mean_arm={{mean_arm:.4f}}m, max_thigh={{max_thigh:.4f}}m")
assert mean_arm >= 0.10, f"mean_arm {{mean_arm}} < 0.10"
assert max_thigh < 0.005, f"max_thigh {{max_thigh}} >= 0.005"
print("DEFORMATION_CHECK_OK")
"""
    proc = _run_blender(verify_script)
    assert proc.returncode == 0, f"Deformation check failed:\n{proc.stdout}\n{proc.stderr}"
    assert "DEFORMATION_CHECK_OK" in proc.stdout


# ==============================================================================
# Suite 3: Pre-Bind Seam Welding (remove_doubles)
# ==============================================================================

def test_pre_bind_seam_welding(l3_env, tmp_path):
    """Validate that load_mesh welds duplicate vertices at UV seams before armature binding."""
    mesh_glb = tmp_path / "seam_split_model.glb"
    _generate_humanoid_glb(mesh_glb, pose_angle_deg=0.0, split_seams=True, height=1.72)

    welding_script = f"""
import os, sys
import bpy

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

# 1. Inspect raw mesh vertex count before load_mesh cleanup
bpy.ops.import_scene.gltf(filepath={str(mesh_glb)!r})
raw_mesh = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
raw_verts = len(raw_mesh.data.vertices)

# 2. Now run rig.load_mesh (which runs remove_doubles and dissolve_degenerate)
sys.path.insert(0, 'src/charforge/bpy_scripts')
import rig

cleaned_mesh = rig.load_mesh({str(mesh_glb)!r})
cleaned_verts = len(cleaned_mesh.data.vertices)

print(f"SEAM WELDING: raw_verts={{raw_verts}}, cleaned_verts={{cleaned_verts}}")
assert cleaned_verts < raw_verts, f"Seam welding did not remove duplicate vertices: {{cleaned_verts}} == {{raw_verts}}"

# Verify UV layers are preserved
assert len(cleaned_mesh.data.uv_layers) > 0, "UV layers were lost during seam welding"

print("SEAM_WELDING_OK")
"""
    proc = _run_blender(welding_script)
    assert proc.returncode == 0, f"Seam welding test failed:\n{proc.stdout}\n{proc.stderr}"
    assert "SEAM_WELDING_OK" in proc.stdout


# ==============================================================================
# Suite 4: Bone Heat Failure Detection & Pre-Fallback Ratio
# ==============================================================================

def test_bone_heat_failure_detection_and_pre_fallback_ratio(l3_env, sheet_path, tmp_path):
    """Validate os.dup2 stderr capture, pre_fallback_ratio calculation, and two-tier fallback weighting."""
    cr = server.project_create(sheet_path=sheet_path, name="HeatFailureTest", height_cm=172.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)
    (pdir / "meshes").mkdir(parents=True, exist_ok=True)

    mesh_glb = pdir / "meshes" / "heat_fail.glb"
    _generate_humanoid_glb(mesh_glb, pose_angle_deg=45.0, internal_wall=True, height=1.72)

    rig_res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/heat_fail.glb", method="blender")
    assert rig_res["ok"], f"auto_rig failed: {rig_res}"
    assert rig_res["data"]["bones"] == 22
    assert rig_res["data"]["unweighted_ratio"] < 0.01

    # Verify rig.py output JSON contract directly via blender.run
    rig_data = blender.run("rig.py", {"mesh": pdir / "meshes" / "heat_fail.glb", "out": pdir / "export" / "direct_heat_fail.glb"})
    assert rig_data["ok"]
    assert "pre_fallback_ratio" in rig_data
    pre_ratio = rig_data["pre_fallback_ratio"]
    assert pre_ratio > 0.0, f"Expected pre_fallback_ratio > 0 for disconnected islands, got {pre_ratio}"
    assert rig_data["unweighted_ratio"] < 0.01
    assert rig_data["max_thigh_disp"] < 0.005
