"""Adversarial stress-test suite for Deformation and Weight Leakage Prevention.

Authored by empirical challenger challenger_m2_2.
Stress-tests:
1. Single-mesh humanoids with arms in A-pose (45 deg) and arms-down (80 deg):
   - On clean welded meshes.
   - On meshes with pre-existing split duplicate vertices at UV seams.
2. Posing LeftArm by 40 deg around local Z:
   - Mean arm/hand displacement > 0.10 m (10 cm).
   - Maximum thigh displacement < 0.005 m (0.5 cm).
   - Maximum lower torso displacement < 0.005 m (0.5 cm).
3. Direct inspection of vertex groups:
   - Zero cross-binding between thigh vertices and any arm bones (LeftArm, LeftForeArm, LeftHand, RightArm, RightForeArm, RightHand).
   - Zero cross-binding between lower torso vertices and arm extremity bones (LeftForeArm, LeftHand, RightForeArm, RightHand).
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path
import pytest

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


def _generate_humanoid_glb(
    out_path: Path,
    pose_angle_deg: float,
    split_seams: bool = False,
    height: float = 1.72,
) -> None:
    """Generate a clean or seam-split 1.72m humanoid mesh with arms at pose_angle_deg."""
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
torso.name = "Torso"

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

bpy.ops.object.select_all(action='SELECT')
bpy.context.view_layer.objects.active = torso
bpy.ops.object.join()
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Add UV map
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project()

if {split_seams}:
    bpy.ops.mesh.edge_split(type='EDGE')

bpy.ops.object.mode_set(mode='OBJECT')

out_path = {str(out_path)!r}
bpy.ops.export_scene.gltf(filepath=out_path, export_format='GLB')
"""
    proc = _run_blender(gen_script)
    assert proc.returncode == 0, f"Mesh generation failed: {proc.stderr}"
    assert out_path.is_file() and out_path.stat().st_size > 0


@pytest.mark.parametrize(
    "pose_deg,split_seams",
    [
        (45.0, False),  # A-pose, clean welded
        (45.0, True),   # A-pose, split duplicate seams
        (80.0, False),  # Arms-down, clean welded
        (80.0, True),   # Arms-down, split duplicate seams
    ],
    ids=[
        "a_pose_clean",
        "a_pose_seam_split",
        "arms_down_clean",
        "arms_down_seam_split",
    ],
)
def test_empirical_deformation_and_leakage(l3_env, sheet_path, tmp_path, pose_deg, split_seams):
    """Stress-test 40 deg local Z LeftArm rotation and inspect vertex groups on clean and seam-split meshes."""
    mesh_glb = tmp_path / f"humanoid_pose_{int(pose_deg)}_{'split' if split_seams else 'clean'}.glb"
    H = 1.72
    _generate_humanoid_glb(mesh_glb, pose_angle_deg=pose_deg, split_seams=split_seams, height=H)

    proj_name = f"StressDeform_{int(pose_deg)}_{'split' if split_seams else 'clean'}"
    cr = server.project_create(sheet_path=sheet_path, name=proj_name, height_cm=H * 100.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)
    dest_mesh = pdir / "meshes" / mesh_glb.name
    dest_mesh.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(mesh_glb, dest_mesh)

    rig_res = tools_full.auto_rig(project_id=pid, mesh_path=f"meshes/{dest_mesh.name}", method="blender")
    assert rig_res["ok"], f"auto_rig failed on {proj_name}: {rig_res}"

    rigged_glb = pdir / rig_res["data"]["mesh_path"]
    assert rigged_glb.is_file()

    verify_script = f"""
import math
import bpy
from mathutils import Vector, Euler

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

bpy.ops.import_scene.gltf(filepath={str(rigged_glb)!r})
arm = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE'][0]
mesh = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]

H = {H}
rad = math.radians({pose_deg})
arm_dir = Vector((math.cos(rad), 0, -math.sin(rad)))
sh_center = Vector((0.18, 0, 0.75 * H))
L_upper = 0.28
L_fore = 0.25
L_hand = 0.16
elbow_true = sh_center + arm_dir * L_upper
wrist_true = elbow_true + arm_dir * L_fore

# Build reference coordinates of body primitives to map vertices unambiguously
part_coords = {{}}

# Torso cylinder: radius 0.14, depth 0.55*H, center (0, 0, 0.58*H)
torso_z_bot = 0.58*H - 0.55*H / 2.0
torso_z_top = 0.58*H + 0.55*H / 2.0
part_coords['Torso'] = [
    Vector((0.14 * math.cos(2*math.pi*i/32), 0.14 * math.sin(2*math.pi*i/32), z))
    for z in [torso_z_bot, torso_z_top]
    for i in range(32)
]

# Left Leg cylinder: radius 0.06*H, depth 0.50*H, center (0.10*H, 0, 0.25*H)
r_leg = 0.06 * H
part_coords['LeftLeg'] = [
    Vector((0.10*H + r_leg * math.cos(2*math.pi*i/24), r_leg * math.sin(2*math.pi*i/24), z))
    for z in [0.0, 0.50*H]
    for i in range(24)
]
# Right Leg cylinder: radius 0.06*H, depth 0.50*H, center (-0.10*H, 0, 0.25*H)
part_coords['RightLeg'] = [
    Vector((-0.10*H + r_leg * math.cos(2*math.pi*i/24), r_leg * math.sin(2*math.pi*i/24), z))
    for z in [0.0, 0.50*H]
    for i in range(24)
]

# Left Forearm: cylinder radius 0.038, depth L_fore, centered at elbow_true + arm_dir*(L_fore/2)
# Oriented along arm_dir
v_ortho_y = Vector((0, 1, 0))
v_ortho_perp = arm_dir.cross(v_ortho_y).normalized()
part_coords['LeftForeArm'] = []
for t in [0.0, L_fore]:
    center = elbow_true + arm_dir * t
    for i in range(24):
        theta = 2 * math.pi * i / 24
        part_coords['LeftForeArm'].append(center + 0.038 * (math.cos(theta) * v_ortho_perp + math.sin(theta) * v_ortho_y))

# Left Hand: cylinder radius 0.030, depth L_hand, centered at wrist_true + arm_dir*(L_hand/2)
part_coords['LeftHand'] = []
for t in [0.0, L_hand]:
    center = wrist_true + arm_dir * t
    for i in range(24):
        theta = 2 * math.pi * i / 24
        part_coords['LeftHand'].append(center + 0.030 * (math.cos(theta) * v_ortho_perp + math.sin(theta) * v_ortho_y))

# Match each mesh vertex to its origin body part
thigh_v_indices = []
lower_torso_v_indices = []
left_fore_hand_v_indices = []

for v in mesh.data.vertices:
    p = v.co
    best_pname = None
    best_d = 1e9
    for pname, pts in part_coords.items():
        for pt in pts:
            d = (p - pt).length
            if d < best_d:
                best_d = d
                best_pname = pname
    if best_d < 0.005:
        if best_pname in ('LeftLeg', 'RightLeg'):
            thigh_v_indices.append(v.index)
        elif best_pname == 'Torso' and p.z <= 0.58 * H:
            lower_torso_v_indices.append(v.index)
        elif best_pname in ('LeftForeArm', 'LeftHand'):
            left_fore_hand_v_indices.append(v.index)

print(f"VERIFY CLASSIFICATION: Thigh verts={{len(thigh_v_indices)}}, Lower torso verts={{len(lower_torso_v_indices)}}, Forearm/Hand verts={{len(left_fore_hand_v_indices)}}")
assert len(thigh_v_indices) > 0, "No thigh vertices identified"
assert len(lower_torso_v_indices) > 0, "No lower torso vertices identified"
assert len(left_fore_hand_v_indices) > 0, "No forearm/hand vertices identified"

# 1. DIRECT VERTEX GROUP CROSS-BINDING CHECK
arm_bone_names = [
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand"
]
arm_vgs = {{
    name: mesh.vertex_groups[name].index
    for name in arm_bone_names
    if name in mesh.vertex_groups
}}

thigh_cross_bound = []
for i in thigh_v_indices:
    v = mesh.data.vertices[i]
    for g in v.groups:
        for bname, vg_idx in arm_vgs.items():
            if g.group == vg_idx and g.weight > 1e-4:
                thigh_cross_bound.append((i, bname, g.weight, v.co))

if thigh_cross_bound:
    sample = thigh_cross_bound[:5]
    print(f"FAIL: Cross-binding detected on thigh vertices: count={{len(thigh_cross_bound)}}, sample={{sample}}")
assert len(thigh_cross_bound) == 0, f"Found {{len(thigh_cross_bound)}} thigh vertices cross-bound to arm bones! Sample: {{thigh_cross_bound[:3]}}"

# 2. POSE LeftArm 40 deg around local Z
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

arm_disps = [(eval_coords[i] - orig_coords[i]).length for i in left_fore_hand_v_indices]
thigh_disps = [(eval_coords[i] - orig_coords[i]).length for i in thigh_v_indices]
torso_disps = [(eval_coords[i] - orig_coords[i]).length for i in lower_torso_v_indices]

mean_arm_disp = sum(arm_disps) / len(arm_disps) if arm_disps else 0.0
max_thigh_disp = max(thigh_disps) if thigh_disps else 0.0
max_lower_torso_disp = max(torso_disps) if torso_disps else 0.0

print(f"RESULTS ({pose_deg} deg, split={split_seams}):")
print(f"  mean_arm_disp = {{mean_arm_disp:.5f}} m (threshold > 0.10 m)")
print(f"  max_thigh_disp = {{max_thigh_disp:.5f}} m (threshold < 0.005 m)")
print(f"  max_lower_torso_disp = {{max_lower_torso_disp:.5f}} m (threshold < 0.005 m)")

assert mean_arm_disp > 0.10, f"mean_arm_disp {{mean_arm_disp:.4f}}m <= 0.10m"
assert max_thigh_disp < 0.005, f"max_thigh_disp {{max_thigh_disp:.5f}}m >= 0.005m (weight leakage)"
assert max_lower_torso_disp < 0.005, f"max_lower_torso_disp {{max_lower_torso_disp:.5f}}m >= 0.005m (weight leakage to torso)"

print("CHALLENGE_VERIFY_SUCCESS")
"""
    proc = _run_blender(verify_script)
    assert proc.returncode == 0, f"Blender verification failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "CHALLENGE_VERIFY_SUCCESS" in proc.stdout
