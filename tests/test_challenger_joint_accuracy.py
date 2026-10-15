"""Adversarial stress-test suite for Anatomical Joint Placement Accuracy.

Designed by Challenger (teamwork_preview_challenger_m2_1).
Mission:
- Construct 1.72 m humanoid meshes in T-pose (0 deg), A-pose (45 deg), arms-down (80 deg)
  plus intermediate and asymmetrical poses.
- Run rig.py to construct armatures.
- Measure Euclidean distance between detected joint heads (LeftForeArm, LeftHand)
  and ground truth anatomical joints.
- Assert that error is strictly <= 8 cm (0.08 m) in all poses.
- Also evaluate RightForeArm and RightHand on asymmetrical and symmetric meshes.
"""
import math
import subprocess
import json
import pytest
from pathlib import Path
from charforge import config

def _run_blender_script(script_code: str) -> subprocess.CompletedProcess:
    exe = config.blender_exe()
    assert exe is not None, "Blender executable not found"
    return subprocess.run(
        [exe, "-b", "--factory-startup", "--python-exit-code", "1", "--python-expr", script_code],
        capture_output=True,
        text=True,
    )

def _evaluate_humanoid_joints(
    l_angle_deg: float,
    r_angle_deg: float | None = None,
    subdivide_limbs: bool = False,
    sh_x_nominal: float = 0.18,
    height: float = 1.72,
) -> dict:
    if r_angle_deg is None:
        r_angle_deg = l_angle_deg

    eval_script = f"""
import math
import json
import sys
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

sh_l = Vector(({sh_x_nominal}, 0, 0.75 * H))
sh_r = Vector((-({sh_x_nominal}), 0, 0.75 * H))

L_upper = 0.28
L_fore = 0.25
L_hand = 0.16

l_rad = math.radians({l_angle_deg})
l_u_dir = Vector((math.cos(l_rad), 0, -math.sin(l_rad))).normalized()
l_elb_true = sh_l + l_u_dir * L_upper
l_wri_true = l_elb_true + l_u_dir * L_fore
l_hnd_true = l_wri_true + l_u_dir * L_hand

def make_limb(start, end, radius):
    vec = end - start
    mid = (start + end) / 2.0
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=vec.length, location=(0,0,0), vertices=24)
    cyl = bpy.context.active_object
    rot = Vector((0, 0, 1)).rotation_difference(vec).to_euler()
    cyl.rotation_euler = rot
    cyl.location = mid
    if {subdivide_limbs}:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=5)
        bpy.ops.object.mode_set(mode='OBJECT')
    return cyl

make_limb(sh_l, l_elb_true, 0.045)
make_limb(l_elb_true, l_wri_true, 0.038)
make_limb(l_wri_true, l_hnd_true, 0.030)

r_rad = math.radians({r_angle_deg})
r_u_dir = Vector((-math.cos(r_rad), 0, -math.sin(r_rad))).normalized()
r_elb_true = sh_r + r_u_dir * L_upper
r_wri_true = r_elb_true + r_u_dir * L_fore
r_hnd_true = r_wri_true + r_u_dir * L_hand

make_limb(sh_r, r_elb_true, 0.045)
make_limb(r_elb_true, r_wri_true, 0.038)
make_limb(r_wri_true, r_hnd_true, 0.030)

# Join all parts
bpy.ops.object.select_all(action='SELECT')
bpy.context.view_layer.objects.active = torso
bpy.ops.object.join()
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Smart UV
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project()
bpy.ops.object.mode_set(mode='OBJECT')

sys.path.insert(0, 'src/charforge/bpy_scripts')
import rig

# Execute armature build
arm_obj = rig.build_armature(torso, neck_ratio=0.15)
bones = arm_obj.data.bones

b_l_elb = Vector(bones["LeftForeArm"].head_local)
b_l_wri = Vector(bones["LeftHand"].head_local)
b_r_elb = Vector(bones["RightForeArm"].head_local)
b_r_wri = Vector(bones["RightHand"].head_local)

l_elb_err = (b_l_elb - l_elb_true).length
l_wri_err = (b_l_wri - l_wri_true).length
r_elb_err = (b_r_elb - r_elb_true).length
r_wri_err = (b_r_wri - r_wri_true).length

# Also test deformation and binding
res_bind = rig.bind_and_verify_weights(torso, arm_obj, neck_ratio=0.15)
deform_res = rig.pose_deformation_check(torso, arm_obj)

output = {{
    "l_elb_err": l_elb_err,
    "l_wri_err": l_wri_err,
    "r_elb_err": r_elb_err,
    "r_wri_err": r_wri_err,
    "l_elb_detected": list(b_l_elb),
    "l_elb_true": list(l_elb_true),
    "l_wri_detected": list(b_l_wri),
    "l_wri_true": list(l_wri_true),
    "r_elb_detected": list(b_r_elb),
    "r_elb_true": list(r_elb_true),
    "r_wri_detected": list(b_r_wri),
    "r_wri_true": list(r_wri_true),
    "unweighted_ratio": float(res_bind),
    "mean_arm_disp": deform_res.mean_arm_disp,
    "max_thigh_disp": deform_res.max_thigh_disp,
}}

print("RESULT_JSON:" + json.dumps(output))
"""
    proc = _run_blender_script(eval_script)
    assert proc.returncode == 0, f"Blender script failed:\n{proc.stdout}\n{proc.stderr}"
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError(f"Could not find RESULT_JSON in Blender stdout:\n{proc.stdout}")


# ==============================================================================
# Suite A: Canonical Pose Verification (0 deg, 45 deg, 80 deg)
# ==============================================================================

@pytest.mark.parametrize("pose_deg", [0.0, 45.0, 80.0])
def test_canonical_poses_joint_accuracy(pose_deg):
    """Verify LeftForeArm (elbow) and LeftHand (wrist) errors <= 8 cm on canonical poses."""
    data = _evaluate_humanoid_joints(pose_deg)
    l_elb_err = data["l_elb_err"]
    l_wri_err = data["l_wri_err"]
    assert l_elb_err <= 0.08, f"Pose {pose_deg} deg: Left elbow error {l_elb_err:.4f}m exceeds 0.08m threshold"
    assert l_wri_err <= 0.08, f"Pose {pose_deg} deg: Left wrist error {l_wri_err:.4f}m exceeds 0.08m threshold"


# ==============================================================================
# Suite B: Intermediate Angles Stress Test (10 to 85 deg)
# ==============================================================================

@pytest.mark.parametrize("pose_deg", [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 75.0, 85.0])
def test_intermediate_angles_joint_accuracy(pose_deg):
    """Stress-test joint placement across intermediate arm angles."""
    data = _evaluate_humanoid_joints(pose_deg)
    l_elb_err = data["l_elb_err"]
    l_wri_err = data["l_wri_err"]
    assert l_elb_err <= 0.08, (
        f"Intermediate pose {pose_deg} deg: Left elbow error {l_elb_err:.4f}m ({l_elb_err*100:.1f}cm) exceeds 0.08m threshold! "
        f"Detected: {data['l_elb_detected']}, True: {data['l_elb_true']}"
    )
    assert l_wri_err <= 0.08, (
        f"Intermediate pose {pose_deg} deg: Left wrist error {l_wri_err:.4f}m ({l_wri_err*100:.1f}cm) exceeds 0.08m threshold! "
        f"Detected: {data['l_wri_detected']}, True: {data['l_wri_true']}"
    )


# ==============================================================================
# Suite C: Subdivided Limbs Geometry Stress Test
# ==============================================================================

@pytest.mark.parametrize("pose_deg", [0.0, 20.0, 30.0, 45.0, 80.0])
def test_subdivided_limbs_joint_accuracy(pose_deg):
    """Stress-test joint placement on meshes with subdivided limb geometry (realistic edge loops)."""
    data = _evaluate_humanoid_joints(pose_deg, subdivide_limbs=True)
    l_elb_err = data["l_elb_err"]
    l_wri_err = data["l_wri_err"]
    assert l_elb_err <= 0.08, (
        f"Subdivided pose {pose_deg} deg: Left elbow error {l_elb_err:.4f}m ({l_elb_err*100:.1f}cm) exceeds 0.08m threshold! "
        f"Detected: {data['l_elb_detected']}, True: {data['l_elb_true']}"
    )
    assert l_wri_err <= 0.08, (
        f"Subdivided pose {pose_deg} deg: Left wrist error {l_wri_err:.4f}m ({l_wri_err*100:.1f}cm) exceeds 0.08m threshold! "
        f"Detected: {data['l_wri_detected']}, True: {data['l_wri_true']}"
    )


# ==============================================================================
# Suite D: Asymmetrical Poses Stress Test
# ==============================================================================

@pytest.mark.parametrize("l_deg,r_deg", [
    (0.0, 45.0),
    (45.0, 80.0),
    (30.0, 60.0),
    (80.0, 0.0),
    (60.0, 20.0),
    (15.0, 75.0),
])
def test_asymmetrical_poses_left_arm_accuracy(l_deg, r_deg):
    """Verify LeftForeArm and LeftHand placement accuracy on asymmetrical models."""
    data = _evaluate_humanoid_joints(l_deg, r_angle_deg=r_deg)
    l_elb_err = data["l_elb_err"]
    l_wri_err = data["l_wri_err"]
    assert l_elb_err <= 0.08, (
        f"Asym L={l_deg}/R={r_deg}: Left elbow error {l_elb_err:.4f}m exceeds 0.08m threshold"
    )
    assert l_wri_err <= 0.08, (
        f"Asym L={l_deg}/R={r_deg}: Left wrist error {l_wri_err:.4f}m exceeds 0.08m threshold"
    )


@pytest.mark.parametrize("l_deg,r_deg", [
    (0.0, 45.0),
    (45.0, 80.0),
    (80.0, 0.0),
])
def test_asymmetrical_poses_right_arm_mirroring_limitation(l_deg, r_deg):
    """Document whether RightForeArm and RightHand follow the right arm or mirror the left arm.

    rig.py currently mirrors RightArm from LeftArm. On an asymmetrical character,
    this creates severe joint errors on the right arm (> 20 cm to 60 cm).
    """
    data = _evaluate_humanoid_joints(l_deg, r_angle_deg=r_deg)
    r_elb_err = data["r_elb_err"]
    r_wri_err = data["r_wri_err"]
    # We assert whether rig.py successfully handles the right arm or fails due to mirroring
    assert r_elb_err <= 0.08 and r_wri_err <= 0.08, (
        f"Asym L={l_deg}/R={r_deg}: Right arm joints failed because rig.py mirrors LeftArm across X=0. "
        f"Right elbow err: {r_elb_err:.4f}m ({r_elb_err*100:.1f}cm), Right wrist err: {r_wri_err:.4f}m ({r_wri_err*100:.1f}cm)"
    )
