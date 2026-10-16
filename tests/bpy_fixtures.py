"""Synthetic rigged characters for Blender tests (no real art needed).

make_character() runs headless Blender once and writes a skinned A-pose humanoid GLB:
capsule body voxel-remeshed into one surface, Mixamo-like skeleton (optionally renamed), bone-heat weights,
optional skirt / ponytail, optional injected weight leak (left hand bound to the left thigh).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from charforge import config

_SCRIPT = r'''
import bpy, json, math, sys
from mathutils import Vector
P = json.loads(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)

def cyl(a, b, r, name):
    a, b = Vector(a), Vector(b)
    d = b - a
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=r, depth=d.length, location=(a + b) / 2)
    o = bpy.context.active_object
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    o.name = name
    return o

def sph(c, r, name):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=r, location=c)
    o = bpy.context.active_object
    o.name = name
    return o

ang = math.radians(P["arm_angle"])          # 0 = arms down along the body, 90 = T-pose
sh = {s: Vector((s * 0.17, 0, 1.36)) for s in (1, -1)}
dirv = {s: Vector((s * math.sin(ang), 0, -math.cos(ang))) for s in (1, -1)}
elb = {s: sh[s] + dirv[s] * 0.27 for s in (1, -1)}
wri = {s: elb[s] + dirv[s] * 0.25 for s in (1, -1)}
hnd = {s: wri[s] + dirv[s] * 0.10 for s in (1, -1)}
lx = P["leg_x"]
parts = [
    cyl((0, 0, 0.86), (0, 0, 1.40), 0.13, "torso"),
    sph((0, 0, 0.92), 0.15, "pelvis"),
    cyl((0, 0, 1.38), (0, 0, 1.50), 0.05, "neck"),
    sph((0, 0, 1.58), 0.11, "head"),
]
for s in (1, -1):
    parts += [
        sph(sh[s], 0.06, "shoulder"),
        cyl(sh[s], elb[s], 0.045, "uparm"), sph(elb[s], 0.045, "elbow"),
        cyl(elb[s], wri[s], 0.038, "forearm"), sph(wri[s], 0.04, "wrist"),
        cyl(wri[s], hnd[s], 0.035, "hand"),
        cyl((s * lx, 0, 0.92), (s * lx, 0, 0.47), 0.07, "thigh"), sph((s * lx, 0, 0.47), 0.06, "knee"),
        cyl((s * lx, 0, 0.47), (s * lx, 0, 0.08), 0.05, "shin"),
        cyl((s * lx, 0.02, 0.05), (s * lx, -0.16, 0.05), 0.045, "foot"),
    ]
if P["skirt"]:
    bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=0.30, radius2=0.155, depth=0.40, location=(0, 0, 0.72), end_fill_type="NOTHING")
    skirt = bpy.context.active_object
    mod = skirt.modifiers.new("thick", "SOLIDIFY")
    mod.thickness = 0.035
    bpy.ops.object.modifier_apply(modifier=mod.name)
    parts.append(skirt)
if P["ponytail"]:
    parts.append(cyl((0, 0.10, 1.62), (0, 0.16, 1.18), 0.035, "ponytail"))
bpy.ops.object.select_all(action="DESELECT")
for o in parts:
    o.select_set(True)
bpy.context.view_layer.objects.active = parts[0]
bpy.ops.object.join()
body = bpy.context.active_object
body.name = "Body"
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
if P["remesh"]:
    body.data.remesh_voxel_size = P["voxel"]
    bpy.ops.object.voxel_remesh()

arm_data = bpy.data.armatures.new("Armature")
arm = bpy.data.objects.new("Armature", arm_data)
bpy.context.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
N = P["names"]
def bone(key, head, tail, parent=None):
    b = arm_data.edit_bones.new(N.get(key, key))
    b.head, b.tail = Vector(head), Vector(tail)
    if parent:
        b.parent = arm_data.edit_bones[N.get(parent, parent)]
        b.use_connect = (b.parent.tail - b.head).length < 1e-4
    return b
bone("Hips", (0, 0, 0.92), (0, 0, 1.02))
bone("Spine", (0, 0, 1.02), (0, 0, 1.14), "Hips")
bone("Spine1", (0, 0, 1.14), (0, 0, 1.26), "Spine")
bone("Spine2", (0, 0, 1.26), (0, 0, 1.38), "Spine1")
bone("Neck", (0, 0, 1.38), (0, 0, 1.48), "Spine2")
bone("Head", (0, 0, 1.48), (0, 0, 1.66), "Neck")
bone("HeadTop_End", (0, 0, 1.66), (0, 0, 1.70), "Head")
for s, side in ((1, "Left"), (-1, "Right")):
    bone(side + "Shoulder", (s * 0.05, 0, 1.36), tuple(sh[s]), "Spine2")
    bone(side + "Arm", tuple(sh[s]), tuple(elb[s]), side + "Shoulder")
    bone(side + "ForeArm", tuple(elb[s]), tuple(wri[s]), side + "Arm")
    bone(side + "Hand", tuple(wri[s]), tuple(hnd[s]), side + "ForeArm")
    bone(side + "UpLeg", (s * lx, 0, 0.92), (s * lx, 0, 0.47), "Hips")
    bone(side + "Leg", (s * lx, 0, 0.47), (s * lx, 0, 0.08), side + "UpLeg")
    bone(side + "Foot", (s * lx, 0, 0.08), (s * lx, -0.10, 0.04), side + "Leg")
    bone(side + "ToeBase", (s * lx, -0.10, 0.04), (s * lx, -0.17, 0.03), side + "Foot")
bpy.ops.object.mode_set(mode="OBJECT")

bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type="ARMATURE_AUTO")

if P["leak"]:
    hand = body.vertex_groups[N.get("LeftHand", "LeftHand")]
    for v in body.data.vertices:
        if v.co.x > 0.02 and 0.55 < v.co.z < 0.85 and math.hypot(v.co.x - lx, v.co.y) < 0.12:
            for g in list(v.groups):
                body.vertex_groups[g.group].add([v.index], g.weight * (1 - P["leak"]), "REPLACE")
            hand.add([v.index], P["leak"], "ADD")

bpy.ops.export_scene.gltf(filepath=P["out"], export_format="GLB", export_skins=True)
with open(P["out"] + ".json", "w") as f:
    json.dump({"verts": len(body.data.vertices), "bones": len(arm_data.bones)}, f)
'''

UE_NAMES = {
    "Hips": "pelvis", "Spine": "spine_01", "Spine1": "spine_02", "Spine2": "spine_03", "Neck": "neck_01", "Head": "head",
    "HeadTop_End": "head_end", "LeftShoulder": "clavicle_l", "LeftArm": "upperarm_l", "LeftForeArm": "lowerarm_l",
    "LeftHand": "hand_l", "RightShoulder": "clavicle_r", "RightArm": "upperarm_r", "RightForeArm": "lowerarm_r",
    "RightHand": "hand_r", "LeftUpLeg": "thigh_l", "LeftLeg": "calf_l", "LeftFoot": "foot_l", "LeftToeBase": "ball_l",
    "RightUpLeg": "thigh_r", "RightLeg": "calf_r", "RightFoot": "foot_r", "RightToeBase": "ball_r",
}
OPAQUE_NAMES = {k: f"Bone.{i:03d}" for i, k in enumerate(UE_NAMES)}


def make_character(
    out: Path,
    *,
    arm_angle: float = 40.0,
    leg_x: float = 0.09,
    skirt: bool = False,
    ponytail: bool = False,
    leak: float = 0.0,
    names: dict[str, str] | None = None,
    remesh: bool = True,
    voxel: float = 0.014,
) -> dict:
    exe = config.blender_exe()
    if not exe:
        raise RuntimeError("Blender not installed")
    params = {
        "out": str(out), "arm_angle": arm_angle, "leg_x": leg_x, "skirt": skirt, "ponytail": ponytail,
        "leak": leak, "names": names or {}, "remesh": remesh, "voxel": voxel,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [exe, "-b", "--factory-startup", "--python-exit-code", "1", "--python-expr", _SCRIPT, "--", json.dumps(params)],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(f"fixture build failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    return json.loads(Path(str(out) + ".json").read_text())
