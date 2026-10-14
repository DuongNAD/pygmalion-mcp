"""Mixamo-compatible humanoid auto-rigging script. Runs INSIDE Blender: bpy/mathutils/stdlib only.

Operations:
1. Load mesh, center transforms, get bounding box and reference metrics.
2. Build 22-bone humanoid armature (Mixamo naming convention).
3. Bind mesh with ARMATURE_AUTO (bone heat), fallback to ARMATURE_ENVELOPE.
4. Verify skin weights and fix any unweighted vertices to closest bone (unweighted_ratio < 0.01).
5. Self-check: rotate LeftArm by 45 deg, verify vertex displacement > 0.01m, reset pose.
6. Export rigged model with armature to GLB.
Result JSON: {ok: true, bones: int, unweighted_ratio: float, max_disp: float, mesh_path: str}
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

IMPORTERS = {
    ".glb": lambda f: bpy.ops.import_scene.gltf(filepath=f),
    ".fbx": lambda f: bpy.ops.wm.fbx_import(filepath=f) if hasattr(bpy.ops.wm, "fbx_import") else bpy.ops.import_scene.fbx(filepath=f),
    ".obj": lambda f: bpy.ops.wm.obj_import(filepath=f),
    ".ply": lambda f: bpy.ops.wm.ply_import(filepath=f),
    ".stl": lambda f: bpy.ops.wm.stl_import(filepath=f),
}

BONE_NAMES = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neck-ratio", type=float, default=0.15)
    ap.add_argument("--ref-json", default=None)
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:])


def load_mesh(path):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    ext = os.path.splitext(path)[1].lower()
    if ext not in IMPORTERS:
        raise ValueError(f"Unsupported mesh format: {ext}")
    IMPORTERS[ext](path)
    bpy.context.view_layer.update()

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("File contains no mesh objects")

    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for m in meshes:
            m.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        mesh_obj = bpy.context.view_layer.objects.active
    else:
        mesh_obj = meshes[0]

    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return mesh_obj


def build_armature(mesh_obj, neck_ratio):
    zs = [v.co.z for v in mesh_obj.data.vertices]
    xs = [v.co.x for v in mesh_obj.data.vertices]
    ys = [v.co.y for v in mesh_obj.data.vertices]
    z_min, z_max = min(zs), max(zs)
    H = max(0.1, z_max - z_min)
    y_mid = (min(ys) + max(ys)) / 2.0

    arm_data = bpy.data.armatures.new("Armature")
    arm_obj = bpy.data.objects.new("Armature", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    # Height benchmarks
    z_neck_base = z_min + (1.0 - neck_ratio) * H
    z_head_base = z_neck_base + (z_max - z_neck_base) * 0.35
    z_head_top = z_max - 0.01 * H

    z_hips = z_min + 0.52 * H
    z_spine = z_min + 0.58 * H
    z_spine1 = z_min + 0.65 * H
    z_spine2 = min(z_neck_base - 0.04 * H, z_min + 0.72 * H)

    # Leg position heuristic from mesh
    leg_verts = [v.co for v in mesh_obj.data.vertices if abs(v.co.z - (z_min + 0.25 * H)) < 0.05 * H]
    left_legs = [v.x for v in leg_verts if v.x > 0]
    if left_legs:
        leg_x = float(sum(left_legs) / len(left_legs))
    else:
        leg_x = 0.08 * H
    leg_x = max(0.05 * H, min(0.18 * H, leg_x))

    # Arm position heuristic from mesh
    z_sh = min(z_neck_base - 0.02 * H, z_min + 0.78 * H)
    sh_verts = [v.co for v in mesh_obj.data.vertices if abs(v.co.z - z_sh) < 0.04 * H]
    if sh_verts:
        max_arm_x = max(abs(v.x) for v in sh_verts)
    else:
        max_arm_x = 0.20 * H
    max_arm_x = max(0.14 * H, min(0.45 * H, max_arm_x))

    sh_x = min(0.05 * H, max_arm_x * 0.25)
    arm_x = max(sh_x + 0.02 * H, min(0.12 * H, max_arm_x * 0.55))
    elbow_x = max(arm_x + 0.01 * H, min(0.14 * H, max_arm_x * 0.75))
    hand_x = max(elbow_x + 0.01 * H, min(0.16 * H, max_arm_x * 0.85))

    z_elbow = z_min + 0.65 * H
    z_wrist = z_min + 0.52 * H
    z_hand = z_min + 0.45 * H

    edit_bones = {}

    def add_bone(name, parent_name, head, tail, use_connect=False):
        b = arm_data.edit_bones.new(name)
        b.head = head
        b.tail = tail
        b.head_radius = 0.04 * H
        b.tail_radius = 0.04 * H
        b.envelope_distance = 0.10 * H
        if parent_name and parent_name in edit_bones:
            b.parent = edit_bones[parent_name]
            b.use_connect = use_connect
        edit_bones[name] = b
        return b

    # Central spine chain
    add_bone("Hips", None, Vector((0, y_mid, z_hips)), Vector((0, y_mid, z_spine)))
    add_bone("Spine", "Hips", Vector((0, y_mid, z_spine)), Vector((0, y_mid, z_spine1)), use_connect=True)
    add_bone("Spine1", "Spine", Vector((0, y_mid, z_spine1)), Vector((0, y_mid, z_spine2)), use_connect=True)
    add_bone("Spine2", "Spine1", Vector((0, y_mid, z_spine2)), Vector((0, y_mid, z_neck_base)), use_connect=True)
    add_bone("Neck", "Spine2", Vector((0, y_mid, z_neck_base)), Vector((0, y_mid, z_head_base)), use_connect=True)
    add_bone("Head", "Neck", Vector((0, y_mid, z_head_base)), Vector((0, y_mid, z_head_top)), use_connect=True)

    # Left Leg
    add_bone("LeftUpLeg", "Hips", Vector((leg_x, y_mid, z_hips)), Vector((leg_x, y_mid, z_min + 0.27 * H)))
    add_bone("LeftLeg", "LeftUpLeg", Vector((leg_x, y_mid, z_min + 0.27 * H)), Vector((leg_x, y_mid, z_min + 0.08 * H)), use_connect=True)
    add_bone("LeftFoot", "LeftLeg", Vector((leg_x, y_mid, z_min + 0.08 * H)), Vector((leg_x, y_mid - 0.08 * H, z_min + 0.03 * H)), use_connect=True)
    add_bone("LeftToeBase", "LeftFoot", Vector((leg_x, y_mid - 0.08 * H, z_min + 0.03 * H)), Vector((leg_x, y_mid - 0.14 * H, z_min + 0.02 * H)), use_connect=True)

    # Right Leg
    add_bone("RightUpLeg", "Hips", Vector((-leg_x, y_mid, z_hips)), Vector((-leg_x, y_mid, z_min + 0.27 * H)))
    add_bone("RightLeg", "RightUpLeg", Vector((-leg_x, y_mid, z_min + 0.27 * H)), Vector((-leg_x, y_mid, z_min + 0.08 * H)), use_connect=True)
    add_bone("RightFoot", "RightLeg", Vector((-leg_x, y_mid, z_min + 0.08 * H)), Vector((-leg_x, y_mid - 0.08 * H, z_min + 0.03 * H)), use_connect=True)
    add_bone("RightToeBase", "RightFoot", Vector((-leg_x, y_mid - 0.08 * H, z_min + 0.03 * H)), Vector((-leg_x, y_mid - 0.14 * H, z_min + 0.02 * H)), use_connect=True)

    # Left Arm
    add_bone("LeftShoulder", "Spine2", Vector((sh_x, y_mid, z_sh)), Vector((arm_x, y_mid, z_sh)))
    add_bone("LeftArm", "LeftShoulder", Vector((arm_x, y_mid, z_sh)), Vector((elbow_x, y_mid, z_elbow)), use_connect=True)
    add_bone("LeftForeArm", "LeftArm", Vector((elbow_x, y_mid, z_elbow)), Vector((hand_x, y_mid, z_wrist)), use_connect=True)
    add_bone("LeftHand", "LeftForeArm", Vector((hand_x, y_mid, z_wrist)), Vector((hand_x, y_mid, z_hand)), use_connect=True)

    # Right Arm
    add_bone("RightShoulder", "Spine2", Vector((-sh_x, y_mid, z_sh)), Vector((-arm_x, y_mid, z_sh)))
    add_bone("RightArm", "RightShoulder", Vector((-arm_x, y_mid, z_sh)), Vector((-elbow_x, y_mid, z_elbow)), use_connect=True)
    add_bone("RightForeArm", "RightArm", Vector((-elbow_x, y_mid, z_elbow)), Vector((-hand_x, y_mid, z_wrist)), use_connect=True)
    add_bone("RightHand", "RightForeArm", Vector((-hand_x, y_mid, z_wrist)), Vector((-hand_x, y_mid, z_hand)), use_connect=True)

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def bind_and_verify_weights(mesh_obj, arm_obj):
    # Ensure all bones have vertex groups
    for name in BONE_NAMES:
        if name not in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.new(name=name)

    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    bound = False
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        bound = True
    except Exception:
        pass

    if not bound:
        try:
            bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
            bound = True
        except Exception as e:
            raise RuntimeError(f"Armature parenting failed: {e}")

    # Inspect weights and assign unweighted vertices to closest bone
    segments = {}
    for name in BONE_NAMES:
        b = arm_obj.data.bones.get(name)
        if b:
            segments[name] = (Vector(b.head_local), Vector(b.tail_local))

    vg_indices = {name: mesh_obj.vertex_groups[name].index for name in BONE_NAMES if name in mesh_obj.vertex_groups}

    total_verts = len(mesh_obj.data.vertices)
    unweighted_count = 0

    for v in mesh_obj.data.vertices:
        tot_w = sum(g.weight for g in v.groups if g.group in vg_indices.values())
        if tot_w < 1e-4:
            unweighted_count += 1
            # Assign to closest bone
            p = v.co
            min_dist = float("inf")
            best_bone = "Hips"
            for name, (head, tail) in segments.items():
                ab = tail - head
                ab_len_sq = ab.length_squared
                if ab_len_sq < 1e-8:
                    dist = (p - head).length
                else:
                    t = max(0.0, min(1.0, (p - head).dot(ab) / ab_len_sq))
                    proj = head + t * ab
                    dist = (p - proj).length
                if dist < min_dist:
                    min_dist = dist
                    best_bone = name

            vg = mesh_obj.vertex_groups[best_bone]
            vg.add([v.index], 1.0, "REPLACE")

    # Re-verify final unweighted count
    final_unweighted = 0
    for v in mesh_obj.data.vertices:
        tot_w = sum(g.weight for g in v.groups if g.group in vg_indices.values())
        if tot_w < 1e-4:
            final_unweighted += 1

    ratio = final_unweighted / total_verts if total_verts > 0 else 0.0
    if ratio >= 0.01:
        raise RuntimeError(f"Unweighted vertex ratio {ratio:.4f} exceeds 0.01")

    return ratio


def pose_deformation_check(mesh_obj, arm_obj):
    bpy.ops.object.select_all(action="DESELECT")
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")

    pbone = arm_obj.pose.bones.get("LeftArm")
    if not pbone:
        bpy.ops.object.mode_set(mode="OBJECT")
        raise RuntimeError("LeftArm bone not found in pose bones")

    pbone.rotation_mode = "XYZ"
    pbone.rotation_euler = (0, math.radians(45), 0)
    bpy.context.view_layer.update()

    dg = bpy.context.evaluated_depsgraph_get()
    eval_mesh = mesh_obj.evaluated_get(dg)

    orig_coords = [v.co.copy() for v in mesh_obj.data.vertices]
    eval_coords = [v.co for v in eval_mesh.data.vertices]

    max_disp = 0.0
    for o_co, e_co in zip(orig_coords, eval_coords):
        d = (e_co - o_co).length
        if d > max_disp:
            max_disp = d

    # Reset pose rotation
    pbone.rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode="OBJECT")

    if max_disp <= 0.01:
        raise RuntimeError(f"Pose deformation check failed: max displacement {max_disp:.4f}m <= 0.01m")

    return max_disp


def main():
    args = parse()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        neck_ratio = args.neck_ratio
        if args.ref_json and os.path.isfile(args.ref_json):
            try:
                with open(args.ref_json) as f:
                    rdata = json.load(f)
                    if "neck_ratio" in rdata:
                        neck_ratio = float(rdata["neck_ratio"])
            except Exception:
                pass

        mesh_obj = load_mesh(args.mesh)
        arm_obj = build_armature(mesh_obj, neck_ratio)
        unweighted_ratio = bind_and_verify_weights(mesh_obj, arm_obj)
        max_disp = pose_deformation_check(mesh_obj, arm_obj)

        # Export rigged model
        bpy.ops.object.select_all(action="DESELECT")
        mesh_obj.select_set(True)
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj

        bpy.ops.export_scene.gltf(
            filepath=args.out,
            export_format="GLB",
            export_apply=False,
            export_skins=True,
        )

        if not os.path.isfile(args.out) or os.path.getsize(args.out) == 0:
            raise RuntimeError(f"Exported rigged GLB is missing or empty: {args.out}")

        res = {
            "ok": True,
            "bones": len(BONE_NAMES),
            "unweighted_ratio": round(unweighted_ratio, 6),
            "max_disp": round(max_disp, 4),
            "mesh_path": args.out,
        }
    except Exception as e:
        res = {
            "ok": False,
            "code": "RIG_FAILED",
            "message": f"{type(e).__name__}: {e}",
            "fix": "Retry with different mesh topology or check armature joint placement.",
        }

    with open(args.result, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
