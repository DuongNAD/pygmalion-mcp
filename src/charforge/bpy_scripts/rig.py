"""Mixamo-compatible humanoid auto-rigging script. Runs INSIDE Blender: bpy/mathutils/stdlib only.

Operations:
1. Load mesh, center transforms, pre-bind seam welding (remove_doubles + dissolve_degenerate).
2. Build 22-bone humanoid armature with geometry-driven medial-axis arm landmark detection.
3. Bind mesh with ARMATURE_AUTO (bone heat capture via os.dup2), measure pre-fallback unweighted ratio.
4. Two-tier fallback weighting: Tier 1 Geodesic BFS diffusion + Tier 2 Anatomical corridor IDW.
5. Robust deformation & weight leakage self-check: swing LeftArm 40 deg around local Z.
6. Export rigged model with armature to GLB.
Result JSON: {ok: true, bones: int, unweighted_ratio: float, pre_fallback_ratio: float, mean_arm_disp: float, max_thigh_disp: float, mesh_path: str}
"""
import argparse
import json
import math
import os
import sys
import tempfile

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


class WeightRatio(float):
    pre_fallback_ratio: float = 0.0


class DeformResult(tuple):
    def __new__(cls, mean_arm_disp, max_thigh_disp):
        return super().__new__(cls, (mean_arm_disp, max_thigh_disp))

    @property
    def mean_arm_disp(self):
        return self[0]

    @property
    def max_thigh_disp(self):
        return self[1]


def parse():
    raw_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(raw_args) >= 3 and not raw_args[0].startswith("--"):
        ns = argparse.Namespace()
        ns.mesh = raw_args[0]
        ns.out = raw_args[1]
        ns.result = raw_args[2]
        ns.neck_ratio = 0.15
        ns.ref_json = None
        return ns

    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neck-ratio", type=float, default=0.15)
    ap.add_argument("--ref-json", default=None)
    return ap.parse_args(raw_args)


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

    # Pre-bind cleanup: weld split UV/normal seams and dissolve zero-area degenerate elements
    if len(mesh_obj.data.vertices) > 0:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.0005)
        bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)
        bpy.ops.object.mode_set(mode="OBJECT")

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
    leg_y = float(sum(v.y for v in leg_verts) / len(leg_verts)) if leg_verts else y_mid

    # Geometry-driven arm landmark detection across T-pose, A-pose, and arms-down
    z_sh_est = z_min + 0.75 * H
    spine_ref = Vector((0, leg_y, z_sh_est))

    def detect_arm_landmarks(is_right):
        sign = -1.0 if is_right else 1.0
        sh_nom_x = sign * 0.10465 * H
        sh_est = Vector((sh_nom_x, leg_y, z_sh_est))

        # Distal arm candidates: strictly on the lateral side, excluding leg corridor
        arm_cands = [
            v.co for v in mesh_obj.data.vertices
            if (v.co.x * sign) > 0.06 * H
            and (v.co.z > z_min + 0.52 * H or (v.co.x * sign) > leg_x + 0.04 * H)
            and v.co.z > z_min + 0.30 * H
        ]
        if arm_cands:
            fingertip = max(arm_cands, key=lambda v: (v - spine_ref).length)
        else:
            fingertip = sh_est + Vector((sign * 0.35 * H, 0, -0.30 * H))

        # Determine shoulder X landmark: nominal 0.10465 * H, bounded inside actual shoulder span on slim characters
        sh_slice_xs = [
            v.co.x * sign for v in mesh_obj.data.vertices
            if abs(v.co.z - z_sh_est) < 0.03 * H and (v.co.x * sign) > 0
        ]
        sh_x = 0.10465 * H
        if sh_slice_xs:
            max_sh_x = max(sh_slice_xs)
            if max_sh_x < 0.125 * H:
                sh_x = min(sh_x, max(0.075 * H, max_sh_x * 0.72))

        sh = Vector((sign * sh_x, leg_y, z_sh_est))
        arm_vec = fingertip - sh
        L_arm = max(0.10 * H, arm_vec.length)
        u_arm = arm_vec.normalized()

        def slice_centroid(t_frac, window=0.08):
            target_proj = t_frac * L_arm
            pts = [
                v for v in arm_cands
                if abs((v - sh).dot(u_arm) - target_proj) < window * L_arm
                and ((v - sh) - (v - sh).dot(u_arm) * u_arm).length < 0.12 * H
            ]
            if pts:
                return Vector((
                    sum(p.x for p in pts) / len(pts),
                    sum(p.y for p in pts) / len(pts),
                    sum(p.z for p in pts) / len(pts),
                ))
            return sh + u_arm * target_proj

        # Anatomical humanoid ratios: upper arm 0.406 of arm length, forearm to wrist 0.768
        elb = slice_centroid(0.406)
        wri = slice_centroid(0.768)
        hnd = slice_centroid(0.95)

        # Ensure strictly non-zero bone lengths
        if (elb - sh).length < 0.02 * H:
            elb = sh + u_arm * (0.406 * L_arm)
        if (wri - elb).length < 0.02 * H:
            wri = elb + u_arm * (0.362 * L_arm)
        if (hnd - wri).length < 0.02 * H:
            hnd = wri + u_arm * (0.182 * L_arm)

        return sh, elb, wri, hnd

    l_sh, l_elb, l_wri, l_hnd = detect_arm_landmarks(False)
    r_sh, r_elb, r_wri, r_hnd = detect_arm_landmarks(True)

    sh_root_x = min(0.06 * H, abs(l_sh.x) * 0.4)
    sh_root_z = l_sh.z
    r_sh_root_x = -min(0.06 * H, abs(r_sh.x) * 0.4)
    r_sh_root_z = r_sh.z

    edit_bones = {}

    def add_bone(name, parent_name, head, tail, use_connect=False):
        b = arm_data.edit_bones.new(name)
        b.head = head
        b.tail = tail
        b.head_radius = 0.04 * H
        b.tail_radius = 0.04 * H
        b.envelope_distance = 0.06 * H
        if parent_name and parent_name in edit_bones:
            b.parent = edit_bones[parent_name]
            b.use_connect = use_connect
        edit_bones[name] = b
        return b

    # Central spine chain
    add_bone("Hips", None, Vector((0, leg_y, z_hips)), Vector((0, leg_y, z_spine)))
    add_bone("Spine", "Hips", Vector((0, leg_y, z_spine)), Vector((0, leg_y, z_spine1)), use_connect=True)
    add_bone("Spine1", "Spine", Vector((0, leg_y, z_spine1)), Vector((0, leg_y, z_spine2)), use_connect=True)
    add_bone("Spine2", "Spine1", Vector((0, leg_y, z_spine2)), Vector((0, leg_y, z_neck_base)), use_connect=True)
    add_bone("Neck", "Spine2", Vector((0, leg_y, z_neck_base)), Vector((0, leg_y, z_head_base)), use_connect=True)
    add_bone("Head", "Neck", Vector((0, leg_y, z_head_base)), Vector((0, leg_y, z_head_top)), use_connect=True)

    # Left Leg
    add_bone("LeftUpLeg", "Hips", Vector((leg_x, leg_y, z_hips)), Vector((leg_x, leg_y, z_min + 0.27 * H)))
    add_bone("LeftLeg", "LeftUpLeg", Vector((leg_x, leg_y, z_min + 0.27 * H)), Vector((leg_x, leg_y, z_min + 0.08 * H)), use_connect=True)
    add_bone("LeftFoot", "LeftLeg", Vector((leg_x, leg_y, z_min + 0.08 * H)), Vector((leg_x, leg_y - 0.08 * H, z_min + 0.03 * H)), use_connect=True)
    add_bone("LeftToeBase", "LeftFoot", Vector((leg_x, leg_y - 0.08 * H, z_min + 0.03 * H)), Vector((leg_x, leg_y - 0.14 * H, z_min + 0.02 * H)), use_connect=True)

    # Right Leg
    add_bone("RightUpLeg", "Hips", Vector((-leg_x, leg_y, z_hips)), Vector((-leg_x, leg_y, z_min + 0.27 * H)))
    add_bone("RightLeg", "RightUpLeg", Vector((-leg_x, leg_y, z_min + 0.27 * H)), Vector((-leg_x, leg_y, z_min + 0.08 * H)), use_connect=True)
    add_bone("RightFoot", "RightLeg", Vector((-leg_x, leg_y, z_min + 0.08 * H)), Vector((-leg_x, leg_y - 0.08 * H, z_min + 0.03 * H)), use_connect=True)
    add_bone("RightToeBase", "RightFoot", Vector((-leg_x, leg_y - 0.08 * H, z_min + 0.03 * H)), Vector((-leg_x, leg_y - 0.14 * H, z_min + 0.02 * H)), use_connect=True)

    # Left Arm
    add_bone("LeftShoulder", "Spine2", Vector((sh_root_x, leg_y, sh_root_z)), l_sh)
    add_bone("LeftArm", "LeftShoulder", l_sh, l_elb, use_connect=True)
    add_bone("LeftForeArm", "LeftArm", l_elb, l_wri, use_connect=True)
    add_bone("LeftHand", "LeftForeArm", l_wri, l_hnd, use_connect=True)

    # Right Arm (independent landmark detection in negative X space)
    add_bone("RightShoulder", "Spine2", Vector((r_sh_root_x, leg_y, r_sh_root_z)), r_sh)
    add_bone("RightArm", "RightShoulder", r_sh, r_elb, use_connect=True)
    add_bone("RightForeArm", "RightArm", r_elb, r_wri, use_connect=True)
    add_bone("RightHand", "RightForeArm", r_wri, r_hnd, use_connect=True)

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def _assign_fallback_weights(mesh_obj, arm_obj, vg_indices, H, z_min, neck_ratio):
    segments = {}
    for name in BONE_NAMES:
        b = arm_obj.data.bones.get(name)
        if b:
            segments[name] = (Vector(b.head_local), Vector(b.tail_local))

    z_hips = z_min + 0.52 * H
    z_neck_base = z_min + (1.0 - neck_ratio) * H

    weighted_verts = set()
    unweighted_verts = set()
    vertex_weights = {}

    for v in mesh_obj.data.vertices:
        vw = {}
        for g in v.groups:
            for bname, idx in vg_indices.items():
                if g.group == idx and g.weight > 1e-4:
                    vw[bname] = g.weight
        tot_w = sum(vw.values())
        if tot_w >= 1e-4:
            vertex_weights[v.index] = {bn: w / tot_w for bn, w in vw.items()}
            weighted_verts.add(v.index)
        else:
            unweighted_verts.add(v.index)

    if not unweighted_verts:
        return

    # TIER 1: Geodesic BFS Edge Neighbor Diffusion for connected surfaces
    adj = {i: [] for i in range(len(mesh_obj.data.vertices))}
    for edge in mesh_obj.data.edges:
        v0, v1 = edge.vertices[0], edge.vertices[1]
        adj[v0].append(v1)
        adj[v1].append(v0)

    resolved_in_tier1 = set()
    frontier = set(unweighted_verts)

    max_passes = max(50, int(len(unweighted_verts) ** 0.5) * 5)
    for _ in range(max_passes):
        newly_weighted = {}
        for v_idx in frontier:
            nbr_weights = [vertex_weights[nbr] for nbr in adj[v_idx] if nbr in vertex_weights]
            if nbr_weights:
                avg_w = {}
                for nw in nbr_weights:
                    for bn, val in nw.items():
                        avg_w[bn] = avg_w.get(bn, 0.0) + val
                tot = sum(avg_w.values())
                newly_weighted[v_idx] = {bn: val / tot for bn, val in avg_w.items()}

        if not newly_weighted:
            break

        for v_idx, w_dict in newly_weighted.items():
            vertex_weights[v_idx] = w_dict
            frontier.remove(v_idx)
            resolved_in_tier1.add(v_idx)

    for v_idx in resolved_in_tier1:
        for bname, weight in vertex_weights[v_idx].items():
            if weight > 1e-4:
                vg = mesh_obj.vertex_groups[bname]
                vg.add([v_idx], weight, "REPLACE")

    unweighted_verts = frontier

    # TIER 2: Anatomical Corridor-Constrained IDW for disconnected islands
    if unweighted_verts:
        arm_bones = {
            "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
            "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        }
        leg_bones = {
            "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
            "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Hips",
        }
        limb_extremities = {
            "LeftForeArm", "LeftHand", "RightForeArm", "RightHand",
            "LeftLeg", "LeftFoot", "LeftToeBase", "RightLeg", "RightFoot", "RightToeBase",
        }

        torso_bones = {"Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"}

        for v_idx in unweighted_verts:
            p = mesh_obj.data.vertices[v_idx].co

            if p.z <= z_hips:
                if p.x >= 0.0:
                    allowed_bones = [b for b in BONE_NAMES if not b.startswith("Right")]
                else:
                    allowed_bones = [b for b in BONE_NAMES if not b.startswith("Left")]
            elif p.x > 0.005 * H:
                allowed_bones = [b for b in BONE_NAMES if not b.startswith("Right")]
            elif p.x < -0.005 * H:
                allowed_bones = [b for b in BONE_NAMES if not b.startswith("Left")]
            else:
                allowed_bones = list(BONE_NAMES)

            # Lower torso & leg corridor: strictly exclude arm bones
            if p.z <= z_hips + 0.02 * H and abs(p.x) <= 0.18 * H:
                allowed_bones = [b for b in allowed_bones if b not in arm_bones]
                if p.z <= z_min + 0.495 * H:
                    allowed_bones = [b for b in allowed_bones if b not in torso_bones]
                if not allowed_bones:
                    side = "Left" if p.x >= 0.0 else "Right"
                    allowed_bones = [f"{side}UpLeg", f"{side}Leg", f"{side}Foot"]
            # Arm corridor: strictly exclude leg bones & hips
            elif abs(p.x) > 0.18 * H or (p.z > z_hips and (p.x > 0.12 * H or p.x < -0.12 * H)):
                allowed_bones = [b for b in allowed_bones if b not in leg_bones]
                if not allowed_bones:
                    allowed_bones = ["LeftArm", "LeftForeArm", "LeftHand"] if p.x > 0 else ["RightArm", "RightForeArm", "RightHand"]
            # Head corridor: strictly head/neck/spine2
            elif p.z >= z_neck_base - 0.02 * H:
                allowed_bones = [b for b in allowed_bones if b in {"Head", "Neck", "Spine2"}]
                if not allowed_bones:
                    allowed_bones = ["Head", "Neck"]
            # Mid torso corridor: exclude extremities
            else:
                allowed_bones = [b for b in allowed_bones if b not in limb_extremities]
                if not allowed_bones:
                    allowed_bones = ["Hips", "Spine", "Spine1", "Spine2"]

            bone_dists = []
            for bname in allowed_bones:
                if bname not in segments:
                    continue
                head, tail = segments[bname]
                ab = tail - head
                ab_len_sq = ab.length_squared
                if ab_len_sq < 1e-8:
                    dist = (p - head).length
                else:
                    t = max(0.0, min(1.0, (p - head).dot(ab) / ab_len_sq))
                    proj = head + t * ab
                    dist = (p - proj).length
                bone_dists.append((dist, bname))

            bone_dists.sort(key=lambda x: x[0])
            if bone_dists:
                if len(bone_dists) == 1 or bone_dists[0][0] < 1e-4:
                    vg = mesh_obj.vertex_groups[bone_dists[0][1]]
                    vg.add([v_idx], 1.0, "REPLACE")
                else:
                    top = bone_dists[:2]
                    w0 = 1.0 / (top[0][0] ** 2 + 1e-6)
                    w1 = 1.0 / (top[1][0] ** 2 + 1e-6)
                    tot = w0 + w1
                    mesh_obj.vertex_groups[top[0][1]].add([v_idx], w0 / tot, "REPLACE")
                    mesh_obj.vertex_groups[top[1][1]].add([v_idx], w1 / tot, "REPLACE")


def bind_and_verify_weights(mesh_obj, arm_obj, neck_ratio=0.15):
    for name in BONE_NAMES:
        if name not in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.new(name=name)

    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    # Capture C-level stdout and stderr during parent_set to catch "Bone Heat Weighting: failed"
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    captured_output = ""
    with tempfile.TemporaryFile(mode="w+t") as tmp:
        os.dup2(tmp.fileno(), stdout_fd)
        os.dup2(tmp.fileno(), stderr_fd)
        try:
            bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        except Exception:
            pass
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)
        tmp.seek(0)
        captured_output = tmp.read()

    heat_failed = "Bone Heat Weighting: failed" in captured_output

    # Ensure Armature modifier is present on mesh_obj
    arm_mod = next((m for m in mesh_obj.modifiers if m.type == "ARMATURE"), None)
    if not arm_mod:
        arm_mod = mesh_obj.modifiers.new("Armature", "ARMATURE")
        arm_mod.object = arm_obj

    # Measure unweighted ratio BEFORE any fallback assignment
    vg_indices = {name: mesh_obj.vertex_groups[name].index for name in BONE_NAMES if name in mesh_obj.vertex_groups}
    total_verts = len(mesh_obj.data.vertices)
    initial_unweighted = 0
    for v in mesh_obj.data.vertices:
        tot_w = sum(g.weight for g in v.groups if g.group in vg_indices.values())
        if tot_w < 1e-4:
            initial_unweighted += 1

    pre_fallback_ratio = initial_unweighted / total_verts if total_verts > 0 else 0.0

    zs = [v.co.z for v in mesh_obj.data.vertices]
    z_min, z_max = min(zs), max(zs)
    H = max(0.1, z_max - z_min)

    if initial_unweighted > 0 or heat_failed:
        _assign_fallback_weights(mesh_obj, arm_obj, vg_indices, H, z_min, neck_ratio)

    # Enforce anatomical limb isolation and bilateral symmetry
    all_arm_bones = [
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    ]
    all_arm_vgs = {mesh_obj.vertex_groups[n].index for n in all_arm_bones if n in mesh_obj.vertex_groups}
    torso_vgs = {
        mesh_obj.vertex_groups[n].index
        for n in ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head")
        if n in mesh_obj.vertex_groups
    }

    l_leg_bones = ["Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"]
    r_leg_bones = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]
    leg_vgs = {mesh_obj.vertex_groups[n].index for n in set(l_leg_bones + r_leg_bones) if n in mesh_obj.vertex_groups}

    b_l_leg = arm_obj.data.bones.get("LeftUpLeg")
    leg_x = abs(b_l_leg.head_local.x) if b_l_leg else 0.10 * H
    y_mid = (min(v.co.y for v in mesh_obj.data.vertices) + max(v.co.y for v in mesh_obj.data.vertices)) / 2.0
    leg_y = b_l_leg.head_local.y if b_l_leg else y_mid

    for v in mesh_obj.data.vertices:
        # Bilateral isolation: strict at x=0 in lower body, 0.005*H margin in upper body
        x_thresh = 0.0 if v.co.z <= z_min + 0.545 * H else 0.005 * H
        if v.co.x < -x_thresh:
            for g in list(v.groups):
                if mesh_obj.vertex_groups[g.group].name.startswith("Left"):
                    mesh_obj.vertex_groups[g.group].remove([v.index])
        elif v.co.x >= x_thresh if x_thresh == 0.0 else v.co.x > x_thresh:
            for g in list(v.groups):
                if mesh_obj.vertex_groups[g.group].name.startswith("Right"):
                    mesh_obj.vertex_groups[g.group].remove([v.index])

        # Anatomical limb corridor isolation in lower body (z <= z_min + 0.545 * H)
        if v.co.z <= z_min + 0.545 * H:
            has_leg_weight = (
                abs(v.co.x) <= max(leg_x + 0.065 * H, 0.16 * H) + 0.005
                and any(g.group in leg_vgs and g.weight > 1e-4 for g in v.groups)
            )
            d_l_leg_xy = math.sqrt((v.co.x - leg_x)**2 + (v.co.y - leg_y)**2)
            d_r_leg_xy = math.sqrt((v.co.x + leg_x)**2 + (v.co.y - leg_y)**2)
            is_in_leg = (d_l_leg_xy <= 0.065 * H + 0.005 or d_r_leg_xy <= 0.065 * H + 0.005) and v.co.z <= z_min + 0.545 * H
            is_in_torso = abs(v.co.x) <= leg_x + 0.02 * H and abs(v.co.y - leg_y) <= 0.16 * H

            if has_leg_weight or is_in_leg or is_in_torso:
                # STRICTLY FORBID ALL ARM BONES in the torso/hip/thigh corridor
                for g in list(v.groups):
                    if g.group in all_arm_vgs:
                        mesh_obj.vertex_groups[g.group].remove([v.index])
                # Below hip pivot (z <= z_min + 0.515 * H), prevent Hips/Spine weight from freezing upper thighs
                if v.co.z <= z_min + 0.515 * H:
                    for g in list(v.groups):
                        if g.group in torso_vgs:
                            mesh_obj.vertex_groups[g.group].remove([v.index])

                # Smooth hip-to-pelvis transition straddling the hip pivot z_hips=0.52*H ([0.515*H, 0.545*H])
                if v.co.z > z_min + 0.515 * H:
                    side = "Left" if v.co.x >= 0.0 else "Right"
                    t_hip = max(0.0, min(1.0, (v.co.z - (z_min + 0.515 * H)) / (0.03 * H)))
                    for g in list(v.groups):
                        mesh_obj.vertex_groups[g.group].remove([v.index])
                    if t_hip < 0.99:
                        mesh_obj.vertex_groups[f"{side}UpLeg"].add([v.index], 1.0 - t_hip, "REPLACE")
                    if t_hip > 0.01:
                        mesh_obj.vertex_groups["Hips"].add([v.index], t_hip, "REPLACE")
                # If vertex is now unweighted, assign to height-appropriate leg bone
                elif not v.groups or sum(g.weight for g in v.groups) < 1e-4:
                    side = "Left" if v.co.x >= 0.0 else "Right"
                    if v.co.z < z_min + 0.08 * H:
                        mesh_obj.vertex_groups[f"{side}Foot"].add([v.index], 1.0, "REPLACE")
                    elif v.co.z < z_min + 0.27 * H:
                        mesh_obj.vertex_groups[f"{side}Leg"].add([v.index], 1.0, "REPLACE")
                    else:
                        mesh_obj.vertex_groups[f"{side}UpLeg"].add([v.index], 1.0, "REPLACE")
            elif v.co.z > z_min + 0.30 * H and abs(v.co.x) > max(leg_x + 0.085 * H, 0.18 * H):
                # Lateral hand/forearm vertices below hip level: strip stray leg/torso weights
                for g in list(v.groups):
                    if g.group in leg_vgs or g.group in torso_vgs:
                        mesh_obj.vertex_groups[g.group].remove([v.index])
                if not v.groups or sum(g.weight for g in v.groups) < 1e-4:
                    side = "Left" if v.co.x >= 0.0 else "Right"
                    mesh_obj.vertex_groups[f"{side}Hand"].add([v.index], 1.0, "REPLACE")

        tot_w = sum(g.weight for g in v.groups)
        if tot_w > 1e-4:
            for g in v.groups:
                mesh_obj.vertex_groups[g.group].add([v.index], g.weight / tot_w, "REPLACE")

    # Re-verify final unweighted count
    final_unweighted = 0
    for v in mesh_obj.data.vertices:
        tot_w = sum(g.weight for g in v.groups if g.group in vg_indices.values())
        if tot_w < 1e-4:
            final_unweighted += 1

    final_ratio = final_unweighted / total_verts if total_verts > 0 else 0.0
    if final_ratio >= 0.01:
        raise RuntimeError(f"Unweighted vertex ratio {final_ratio:.4f} exceeds 0.01")

    res_ratio = WeightRatio(final_ratio)
    res_ratio.pre_fallback_ratio = pre_fallback_ratio
    return res_ratio


def pose_deformation_check(mesh_obj, arm_obj):
    zs = [v.co.z for v in mesh_obj.data.vertices]
    z_min, z_max = min(zs), max(zs)
    H = max(0.1, z_max - z_min)
    z_hips = z_min + 0.52 * H

    orig_coords = [v.co.copy() for v in mesh_obj.data.vertices]

    bpy.ops.object.select_all(action="DESELECT")
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")

    pbone = arm_obj.pose.bones.get("LeftArm")
    if not pbone:
        bpy.ops.object.mode_set(mode="OBJECT")
        raise RuntimeError("LeftArm bone not found in pose bones")

    # Arm bone landmarks from armature
    b_arm = arm_obj.data.bones.get("LeftArm")
    b_fore = arm_obj.data.bones.get("LeftForeArm")
    b_hand = arm_obj.data.bones.get("LeftHand")

    if b_arm and b_hand:
        l_sh = Vector(b_arm.head_local)
        l_hnd = Vector(b_hand.tail_local)
        u_arm = (l_hnd - l_sh).normalized()
        L_arm = max(0.05 * H, (l_hnd - l_sh).length)
    else:
        l_sh = Vector((0.15 * H, 0, z_min + 0.75 * H))
        u_arm = Vector((1, 0, 0))
        L_arm = 0.35 * H

    b_leg = arm_obj.data.bones.get("LeftUpLeg")
    leg_x = abs(b_leg.head_local.x) if b_leg else 0.08 * H

    pbone.rotation_mode = "XYZ"
    pbone.rotation_euler = (0, 0, math.radians(40))
    bpy.context.view_layer.update()

    dg = bpy.context.evaluated_depsgraph_get()
    eval_mesh = mesh_obj.evaluated_get(dg)
    eval_coords = [v.co.copy() for v in eval_mesh.data.vertices]

    # Reset pose rotation
    pbone.rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode="OBJECT")

    # Arm vertices to test for active displacement (LeftForeArm and LeftHand)
    arm_vgs = [mesh_obj.vertex_groups[n].index for n in ["LeftForeArm", "LeftHand"] if n in mesh_obj.vertex_groups]
    arm_v_indices = [v.index for v in mesh_obj.data.vertices if any(g.group in arm_vgs and g.weight > 0.1 for g in v.groups)]
    if not arm_v_indices:
        arm_v_indices = [
            i for i, v in enumerate(orig_coords)
            if (v - l_sh).dot(u_arm) > 0.35 * L_arm
            and ((v - l_sh) - (v - l_sh).dot(u_arm) * u_arm).length < 0.12 * H
        ]

    arm_disps = [(eval_coords[i] - orig_coords[i]).length for i in arm_v_indices]
    mean_arm_disp = sum(arm_disps) / len(arm_disps) if arm_disps else 0.0

    # Thigh and lower-torso vertices: lower-body vertices in the hip/leg envelope
    # that are geometrically closer to leg/hip bones than arm bones.
    arm_bone_names = [
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    ]
    leg_bone_names = [
        "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
        "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    ]

    def _seg_dist(p, b):
        h, t = Vector(b.head_local), Vector(b.tail_local)
        ab = t - h
        l2 = ab.length_squared
        if l2 < 1e-8:
            return (p - h).length
        t_val = max(0.0, min(1.0, (p - h).dot(ab) / l2))
        return (p - (h + t_val * ab)).length

    arm_bones = [arm_obj.data.bones[n] for n in arm_bone_names if n in arm_obj.data.bones]
    leg_bones = [arm_obj.data.bones[n] for n in leg_bone_names if n in arm_obj.data.bones]

    thigh_v_indices = [
        i for i, v in enumerate(orig_coords)
        if v.z <= z_hips
        and abs(v.x) <= leg_x + 0.06 * H
        and (not leg_bones or not arm_bones or min(_seg_dist(v, b) for b in leg_bones) <= min(_seg_dist(v, b) for b in arm_bones))
    ]
    thigh_disps = [(eval_coords[i] - orig_coords[i]).length for i in thigh_v_indices]
    max_thigh_disp = max(thigh_disps) if thigh_disps else 0.0

    # Direct assertion: zero arm bone cross-binding on any thigh corridor vertex
    all_arm_vgs = [mesh_obj.vertex_groups[n].index for n in [
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand"
    ] if n in mesh_obj.vertex_groups]
    for i in thigh_v_indices:
        v = mesh_obj.data.vertices[i]
        for g in v.groups:
            if g.group in all_arm_vgs and g.weight > 1e-4:
                bname = mesh_obj.vertex_groups[g.group].name
                raise RuntimeError(f"Weight leakage: thigh vertex {i} cross-bound to arm bone {bname} with weight {g.weight:.4f}")

    min_arm_disp = min(0.10, 0.06 * H) if arm_v_indices else 0.10
    if mean_arm_disp < min_arm_disp:
        raise RuntimeError(f"Pose deformation check failed: mean arm displacement {mean_arm_disp:.4f}m < {min_arm_disp:.4f}m")
    if thigh_v_indices and max_thigh_disp >= 0.005:
        raise RuntimeError(f"Weight leakage check failed: max thigh displacement {max_thigh_disp:.4f}m >= 0.005m")

    return DeformResult(mean_arm_disp, max_thigh_disp)


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
        unweighted_ratio = bind_and_verify_weights(mesh_obj, arm_obj, neck_ratio)
        mean_arm_disp, max_thigh_disp = pose_deformation_check(mesh_obj, arm_obj)

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

        pre_ratio = getattr(unweighted_ratio, "pre_fallback_ratio", float(unweighted_ratio))
        res = {
            "ok": True,
            "bones": len(BONE_NAMES),
            "unweighted_ratio": round(float(unweighted_ratio), 6),
            "pre_fallback_ratio": round(float(pre_ratio), 6),
            "mean_arm_disp": round(float(mean_arm_disp), 4),
            "max_thigh_disp": round(float(max_thigh_disp), 4),
            "max_disp": round(float(mean_arm_disp), 4),
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
