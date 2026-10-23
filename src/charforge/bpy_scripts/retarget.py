"""Retarget motion clips (BVH, glTF/GLB, FBX) onto any humanoid rig by bone role. Runs INSIDE Blender:
bpy/mathutils/stdlib only (+ the sibling cf_humanoid module). merge_animations.py imports it for library clips.

Both skeletons are mapped to humanoid roles (cf_humanoid.map_roles; bone names as fallback, e.g. CMU/Mixamo BVH).
Then, frame by frame in world space:
  facing    G turns the source rest (left-right hip axis) onto the target's, so a clip facing +Y or +X still works
  rest      arms and legs: A_r turns the target rest segment onto the source rest segment, so A-pose and T-pose
            rests both work; hands keep the forearm's A_r; torso, head, shoulders and feet need none (both rests
            stand upright, look ahead, feet flat)
  rotation  target world rotation of role r = G.D_r.G^-1 . A_r . rest_r, where D_r is the source bone's world
            rotation relative to its rest; bones without a role keep their rest pose relative to their parent
  root      the pelvis centre (midpoint of the hip joints) follows the source, scaled by the leg-length ratio:
            vertical motion relative to the rest height, horizontal motion relative to the first frame
  ground    one vertical offset for the whole clip puts its lowest sole point on the rest ground (not the first
            frame's: runs and jumps may start in the air); a clip that never touches the ground keeps its clearance
  bake      one rotation key per mapped bone (+ a hips location key) per frame at the scene fps; optional trim
            [t0, t1] in seconds; loops get their last quarter blended into the first frame
"""
import math
import os

import bpy
from mathutils import Matrix, Vector

import cf_humanoid as hum

LIMBS = {  # role -> role whose joint ends its segment (rest alignment)
    "leftUpperArm": "leftLowerArm", "leftLowerArm": "leftHand", "rightUpperArm": "rightLowerArm", "rightLowerArm": "rightHand",
    "leftUpperLeg": "leftLowerLeg", "leftLowerLeg": "leftFoot", "rightUpperLeg": "rightLowerLeg", "rightLowerLeg": "rightFoot",
}
FOLLOW = {"leftHand": "leftLowerArm", "rightHand": "rightLowerArm"}  # same rest alignment as the forearm
LEGS = ("leftUpperLeg", "rightUpperLeg", "leftLowerLeg", "rightLowerLeg", "leftFoot", "rightFoot")
PROCEDURAL_FIX = "Use a procedural action (idle, walk, run, wave, jump) or another library clip."


class ClipError(RuntimeError):
    """A clip that cannot be retargeted; `fix` says what the driving model should do instead."""

    def __init__(self, message, fix=PROCEDURAL_FIX):
        super().__init__(message)
        self.fix = fix


def _rot(m):
    return m.to_3x3().normalized()


def _smooth(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def assign(arm, act):
    """Make `act` the evaluated action of `arm` (NLA muted, pose reset, Blender 4.4+ action slot)."""
    ad = arm.animation_data or arm.animation_data_create()
    for t in ad.nla_tracks:
        t.mute = True
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    ad.action = act
    if act is not None and hasattr(ad, "action_slot") and getattr(ad, "action_slot", None) is None and len(getattr(act, "slots", [])):
        ad.action_slot = act.slots[0]


# ---------------------------------------------------------------- import

def import_clip(path, action=None):
    """Import a clip file into the current scene. Returns (armature, action, new objects, new actions)."""
    if not os.path.isfile(path):
        raise ClipError(f"Motion clip not found: {path}", "Reinstall CharForge from its repository (assets/motions is missing), or use procedural actions.")
    objs, acts = set(bpy.context.scene.objects), set(bpy.data.actions)
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".bvh":  # standard BVH: Y up, character facing +Z; keys at the scene fps
            bpy.ops.import_anim.bvh(filepath=path, axis_forward="-Z", axis_up="Y", rotate_mode="QUATERNION", use_fps_scale=True,
                                    frame_start=1, update_scene_fps=False, update_scene_duration=False)
        elif ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=path)
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=path, use_anim=True)
        else:
            raise ClipError(f"Unsupported clip format {ext!r}: {path}", "Use a .bvh, .glb or .fbx clip.")
    except ClipError:
        raise
    except Exception as e:
        raise ClipError(f"Could not import clip {os.path.basename(path)}: {e}") from e
    new = [o for o in bpy.context.scene.objects if o not in objs]
    arms = [o for o in new if o.type == "ARMATURE"]
    if not arms:
        raise ClipError(f"Clip {os.path.basename(path)} has no skeleton.")
    arm = max(arms, key=lambda o: len(o.data.bones))
    new_acts = [a for a in bpy.data.actions if a not in acts]
    if action:
        act = (next((a for a in new_acts if a.name == action), None) or next((a for a in new_acts if a.name.lower() == action.lower()), None)
               or next((a for a in new_acts if a.name.lower().startswith(action.lower())), None))  # importers may suffix names
        if act is None:
            raise ClipError(f"Clip {os.path.basename(path)} has no animation {action!r} (has: {', '.join(sorted(a.name for a in new_acts))}).")
    else:
        current = arm.animation_data.action if arm.animation_data else None
        act = current if current in new_acts else max(new_acts, key=lambda a: a.frame_range[1] - a.frame_range[0], default=None)
    if act is None:
        raise ClipError(f"Clip {os.path.basename(path)} has no animation.")
    return arm, act, new, new_acts


def remove_clip(objects, actions):
    """Delete what import_clip brought in: its objects, its actions and the data they leave orphaned."""
    for o in objects:
        bpy.data.objects.remove(o, do_unlink=True)
    for a in actions:
        bpy.data.actions.remove(a)
    for coll in (bpy.data.armatures, bpy.data.meshes):
        for d in [d for d in coll if d.users == 0]:
            coll.remove(d)


# ---------------------------------------------------------------- roles

def source_roles(arm):
    """Roles of a clip skeleton: structure first; bone names decide the sides when the clip does not face -Y."""
    st = hum.map_roles(arm)
    named = hum.map_roles_by_name(arm)
    notes = []
    if st["ok"]:
        roles = dict(st["roles"])
        sided = [(r, b) for r, b in roles.items() if r.startswith(("left", "right")) and hum.name_side(b)]
        wrong = sum(1 for r, b in sided if not r.startswith(hum.name_side(b)))
        if sided and wrong * 2 > len(sided) and named["ok"]:
            roles = dict(named["roles"])
            notes.append("clip skeleton does not face -Y: sides taken from bone names")
        for r, b in named["roles"].items():
            if r not in roles and b not in roles.values():
                roles[r] = b
    elif named["ok"]:
        roles = dict(named["roles"])
        notes.append("clip roles taken from bone names")
    else:
        raise ClipError("Clip skeleton is not humanoid: " + "; ".join(st["notes"] or ["no hips, arms and legs found"]))
    missing = [r for r in ("hips",) + LEGS if r not in roles]
    if missing:
        raise ClipError(f"Clip skeleton has no {', '.join(missing)}.")
    return roles, notes


def target_roles(arm):
    st = hum.map_roles(arm)
    roles = dict(st["roles"])
    for r, b in hum.map_roles_by_name(arm)["roles"].items():
        if r not in roles and b not in roles.values():
            roles[r] = b
    missing = [r for r in ("hips",) + LEGS if r not in roles]
    if missing:
        raise ClipError(f"The rig has no humanoid {', '.join(missing)} bone(s): library clips cannot drive it.",
                        "Rerun auto_rig(method='blender') to get a humanoid skeleton, or use procedural actions.")
    return roles


# ---------------------------------------------------------------- rest measurements

def _joints(arm, roles):
    mw = arm.matrix_world
    return {r: mw @ arm.data.bones[b].head_local for r, b in roles.items()}


def _seg_end(arm, roles, J, role):
    child = LIMBS[role]
    if child in J:
        return J[child]
    return arm.matrix_world @ arm.data.bones[roles[role]].tail_local


def _leg_length(J):
    return 0.5 * sum((J[f"{s}UpperLeg"] - J[f"{s}Foot"]).length for s in ("left", "right"))


def _facing(Js, Jt):
    """Rotation about Z taking the source left-right hip axis onto the target's (both rests upright)."""
    top = next(Js[r] for r in ("head", "neck", "upperChest", "chest", "spine", "hips") if r in Js)
    up = top - Js["hips"]
    if up.length < 1e-6 or up.normalized().z < 0.5:
        raise ClipError("Clip skeleton is not upright in its rest pose (wrong up axis).", "Export the clip Y-up (BVH) or Z-up, standing, then retry.")
    ls = Js["leftUpperLeg"] - Js["rightUpperLeg"]
    lt = Jt["leftUpperLeg"] - Jt["rightUpperLeg"]
    a = math.atan2(lt.y, lt.x) - math.atan2(ls.y, ls.x)
    return Matrix.Rotation(a, 3, "Z")


def _order(arm):
    """Bone names, parents before children."""
    out, queue = [], [b for b in arm.data.bones if b.parent is None]
    while queue:
        b = queue.pop(0)
        out.append(b.name)
        queue.extend(b.children)
    return out


def sole_points(arm, roles, meshes):
    """(bone, point in bone rest space) pairs at the soles: the lowest mesh vertices near each foot, so the ground
    check follows the deformed feet without evaluating the mesh. Joints only when the rig has no mesh."""
    mw = arm.matrix_world
    J = _joints(arm, roles)
    pts = []
    rest = []
    if meshes:
        hum.mute_animation(arm)
        hum.rest_pose(arm)
        rest = hum.evaluated_coords(meshes)
    ground = min((v.z for v in rest), default=min(J[f"{s}Foot"].z for s in ("left", "right")))
    feet = {}
    for side in ("left", "right"):
        foot, toes = roles[f"{side}Foot"], roles.get(f"{side}Toes")
        a = J[f"{side}Foot"]
        b = J[f"{side}Toes"] if f"{side}Toes" in J else mw @ arm.data.bones[foot].tail_local
        tip = mw @ arm.data.bones[toes].tail_local if toes else b
        feet[side] = (foot, toes, a, b, tip)
    near = {"left": [], "right": []}
    z_max = ground + 0.5 * (max(f[2].z for f in feet.values()) - ground) + 0.005  # lower half of the feet
    for v in rest:
        if v.z > z_max:
            continue
        d = {s: min(_seg_dist(v, a, b), _seg_dist(v, b, tip)) for s, (_, _, a, b, tip) in feet.items()}
        s = min(d, key=d.get)
        foot, toes, a, b, tip = feet[s]
        if v.z <= a.z and d[s] < 0.6 * max((tip - a).length, 0.05):
            near[s].append((v, toes if toes and _seg_dist(v, b, tip) < _seg_dist(v, a, b) else foot))
    for side, (foot, toes, a, b, tip) in feet.items():
        cand = near[side] or [(a, foot), (b, toes or foot), (tip, toes or foot)]
        cand.sort(key=lambda t: t[0].z)
        pick = cand[:48] + [min(cand, key=lambda t: t[0].y), max(cand, key=lambda t: t[0].y)]
        for v, bone in pick:
            pts.append((bone, arm.data.bones[bone].matrix_local.inverted() @ (mw.inverted() @ v)))
    return pts


def _seg_dist(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-12)))
    return (p - (a + t * ab)).length


# ---------------------------------------------------------------- retarget

def retarget(src, act, tgt, meshes, name, fps=30.0, loop=False, trim=None):
    """Bake action `act` of clip armature `src` onto armature `tgt` as a new action `name`. Returns (action, info)."""
    roles_s, notes = source_roles(src)
    roles_t = target_roles(tgt)
    Js, Jt = _joints(src, roles_s), _joints(tgt, roles_t)
    G = _facing(Js, Jt)
    Ginv = G.transposed()
    Ls, Lt = _leg_length(Js), _leg_length(Jt)
    if Ls < 1e-4 or Lt < 1e-4:
        raise ClipError("Clip or rig legs have zero length.")
    k = Lt / Ls

    mws, mwt = src.matrix_world, tgt.matrix_world
    rest_s = {r: _rot(mws @ src.data.bones[b].matrix_local) for r, b in roles_s.items()}
    rest_t = {r: _rot(mwt @ tgt.data.bones[b].matrix_local) for r, b in roles_t.items()}
    align = {}
    for r in LIMBS:
        if r in roles_s and r in roles_t:
            ds = G @ (_seg_end(src, roles_s, Js, r) - Js[r])
            dt = _seg_end(tgt, roles_t, Jt, r) - Jt[r]
            if ds.length > 1e-6 and dt.length > 1e-6:
                align[r] = dt.normalized().rotation_difference(ds.normalized()).to_matrix()
    for r, base in FOLLOW.items():
        if base in align:
            align[r] = align[base]
    shared = [r for r in hum.ROLES if r in roles_s and r in roles_t]
    notes += [f"clip has no {r}: rig keeps its rest pose there" for r in hum.ROLES if r in roles_t and r not in roles_s and r not in ("upperChest", "leftToes", "rightToes")]

    # ---- sample the clip (rig not animated meanwhile, so only the small clip skeleton is evaluated)
    sc = bpy.context.scene
    if tgt.animation_data:
        tgt.animation_data.action = None
    hum.rest_pose(tgt)
    assign(src, act)
    f_start, f_end = act.frame_range
    span = max(0.0, (f_end - f_start) / fps)
    t0, t1 = (trim or (0.0, span))
    t0, t1 = max(0.0, min(t0, span)), max(0.0, min(t1, span))
    if t1 <= t0:
        t1 = span
    n = max(1, int(round((t1 - t0) * fps)))
    rest_s_inv = {r: m.inverted() for r, m in rest_s.items()}
    feet_s = [roles_s[r] for r in ("leftFoot", "rightFoot", "leftToes", "rightToes") if r in roles_s]
    floor_s = min((mws @ src.data.bones[b].head_local).z for b in feet_s)
    frames, heights = [], []  # heights: lowest clip foot joint above its rest height
    for i in range(n + 1):
        f = f_start + t0 * fps + i
        sc.frame_set(int(math.floor(f)), subframe=f - math.floor(f))
        D = {r: _rot(mws @ src.pose.bones[roles_s[r]].matrix) @ rest_s_inv[r] for r in shared}
        c = 0.5 * ((mws @ src.pose.bones[roles_s["leftUpperLeg"]].head) + (mws @ src.pose.bones[roles_s["rightUpperLeg"]].head))
        frames.append((D, c))
        heights.append(min((mws @ src.pose.bones[b].head).z for b in feet_s) - floor_s)
    assign(src, None)

    # ---- target world rotations and pelvis path
    cs_rest = 0.5 * (Js["leftUpperLeg"] + Js["rightUpperLeg"])
    ct_rest = 0.5 * (Jt["leftUpperLeg"] + Jt["rightUpperLeg"])
    c0 = frames[0][1]
    ref = Vector((c0.x, c0.y, cs_rest.z))  # horizontal from the first frame, vertical from the rest height
    travel = (Vector((frames[-1][1].x, frames[-1][1].y, 0)) - Vector((c0.x, c0.y, 0))).length > 0.25 * Ls
    W, C = [], []
    for D, c in frames:
        W.append({r: G @ D[r] @ Ginv @ align.get(r, Matrix.Identity(3)) @ rest_t[r] for r in shared})
        C.append(ct_rest + k * (G @ (c - ref)))

    # ---- forward kinematics of the rig, keys per bone
    order = _order(tgt)
    bones = tgt.data.bones
    role_of = {roles_t[r]: r for r in shared}
    hips = roles_t["hips"]
    mover = hips
    while bones[mover].use_connect and bones[mover].parent is not None:
        mover = bones[mover].parent.name
    rest_l = {b: bones[b].matrix_local for b in order}
    rest_l_inv = {b: m.inverted() for b, m in rest_l.items()}
    Rmw_inv = _rot(mwt).inverted()
    mwt_inv = mwt.inverted()
    hips_off = (mwt @ rest_l[hips].translation) - ct_rest  # rest hips head relative to the pelvis centre (world)

    def solve(Wf, cf):
        pose, basis = {}, {}
        dh = Wf["hips"] @ rest_t["hips"].inverted() if "hips" in Wf else Matrix.Identity(3)
        head_want = mwt_inv @ (cf + dh @ hips_off)
        for b in order:
            par = bones[b].parent
            X = pose[par.name] @ rest_l_inv[par.name] @ rest_l[b] if par else rest_l[b]
            r = role_of.get(b)
            if r is not None:
                Rarm = (Rmw_inv @ Wf[r]).to_4x4()
                head = head_want if b == hips and mover == hips else X.translation
                P = Matrix.Translation(head) @ Rarm
                B = X.inverted() @ P
                if b != mover:
                    B = B.to_quaternion().to_matrix().to_4x4()
                    P = X @ B
                basis[b] = B
                pose[b] = P
            else:
                pose[b] = X
        if mover != hips:  # hips connected to a parent chain: move the first free ancestor instead
            delta = head_want - pose[hips].translation
            X = pose[mover]
            loc = X.to_3x3().inverted() @ delta
            basis[mover] = Matrix.Translation(loc)
            return solve_with(basis)
        return pose, basis

    def solve_with(fixed):
        pose = {}
        for b in order:
            par = bones[b].parent
            X = pose[par.name] @ rest_l_inv[par.name] @ rest_l[b] if par else rest_l[b]
            pose[b] = X @ fixed[b] if b in fixed else X
        return pose, fixed

    soles = sole_points(tgt, roles_t, meshes)
    rest_low = min((mwt @ (rest_l[b] @ p)).z for b, p in soles)

    def lowest(pose):
        return min((mwt @ (pose[b] @ p)).z for b, p in soles)

    solved = [solve(W[i], C[i]) for i in range(len(frames))]
    offset = rest_low + max(0.0, k * min(heights)) - min(lowest(pose) for pose, _ in solved)
    if abs(offset) > 1e-7:
        C = [c + Vector((0.0, 0.0, offset)) for c in C]
        solved = [solve(W[i], C[i]) for i in range(len(frames))]

    keyed = sorted(set(basis_name for _, basis in solved for basis_name in basis), key=order.index)
    quats = {b: [] for b in keyed}
    locs = []
    for pose, basis in solved:
        for b in keyed:
            q = basis[b].to_quaternion() if b in basis else Matrix.Identity(4).to_quaternion()
            if quats[b] and quats[b][-1].dot(q) < 0:
                q.negate()
            quats[b].append(q)
        locs.append(basis[mover].translation.copy() if mover in basis else Vector())
    if loop and len(frames) >= 8:
        m = max(2, (len(frames) - 1) // 4)
        last = len(frames) - 1
        for i in range(last - m, last + 1):
            w = _smooth((i - (last - m)) / m)
            for b in keyed:
                q0 = quats[b][0] if quats[b][0].dot(quats[b][i]) >= 0 else -quats[b][0]
                quats[b][i] = quats[b][i].slerp(q0, w)
            if not travel:  # a clip that moves forward keeps its path (bone-space location is not split by axis)
                locs[i] = locs[i].lerp(locs[0], w)

    # ---- bake
    new = bpy.data.actions.new(name=name)
    assign(tgt, new)
    pbs = tgt.pose.bones
    for b in keyed:
        pbs[b].rotation_mode = "QUATERNION"
    for i in range(len(frames)):
        for b in keyed:
            pbs[b].rotation_quaternion = quats[b][i]
            pbs[b].keyframe_insert("rotation_quaternion", frame=i, group=b)
        pbs[mover].location = locs[i]
        pbs[mover].keyframe_insert("location", frame=i, group=mover)
    return new, {
        "frames": len(frames),
        "seconds": round((len(frames) - 1) / fps, 3),
        "leg_scale": round(k, 3),
        "ground_offset_cm": round(100 * offset, 2),
        "loop": bool(loop),
        "source_roles_missing": [r for r in hum.ROLES if r not in roles_s],
        "notes": notes,
    }


def retarget_file(path, tgt, meshes, name, fps=30.0, loop=False, trim=None, action=None):
    """Import a clip file, bake it onto `tgt` as action `name`, delete the clip. Returns (action, info)."""
    src, act, objs, acts = import_clip(path, action)
    try:
        new, info = retarget(src, act, tgt, meshes, name, fps=fps, loop=loop, trim=trim)
    finally:
        remove_clip(objs, acts)
    new.name = name  # the clip's own action may have held the name (BVH actions are named after the file)
    info["file"] = os.path.basename(path)
    return new, info
