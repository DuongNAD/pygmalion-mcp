"""Humanoid skeleton analysis shared by the bpy_scripts: role mapping, deformation (weight-leak) check,
contact gaps, weight repair and the humanoid map. Runs INSIDE Blender: bpy/mathutils/stdlib only.

Import from a sibling script with:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import cf_humanoid as hum

Conventions: Z-up, meters, character faces -Y (Blender glTF import default), character left = +X.
Bone names are never trusted alone: roles come from the bone hierarchy and joint positions, names only refine.
"""
import math
import re
from array import array

import bpy
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

ROLES = (
    "hips", "spine", "chest", "upperChest", "neck", "head",
    "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
    "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes",
    "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes",
)
MIXAMO = {
    "hips": "Hips", "spine": "Spine", "chest": "Spine1", "upperChest": "Spine2", "neck": "Neck", "head": "Head",
    "leftShoulder": "LeftShoulder", "leftUpperArm": "LeftArm", "leftLowerArm": "LeftForeArm", "leftHand": "LeftHand",
    "rightShoulder": "RightShoulder", "rightUpperArm": "RightArm", "rightLowerArm": "RightForeArm", "rightHand": "RightHand",
    "leftUpperLeg": "LeftUpLeg", "leftLowerLeg": "LeftLeg", "leftFoot": "LeftFoot", "leftToes": "LeftToeBase",
    "rightUpperLeg": "RightUpLeg", "rightLowerLeg": "RightLeg", "rightFoot": "RightFoot", "rightToes": "RightToeBase",
}
GROUP_OF_ROLE = {
    "hips": "torso", "spine": "torso", "chest": "torso", "upperChest": "torso", "neck": "head", "head": "head",
    "leftShoulder": "shoulderL", "leftUpperArm": "armL", "leftLowerArm": "armL", "leftHand": "armL",
    "rightShoulder": "shoulderR", "rightUpperArm": "armR", "rightLowerArm": "armR", "rightHand": "armR",
    "leftUpperLeg": "legL", "leftLowerLeg": "legL", "leftFoot": "legL", "leftToes": "legL",
    "rightUpperLeg": "legR", "rightLowerLeg": "legR", "rightFoot": "legR", "rightToes": "legR",
}
AMBIG_RATIO = 1.3  # a vertex is ambiguous when another body group is within 1.3x its nearest-bone distance

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_NOISE = {"mixamorig", "def", "org", "mch", "j", "bip", "bip01", "c", "cc", "base", "rig", "jnt", "bn", "b", "jt", "sk"}


# ---------------------------------------------------------------- names

def tokens(name):
    """'mixamorig:LeftForeArm' -> ['left', 'fore', 'arm']; 'upperarm_l' -> ['upperarm', 'l']."""
    base = re.split(r"[:|]", name)[-1]
    out = []
    for part in re.split(r"[^A-Za-z0-9]+", base):
        out += [t.lower() for t in _CAMEL.findall(part)]
    return out


def name_side(name):
    t = set(tokens(name))
    left, right = bool(t & {"left", "l"}), bool(t & {"right", "r"})
    return "left" if left and not right else "right" if right and not left else None


def name_kind(name):
    """Coarse anatomical kind from a bone name, or None."""
    t = [x for x in tokens(name) if x not in _NOISE and x not in ("left", "right", "l", "r") and not x.isdigit()]
    s = "".join(t)
    if not s:
        return None
    if t and t[-1] in ("end", "nub", "tip") or s.endswith("end"):
        return "end"
    if any(k in s for k in ("thumb", "index", "middle", "ring", "pinky", "little", "finger")):
        return "finger"
    if any(k in s for k in ("forearm", "lowerarm", "elbow")):
        return "lowerArm"
    if any(k in s for k in ("upperarm", "uparm")) or s == "arm":
        return "upperArm"
    if any(k in s for k in ("hand", "wrist", "palm")):
        return "hand"
    if any(k in s for k in ("shoulder", "clavicle", "collar")):
        return "shoulder"
    if any(k in s for k in ("upleg", "thigh", "upperleg")):
        return "upperLeg"
    if any(k in s for k in ("calf", "shin", "lowerleg", "knee")) or s == "leg":
        return "lowerLeg"
    if any(k in s for k in ("toe", "ball")):
        return "toes"
    if any(k in s for k in ("foot", "ankle")):
        return "foot"
    if "neck" in s:
        return "neck"
    if "head" in s:
        return "head"
    if any(k in s for k in ("spine", "chest", "torso", "abdomen")):
        return "spine"
    if any(k in s for k in ("hips", "pelvis")) or s == "hip":
        return "hips"
    return None


# ---------------------------------------------------------------- scene

def import_any(path):
    """Import a mesh/rig file into an empty scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = path.lower().rsplit(".", 1)[-1]
    if ext in ("glb", "gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == "fbx":
        if hasattr(bpy.ops.wm, "fbx_import"):
            bpy.ops.wm.fbx_import(filepath=path)
        else:
            bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise ValueError(f"Unsupported rig format: .{ext}")
    bpy.context.view_layer.update()


def find_rig():
    """(armature, [skinned meshes]) of the scene: the armature with most bones and the meshes it deforms."""
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError("No armature in the file: the model is not rigged.")
    arm = max(arms, key=lambda o: len(o.data.bones))
    meshes = [
        o for o in bpy.context.scene.objects
        if o.type == "MESH" and len(o.data.vertices) and any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)
    ]
    if not meshes:
        raise RuntimeError("The armature deforms no mesh (no Armature modifier).")
    return arm, meshes


def bone_shape_objects():
    """Objects only used as pose-bone custom shapes (the glTF importer adds an icosphere); never export them."""
    shapes = set()
    for o in bpy.context.scene.objects:
        if o.type == "ARMATURE" and o.pose:
            shapes.update(pb.custom_shape for pb in o.pose.bones if pb.custom_shape)
    return shapes


def mute_animation(arm):
    """Detach actions and NLA so the pose is fully controlled by matrix_basis."""
    ad = arm.animation_data
    if ad:
        ad.action = None
        for t in ad.nla_tracks:
            t.mute = True


def rest_pose(arm):
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def evaluated_coords(meshes):
    """World-space coordinates of all evaluated (deformed) vertices, concatenated over meshes."""
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for obj in meshes:
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
        n = len(me.vertices)
        buf = array("f", [0.0]) * (3 * n)
        me.vertices.foreach_get("co", buf)
        mw = ev.matrix_world.copy()
        ev.to_mesh_clear()
        ident = mw == Matrix.Identity(4)
        for i in range(n):
            v = Vector((buf[3 * i], buf[3 * i + 1], buf[3 * i + 2]))
            out.append(v if ident else mw @ v)
    return out


def mesh_edges(meshes):
    """Edges as (i, j) pairs in the concatenated vertex index space of evaluated_coords()."""
    edges, offset = [], 0
    for obj in meshes:
        me = obj.data
        buf = array("i", [0]) * (2 * len(me.edges))
        me.edges.foreach_get("vertices", buf)
        edges += [(offset + buf[2 * k], offset + buf[2 * k + 1]) for k in range(len(me.edges))]
        offset += len(me.vertices)
    return edges


# ---------------------------------------------------------------- role mapping

def _tables(arm):
    mw = arm.matrix_world
    B = {}
    for b in arm.data.bones:
        B[b.name] = {
            "head": mw @ b.head_local,
            "tail": mw @ b.tail_local,
            "parent": b.parent.name if b.parent else None,
            "children": [c.name for c in b.children],
        }
    return B


def _ancestors(B, n):
    out = []
    while n is not None:
        out.append(n)
        n = B[n]["parent"]
    return out


def _lca(B, a, b):
    up = set(_ancestors(B, a))
    for n in _ancestors(B, b):
        if n in up:
            return n
    return None


def _subtree(B, n):
    out, stack = [], [n]
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(B[cur]["children"])
    return out


def _path(B, top, leaf):
    """Bones from top's child down to leaf (top excluded)."""
    chain = []
    for n in _ancestors(B, leaf):
        if n == top:
            return list(reversed(chain))
        chain.append(n)
    return []


def map_roles(arm):
    """Humanoid roles from structure + joint positions, refined by names.
    Returns {"ok", "roles": {role: bone}, "missing": [...], "notes": [...], "height_m", "source"}."""
    B = _tables(arm)
    notes = []
    if len(B) < 8:
        return {"ok": False, "roles": {}, "missing": list(ROLES), "notes": ["Fewer than 8 bones: not a humanoid rig."], "height_m": 0.0, "source": "structure"}
    pts = [p for d in B.values() for p in (d["head"], d["tail"])]
    z_lo, z_hi = min(p.z for p in pts), max(p.z for p in pts)
    x_mid = (min(p.x for p in pts) + max(p.x for p in pts)) / 2.0
    H = max(1e-3, z_hi - z_lo)
    leaves = [n for n, d in B.items() if not d["children"]]

    def lowz(n):
        return min(B[n]["head"].z, B[n]["tail"].z)

    def topz(n):
        return max(B[n]["head"].z, B[n]["tail"].z)

    def side_x(n):
        return (B[n]["head"].x + B[n]["tail"].x) / 2.0 - x_mid

    def lateral(n):
        return max(abs(p.x - x_mid) for m in _subtree(B, n) for p in (B[m]["head"], B[m]["tail"]))

    def reach(n):
        """Length of the longest bone chain below n (arms are long even when they hang down)."""
        best = 0.0
        stack = [(n, 0.0)]
        while stack:
            cur, acc = stack.pop()
            acc += (B[cur]["tail"] - B[cur]["head"]).length
            best = max(best, acc)
            stack.extend((c, acc) for c in B[cur]["children"])
        return best

    def is_arm(c, sign):
        return lateral(c) >= 0.08 * H and reach(c) >= 0.2 * H and any(sign * side_x(m) > 0.04 * H for m in _subtree(B, c))

    left_feet = [n for n in leaves if side_x(n) > 0.015 * H]
    right_feet = [n for n in leaves if side_x(n) < -0.015 * H]
    if not left_feet or not right_feet:
        return {"ok": False, "roles": {}, "missing": list(ROLES), "notes": ["No left/right leg chains found."], "height_m": round(H, 4), "source": "structure"}
    lf, rf = min(left_feet, key=lowz), min(right_feet, key=lowz)
    if lowz(lf) > z_lo + 0.2 * H or lowz(rf) > z_lo + 0.2 * H:
        return {"ok": False, "roles": {}, "missing": list(ROLES), "notes": ["No bones reach the feet."], "height_m": round(H, 4), "source": "structure"}
    pelvis = _lca(B, lf, rf)
    top_leaf = max(leaves, key=topz)
    hips = _lca(B, pelvis, top_leaf) if pelvis else None
    if hips is None:
        return {"ok": False, "roles": {}, "missing": list(ROLES), "notes": ["Legs and spine share no root bone."], "height_m": round(H, 4), "source": "structure"}
    roles = {"hips": hips}

    def joint(chain, i):
        return B[chain[i]]["head"] if i < len(chain) else B[chain[-1]]["tail"]

    def drop(chain, i):
        return joint(chain, i).z - joint(chain, i + 1).z

    for side, leaf in (("left", lf), ("right", rf)):
        chain = _path(B, hips, leaf)
        k = next((i for i, n in enumerate(chain) if abs(B[n]["head"].x - x_mid) >= 0.02 * H and drop(chain, i) >= 0.12 * H), None)
        if k is None:
            notes.append(f"No {side} thigh bone found.")
            continue
        roles[f"{side}UpperLeg"] = chain[k]
        low = next((i for i in range(k + 1, len(chain)) if drop(chain, i) >= 0.10 * H), None)
        if low is not None:
            roles[f"{side}LowerLeg"] = chain[low]
            if low + 1 < len(chain):
                roles[f"{side}Foot"] = chain[low + 1]
            if low + 2 < len(chain):
                roles[f"{side}Toes"] = chain[low + 2]

    spine_path = _path(B, hips, top_leaf)
    branch = None
    for n in spine_path:
        side_kids = [c for c in B[n]["children"] if c not in spine_path]
        if any(is_arm(c, 1.0) for c in side_kids) and any(is_arm(c, -1.0) for c in side_kids):
            branch = n
            break
    if branch is None:
        notes.append("No chest bone with two arm branches: arm roles missing.")
        torso = spine_path[:-2] if len(spine_path) > 2 else spine_path[:1]
        rest = spine_path[len(torso):]
    else:
        torso = spine_path[: spine_path.index(branch) + 1]
        rest = spine_path[spine_path.index(branch) + 1:]
    if torso:
        roles["spine"] = torso[0]
        if len(torso) >= 3:
            roles["chest"], roles["upperChest"] = torso[-2], torso[-1]
        elif len(torso) == 2:
            roles["chest"] = torso[1]
    if rest:
        roles["neck"] = rest[0]
        roles["head"] = rest[1] if len(rest) >= 2 else rest[0]
        named_head = [n for n in rest if name_kind(n) == "head"]
        if len(named_head) == 1:
            roles["head"] = named_head[0]

    if branch is not None:
        for side in ("left", "right"):
            sign = 1.0 if side == "left" else -1.0
            kids = [c for c in B[branch]["children"] if c not in spine_path and is_arm(c, sign)]
            if not kids:
                notes.append(f"No {side} arm chain found.")
                continue
            root = max(kids, key=lambda c: max(sign * side_x(m) for m in _subtree(B, c)))
            origin = B[branch]["head"]
            tip = max((m for m in _subtree(B, root) if not B[m]["children"]), key=lambda m: (B[m]["tail"] - origin).length)
            chain = [root] + _path(B, root, tip) if tip != root else [root]
            hand_i = next((i for i, n in enumerate(chain) if len(B[n]["children"]) >= 2), None)
            if hand_i is None:
                hand_i = min(3, len(chain) - 1) if len(chain) >= 4 else len(chain) - 1
            named = [i for i, n in enumerate(chain) if name_kind(n) == "hand"]
            if len(named) == 1:
                hand_i = named[0]
            roles[f"{side}Hand"] = chain[hand_i]
            pre = chain[:hand_i]
            if len(pre) >= 1:
                roles[f"{side}LowerArm"] = pre[-1]
            if len(pre) >= 2:
                roles[f"{side}UpperArm"] = pre[-2]
            if len(pre) >= 3:
                roles[f"{side}Shoulder"] = pre[-3]

    # Names only annotate: a structural pick whose name reads another body part is reported, not trusted.
    for role, bone in roles.items():
        want = role.replace("left", "").replace("right", "")
        want = want[0].lower() + want[1:]
        want = {"chest": "spine", "upperChest": "spine"}.get(want, want)
        got = name_kind(bone)
        if got and got not in (want, "end") and not (want == "spine" and got == "hips"):
            notes.append(f"Bone {bone!r} mapped to {role} by structure but its name reads {got}.")

    # Side sanity: a bone named left must sit on +X (the character faces -Y).
    for role, bone in roles.items():
        ns = name_side(bone)
        if ns and role.startswith(("left", "right")) and not role.startswith(ns):
            side = "left" if role.startswith("left") else "right"
            notes.append(f"Bone {bone!r} is named {ns} but sits on the character's {side} (+X is left): rig may be mirrored.")
            break

    missing = [r for r in ROLES if r not in roles]
    core = ("hips", "leftUpperArm", "rightUpperArm", "leftUpperLeg", "rightUpperLeg", "head")
    return {
        "ok": all(r in roles for r in core),
        "roles": roles,
        "missing": missing,
        "notes": notes,
        "height_m": round(H, 4),
        "source": "structure+names",
    }


def map_roles_by_name(arm):
    """Humanoid roles from bone names alone (Mixamo, CMU BVH, UE, 3ds Max Biped styles), for skeletons whose structure
    is ambiguous, e.g. a motion clip that does not face -Y. The bone nearest the root wins each role.
    Returns the same dict shape as map_roles."""
    B = _tables(arm)
    order, queue = [], [n for n, d in B.items() if d["parent"] is None]
    while queue:
        n = queue.pop(0)
        order.append(n)
        queue.extend(B[n]["children"])
    roles, torso = {}, []
    for n in order:
        kind = name_kind(n)
        if kind in (None, "end", "finger"):
            continue
        if kind == "spine":
            torso.append(n)
        elif kind in ("hips", "neck", "head"):
            roles.setdefault(kind, n)
        elif name_side(n):
            roles.setdefault(name_side(n) + kind[0].upper() + kind[1:], n)
    if torso:
        roles["spine"] = torso[0]
        if len(torso) >= 3:
            roles["chest"], roles["upperChest"] = torso[-2], torso[-1]
        elif len(torso) == 2:
            roles["chest"] = torso[1]
    pts = [p for d in B.values() for p in (d["head"], d["tail"])]
    core = ("hips", "leftUpperArm", "rightUpperArm", "leftUpperLeg", "rightUpperLeg", "head")
    return {
        "ok": all(r in roles for r in core),
        "roles": roles,
        "missing": [r for r in ROLES if r not in roles],
        "notes": [],
        "height_m": round(max(p.z for p in pts) - min(p.z for p in pts), 4) if pts else 0.0,
        "source": "names",
    }


def bone_roles(arm, roles):
    """bone -> role of its nearest mapped ancestor-or-self (fingers -> hand, toe ends -> toes, hair -> head)."""
    B = _tables(arm)
    role_of = {b: r for r, b in roles.items()}
    out = {}
    for n in B:
        cur = n
        while cur is not None and cur not in role_of:
            cur = B[cur]["parent"]
        out[n] = role_of.get(cur)
    return out


def humanoid_map(arm, mapping):
    roles = mapping["roles"]
    mapped = set(roles.values())
    return {
        "format": "charforge.humanoid/1",
        "rest_pose": "as bound (A-pose for provider rigs)",
        "coordinates": "glTF: +Y up, +Z forward (character faces +Z), meters",
        "bones": {r: roles[r] for r in ROLES if r in roles},
        "mixamo": {roles[r]: MIXAMO[r] for r in ROLES if r in roles},
        "missing": mapping["missing"],
        "unmapped": sorted(b.name for b in arm.data.bones if b.name not in mapped),
        "height_m": mapping["height_m"],
    }


# ---------------------------------------------------------------- vertex classification

class Body:
    """Per-vertex body-part classification of the rest pose, from bone segments only (never from weights)."""

    def __init__(self, arm, meshes, mapping):
        self.arm, self.meshes, self.mapping = arm, meshes, mapping
        self.roles = mapping["roles"]
        self.role_of_bone = bone_roles(arm, self.roles)
        B = _tables(arm)
        self.B = B
        mute_animation(arm)
        rest_pose(arm)
        self.rest = evaluated_coords(meshes)
        zs = [v.z for v in self.rest] or [0.0, 1.0]
        self.z_lo, self.z_hi = min(zs), max(zs)
        self.H = max(0.1, self.z_hi - self.z_lo)
        self.step = 0.006 * self.H

        kd_pts, self.sample_bone = [], []
        for n, d in B.items():
            role = self.role_of_bone.get(n)
            if role is None:
                continue
            a, b = d["head"], d["tail"]
            k = max(2, int(math.ceil((b - a).length / self.step)) + 1)
            for j in range(k):
                kd_pts.append(a.lerp(b, j / (k - 1)))
                self.sample_bone.append(n)
        kd = KDTree(len(kd_pts))
        for i, p in enumerate(kd_pts):
            kd.insert(p, i)
        kd.balance()

        self.bone, self.dist, self.ambig = [], [], []
        for v in self.rest:
            _, idx, d = kd.find(v)
            b = self.sample_bone[idx]
            g = self.group_of(b)
            amb = None
            for _, idx2, _ in kd.find_range(v, d * AMBIG_RATIO + 0.5 * self.step):
                g2 = self.group_of(self.sample_bone[idx2])
                if g2 != g:
                    amb = g2
                    break
            self.bone.append(b)
            self.dist.append(d)
            self.ambig.append(amb)

        # Core radius per bone: vertices beyond it are outer layers (hair, skirt, coat) and skipped by repair.
        per_bone = {}
        for b, d in zip(self.bone, self.dist):
            per_bone.setdefault(b, []).append(d)
        self.core_r = {}
        for b, ds in per_bone.items():
            ds.sort()
            self.core_r[b] = 1.5 * ds[int(0.4 * (len(ds) - 1))] + self.step

    def role(self, i):
        return self.role_of_bone.get(self.bone[i])

    def group_of(self, bone):
        r = self.role_of_bone.get(bone)
        return GROUP_OF_ROLE.get(r) if r else None

    def group(self, i):
        return self.group_of(self.bone[i])

    def joint(self, role):
        b = self.roles.get(role)
        return self.B[b]["head"] if b else None

    def seg_len(self, role, child_role):
        a, b = self.joint(role), self.joint(child_role)
        if a is None or b is None:
            bone = self.roles.get(role)
            return (self.B[bone]["tail"] - self.B[bone]["head"]).length if bone else 0.0
        return (b - a).length

    def is_core(self, i):
        return self.dist[i] <= self.core_r.get(self.bone[i], 0.0)


# ---------------------------------------------------------------- deformation test

def rotate_bone_world(arm, bone, axis_world, angle):
    """Rotate a pose bone about a world axis through its head (pose must be at rest)."""
    pb = arm.pose.bones[bone]
    axis = (arm.matrix_world.to_3x3().inverted() @ Vector(axis_world)).normalized()
    head = pb.bone.head_local.copy()
    pb.matrix = Matrix.Translation(head) @ Matrix.Rotation(angle, 4, axis) @ Matrix.Translation(-head) @ pb.matrix
    bpy.context.view_layer.update()


def _sub_roles(prefix):
    return {f"{prefix}{k}" for k in ("UpperArm", "LowerArm", "Hand")}


def _stats(vals, tol):
    if not vals:
        return {"n": 0, "max_m": 0.0, "p99_m": 0.0, "leak_ratio": 0.0}
    vals = sorted(vals)
    n = len(vals)
    return {
        "n": n,
        "max_m": round(vals[-1], 4),
        "p99_m": round(vals[min(n - 1, int(0.99 * (n - 1)))], 4),
        "leak_ratio": round(sum(1 for x in vals if x > tol) / n, 4),
    }


def _flood(adj, src, through):
    """`src` plus every vertex reachable from it walking only through `through`."""
    seen, stack = set(src), list(src)
    while stack:
        v = stack.pop()
        for u in adj.get(v, ()):
            if u not in seen and u in through:
                seen.add(u)
                stack.append(u)
    return seen


def _side_contact(adj, rest, side_src, other_set, band, gap_min):
    """Grow a part over its own rim (ambiguous vertices in `band` reachable from it), then:
    fused = rim vertices that touch `other_set` by an edge; gap = closest distance from the grown part to the
    other side (other_set + band vertices the part did not claim)."""
    part = _flood(adj, side_src, band)
    fused = {u for v in part for u in adj.get(v, ()) if u in other_set}
    others = sorted(other_set | (band - part))
    gap = None
    if others and part:
        kd = KDTree(len(others))
        for k, i in enumerate(others):
            kd.insert(rest[i], k)
        kd.balance()
        gap = min(kd.find(rest[i])[2] for i in part)
    return {
        "gap_m": None if gap is None else round(gap, 4),
        "fused_vertices": len(fused),
        "separated": None if gap is None else bool(not fused and gap >= gap_min),
    }


def contacts(body, gap_min=0.01):
    """Fusion and geometric gaps between each forearm/hand and the body, and between the two legs.
    A part is its unambiguous vertices plus its own rim of ambiguous vertices (where two parts meet), so the gap is
    measured surface to surface and fused = the rim runs into the other part. The shoulder (arms) and the crotch
    (legs) are left out, where parts meet by design; legs use core vertices, so a skirt is not fused thighs."""
    n, rest = len(body.rest), body.rest
    edges = mesh_edges(body.meshes)
    first = {}
    for i, v in enumerate(rest):  # glTF splits vertices along UV seams: weld copies that share a position
        j = first.setdefault((round(v.x * 1e5), round(v.y * 1e5), round(v.z * 1e5)), i)
        if j != i:
            edges.append((j, i))
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    amb = {i for i in range(n) if body.ambig[i] is not None}
    res = {}
    distal = {"left": {"leftLowerArm", "leftHand"}, "right": {"rightLowerArm", "rightHand"}}
    body_set = {i for i in range(n) if body.group(i) in ("torso", "legL", "legR") and i not in amb}
    for side in ("left", "right"):
        sh = body.joint(f"{side}UpperArm")
        if sh is None:
            res[f"{side}_arm"] = {"gap_m": None, "fused_vertices": 0, "separated": None}
            continue
        arm_len = body.seg_len(f"{side}UpperArm", f"{side}LowerArm") + body.seg_len(f"{side}LowerArm", f"{side}Hand")
        far = {i for i in range(n) if (rest[i] - sh).length > 0.35 * arm_len}
        src = {i for i in far if i not in amb and body.role(i) in distal[side] and (rest[i] - sh).length > 0.5 * arm_len}
        res[f"{side}_arm"] = _side_contact(adj, rest, src, body_set & far, amb & far, gap_min) if src else {
            "gap_m": None, "fused_vertices": 0, "separated": None}
    hl, hr = body.joint("leftUpperLeg"), body.joint("rightUpperLeg")
    if hl is None or hr is None:
        res["legs"] = {"gap_m": None, "fused_vertices": 0, "separated": None}
        return res
    thigh = 0.5 * (body.seg_len("leftUpperLeg", "leftLowerLeg") + body.seg_len("rightUpperLeg", "rightLowerLeg"))
    z_cut = min(hl.z, hr.z) - 0.35 * thigh
    low = {i for i in range(n) if rest[i].z < z_cut}
    core_low = {i for i in low if body.is_core(i)}
    lset = {i for i in core_low if i not in amb and body.group(i) == "legL"}
    rset = {i for i in core_low if i not in amb and body.group(i) == "legR"}
    legs = _side_contact(adj, rest, lset, rset, amb & core_low, gap_min) if lset else {"gap_m": None, "fused_vertices": 0, "separated": None}
    outer = {i for i in low if body.group(i) in ("legL", "legR") and not body.is_core(i)}
    reach = _flood(adj, lset, (amb & low) | outer) if lset and outer else set()
    cloth = bool(lset) and not legs["fused_vertices"] and any(u in rset for v in reach for u in adj.get(v, ()))
    legs["bridged_by_cloth"] = cloth
    if cloth:  # a skirt/dress joins the legs: thigh-to-thigh contact cannot be told from cloth, report it as unknown
        legs["separated"] = None
    res["legs"] = legs
    return res


def deformation_tests(body, angle_deg=40.0, tol_m=0.005, max_ratio=0.01, hard_m=0.03, follow_min=0.6, legs_apart=True):
    """Swing each upper arm (forward and up) and each thigh (forward) by angle_deg; measure what must stay still.
    Returns {"tests": [...], "passed": bool, "issues": [str]}."""
    arm = body.arm
    ang = math.radians(angle_deg)
    tol = max(tol_m, 0.003 * body.H)
    plan = [
        ("left_arm_swing", "left", "arm", (1, 0, 0), -ang),
        ("left_arm_raise", "left", "arm", (0, 1, 0), -ang),
        ("right_arm_swing", "right", "arm", (1, 0, 0), -ang),
        ("right_arm_raise", "right", "arm", (0, 1, 0), ang),
        ("left_leg_swing", "left", "leg", (1, 0, 0), -ang),
        ("right_leg_swing", "right", "leg", (1, 0, 0), -ang),
    ]
    tests, issues = [], []
    for name, side, limb, axis, a in plan:
        other = "right" if side == "left" else "left"
        role = f"{side}UpperArm" if limb == "arm" else f"{side}UpperLeg"
        bone = body.roles.get(role)
        if not bone:
            tests.append({"name": name, "skipped": f"no {role} bone"})
            continue
        J = body.joint(role)
        if limb == "arm":
            r_blend = max(0.6 * body.seg_len(role, f"{side}LowerArm"), 0.06 * body.H)
            moving = _sub_roles(side)
            static = {
                "legs": lambda i: body.group(i) in ("legL", "legR"),
                "torso": lambda i, J=J, r=r_blend: body.group(i) == "torso" and (body.rest[i] - J).length > r,
                "head": lambda i: body.group(i) == "head",
                "other_arm": lambda i, o=other: body.role(i) in _sub_roles(o),
            }
        else:
            r_blend = max(0.5 * body.seg_len(role, f"{side}LowerLeg"), 0.06 * body.H)
            moving = {f"{side}UpperLeg", f"{side}LowerLeg", f"{side}Foot", f"{side}Toes"}
            static = {
                "arms": lambda i: body.role(i) in ("leftLowerArm", "leftHand", "rightLowerArm", "rightHand"),
                "upper_body": lambda i, J=J, r=r_blend: (body.role(i) in ("chest", "upperChest") or body.group(i) == "head") and (body.rest[i] - J).length > r,
            }
            if legs_apart:
                static["other_leg"] = lambda i, o=other: body.role(i) in (f"{o}LowerLeg", f"{o}Foot", f"{o}Toes")

        rest_pose(arm)
        rotate_bone_world(arm, bone, axis, a)
        posed = evaluated_coords(body.meshes)
        rest_pose(arm)
        R = Matrix.Rotation(a, 3, Vector(axis))

        regions = {k: [] for k in static}
        follow = []
        for i, p0 in enumerate(body.rest):
            if body.ambig[i] is not None:
                continue
            disp = (posed[i] - p0).length
            if body.role(i) in moving:
                if (p0 - J).length > r_blend and body.is_core(i):
                    ideal = (R @ (p0 - J) - (p0 - J)).length
                    if ideal > 0.01:
                        follow.append(disp / ideal)
                continue
            for k, pred in static.items():
                if pred(i):
                    regions[k].append(disp)
                    break
        stats = {k: _stats(v, tol) for k, v in regions.items()}
        follow.sort()
        f_med = round(follow[len(follow) // 2], 3) if follow else None
        bad = [k for k, s in stats.items() if s["n"] and (s["leak_ratio"] > max_ratio or s["max_m"] > hard_m)]
        test = {"name": name, "bone": bone, "regions": stats, "follow": f_med, "moving_n": len(follow), "passed": True}
        if bad:
            test["passed"] = False
            worst = max(bad, key=lambda k: stats[k]["max_m"])
            issues.append(
                f"{name}: {', '.join(bad)} moved with the {limb} (worst {worst}: max {stats[worst]['max_m'] * 100:.1f} cm, "
                f"{stats[worst]['leak_ratio'] * 100:.1f}% of vertices)"
            )
        if f_med is not None and len(follow) >= 20 and f_med < follow_min:
            test["passed"] = False
            issues.append(f"{name}: the {limb} follows its bone only {f_med * 100:.0f}% (missing weights)")
        tests.append(test)
    ran = [t for t in tests if "skipped" not in t]
    return {"tests": tests, "passed": bool(ran) and all(t["passed"] for t in ran), "issues": issues}


# ---------------------------------------------------------------- weight repair

def repair_weights(body, legs_apart=True):
    """Strip weights of anatomically wrong bones (the "sticking" leak), on unambiguous vertices only.
    - Arms never drive legs, torso, skirt or head beyond the shoulder blend zone (outer layers included).
    - Legs and the other arm never drive an arm (outer layers included).
    - Core vertices only: forearm/hand are not driven by the torso; shins/feet are not driven by the other leg
      when the legs are apart; the head is not driven by legs.
    Head bones are never stripped (hair is often weighted to the head on purpose).
    Returns {"vertices_changed", "weight_removed"}."""
    arm = body.arm
    bones = {b.name for b in arm.data.bones}
    sh = {s: body.joint(f"{s}UpperArm") for s in ("left", "right")}
    r_sh = {s: max(0.6 * body.seg_len(f"{s}UpperArm", f"{s}LowerArm"), 0.06 * body.H) for s in ("left", "right")}

    def disallowed(i):
        g, role, p, core = body.group(i), body.role(i) or "", body.rest[i], body.is_core(i)
        out = set()
        if g in ("legL", "legR", "torso", "head"):
            out.add("distal")
            for s, ga in (("left", "armL"), ("right", "armR")):
                if sh[s] is None or (p - sh[s]).length > r_sh[s]:
                    out.add(ga)
            if g in ("legL", "legR"):
                out |= {"shoulderL", "shoulderR"}
        elif g in ("armL", "armR"):
            o = "R" if g == "armL" else "L"
            out |= {"legL", "legR", "arm" + o, "shoulder" + o}
        if not core:
            return out
        if g in ("legL", "legR") and legs_apart and role.endswith(("LowerLeg", "Foot", "Toes")):
            out.add("legR" if g == "legL" else "legL")
        elif g == "head":
            out |= {"legL", "legR"}
        elif g in ("armL", "armR"):
            s = "left" if g == "armL" else "right"
            if role.endswith(("LowerArm", "Hand")) or (sh[s] is not None and (p - sh[s]).length > r_sh[s]):
                out.add("torso")
        return out

    changed, removed, offset = 0, 0.0, 0
    for obj in body.meshes:
        vgs = {vg.index: vg for vg in obj.vertex_groups if vg.name in bones}
        touched = False
        for v in obj.data.vertices:
            i = offset + v.index
            if body.ambig[i] is not None or body.group(i) is None:
                continue
            bad_groups = disallowed(i)
            if not bad_groups:
                continue
            ws = {g.group: g.weight for g in v.groups if g.group in vgs and g.weight > 1e-6}
            drop = []
            for gi in ws:
                bname = vgs[gi].name
                brole = body.role_of_bone.get(bname) or ""
                if body.group_of(bname) in bad_groups or ("distal" in bad_groups and brole.endswith(("LowerArm", "Hand"))):
                    drop.append(gi)
            if not drop:
                continue
            for gi in drop:
                removed += ws.pop(gi)
                vgs[gi].remove([v.index])
            total = sum(ws.values())
            if total < 1e-4:
                own = obj.vertex_groups.get(body.bone[i]) or obj.vertex_groups.new(name=body.bone[i])
                own.add([v.index], 1.0, "REPLACE")
            else:
                for gi, w in ws.items():
                    vgs[gi].add([v.index], w / total, "REPLACE")
            changed += 1
            touched = True
        if touched:
            obj.data.update()
        offset += len(obj.data.vertices)
    bpy.context.view_layer.update()
    return {"vertices_changed": changed, "weight_removed": round(removed, 3)}


def unweighted_ratio(body):
    bones = {b.name for b in body.arm.data.bones}
    n = un = 0
    for obj in body.meshes:
        idx = {vg.index for vg in obj.vertex_groups if vg.name in bones}
        for v in obj.data.vertices:
            n += 1
            if sum(g.weight for g in v.groups if g.group in idx) < 1e-4:
                un += 1
    return un / n if n else 0.0


def snapshot_weights(meshes):
    return [[[(g.group, g.weight) for g in v.groups] for v in obj.data.vertices] for obj in meshes]


def restore_weights(meshes, snap):
    for obj, verts in zip(meshes, snap):
        groups = list(obj.vertex_groups)
        for v, saved in zip(obj.data.vertices, verts):
            for gi in [g.group for g in v.groups]:  # copy first: remove() reallocates v.groups
                groups[gi].remove([v.index])
            for gi, w in saved:
                groups[gi].add([v.index], w, "REPLACE")
        obj.data.update()
    bpy.context.view_layer.update()


# ---------------------------------------------------------------- one-call QA

def leak_score(dt):
    """Lower is better: failed tests weigh most, then the summed worst displacement of still regions."""
    score = 0.0
    for t in dt["tests"]:
        if "skipped" in t:
            continue
        score += 0.0 if t["passed"] else 10.0
        score += sum(r["max_m"] for r in t["regions"].values())
    return round(score, 4)


def contact_issues(con):
    out = []
    for side in ("left", "right"):
        c = con.get(f"{side}_arm") or {}
        if c.get("separated") is False:
            out.append(
                f"{side} forearm/hand touches the body (gap {100 * (c['gap_m'] or 0):.1f} cm, {c['fused_vertices']} fused vertices): "
                "weights bleed there; a sheet/mesh with a clear A-pose gap fixes it"
            )
    c = con.get("legs") or {}
    if c.get("separated") is False:
        out.append(
            f"legs touch below the crotch (gap {100 * (c['gap_m'] or 0):.1f} cm, {c['fused_vertices']} fused vertices): "
            "thighs may stick when walking"
        )
    return out


def run_qa(arm, meshes, mapping, angle=40.0, tol_m=0.005, max_ratio=0.01, hard_m=0.03, follow_min=0.6, gap_min_m=0.01, repair=True):
    """Deformation check, then weight repair when it fails; the repair is kept only if it measurably helps.
    Returns {passed, issues, contact_issues, contacts, legs_apart, before, after, repaired, repair, unweighted_ratio}."""
    if not mapping["ok"]:
        msg = "Could not map a humanoid skeleton: " + "; ".join(mapping["notes"] or mapping["missing"])
        return {"passed": None, "issues": [msg], "contact_issues": [], "contacts": {}, "legs_apart": None,
                "before": None, "after": None, "repaired": False, "repair": None, "unweighted_ratio": None}
    body = Body(arm, meshes, mapping)
    con = contacts(body, gap_min_m)
    legs_apart = con["legs"]["separated"] is not False
    kw = dict(angle_deg=angle, tol_m=tol_m, max_ratio=max_ratio, hard_m=hard_m, follow_min=follow_min, legs_apart=legs_apart)
    before = deformation_tests(body, **kw)
    before["leak_score"] = leak_score(before)
    res = {"contacts": con, "legs_apart": legs_apart, "before": before, "after": None, "repaired": False, "repair": None}
    if repair and not before["passed"]:
        snap = snapshot_weights(meshes)
        rep = repair_weights(body, legs_apart)
        res["repair"] = rep
        if rep["vertices_changed"]:
            after = deformation_tests(body, **kw)
            after["leak_score"] = leak_score(after)
            res["after"] = after
            improved = (after["passed"] and not before["passed"]) or after["leak_score"] < before["leak_score"] - 1e-4
            if improved and unweighted_ratio(body) < 0.01:
                res["repaired"] = True
            else:
                restore_weights(meshes, snap)
    final = res["after"] if res["repaired"] else before
    ci = contact_issues(con)
    res.update(passed=final["passed"], issues=final["issues"] + ci, contact_issues=ci, unweighted_ratio=round(unweighted_ratio(body), 5))
    return res
