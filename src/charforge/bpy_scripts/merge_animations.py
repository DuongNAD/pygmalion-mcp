"""Multi-action animation assembly for rigged characters. Runs INSIDE Blender: bpy/mathutils/stdlib only
(+ the sibling cf_humanoid / cf_secondary / retarget modules).

Modes:
  procedural  one clip per --plan item on any humanoid rig: {"name", "kind"} keys idle/walk/run/wave/jump (bones found
              by role, rotated about anatomical world axes, so bone names and rolls do not matter); {"name", "clip",
              "loop", "trim", "action"} retargets a library clip by role (retarget.py). Without --plan each action
              name must start with a procedural kind; any other name fails (no silent fallback).
  merge       import clip GLBs and merge their actions onto the first rig (legacy).
  rename      rename animation tracks in a GLB's JSON chunk (lossless, no Blender re-export).
  post        start from an animated GLB (--mesh): merge extra clips (--clips, named by the tail of --actions),
              copy repaired skin weights (--weights-from), add secondary motion (--secondary auto), export --out.
              Nothing to do -> no export and "changed": false.
procedural and post both honour --secondary auto|off (spring chains for hanging hair and skirts, kept only if
they do not worsen surface stretch).

Protocol:
blender -b --factory-startup --python-exit-code 1 -P merge_animations.py -- --result <json> --mesh <glb> --out <glb>
        [--actions a,b,c] [--mode procedural|merge|rename|post] [--clips x.glb,y.glb] [--fps 30] [--plan <json list>]
        [--secondary auto|off] [--weights-from rigged.glb] [--stiffness 1.2] [--drag 0.35] [--gravity 0.15]
        [--skirt-chains 8] [--max-hair-chains 4]
Result JSON: {ok, actions, clip_count, mesh_path, changed, secondary: {...}, weights: {...}, retarget: {clip: {...}}}
"""
import argparse
import json
import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
from mathutils import Matrix, Quaternion, Vector  # noqa: E402
from mathutils.kdtree import KDTree  # noqa: E402

import cf_humanoid as hum  # noqa: E402
import cf_secondary as sec  # noqa: E402
import retarget as rt  # noqa: E402

PROCEDURAL = ("idle", "walk", "run", "wave", "jump")
BONE_NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--actions", default="idle,walk,wave")
    ap.add_argument("--mode", default="procedural", choices=["procedural", "merge", "rename", "post"])
    ap.add_argument("--clips", default="")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--secondary", default="off", choices=["auto", "off"])
    ap.add_argument("--weights-from", default=None)
    ap.add_argument("--stiffness", type=float, default=1.2)
    ap.add_argument("--drag", type=float, default=0.35)
    ap.add_argument("--gravity", type=float, default=0.15)
    ap.add_argument("--skirt-chains", type=int, default=8)
    ap.add_argument("--max-hair-chains", type=int, default=4)
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:])


# ---------------------------------------------------------------- scene helpers

def get_pose_bone(arm_obj, name):
    """Find pose bone by standard name, prefix-stripped name, or case-insensitive match."""
    if not arm_obj or not arm_obj.pose:
        return None
    if name in arm_obj.pose.bones:
        return arm_obj.pose.bones[name]
    name_lower = name.lower()
    for b in arm_obj.pose.bones:
        b_lower = b.name.lower()
        if b_lower == name_lower or b_lower.endswith(":" + name_lower) or b_lower.endswith("_" + name_lower):
            return b
    return None


def clear_scene(fps=30.0):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = int(round(fps))
    bpy.context.scene.render.fps_base = 1.0


def find_armature():
    armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature object found in scene.")
    return max(armatures, key=lambda o: len(o.data.bones))


def skinned_meshes(arm):
    return [o for o in bpy.context.scene.objects if o.type == "MESH" and any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)]


def clean_orphaned_data():
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)
    for a in list(bpy.data.armatures):
        if a.users == 0:
            bpy.data.armatures.remove(a)


def assign(arm, act):
    """Make `act` the evaluated action of the armature (NLA muted; Blender 4.4+ action slots handled).
    The pose is reset first: bones the clip does not key would otherwise keep the previous clip's values."""
    ad = arm.animation_data or arm.animation_data_create()
    for t in ad.nla_tracks:
        t.mute = True
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    ad.action = act
    if hasattr(ad, "action_slot") and getattr(ad, "action_slot", None) is None and len(getattr(act, "slots", [])):
        ad.action_slot = act.slots[0]


def stash(arm, actions):
    """One unmuted NLA track per action, no active action: the glTF exporter writes every clip."""
    ad = arm.animation_data or arm.animation_data_create()
    ad.action = None
    for t in list(ad.nla_tracks):
        ad.nla_tracks.remove(t)
    for act in actions:
        act.use_fake_user = True
        track = ad.nla_tracks.new()
        track.name = act.name
        track.strips.new(act.name, int(act.frame_range[0]), act)


def export_glb(path, arm, actions):
    """Export exactly `actions` (the exporter writes every action in the file, so others are dropped first)."""
    for o in hum.bone_shape_objects():
        bpy.data.objects.remove(o, do_unlink=True)
    keep = set(actions)
    for act in [a for a in bpy.data.actions if a not in keep]:
        bpy.data.actions.remove(act)
    stash(arm, actions)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_current_frame=False,
        export_skins=True,
        export_apply=False,
    )
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"Exported GLB is missing or empty: {path}")


def glb_animation_names(path):
    """Animation names in file order, read from the GLB JSON chunk."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20:
        return []
    n, kind = struct.unpack("<I4s", data[12:20])
    if kind != b"JSON":
        return []
    return [a.get("name", f"action_{i}") for i, a in enumerate(json.loads(data[20:20 + n]).get("animations", []))]


def actions_in_file_order(path, new_actions):
    """Imported actions ordered like the GLB animations (bpy.data sorts by name)."""
    order = glb_animation_names(path)
    left = list(new_actions)
    out = []
    for name in order:
        hit = next((a for a in left if a.name == name), None) or next((a for a in left if a.name.startswith(name)), None)
        if hit:
            out.append(hit)
            left.remove(hit)
    return out + sorted(left, key=lambda a: a.name)


# ---------------------------------------------------------------- procedural animation

class Animator:
    """Keys rotations about anatomical world axes on bones found by role (works on any humanoid rig).
    World X = character left, -Y = forward, Z = up; axes are converted to armature space once."""

    def __init__(self, arm):
        self.arm = arm
        self.m3inv = arm.matrix_world.to_3x3().inverted()
        self.X, self.Y, self.Z = ((self.m3inv @ Vector(v)).normalized() for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        mapping = hum.map_roles(arm)
        self.roles = dict(mapping["roles"])
        for role, mixamo in hum.MIXAMO.items():  # rigs that map only partially still animate by name
            if role not in self.roles:
                pb = get_pose_bone(arm, mixamo)
                if pb:
                    self.roles[role] = pb.name
        zs = [(arm.matrix_world @ b.head_local).z for b in arm.data.bones] + [(arm.matrix_world @ b.tail_local).z for b in arm.data.bones]
        self.H = max(0.3, max(zs) - min(zs))
        self.rest_abd = {s: self._abduction(s) for s in ("left", "right")}

    def bone(self, role):
        n = self.roles.get(role)
        return self.arm.pose.bones.get(n) if n else None

    def _abduction(self, side):
        """Angle (rad) between the rest arm (shoulder -> wrist, robust to a bent elbow) and straight down,
        in the frontal plane."""
        b = self.arm.data.bones.get(self.roles.get(f"{side}UpperArm") or "")
        if not b:
            return 0.0
        h = self.arm.data.bones.get(self.roles.get(f"{side}Hand") or "")
        end = h.head_local if h else b.tail_local
        d = (self.arm.matrix_world.to_3x3() @ (end - b.head_local)).normalized()
        return math.atan2(abs(d.x), max(1e-6, -d.z)) if d.z < 0 else math.pi / 2

    def quat(self, role, rotations):
        """Local quaternion for a sequence of (armature axis, angle) applied in order to the rest bone."""
        pb = self.bone(role)
        R = pb.bone.matrix_local.to_3x3()
        Q = Matrix.Identity(3)
        for axis, ang in rotations:
            if ang:
                Q = Matrix.Rotation(ang, 3, axis) @ Q
        return (R.inverted() @ Q @ R).to_quaternion()

    def key_rot(self, role, frame, rotations):
        pb = self.bone(role)
        if not pb:
            return
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = self.quat(role, rotations)
        pb.keyframe_insert("rotation_quaternion", frame=frame, group=pb.name)

    def key_loc(self, role, frame, delta):
        """Translate a bone by a world-space offset in meters (scaled to the character height)."""
        pb = self.bone(role)
        if not pb:
            return
        pb.location = pb.bone.matrix_local.to_3x3().inverted() @ (self.m3inv @ (Vector(delta) * (self.H / 1.7)))
        pb.keyframe_insert("location", frame=frame, group=pb.name)

    def relaxed_arms(self, frame, extra=None, sway=0.0, elbow=0.18):
        """Lower A-pose arms to ~22 deg from the body; `extra` = {side: [(axis, angle)]} on top."""
        extra = extra or {}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            target = math.radians(22.0)
            down = (self.rest_abd[side] - target) * sign  # rotate toward the body about Y
            rots = [(self.Y, down), (self.X, -sway * sign)] + list(extra.get(side, []))
            self.key_rot(f"{side}UpperArm", frame, rots)
            self.key_rot(f"{side}LowerArm", frame, [(self.X, -elbow)])

    def make(self, name, fps, kind=None):
        kind = kind or next((k for k in PROCEDURAL if name.lower().startswith(k)), None)
        if kind not in PROCEDURAL:
            raise rt.ClipError(f"Unknown procedural action {name!r}.", f"Use one of: {', '.join(PROCEDURAL)}.")
        act = bpy.data.actions.new(name=name)
        assign(self.arm, act)
        getattr(self, kind)(fps)
        return act

    # ---- clips (fps-independent timing)

    def idle(self, fps):
        n = int(round(3.0 * fps))
        for f in range(0, n + 1):
            t = 2 * math.pi * f / n
            breath = math.sin(t)
            self.key_loc("hips", f, (0.006 * math.sin(t / 1), 0, 0.004 * math.sin(2 * t)))
            self.key_rot("hips", f, [(self.Z, math.radians(1.5) * math.sin(t))])
            self.key_rot("spine", f, [(self.X, math.radians(0.8) * breath)])
            self.key_rot("chest", f, [(self.X, math.radians(1.2) * breath)])
            self.key_rot("neck", f, [(self.X, math.radians(-1.0) * breath)])
            self.key_rot("head", f, [(self.Z, math.radians(3.0) * math.sin(t + 0.6)), (self.X, math.radians(1.5) * math.sin(2 * t))])
            self.relaxed_arms(f, sway=math.radians(2.0) * math.sin(t + 0.3), elbow=math.radians(10))
            for side, sign in (("left", 1.0), ("right", -1.0)):
                self.key_rot(f"{side}UpperLeg", f, [(self.Y, math.radians(1.0) * sign * math.sin(t))])

    def _gait(self, fps, period_s, hip_deg, knee_deg, arm_deg, elbow_deg, lean_deg, bob_m):
        n = int(round(period_s * fps))
        for f in range(0, n + 1):
            t = 2 * math.pi * f / n  # left thigh angle = hip_deg * sin(t); left mid-swing at t = 0
            # pelvis: over the stance leg at mid-stance, lowest at double support, swing-side hip forward and down
            self.key_loc("hips", f, (-0.012 * math.cos(t), 0, bob_m * math.cos(2 * t)))
            self.key_rot("hips", f, [(self.Z, math.radians(-5) * math.sin(t)), (self.Y, math.radians(3) * math.cos(t))])
            self.key_rot("spine", f, [(self.X, math.radians(lean_deg)), (self.Z, math.radians(2) * math.sin(t))])
            self.key_rot("chest", f, [(self.Z, math.radians(4) * math.sin(t))])
            self.key_rot("head", f, [(self.Z, math.radians(-1.5) * math.sin(t)), (self.X, math.radians(-lean_deg * 0.6))])
            for side, ph in (("left", 0.0), ("right", math.pi)):
                s = t + ph
                swing = math.radians(hip_deg) * math.sin(s)  # + = thigh forward
                knee = math.radians(5 + knee_deg * max(0.0, math.cos(s + 0.35)) ** 2)  # peak flexion in mid-swing
                ankle = math.radians(10) * math.sin(s - 0.5)
                self.key_rot(f"{side}UpperLeg", f, [(self.X, -swing)])
                self.key_rot(f"{side}LowerLeg", f, [(self.X, knee)])
                self.key_rot(f"{side}Foot", f, [(self.X, -ankle)])
            arm_swing = -math.radians(arm_deg) * math.sin(t)  # left arm forward while the right thigh is forward
            self.relaxed_arms(
                f,
                extra={"left": [(self.X, -arm_swing)], "right": [(self.X, arm_swing)]},
                elbow=math.radians(elbow_deg),
            )

    def walk(self, fps):
        self._gait(fps, 1.07, hip_deg=24, knee_deg=55, arm_deg=16, elbow_deg=14, lean_deg=2, bob_m=0.012)

    def run(self, fps):
        self._gait(fps, 0.7, hip_deg=38, knee_deg=90, arm_deg=34, elbow_deg=70, lean_deg=9, bob_m=0.03)

    def wave(self, fps):
        n = int(round(2.4 * fps))
        up_end, down_start = 0.25 * n, 0.75 * n
        for f in range(0, n + 1):
            up = sec._smoothstep(0, up_end, f) * (1 - sec._smoothstep(down_start, n, f))
            wave = math.sin(2 * math.pi * 3 * (f - up_end) / (down_start - up_end)) if up_end <= f <= down_start else 0.0
            # upper arm just above shoulder height, forearm upright beside the head, hand sweeping in the frontal plane
            # (+angle about Y lifts the right arm, which points to -X)
            raise_ang = (math.radians(100) - self.rest_abd["right"]) * up
            self.key_rot("rightUpperArm", f, [(self.Y, raise_ang), (self.X, math.radians(-10) * up)])
            self.key_rot("rightLowerArm", f, [(self.Y, (math.radians(80) + math.radians(22) * wave) * up)])
            self.key_rot("rightHand", f, [(self.Y, math.radians(15) * wave * up)])
            self.key_rot("spine", f, [(self.Y, math.radians(2.5) * up)])
            self.key_rot("head", f, [(self.Z, math.radians(-6) * up), (self.Y, math.radians(4) * wave * up)])
            self.key_rot("leftUpperArm", f, [(self.Y, self.rest_abd["left"] - math.radians(22))])
            self.key_rot("leftLowerArm", f, [(self.X, math.radians(-10))])

    def jump(self, fps):
        n = int(round(1.4 * fps))
        for f in range(0, n + 1):
            u = f / n
            crouch = math.sin(math.pi * min(1.0, u / 0.3)) if u < 0.3 else 0.0
            land = math.sin(math.pi * (u - 0.75) / 0.25) if u > 0.75 else 0.0
            air = math.sin(math.pi * (u - 0.3) / 0.45) if 0.3 <= u <= 0.75 else 0.0
            bend = max(crouch, land)
            self.key_loc("hips", f, (0, 0, -0.12 * bend + 0.30 * air))
            self.key_rot("spine", f, [(self.X, math.radians(12) * bend)])
            for side in ("left", "right"):
                self.key_rot(f"{side}UpperLeg", f, [(self.X, -math.radians(45) * bend - math.radians(10) * air)])
                self.key_rot(f"{side}LowerLeg", f, [(self.X, math.radians(90) * bend + math.radians(25) * air)])
                self.key_rot(f"{side}Foot", f, [(self.X, -math.radians(45) * bend + math.radians(20) * air)])
            arms_up = air
            for side, sign in (("left", 1.0), ("right", -1.0)):
                ang = (math.radians(22) - self.rest_abd[side]) * (1 - arms_up) + (math.radians(150) - self.rest_abd[side]) * arms_up
                self.key_rot(f"{side}UpperArm", f, [(self.Y, -ang * sign), (self.X, math.radians(30) * bend)])
                self.key_rot(f"{side}LowerArm", f, [(self.X, -math.radians(20))])


def run_procedural(args, action_names):
    if not args.mesh or not os.path.isfile(args.mesh):
        raise RuntimeError(f"Base mesh not found: {args.mesh}")
    clear_scene(args.fps)
    bpy.ops.import_scene.gltf(filepath=args.mesh)
    arm = find_armature()
    hum.mute_animation(arm)
    plan = json.loads(args.plan) if args.plan else [{"name": n} for n in action_names]
    anim = Animator(arm)
    meshes = skinned_meshes(arm)
    acts, clips = [], {}
    for item in plan:
        if item.get("clip"):
            trim = item.get("trim")
            act, clips[item["name"]] = rt.retarget_file(
                item["clip"], arm, meshes, item["name"], fps=args.fps, loop=bool(item.get("loop")),
                trim=tuple(trim) if trim else None, action=item.get("action"),
            )
        else:
            act = anim.make(item["name"], args.fps, item.get("kind"))
        act.use_fake_user = True  # unassigned between clips; keep it alive until the export
        acts.append(act)
    hum.rest_pose(arm)
    report = {"retarget": clips} if clips else {}
    if args.secondary == "auto":
        report["secondary"] = apply_secondary(arm, skinned_meshes(arm), acts, args)
    export_glb(args.out, arm, acts)
    return report


# ---------------------------------------------------------------- merge / rename (legacy modes)

def run_merge(args, action_names):
    clip_paths = [c.strip() for c in args.clips.split(",") if c.strip()]
    if not clip_paths and args.mesh:
        clip_paths = [args.mesh]
    if not clip_paths:
        raise RuntimeError("No animation clips provided for merge mode.")
    clear_scene(args.fps)
    base_file = args.mesh if (args.mesh and os.path.isfile(args.mesh)) else clip_paths[0]
    subsequent_clips = clip_paths if base_file == args.mesh else clip_paths[1:]
    bpy.ops.import_scene.gltf(filepath=base_file)
    arm_obj = find_armature()
    if not arm_obj.animation_data:
        arm_obj.animation_data_create()
    kept = []
    if bpy.data.actions:
        first_act = bpy.data.actions[0]
        if action_names:
            first_act.name = action_names[0]
        kept.append(first_act)
    for idx, clip_file in enumerate(subsequent_clips):
        name_idx = idx if base_file == args.mesh else idx + 1
        act_name = action_names[name_idx] if name_idx < len(action_names) else f"action_{name_idx}"
        kept += import_clip_actions(clip_file, [act_name])
    export_glb(args.out, arm_obj, kept)


def import_clip_actions(path, names):
    """Import a clip GLB, keep its actions (renamed to `names` in file order), delete its objects."""
    old_objs, old_acts = set(bpy.context.scene.objects), set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=path)
    new_acts = actions_in_file_order(path, [a for a in bpy.data.actions if a not in old_acts])
    for i, act in enumerate(new_acts):
        if i < len(names):
            act.name = names[i]
        act.use_fake_user = True
    for o in [o for o in bpy.context.scene.objects if o not in old_objs]:
        bpy.data.objects.remove(o, do_unlink=True)
    clean_orphaned_data()
    return new_acts[: max(1, len(names))] if names else new_acts


def run_rename(args, action_names):
    """Rename animation tracks inside a multi-action GLB (e.g. from Meshy action_ids) to match requested action_names."""
    src_file = args.mesh if (args.mesh and os.path.isfile(args.mesh)) else args.out
    if not src_file or not os.path.isfile(src_file):
        raise RuntimeError(f"Source GLB for rename not found: {src_file}")
    with open(src_file, "rb") as f:
        glb_bytes = f.read()
    if len(glb_bytes) < 20:
        raise RuntimeError("Invalid GLB: file too small")
    magic, version, _ = struct.unpack("<4sII", glb_bytes[:12])
    if magic != b"glTF" or version != 2:
        raise RuntimeError(f"Invalid GLB header: magic={magic!r}, version={version}")
    chunk0_len, chunk0_type = struct.unpack("<I4s", glb_bytes[12:20])
    if chunk0_type != b"JSON":
        raise RuntimeError(f"Expected JSON chunk 0, got {chunk0_type!r}")
    gltf_obj = json.loads(glb_bytes[20 : 20 + chunk0_len].decode("utf-8"))
    anims = gltf_obj.get("animations") or []
    if not anims:
        raise RuntimeError("GLB contains no animation tracks to rename.")
    gltf_obj["animations"] = [dict(anims[i] if i < len(anims) else anims[-1], name=n) for i, n in enumerate(action_names)]
    new_json_bytes = json.dumps(gltf_obj, ensure_ascii=False).encode("utf-8")
    new_json_bytes += b" " * ((4 - (len(new_json_bytes) % 4)) % 4)
    rest = glb_bytes[20 + chunk0_len :]
    total_len = 12 + 8 + len(new_json_bytes) + len(rest)
    new_glb = struct.pack("<4sII", b"glTF", 2, total_len) + struct.pack("<I4s", len(new_json_bytes), b"JSON") + new_json_bytes + rest
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(new_glb)


# ---------------------------------------------------------------- post-processing

def transfer_weights(arm, meshes, src_path):
    """Copy skin weights by rest position from a rig with the same skeleton (the repaired rig)."""
    old = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=src_path)
    new = [o for o in bpy.context.scene.objects if o not in old]
    try:
        src_arm = max((o for o in new if o.type == "ARMATURE"), key=lambda o: len(o.data.bones))
        src_meshes = [o for o in new if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)]
        if not src_meshes:
            return {"applied": False, "reason": "repaired rig has no skinned mesh"}
        for a in (arm, src_arm):
            hum.mute_animation(a)
            hum.rest_pose(a)
        src_co = hum.evaluated_coords(src_meshes)
        tgt_co = hum.evaluated_coords(meshes)
        zs = [v.z for v in tgt_co] or [0.0, 1.0]
        tol = 2e-3 * max(0.1, max(zs) - min(zs))
        kd = KDTree(len(src_co))
        for i, v in enumerate(src_co):
            kd.insert(v, i)
        kd.balance()
        match = [kd.find(v) for v in tgt_co]
        bad = sum(1 for _, _, d in match if d > tol)
        if bad > 0.01 * max(1, len(tgt_co)):
            return {"applied": False, "reason": f"{bad} of {len(tgt_co)} vertices have no twin in the repaired rig"}
        bones = {b.name for b in arm.data.bones}
        src_w, offset = [], 0
        for o in src_meshes:
            names = {vg.index: vg.name for vg in o.vertex_groups}
            for v in o.data.vertices:
                src_w.append([(names[g.group], g.weight) for g in v.groups if g.group in names and names[g.group] in bones])
            offset += len(o.data.vertices)
        k = 0
        for o in meshes:
            groups = {vg.name: vg for vg in o.vertex_groups}
            by_index = {vg.index: vg for vg in o.vertex_groups}
            for v in o.data.vertices:
                _, j, _ = match[k]
                k += 1
                for gi in [g.group for g in v.groups]:  # copy first: remove() reallocates v.groups
                    if by_index[gi].name in bones:
                        by_index[gi].remove([v.index])
                for name, w in src_w[j]:
                    vg = groups.get(name) or o.vertex_groups.new(name=name)
                    groups[name] = vg
                    vg.add([v.index], w, "REPLACE")
            o.data.update()
        return {"applied": True, "vertices": len(tgt_co), "max_match_m": round(max((d for _, _, d in match), default=0.0), 6)}
    finally:
        for o in new:
            bpy.data.objects.remove(o, do_unlink=True)
        clean_orphaned_data()


def _stretch_and_clip(arm, meshes, actions, edges, rest, shell=None, caps=None, frames_per=4):
    """(p99 edge stretch, share of skirt vertices inside leg capsules) over a few frames of every action."""
    sc = bpy.context.scene
    rest_len = [max((rest[a] - rest[b]).length, 1e-6) for a, b in edges]
    worst, clip_hits, clip_n = 1.0, 0, 0
    for act in actions:
        assign(arm, act)
        f0, f1 = act.frame_range
        for k in range(frames_per):
            sc.frame_set(int(round(f0 + (f1 - f0) * (k + 0.5) / frames_per)))
            co = hum.evaluated_coords(meshes)
            ratios = sorted((co[a] - co[b]).length / l0 for (a, b), l0 in zip(edges, rest_len))
            if ratios:
                worst = max(worst, ratios[int(0.99 * (len(ratios) - 1))])
            if shell and caps:
                segs = [(arm.matrix_world @ arm.pose.bones[b].head, arm.matrix_world @ arm.pose.bones[b].tail, r) for b, r in caps if b in arm.pose.bones]
                for i in shell:
                    p = co[i]
                    clip_n += 1
                    for a, b, r in segs:
                        ab = b - a
                        t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-12)))
                        if (p - (a + t * ab)).length < 0.8 * r:
                            clip_hits += 1
                            break
    return round(worst, 4), (round(clip_hits / clip_n, 4) if clip_n else None)


def apply_secondary(arm, meshes, actions, args):
    """Detect hair/skirt, add spring chains, bake, and keep them only if surface stretch does not get worse."""
    mapping = hum.map_roles(arm)
    if not mapping["ok"] or not meshes or not actions:
        return {"applied": False, "reason": "no humanoid skeleton" if not mapping["ok"] else "nothing to animate"}
    body = hum.Body(arm, meshes, mapping)
    edges = hum.mesh_edges(meshes)
    det = sec.detect(body, edges, skirt_chains=args.skirt_chains, max_hair_chains=args.max_hair_chains)
    if not det["chains"]:
        return {"applied": False, "reason": "no hanging hair or skirt found", "notes": det["notes"]}
    sample = edges if len(edges) <= 40000 else random.Random(3).sample(edges, 40000)
    skirt = next((c for c in det["chains"] if c["kind"] == "skirt"), None)
    shell = skirt["param"]["shell"][:4000] if skirt else None
    caps = sec._colliders(body, "skirt") if skirt else None
    s0, c0 = _stretch_and_clip(arm, meshes, actions, sample, body.rest, shell, caps)
    snap = hum.snapshot_weights(meshes)
    chains = sec.build(body, det["chains"])
    keyed = sec.bake(body, chains, actions, assign, args.fps, stiffness=args.stiffness, drag=args.drag, gravity=args.gravity)
    s1, c1 = _stretch_and_clip(arm, meshes, actions, sample, body.rest, shell, caps)
    info = {
        "chains": [{"kind": c["kind"], "bones": c["bones"]} for c in chains],
        "stretch_before": s0,
        "stretch_after": s1,
        "skirt_in_legs_before": c0,
        "skirt_in_legs_after": c1,
        "keys": keyed,
        "notes": det["notes"],
    }
    if s1 > s0 * 1.15 + 0.05:
        sec.remove(body, chains, snap, actions)
        return {"applied": False, "reason": f"surface stretch got worse ({s0} -> {s1}); chains removed", **info}
    return {"applied": True, **info}


def run_post(args, names):
    clear_scene(args.fps)
    bpy.ops.import_scene.gltf(filepath=args.mesh)
    arm = find_armature()
    base = actions_in_file_order(args.mesh, list(bpy.data.actions))
    clips = [c for c in args.clips.split(",") if c.strip()]
    report = {"changed": False}
    if len(names) < len(base) + len(clips):
        names = names + [f"action_{i}" for i in range(len(names), len(base) + len(clips))]
    for act, name in zip(base, names):
        if act.name != name:
            act.name = name
    acts = list(base[: len(names)])
    for path, name in zip(clips, names[len(base):]):
        acts += import_clip_actions(path, [name])
        report["changed"] = True
    meshes = skinned_meshes(arm)
    if args.weights_from and os.path.isfile(args.weights_from):
        report["weights"] = transfer_weights(arm, meshes, args.weights_from)
        report["changed"] |= bool(report["weights"].get("applied"))
    if args.secondary == "auto":
        report["secondary"] = apply_secondary(arm, meshes, acts, args)
        report["changed"] |= bool(report["secondary"].get("applied"))
    hum.rest_pose(arm)
    if report["changed"]:
        export_glb(args.out, arm, acts)
    report["actions"] = [a.name for a in acts]
    return report


def main():
    args = parse_args()
    action_names = [a.strip() for a in args.actions.split(",") if a.strip()] or ["idle", "walk", "wave"]
    try:
        report = {}
        if args.mode == "rename":
            run_rename(args, action_names)
        elif args.mode == "merge":
            run_merge(args, action_names)
        elif args.mode == "post":
            report = run_post(args, action_names)
            action_names = report.pop("actions", action_names)
        else:
            report = run_procedural(args, action_names)
        res = {
            "ok": True,
            "actions": action_names,
            "clip_count": len(action_names),
            "mesh_path": args.out,
            "changed": report.pop("changed", True),
            **report,
        }
    except Exception as e:
        import traceback

        res = {
            "ok": False,
            "code": "ANIM_FAILED",
            "message": f"{type(e).__name__}: {e}",
            "fix": getattr(e, "fix", "Check mesh armature or animation clips."),
            "trace": traceback.format_exc()[-1200:],
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.result)), exist_ok=True)
    with open(args.result, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
