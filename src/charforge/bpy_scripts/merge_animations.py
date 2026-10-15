"""Multi-action animation generation and GLB merging script.
Runs INSIDE Blender: bpy/mathutils/stdlib only.

Protocol:
blender -b --factory-startup --python-exit-code 1 -P merge_animations.py -- --result <result_json> [--mesh <path>] [--out <path>] [--actions <comma_separated>] [--mode <procedural|merge>] [--clips <comma_separated>]
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector


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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--actions", default="idle,walk,wave")
    ap.add_argument("--mode", default="procedural", choices=["procedural", "merge"])
    ap.add_argument("--clips", default="")
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:])


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


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def find_armature():
    armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature object found in scene.")
    return armatures[0]


def clean_orphaned_data():
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)
    for a in list(bpy.data.armatures):
        if a.users == 0:
            bpy.data.armatures.remove(a)


def generate_procedural_action(arm_obj, action_name):
    """Generate genuine keyframe curves on Mixamo bones for requested action."""
    act = bpy.data.actions.new(name=action_name)
    if not arm_obj.animation_data:
        arm_obj.animation_data_create()
    arm_obj.animation_data.action = act

    act_key = action_name.lower()

    if act_key.startswith("idle"):
        # idle: 60 frames (30 fps), subtle cyclic vertical breathing translation on Hips (+-1.2 cm)
        # and gentle Spine / Neck rotational sway
        for f in range(0, 61, 2):
            t = 2.0 * math.pi * f / 60.0

            pb_hips = get_pose_bone(arm_obj, "Hips")
            if pb_hips:
                pb_hips.location = (0.0, 0.0, 0.012 * math.sin(t))
                pb_hips.keyframe_insert(data_path="location", frame=f)

            pb_spine = get_pose_bone(arm_obj, "Spine1") or get_pose_bone(arm_obj, "Spine")
            if pb_spine:
                pb_spine.rotation_mode = "XYZ"
                pb_spine.rotation_euler = (0.02 * math.sin(t), 0.0, 0.0)
                pb_spine.keyframe_insert(data_path="rotation_euler", frame=f)

            pb_neck = get_pose_bone(arm_obj, "Neck")
            if pb_neck:
                pb_neck.rotation_mode = "XYZ"
                pb_neck.rotation_euler = (0.015 * math.sin(t), 0.0, 0.0)
                pb_neck.keyframe_insert(data_path="rotation_euler", frame=f)

    elif act_key.startswith("walk"):
        # walk: 40 frames (30 fps), cyclic 180° out-of-phase leg swing on LeftUpLeg / RightUpLeg,
        # knee flexion on LeftLeg / RightLeg, counter-swings on LeftArm / RightArm, gait bounce on Hips
        for f in range(0, 41):
            t = 2.0 * math.pi * f / 40.0

            # 180 deg out-of-phase leg swing (+-0.35 rad)
            pb_l_leg = get_pose_bone(arm_obj, "LeftUpLeg")
            pb_r_leg = get_pose_bone(arm_obj, "RightUpLeg")
            if pb_l_leg:
                pb_l_leg.rotation_mode = "XYZ"
                pb_l_leg.rotation_euler = (0.35 * math.sin(t), 0.0, 0.0)
                pb_l_leg.keyframe_insert(data_path="rotation_euler", frame=f)
            if pb_r_leg:
                pb_r_leg.rotation_mode = "XYZ"
                pb_r_leg.rotation_euler = (-0.35 * math.sin(t), 0.0, 0.0)
                pb_r_leg.keyframe_insert(data_path="rotation_euler", frame=f)

            # Knee flexion during backswing (+0.45 rad)
            pb_l_knee = get_pose_bone(arm_obj, "LeftLeg")
            pb_r_knee = get_pose_bone(arm_obj, "RightLeg")
            if pb_l_knee:
                pb_l_knee.rotation_mode = "XYZ"
                pb_l_knee.rotation_euler = (0.45 * max(0.0, -math.sin(t)), 0.0, 0.0)
                pb_l_knee.keyframe_insert(data_path="rotation_euler", frame=f)
            if pb_r_knee:
                pb_r_knee.rotation_mode = "XYZ"
                pb_r_knee.rotation_euler = (0.45 * max(0.0, math.sin(t)), 0.0, 0.0)
                pb_r_knee.keyframe_insert(data_path="rotation_euler", frame=f)

            # Counter-swings on LeftArm / RightArm
            pb_l_arm = get_pose_bone(arm_obj, "LeftArm")
            pb_r_arm = get_pose_bone(arm_obj, "RightArm")
            if pb_l_arm:
                pb_l_arm.rotation_mode = "XYZ"
                pb_l_arm.rotation_euler = (-0.30 * math.sin(t), 0.0, 0.0)
                pb_l_arm.keyframe_insert(data_path="rotation_euler", frame=f)
            if pb_r_arm:
                pb_r_arm.rotation_mode = "XYZ"
                pb_r_arm.rotation_euler = (0.30 * math.sin(t), 0.0, 0.0)
                pb_r_arm.keyframe_insert(data_path="rotation_euler", frame=f)

            # Gait bounce on Hips (2 cycles per 40 frames)
            pb_hips = get_pose_bone(arm_obj, "Hips")
            if pb_hips:
                pb_hips.location = (0.0, 0.0, -0.015 * abs(math.sin(t)))
                pb_hips.keyframe_insert(data_path="location", frame=f)

    elif act_key.startswith("wave"):
        # wave: 60 frames (30 fps), raising RightArm at frame 15,
        # sinusoidal waving oscillation on RightForeArm / RightHand (+-0.4 rad) between frames 15-45,
        # returning to rest by frame 60
        for f in range(0, 61):
            pb_r_arm = get_pose_bone(arm_obj, "RightArm")
            pb_r_forearm = get_pose_bone(arm_obj, "RightForeArm")
            pb_r_hand = get_pose_bone(arm_obj, "RightHand")

            # Arm raise: 0 at f=0, reaches 1.2 rad at f=15, stays at 1.2 until f=45, returns to 0 by f=60
            if f <= 15:
                arm_rot = 1.2 * (f / 15.0)
            elif f <= 45:
                arm_rot = 1.2
            else:
                arm_rot = 1.2 * (60.0 - f) / 15.0

            if pb_r_arm:
                pb_r_arm.rotation_mode = "XYZ"
                pb_r_arm.rotation_euler = (0.0, 0.0, arm_rot)
                pb_r_arm.keyframe_insert(data_path="rotation_euler", frame=f)

            # Oscillating wave between frames 15 and 45
            if 15 <= f <= 45:
                wave_t = 2.0 * math.pi * (f - 15) / 10.0  # 3 complete cycles
                osc = 0.4 * math.sin(wave_t)
            else:
                osc = 0.0

            if pb_r_forearm:
                pb_r_forearm.rotation_mode = "XYZ"
                pb_r_forearm.rotation_euler = (0.0, osc, 0.0)
                pb_r_forearm.keyframe_insert(data_path="rotation_euler", frame=f)
            if pb_r_hand:
                pb_r_hand.rotation_mode = "XYZ"
                pb_r_hand.rotation_euler = (0.0, osc * 0.8, 0.0)
                pb_r_hand.keyframe_insert(data_path="rotation_euler", frame=f)

    else:
        # Generic fallback action: 60 frames of rhythmic limb and torso motion
        for f in range(0, 61, 2):
            t = 2.0 * math.pi * f / 60.0
            pb_hips = get_pose_bone(arm_obj, "Hips")
            if pb_hips:
                pb_hips.location = (0.0, 0.0, 0.010 * math.sin(t))
                pb_hips.keyframe_insert(data_path="location", frame=f)

            pb_spine = get_pose_bone(arm_obj, "Spine")
            if pb_spine:
                pb_spine.rotation_mode = "XYZ"
                pb_spine.rotation_euler = (0.03 * math.sin(t), 0.0, 0.0)
                pb_spine.keyframe_insert(data_path="rotation_euler", frame=f)

            pb_l_arm = get_pose_bone(arm_obj, "LeftArm")
            pb_r_arm = get_pose_bone(arm_obj, "RightArm")
            if pb_l_arm:
                pb_l_arm.rotation_mode = "XYZ"
                pb_l_arm.rotation_euler = (0.2 * math.sin(t), 0.0, 0.0)
                pb_l_arm.keyframe_insert(data_path="rotation_euler", frame=f)
            if pb_r_arm:
                pb_r_arm.rotation_mode = "XYZ"
                pb_r_arm.rotation_euler = (-0.2 * math.sin(t), 0.0, 0.0)
                pb_r_arm.keyframe_insert(data_path="rotation_euler", frame=f)

    # Push to NLA track on armature and assign fake user
    track = arm_obj.animation_data.nla_tracks.new()
    track.name = act.name
    track.strips.new(act.name, int(act.frame_range[0]), act)
    act.use_fake_user = True
    return act


def run_procedural(args, action_names):
    if not args.mesh or not os.path.isfile(args.mesh):
        raise RuntimeError(f"Base mesh not found: {args.mesh}")

    clear_scene()
    bpy.ops.import_scene.gltf(filepath=args.mesh)
    arm_obj = find_armature()

    # Clear any active action
    if arm_obj.animation_data:
        arm_obj.animation_data.action = None

    for act_name in action_names:
        generate_procedural_action(arm_obj, act_name)

    # Ensure no single action is active so all stashed/NLA actions export
    arm_obj.animation_data.action = None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=args.out,
        export_format="GLB",
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_current_frame=False,
    )

    if not os.path.isfile(args.out) or os.path.getsize(args.out) == 0:
        raise RuntimeError(f"Exported animated GLB is missing or empty: {args.out}")


def run_merge(args, action_names):
    clip_paths = [c.strip() for c in args.clips.split(",") if c.strip()]
    if not clip_paths and args.mesh:
        clip_paths = [args.mesh]

    if not clip_paths:
        raise RuntimeError("No animation clips provided for merge mode.")

    clear_scene()

    # Import base model (first clip or explicit base mesh)
    base_file = args.mesh if (args.mesh and os.path.isfile(args.mesh)) else clip_paths[0]
    subsequent_clips = clip_paths if base_file == args.mesh else clip_paths[1:]

    bpy.ops.import_scene.gltf(filepath=base_file)
    arm_obj = find_armature()

    if not arm_obj.animation_data:
        arm_obj.animation_data_create()

    # If base file has an imported action, rename it to first action name
    if bpy.data.actions:
        first_act = bpy.data.actions[0]
        if action_names:
            first_act.name = action_names[0]
        first_act.use_fake_user = True
        track = arm_obj.animation_data.nla_tracks.new()
        track.name = first_act.name
        track.strips.new(first_act.name, int(first_act.frame_range[0]), first_act)

    # Import each subsequent clip
    for idx, clip_file in enumerate(subsequent_clips):
        # Determine intended action name
        name_idx = idx if base_file == args.mesh else idx + 1
        act_name = action_names[name_idx] if name_idx < len(action_names) else f"action_{name_idx}"

        old_objs = set(bpy.context.scene.objects)
        old_acts = set(bpy.data.actions)

        bpy.ops.import_scene.gltf(filepath=clip_file)

        new_acts = [a for a in bpy.data.actions if a not in old_acts]
        for new_act in new_acts:
            new_act.name = act_name
            new_act.use_fake_user = True
            track = arm_obj.animation_data.nla_tracks.new()
            track.name = act_name
            track.strips.new(act_name, int(new_act.frame_range[0]), new_act)

        # Cleanup imported duplicate objects
        new_objs = [o for o in bpy.context.scene.objects if o not in old_objs]
        for o in new_objs:
            bpy.data.objects.remove(o, do_unlink=True)

        clean_orphaned_data()

    arm_obj.animation_data.action = None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=args.out,
        export_format="GLB",
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_current_frame=False,
    )

    if not os.path.isfile(args.out) or os.path.getsize(args.out) == 0:
        raise RuntimeError(f"Exported merged GLB is missing or empty: {args.out}")


def main():
    args = parse_args()
    action_names = [a.strip() for a in args.actions.split(",") if a.strip()]
    if not action_names:
        action_names = ["idle", "walk", "wave"]

    try:
        if args.mode == "merge":
            run_merge(args, action_names)
        else:
            run_procedural(args, action_names)

        res = {
            "ok": True,
            "actions": action_names,
            "clip_count": len(action_names),
            "mesh_path": args.out,
        }
    except Exception as e:
        res = {
            "ok": False,
            "code": "ANIM_FAILED",
            "message": f"{type(e).__name__}: {e}",
            "fix": "Check mesh armature or animation clips.",
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.result)), exist_ok=True)
    with open(args.result, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
