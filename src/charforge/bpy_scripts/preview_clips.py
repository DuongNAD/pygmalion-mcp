"""Render a few frames of every animation clip from the front and side, and measure motion quality.
Runs INSIDE Blender: bpy/mathutils/stdlib only (+ the sibling cf_humanoid module).

Per clip: N frames evenly spread over the action (loop end excluded), orthographic Workbench renders with a
camera framed on the whole clip, plus metrics on the deformed mesh at those frames:
  stretch_p99 / stretch_max  edge length ratio posed/rest (bad weights tear or balloon the surface)
  ground_cm                  how far the lowest vertex sinks below the rest ground (feet through the floor)
  travel_m                   hips displacement over the clip (root motion vs in-place)

Protocol: blender -b --factory-startup --python-exit-code 1 -P preview_clips.py -- --result <json> --mesh <animated.glb>
          --out <dir> [--frames 6] [--res 320] [--views front,side] [--clips a,b]
Result JSON: {ok, fps, clips: [{name, frame_start, frame_end, frames, files: {view: [png...]}, metrics}]}
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import cf_humanoid as hum  # noqa: E402

CAMS = {"front": ((0, -1, 0), (90, 0, 0)), "side": ((1, 0, 0), (90, 0, 90)), "back": ((0, 1, 0), (90, 0, 180))}
MAX_EDGES = 60000


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--views", default="front,side")
    ap.add_argument("--clips", default="")
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:])


def safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60] or "clip"


def assign(arm, act):
    """Evaluate `act` alone, from the rest pose (bones it does not key must not keep the previous clip's pose)."""
    ad = arm.animation_data or arm.animation_data_create()
    for t in ad.nla_tracks:
        t.mute = True
    hum.rest_pose(arm)
    ad.action = act
    if hasattr(ad, "action_slot") and getattr(ad, "action_slot", None) is None and len(getattr(act, "slots", [])):
        ad.action_slot = act.slots[0]


def setup_render(res, meshes):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"
    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.show_cavity = sh.show_specular_highlight = False
    sh.show_object_outline = True
    mats = [s.material for o in meshes for s in o.material_slots if s.material and s.material.node_tree]
    items = {i.identifier for i in bpy.types.View3DShading.bl_rna.properties["color_type"].enum_items}
    if any(n.type == "TEX_IMAGE" for m in mats for n in m.node_tree.nodes):
        sh.color_type = "TEXTURE"
    elif any(len(o.data.color_attributes) for o in meshes):
        sh.color_type = "VERTEX" if "VERTEX" in items else "ATTRIBUTE"
    else:
        sh.color_type = "MATERIAL"
    cam = bpy.data.objects.new("cf_cam", bpy.data.cameras.new("cf_cam"))
    sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    sc.camera = cam
    return cam


def frame_cam(cam, view, lo, hi):
    center, size = (lo + hi) / 2, hi - lo
    direction, rot = CAMS[view]
    dist = size.length * 2 + 1
    cam.location = (center.x + direction[0] * dist, center.y + direction[1] * dist, center.z)
    cam.rotation_euler = [math.radians(a) for a in rot]
    cam.data.ortho_scale = max(size.y if view == "side" else size.x, size.z) * 1.1
    cam.data.clip_end = dist * 4


def edge_sample(edges):
    if len(edges) <= MAX_EDGES:
        return edges
    rnd = random.Random(7)
    return rnd.sample(edges, MAX_EDGES)


def main():
    args = parse()
    try:
        hum.import_any(args.mesh)
        arm, meshes = hum.find_rig()
        for o in hum.bone_shape_objects():
            o.hide_render = True
        acts = list(bpy.data.actions)
        wanted = [c for c in args.clips.split(",") if c]
        if wanted:
            acts = [a for a in acts if a.name in wanted]
        if not acts:
            raise RuntimeError("The file has no animation clips to preview.")
        sc = bpy.context.scene
        fps = sc.render.fps / max(sc.render.fps_base, 1e-6)

        hum.mute_animation(arm)
        hum.rest_pose(arm)
        rest = hum.evaluated_coords(meshes)
        rest_ground = min(v.z for v in rest)
        edges = edge_sample(hum.mesh_edges(meshes))
        rest_len = [max((rest[a] - rest[b]).length, 1e-6) for a, b in edges]
        hips_bone = (hum.map_roles(arm)["roles"] or {}).get("hips")

        cam = setup_render(args.res, meshes)
        views = [v for v in args.views.split(",") if v in CAMS]
        clips = []
        for act in acts:
            f0, f1 = (float(x) for x in act.frame_range)
            n = max(1, args.frames)
            span = max(f1 - f0, 1.0)
            frames = [round(f0 + span * k / n, 2) for k in range(n)] if f1 > f0 else [f0]
            assign(arm, act)
            posed, stretch_p99, stretch_max, ground, hips = [], 0.0, 0.0, 0.0, []
            for f in frames:
                sc.frame_set(int(math.floor(f)), subframe=f - math.floor(f))
                co = hum.evaluated_coords(meshes)
                posed.append(co)
                ratios = sorted((co[a] - co[b]).length / l0 for (a, b), l0 in zip(edges, rest_len))
                if ratios:
                    stretch_p99 = max(stretch_p99, ratios[int(0.99 * (len(ratios) - 1))])
                    stretch_max = max(stretch_max, ratios[-1])
                ground = max(ground, rest_ground - min(v.z for v in co))
                if hips_bone and hips_bone in arm.pose.bones:
                    hips.append(arm.matrix_world @ arm.pose.bones[hips_bone].head)
            lo = Vector([min(v[i] for co in posed for v in co) for i in range(3)])
            hi = Vector([max(v[i] for co in posed for v in co) for i in range(3)])
            name = safe(act.name)
            files = {v: [] for v in views}
            os.makedirs(os.path.join(args.out, name), exist_ok=True)
            for view in views:
                frame_cam(cam, view, lo, hi)
                for k, f in enumerate(frames):
                    sc.frame_set(int(math.floor(f)), subframe=f - math.floor(f))
                    path = os.path.join(args.out, name, f"{view}_{k:02d}.png")
                    sc.render.filepath = path
                    bpy.ops.render.render(write_still=True)
                    files[view].append(path)
            travel = max(((h - hips[0]).length for h in hips), default=0.0)
            clips.append({
                "name": act.name,
                "frame_start": f0,
                "frame_end": f1,
                "seconds": round((f1 - f0) / fps, 3) if fps else None,
                "frames": frames,
                "files": files,
                "metrics": {
                    "stretch_p99": round(stretch_p99, 3),
                    "stretch_max": round(stretch_max, 3),
                    "ground_cm": round(100 * ground, 2),
                    "travel_m": round(travel, 3),
                },
            })
        res = {"ok": True, "fps": fps, "clips": clips}
    except Exception as e:
        res = {"ok": False, "code": "ANIM_FAILED", "message": f"{type(e).__name__}: {e}", "fix": "Previews failed; the animated GLB may be corrupt: rerun animate."}
    with open(args.result, "w", encoding="utf-8") as f:
        json.dump(res, f)


main()
