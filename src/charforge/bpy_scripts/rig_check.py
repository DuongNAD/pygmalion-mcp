"""Deformation check for any humanoid rig (provider or local), optional weight repair, humanoid map.
Runs INSIDE Blender: bpy/mathutils/stdlib only (+ the sibling cf_humanoid module).

pose_deformation_check: swing each upper arm 40 deg forward and up, swing each thigh 40 deg forward, and measure
whether parts that must stay still (thighs, torso, the other limbs) are dragged along. Body parts come from bone
segments, never from the weights under test. A failed check triggers a weight repair that is kept only if the
re-check improves; the repaired rig is then exported to --out with the same skeleton.

Protocol: blender -b --factory-startup --python-exit-code 1 -P rig_check.py -- --result <json> --mesh <rigged.glb>
          [--repair --out <fixed.glb>] [--humanoid <map.json>] [--angle 40] [--tol-m 0.005] [--max-ratio 0.01]
          [--hard-m 0.03] [--follow-min 0.6] [--gap-min-m 0.01]
Result JSON: {ok, bones, mapping, passed, issues, contact_issues, contacts, legs_apart, before, after,
              repaired, repair, unweighted_ratio, out, humanoid}
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402

import cf_humanoid as hum  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--repair", action="store_true", default=False)
    ap.add_argument("--humanoid", default=None)
    ap.add_argument("--angle", type=float, default=40.0)
    ap.add_argument("--tol-m", type=float, default=0.005)
    ap.add_argument("--max-ratio", type=float, default=0.01)
    ap.add_argument("--hard-m", type=float, default=0.03)
    ap.add_argument("--follow-min", type=float, default=0.6)
    ap.add_argument("--gap-min-m", type=float, default=0.01)
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:])


def export_rig(path):
    for o in hum.bone_shape_objects():
        bpy.data.objects.remove(o, do_unlink=True)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", export_skins=True, export_animations=False, export_apply=False)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"Repaired rig export is missing or empty: {path}")


def main():
    args = parse()
    try:
        hum.import_any(args.mesh)
        arm, meshes = hum.find_rig()
        mapping = hum.map_roles(arm)
        res = {
            "ok": True,
            "bones": len(arm.data.bones),
            "mapping": {k: mapping[k] for k in ("ok", "roles", "missing", "notes", "source")},
            "out": None,
            "humanoid": None,
        }
        if args.humanoid:
            os.makedirs(os.path.dirname(os.path.abspath(args.humanoid)), exist_ok=True)
            with open(args.humanoid, "w", encoding="utf-8") as f:
                json.dump(hum.humanoid_map(arm, mapping), f, indent=2)
            res["humanoid"] = args.humanoid
        qa = hum.run_qa(
            arm, meshes, mapping, angle=args.angle, tol_m=args.tol_m, max_ratio=args.max_ratio, hard_m=args.hard_m,
            follow_min=args.follow_min, gap_min_m=args.gap_min_m, repair=bool(args.repair and args.out),
        )
        res.update(qa)
        if qa["repaired"]:
            export_rig(args.out)
            res["out"] = args.out
    except Exception as e:
        res = {
            "ok": False,
            "code": "RIG_FAILED",
            "message": f"{type(e).__name__}: {e}",
            "fix": "The rigged file could not be checked; use auto_rig(method='blender') or another candidate mesh.",
        }
    with open(args.result, "w", encoding="utf-8") as f:
        json.dump(res, f)


main()
