"""Model export script for final GLB and FBX deliverables. Runs INSIDE Blender: bpy/mathutils/stdlib only.

Operations:
1. Apply all transforms (location, rotation, scale)
2. Set metric units (meters)
3. Export GLB (embedded textures) and/or FBX
Result JSON: {ok: true, files: {format: abs_path}, size_mb: float, tris: int}
"""
import argparse
import json
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


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--format", choices=["glb", "fbx", "both"], default="glb")
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:])


def load_mesh(path):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    ext = os.path.splitext(path)[1].lower()
    IMPORTERS[ext](path)
    bpy.context.view_layer.update()

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("File contains no mesh objects")
    return meshes


def count_tris(meshes):
    dg = bpy.context.evaluated_depsgraph_get()
    tris = 0
    for obj in meshes:
        ev = obj.evaluated_get(dg)
        tris += sum(len(poly.vertices) - 2 for poly in ev.data.polygons)
    return tris


def main():
    args = parse()
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        meshes = load_mesh(args.mesh)

        # Apply all transforms
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Set scene units to METRIC
        sc = bpy.context.scene
        sc.unit_settings.system = 'METRIC'
        sc.unit_settings.scale_length = 1.0

        tris = count_tris(meshes)
        files = {}
        total_bytes = 0

        # Export GLB
        if args.format in ("glb", "both"):
            glb_path = os.path.join(args.out_dir, f"{args.stem}.glb")
            bpy.ops.export_scene.gltf(filepath=glb_path, export_format='GLB', export_apply=True)
            files["glb"] = glb_path
            total_bytes += os.path.getsize(glb_path)

        # Export FBX
        if args.format in ("fbx", "both"):
            fbx_path = os.path.join(args.out_dir, f"{args.stem}.fbx")
            try:
                bpy.ops.wm.fbx_export(filepath=fbx_path, apply_unit_scale=True, apply_scale_options='FBX_SCALE_ALL')
            except AttributeError:
                bpy.ops.export_scene.fbx(filepath=fbx_path, apply_unit_scale=True, apply_scale_options='FBX_SCALE_ALL')
            files["fbx"] = fbx_path
            total_bytes += os.path.getsize(fbx_path)

        size_mb = round(total_bytes / (1024 * 1024), 2)
        res = {"ok": True, "files": files, "size_mb": size_mb, "tris": tris}
    except Exception as e:
        res = {"ok": False, "code": "BLENDER_FAILED", "message": f"{type(e).__name__}: {e}", "fix": "Export failed: verify file paths and permissions."}

    with open(args.result, "w") as f:
        json.dump(res, f)


main()
