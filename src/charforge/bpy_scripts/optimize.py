"""Mesh cleanup, decimation, and UV projection script. Runs INSIDE Blender: bpy/mathutils/stdlib only.

Operations:
1. Merge vertices by distance (1e-4)
2. Delete loose geometry
3. Recalculate normals (make consistent)
4. Decimate (collapse) to target triangle count
5. Smart UV project if mesh has no UV layer
6. Export cleaned GLB to output path
Result JSON: {ok: true, mesh: path, tris_before: int, tris_after: int}
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-tris", type=int, default=30000)
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
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        meshes = load_mesh(args.mesh)

        # Join into single object
        bpy.ops.object.select_all(action='DESELECT')
        for m in meshes:
            m.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        if len(meshes) > 1:
            bpy.ops.object.join()
        mesh_obj = bpy.context.view_layer.objects.active

        tris_before = count_tris([mesh_obj])

        # 1. Edit mode operations: merge by distance, delete loose, fix normals
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=1e-4)
        bpy.ops.mesh.delete_loose()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')

        # 2. Decimate to target_tris
        cur_tris = count_tris([mesh_obj])
        if cur_tris > args.target_tris:
            ratio = max(0.001, min(1.0, args.target_tris / cur_tris))
            mod = mesh_obj.modifiers.new(name="Decimate", type='DECIMATE')
            mod.decimate_type = 'COLLAPSE'
            mod.ratio = ratio
            bpy.ops.object.modifier_apply(modifier="Decimate")

        # 3. Smart UV project if missing UVs
        if not mesh_obj.data.uv_layers:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
            bpy.ops.object.mode_set(mode='OBJECT')

        tris_after = count_tris([mesh_obj])

        # Export result
        bpy.ops.export_scene.gltf(filepath=args.out, export_format='GLB')
        res = {"ok": True, "mesh": args.out, "tris_before": tris_before, "tris_after": tris_after}
    except Exception as e:
        res = {"ok": False, "code": "BLENDER_FAILED", "message": f"{type(e).__name__}: {e}", "fix": "Check mesh geometry or retry with different target_tris."}

    with open(args.result, "w") as f:
        json.dump(res, f)


main()
