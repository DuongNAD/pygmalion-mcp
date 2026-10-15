"""Profile-guided lattice deformation script. Runs INSIDE Blender: bpy/mathutils/stdlib only.

Applies:
1. rotate_z
2. Feet grounding at Z=0, XY centering at (0, 0), uniform height scaling to height_m
3. 2x2x16 Lattice deformation per height band
4. Exports deformed mesh to output .glb
Result JSON: {ok: true, mesh: path}
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

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
    ap.add_argument("--height-m", type=float, required=True)
    ap.add_argument("--rotate-z", type=float, default=0.0)
    ap.add_argument("--scale-x", required=True)
    ap.add_argument("--scale-y", required=True)
    ap.add_argument("--damp-arms", action="store_true", default=False)
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:])


def load_mesh(path, rotate_z):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    ext = os.path.splitext(path)[1].lower()
    IMPORTERS[ext](path)

    rot = Matrix.Rotation(math.radians(rotate_z), 4, "Z")
    for obj in bpy.context.scene.objects:
        if obj.parent is None:
            obj.matrix_world = rot @ obj.matrix_world
    bpy.context.view_layer.update()

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("File contains no mesh objects")
    return meshes


def bounds(meshes):
    dg = bpy.context.evaluated_depsgraph_get()
    pts, tris = [], 0
    for obj in meshes:
        ev = obj.evaluated_get(dg)
        pts += [ev.matrix_world @ Vector(c) for c in ev.bound_box]
        tris += sum(len(poly.vertices) - 2 for poly in ev.data.polygons)
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    return lo, hi, tris


def main():
    args = parse()
    try:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        meshes = load_mesh(args.mesh, args.rotate_z)

        # Join into single object if needed
        bpy.ops.object.select_all(action='DESELECT')
        for m in meshes:
            m.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        if len(meshes) > 1:
            bpy.ops.object.join()
        mesh_obj = bpy.context.view_layer.objects.active

        # Grounding, centering, and height scaling
        lo, hi, _ = bounds([mesh_obj])
        cur_h = hi.z - lo.z
        if cur_h <= 0:
            cur_h = 1.0
        scale = args.height_m / cur_h

        # Center XY and ground feet at Z=0
        mesh_obj.location.x -= (lo.x + hi.x) / 2.0
        mesh_obj.location.y -= (lo.y + hi.y) / 2.0
        mesh_obj.location.z -= lo.z
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)

        mesh_obj.scale = (scale, scale, scale)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # Create 2x2x16 Lattice
        lo, hi, _ = bounds([mesh_obj])
        wx = max(hi.x - lo.x, 0.05)
        wy = max(hi.y - lo.y, 0.05)

        lat_data = bpy.data.lattices.new("FitLattice")
        lat_data.points_u = 2
        lat_data.points_v = 2
        lat_data.points_w = 16
        lat_data.interpolation_type_u = 'KEY_LINEAR'
        lat_data.interpolation_type_v = 'KEY_LINEAR'
        lat_data.interpolation_type_w = 'KEY_LINEAR'

        lat_obj = bpy.data.objects.new("FitLattice", lat_data)
        bpy.context.scene.collection.objects.link(lat_obj)

        lat_obj.location = (0, 0, args.height_m / 2.0)
        lat_obj.scale = (wx * 1.08, wy * 1.08, args.height_m / (lat_data.points_w - 1))
        bpy.context.view_layer.update()

        # Add lattice modifier to mesh
        mod = mesh_obj.modifiers.new(name="Lattice", type='LATTICE')
        mod.object = lat_obj

        # Deform control points
        sx = [float(x) for x in args.scale_x.split(",")]
        sy = [float(y) for y in args.scale_y.split(",")]

        if getattr(args, "damp_arms", False):
            # Arm height bands: w in [7, 13]
            for w in range(7, 14):
                if w < len(sx):
                    sx[w] = 1.0 + (sx[w] - 1.0) * 0.20

        # In Blender lattice: w=0 is Z bottom (-0.5), w=15 is Z top (+0.5)
        for i, pt in enumerate(lat_data.points):
            w = i // 4
            if w < len(sx) and w < len(sy):
                pt.co_deform.x *= sx[w]
                pt.co_deform.y *= sy[w]

        # Apply modifier
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.modifier_apply(modifier="Lattice")
        bpy.data.objects.remove(lat_obj, do_unlink=True)

        # Export result
        bpy.ops.export_scene.gltf(filepath=args.out, export_format='GLB')
        res = {"ok": True, "mesh": args.out}
    except Exception as e:
        res = {"ok": False, "code": "FIT_FAILED", "message": f"{type(e).__name__}: {e}", "fix": "Check mesh geometry or retry with different candidate."}

    with open(args.result, "w") as f:
        json.dump(res, f)


main()
