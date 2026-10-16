"""Render orthographic front/side/back views of a mesh. Runs INSIDE Blender: bpy/mathutils/stdlib only.

Conventions: Z-up, meters, character faces -Y (Blender's glTF import default).
  front: camera at -Y looking +Y | side: at +X looking -X | back: at +Y looking -Y
mode=mask  -> Workbench, transparent film; the mask is the PNG alpha channel.
mode=color -> Workbench flat light + texture/attribute colors (~albedo, comparable to flat anime art).
Result JSON: {ok, files: {view: abs path}, dims_m: [x, y, z], tris, engine}
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

CAMS = {"front": ((0, -1, 0), (90, 0, 0)), "side": ((1, 0, 0), (90, 0, 90)), "back": ((0, 1, 0), (90, 0, 180))}


def _fbx(path):
    try:
        bpy.ops.wm.fbx_import(filepath=path)  # C++ importer (Blender 4.5+)
    except AttributeError:
        bpy.ops.import_scene.fbx(filepath=path)


IMPORTERS = {
    ".glb": lambda f: bpy.ops.import_scene.gltf(filepath=f),
    ".fbx": _fbx,
    ".obj": lambda f: bpy.ops.wm.obj_import(filepath=f),
    ".ply": lambda f: bpy.ops.wm.ply_import(filepath=f),
    ".stl": lambda f: bpy.ops.wm.stl_import(filepath=f),
}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="front,side,back")
    ap.add_argument("--mode", choices=["mask", "color"], default="mask")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--rotate-z", type=float, default=0.0)
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:])


def load_mesh(path, rotate_z):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    IMPORTERS[os.path.splitext(path)[1].lower()](path)
    rot = Matrix.Rotation(math.radians(rotate_z), 4, "Z")
    for obj in bpy.context.scene.objects:
        if obj.parent is None:
            obj.matrix_world = rot @ obj.matrix_world
    bpy.context.view_layer.update()
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("file contains no mesh objects")
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


def setup(mode, res, meshes):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"
    sh = sc.display.shading
    sh.light = "FLAT"
    sh.show_cavity = sh.show_object_outline = sh.show_specular_highlight = False
    if mode == "mask":
        sh.color_type, sh.single_color = "SINGLE", (1.0, 1.0, 1.0)
        return
    mats = [s.material for o in meshes for s in o.material_slots if s.material and s.material.node_tree]
    if any(n.type == "TEX_IMAGE" for m in mats for n in m.node_tree.nodes):
        sh.color_type = "TEXTURE"
    elif any(len(o.data.color_attributes) for o in meshes):
        sh.color_type = attribute_color_type()
    else:
        sh.color_type = "MATERIAL"


def attribute_color_type():
    """Workbench 'Attribute' coloring is the enum item VERTEX in Blender 4.x (ATTRIBUTE in some builds)."""
    items = {i.identifier for i in bpy.types.View3DShading.bl_rna.properties["color_type"].enum_items}
    return "VERTEX" if "VERTEX" in items else "ATTRIBUTE"


def render(view, lo, hi, out, mode):
    sc = bpy.context.scene
    cam = bpy.data.objects.get("cf_cam")
    if cam is None:
        cam = bpy.data.objects.new("cf_cam", bpy.data.cameras.new("cf_cam"))
        sc.collection.objects.link(cam)
        cam.data.type = "ORTHO"
    center, size = (lo + hi) / 2, hi - lo
    direction, rot = CAMS[view]
    dist = size.length * 2 + 1
    cam.location = (center.x + direction[0] * dist, center.y + direction[1] * dist, center.z)
    cam.rotation_euler = [math.radians(a) for a in rot]
    cam.data.ortho_scale = max(size.y if view == "side" else size.x, size.z) * 1.08
    cam.data.clip_end = dist * 4
    sc.camera = cam
    sc.render.filepath = os.path.join(out, f"{view}_{mode}.png")
    try:
        bpy.ops.render.render(write_still=True)
    except RuntimeError:
        if mode != "mask":
            raise
        sc.render.engine, sc.cycles.samples = "CYCLES", 1  # headless fallback; alpha is still exact
        bpy.ops.render.render(write_still=True)
    return sc.render.filepath


def main():
    args = parse()
    try:
        os.makedirs(args.out, exist_ok=True)
        meshes = load_mesh(args.mesh, args.rotate_z)
        lo, hi, tris = bounds(meshes)
        setup(args.mode, args.res, meshes)
        files = {v: render(v, lo, hi, args.out, args.mode) for v in args.views.split(",")}
        res = {"ok": True, "files": files, "dims_m": [round(x, 4) for x in hi - lo], "tris": tris, "engine": bpy.context.scene.render.engine}
    except Exception as e:
        res = {"ok": False, "code": "RENDER_FAILED", "message": f"{type(e).__name__}: {e}", "fix": "The mesh failed to import or render; use another candidate."}
    with open(args.result, "w") as f:
        json.dump(res, f)


main()
