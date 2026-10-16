"""Render orthographic front/side/back views of a mesh. Runs INSIDE Blender: bpy/mathutils/stdlib only.

Conventions: Z-up, meters, character faces -Y (Blender's glTF import default).
  front: camera at -Y looking +Y | side: at +X looking -X | back: at +Y looking -Y
mode=mask  -> Workbench, transparent film; the mask is the PNG alpha channel.
mode=color -> Workbench flat light + texture/attribute colors (~albedo, comparable to flat anime art).
--integrity -> also measure the geometry (see integrity()): floating parts, holes, non-manifold edges,
               self-intersections, degenerate faces. Numbers only, no rendering involved.
Result JSON: {ok, files: {view: abs path}, dims_m: [x, y, z], tris, engine, integrity?}
"""
import argparse
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

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
    ap.add_argument("--integrity", action="store_true", default=False)
    ap.add_argument("--gap-frac", type=float, default=0.006)
    ap.add_argument("--debris-share", type=float, default=0.002)
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:])


def integrity(meshes, gap_frac=0.006, debris_share=0.002):
    """Geometry health of the evaluated meshes (world space), with UV-seam copies welded first:
    components   connected pieces; a piece that neither cuts into nor comes within gap_frac x height of the
                 surface of a piece chained to the largest one floats: < debris_share of the vertices = debris,
                 larger = detached part
    holes        open boundary loops; non_manifold edges shared by 3+ faces (or wire edges)
    self_intersecting_faces  triangles cutting through another triangle of the same connected piece (a surface
                             folding through itself); separate pieces passing through each other (eyes in a head,
                             hair over a scalp) are counted apart as interpenetrating_faces
    degenerate_faces         zero-area triangles"""
    dg = bpy.context.evaluated_depsgraph_get()
    shapes = {pb.custom_shape for o in bpy.context.scene.objects if o.type == "ARMATURE" and o.pose for pb in o.pose.bones if pb.custom_shape}
    coords, comp_of, tris = [], [], []
    stats = {"edges": 0, "boundary_edges": 0, "holes": 0, "non_manifold_edges": 0, "faces": 0,
             "self_intersecting_faces": 0, "interpenetrating_faces": 0, "degenerate_faces": 0}
    n_comp = 0
    for obj in meshes:
        if obj in shapes:
            continue
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(me)
        ev.to_mesh_clear()
        bm.transform(obj.matrix_world)
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-6)
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        if not bm.verts:
            bm.free()
            continue
        stats["edges"] += len(bm.edges)
        stats["faces"] += len(bm.faces)
        boundary = [e for e in bm.edges if e.is_boundary]
        stats["boundary_edges"] += len(boundary)
        stats["non_manifold_edges"] += sum(1 for e in bm.edges if e.is_wire or len(e.link_faces) > 2)
        stats["degenerate_faces"] += sum(1 for f in bm.faces if f.calc_area() < 1e-12)
        seen = set()  # holes = connected chains of boundary edges
        for e in boundary:
            if e.index in seen:
                continue
            stats["holes"] += 1
            stack = [e]
            seen.add(e.index)
            while stack:
                cur = stack.pop()
                for v in cur.verts:
                    for e2 in v.link_edges:
                        if e2.is_boundary and e2.index not in seen:
                            seen.add(e2.index)
                            stack.append(e2)
        label = [-1] * len(bm.verts)  # connected pieces
        for v in bm.verts:
            if label[v.index] >= 0:
                continue
            label[v.index] = n_comp
            stack = [v]
            while stack:
                cur = stack.pop()
                for e in cur.link_edges:
                    o = e.other_vert(cur)
                    if label[o.index] < 0:
                        label[o.index] = n_comp
                        stack.append(o)
            n_comp += 1
        tree = BVHTree.FromBMesh(bm)  # crossing triangle pairs that share no vertex
        tri_verts = [{v.index for v in f.verts} for f in bm.faces]
        same, other = set(), set()
        for i, j in tree.overlap(tree):
            if i != j and not (tri_verts[i] & tri_verts[j]):
                same_piece = label[next(iter(tri_verts[i]))] == label[next(iter(tri_verts[j]))]
                (same if same_piece else other).update((i, j))
        stats["self_intersecting_faces"] += len(same)
        stats["interpenetrating_faces"] += len(other)
        offset = len(coords)
        coords += [v.co.copy() for v in bm.verts]
        comp_of += label
        tris += [tuple(offset + v.index for v in f.verts) for f in bm.faces]
        bm.free()
    total = len(coords)
    if not total:
        return {"ok": False, "issues": ["no geometry"], **stats}
    zs = [c.z for c in coords]
    height = max(1e-6, max(zs) - min(zs))
    gap = gap_frac * height
    members, faces_of = {}, {}
    for i, c in enumerate(comp_of):
        members.setdefault(c, []).append(i)
    for t in tris:
        faces_of.setdefault(comp_of[t[0]], []).append(t)
    main = max(members, key=lambda c: len(members[c]))
    anchored, pending = {main}, set(members) - {main}

    def touches(tree, c):
        if any(tree.find_nearest(coords[i], gap)[0] is not None for i in members[c]):
            return True
        return bool(faces_of.get(c)) and bool(tree.overlap(BVHTree.FromPolygons(coords, faces_of[c], all_triangles=True)))

    while pending:
        tree = BVHTree.FromPolygons(coords, [t for c in anchored for t in faces_of.get(c, [])], all_triangles=True)
        newly = {c for c in pending if touches(tree, c)}
        if not newly:
            break
        anchored |= newly
        pending -= newly
    debris = [c for c in pending if len(members[c]) < debris_share * total]
    detached = [c for c in pending if c not in debris]
    share = lambda cs: round(100.0 * sum(len(members[c]) for c in cs) / total, 2)  # noqa: E731
    return {
        "components": len(members),
        "debris_parts": len(debris),
        "debris_pct": share(debris),
        "detached_parts": len(detached),
        "detached_pct": share(detached),
        "vertices": total,
        **stats,
    }


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
        if args.integrity:
            res["integrity"] = integrity(meshes, args.gap_frac, args.debris_share)
    except Exception as e:
        res = {"ok": False, "code": "RENDER_FAILED", "message": f"{type(e).__name__}: {e}", "fix": "The mesh failed to import or render; use another candidate."}
    with open(args.result, "w") as f:
        json.dump(res, f)


main()
