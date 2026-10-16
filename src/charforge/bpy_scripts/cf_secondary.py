"""Secondary motion for generated characters: spring-bone chains for hanging hair and skirts, baked into every
action so any engine plays them (no runtime physics needed). Runs INSIDE Blender: bpy/mathutils/stdlib only.

Detection uses the rest pose and the mapped skeleton (cf_humanoid.Body):
  hair  = vertices driven by the head that hang below and away from the skull (ponytail, long hair)
  skirt = surface that crosses the gap between the thighs below the crotch (skirt, dress)
Chains: hair follows the hanging mass; skirts get N chains around the hips. Weights blend from the original
binding at the attachment to the chain toward the tip. Simulation is the VRM spring-bone update
(inertia with drag, stiffness toward the animated rest direction, gravity, fixed bone length) with
capsule colliders (legs for skirts, torso and skull for hair), pre-rolled so loops start settled.
"""
import math

import bpy
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

PREFIX = "CF_"


def _smoothstep(a, b, x):
    if b <= a:
        return 1.0 if x >= b else 0.0
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def _weld_groups(coords, eps=1e-5):
    """Vertex index -> representative index for vertices at the same position (glTF splits seams)."""
    rep, seen = list(range(len(coords))), {}
    for i, v in enumerate(coords):
        key = (round(v.x / eps), round(v.y / eps), round(v.z / eps))
        rep[i] = seen.setdefault(key, i)
    return rep


def _components(members, edges, rep):
    """Connected components of `members` (set of vertex ids) through mesh edges and welded seams."""
    parent = {}

    def find(a):
        while parent.get(a, a) != a:
            parent[a] = parent.get(parent[a], parent[a])
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in members:
        parent.setdefault(i, i)
        if rep[i] != i and rep[i] in members:
            union(i, rep[i])
    for a, b in edges:
        if a in members and b in members:
            union(a, b)
    comps = {}
    for i in members:
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def dominant_bones(body):
    """Per concatenated vertex: name of the bone with the largest weight (or None)."""
    out = []
    bones = {b.name for b in body.arm.data.bones}
    for obj in body.meshes:
        names = {vg.index: vg.name for vg in obj.vertex_groups if vg.name in bones}
        for v in obj.data.vertices:
            best, bw = None, 0.0
            for g in v.groups:
                if g.group in names and g.weight > bw:
                    best, bw = names[g.group], g.weight
            out.append(best)
    return out


def _centroid(pts):
    s = Vector((0.0, 0.0, 0.0))
    for p in pts:
        s += p
    return s / max(1, len(pts))


def _back_hanging(body, c_h, r_h):
    """Vertices hanging behind the neck/back (ponytail, long hair) whatever bone drives them: farther behind
    the spine than the back surface at their height (75th percentile of torso depth per 2 cm slice) + 1.5 cm."""
    rest, roles = body.rest, body.roles
    hips = body.joint("hips")
    if hips is None or roles.get("neck") is None:
        return set()
    spine_y = hips.y
    z_lo = hips.z
    slices = {}
    for i, p in enumerate(rest):
        if body.group(i) in ("torso", "head") and p.y > spine_y and abs(p.x - hips.x) < 0.6 * r_h + 0.05 * body.H:
            slices.setdefault(int((p.z - z_lo) / 0.02), []).append(p.y - spine_y)
    back = {}
    for k, ys in slices.items():
        ys.sort()
        back[k] = ys[int(0.75 * (len(ys) - 1))]
    out = set()
    for i, p in enumerate(rest):
        if p.z < z_lo or p.z > c_h.z or body.group(i) not in ("torso", "head"):
            continue
        if abs(p.x - hips.x) > 1.2 * r_h or (p - c_h).length < 1.2 * r_h:
            continue
        depth = back.get(int((p.z - z_lo) / 0.02))
        if depth is not None and p.y - spine_y > depth + 0.015:
            out.add(i)
    return out


def detect(body, edges, skirt_chains=8, max_hair_chains=4):
    """Chain specs in armature-independent world space: [{"kind", "parent", "joints": [Vector], "verts": [ids], "param": {...}}]."""
    roles, rest, H = body.roles, body.rest, body.H
    dom = dominant_bones(body)
    rep = _weld_groups(rest)
    specs, notes = [], []

    # ---- hair: head-driven mass hanging below and away from the skull
    head_bone = roles.get("head")
    if head_bone:
        skull = [rest[i] for i in range(len(rest)) if body.role(i) == "head" and body.is_core(i) and body.ambig[i] is None]
        if len(skull) >= 20:
            c_h = _centroid(skull)
            ds = sorted((p - c_h).length for p in skull)
            r_h = ds[len(ds) // 2]
            neck_bone = roles.get("neck")
            cand = set()
            for i, p in enumerate(rest):
                b = dom[i]
                if b is None or body.group_of(b) != "head":
                    continue
                if (p - c_h).length < 1.45 * r_h or p.z > c_h.z + 0.25 * r_h:
                    continue
                if body.bone[i] == neck_bone and body.is_core(i):
                    continue
                cand.add(i)
            cand |= _back_hanging(body, c_h, r_h)
            comps = [c for c in _components(cand, edges, rep) if len(c) >= 30]
            comps = [c for c in comps if min((rest[i] - c_h).length for i in c) < 2.2 * r_h]  # must hang from the head
            chains = []
            for comp in comps:
                pts = [rest[i] for i in comp]
                near = sorted(comp, key=lambda i: (rest[i] - c_h).length)[: max(3, len(comp) // 20)]
                root = _centroid([rest[i] for i in near])
                L = max((p - root).length for p in pts)
                if L < 0.08 * H:
                    continue
                xs = [p.x for p in pts]
                width = max(xs) - min(xs)
                splits = 1 if width < 0.12 * H else min(3, int(width / (0.08 * H)) + 1)
                if splits == 1:
                    groups = [comp]
                else:
                    lo, hi = min(xs), max(xs)
                    groups = [[i for i in comp if min(splits - 1, int((rest[i].x - lo) / max(hi - lo, 1e-6) * splits)) == k] for k in range(splits)]
                for g in groups:
                    if len(g) >= 20:
                        chains.append(g)
            chains.sort(key=len, reverse=True)
            for g in chains[:max_hair_chains]:
                pts = [rest[i] for i in g]
                near = sorted(g, key=lambda i: (rest[i] - c_h).length)[: max(3, len(g) // 20)]
                root = _centroid([rest[i] for i in near])
                L = max((p - root).length for p in pts)
                nb = max(2, min(4, int(round(L / (0.07 * H)))))
                joints = [root]
                for k in range(1, nb + 1):
                    lo_t, hi_t = (k - 0.6) / nb, k / nb
                    band = [p for p in pts if lo_t * L <= (p - root).length <= hi_t * L + 1e-6]
                    joints.append(_centroid(band) if band else joints[-1] + (joints[-1] - joints[max(0, len(joints) - 2)]))
                if any((joints[k + 1] - joints[k]).length < 0.01 * H for k in range(nb)):
                    continue
                specs.append({"kind": "hair", "parent": head_bone, "joints": joints, "verts": g, "param": {"c_h": c_h, "r_h": r_h}})
            if not specs:
                notes.append("no hanging hair found")

    # ---- skirt: surface crossing between the thighs below the crotch
    hips = roles.get("hips")
    hl, hr = body.joint("leftUpperLeg"), body.joint("rightUpperLeg")
    kl, kr = body.joint("leftLowerLeg"), body.joint("rightLowerLeg")
    if hips and hl and hr and kl and kr:
        x_mid = 0.5 * (hl.x + hr.x)
        half = 0.5 * abs(hl.x - hr.x)
        thigh = 0.5 * ((hl - kl).length + (hr - kr).length)
        z_hip = 0.5 * (hl.z + hr.z)
        z_knee = 0.5 * (kl.z + kr.z)
        shin_r = []
        for i in range(len(rest)):
            if body.role(i) in ("leftLowerLeg", "rightLowerLeg") and body.is_core(i) and body.ambig[i] is None:
                shin_r.append(body.dist[i])
        shin_r.sort()
        leg_r = 1.6 * (shin_r[len(shin_r) // 2] if shin_r else 0.04 * H)  # thighs are ~1.4-1.6x the shin radius
        between = [
            i for i, p in enumerate(rest)
            if abs(p.x - x_mid) < 0.45 * half and z_knee + 0.1 * thigh < p.z < z_hip - 0.4 * thigh
            and body.group(i) in ("legL", "legR", "torso")
        ]
        if len(between) >= 30:
            def leg_d(p):
                d = []
                for a, b in ((hl, kl), (hr, kr)):
                    ab = Vector((b.x - a.x, b.y - a.y, 0.0))
                    ap = Vector((p.x - a.x, p.y - a.y, 0.0))
                    t = max(0.0, min(1.0, ap.dot(ab) / max(ab.length_squared, 1e-9)))
                    d.append((ap - t * ab).length)
                return min(d)

            z_top = z_hip + 0.03 * H
            z_hem = min(rest[i].z for i in between) - 0.03 * H  # the lowest cloth crossing between the legs
            feet = ("leftFoot", "rightFoot", "leftToes", "rightToes")
            shell = [
                i for i, p in enumerate(rest)
                if z_hem <= p.z < z_top and body.group(i) in ("legL", "legR", "torso")
                and body.role(i) not in feet and leg_d(rest[i]) > leg_r
            ]
            if shell:
                c = Vector((x_mid, _centroid([rest[i] for i in shell]).y, 0.0))
                span = max(z_top - z_hem, 0.05 * H)
                rings = {}

                def ring_key(p):
                    phi = math.atan2(p.y - c.y, p.x - c.x) % (2 * math.pi)
                    return int(round(phi / (2 * math.pi) * skirt_chains)) % skirt_chains

                for i in shell:
                    p = rest[i]
                    t = (z_top - p.z) / span
                    if t >= 0.2:
                        rings.setdefault((ring_key(p), 2 if t > 0.8 else 1), []).append(p)
                # roots: outermost body surface at the waist (the shell may not reach it beside the hips)
                waist = {}
                for i, p in enumerate(rest):
                    if body.group(i) in ("legL", "legR", "torso") and 0.0 <= (z_top - p.z) / span < 0.2:
                        waist.setdefault(ring_key(p), []).append(p)
                for k, pts in waist.items():
                    pts.sort(key=lambda q: (q.x - c.x) ** 2 + (q.y - c.y) ** 2)
                    rings[(k, 0)] = pts[int(0.7 * len(pts)):] or pts
                made = 0
                for k in range(skirt_chains):
                    if not all((k, b) in rings for b in (0, 1, 2)):
                        continue
                    joints = [_centroid(rings[(k, b)]) for b in (0, 1, 2)]
                    if any((joints[j + 1] - joints[j]).length < 0.02 * H for j in range(2)):
                        continue
                    specs.append({"kind": "skirt", "parent": hips, "joints": joints, "verts": [], "param": {"k": k, "n": skirt_chains}})
                    made += 1
                if made >= 3:
                    skirt_specs = [s for s in specs if s["kind"] == "skirt"]
                    for s in skirt_specs:
                        s["param"].update({"center": c, "z_top": z_top, "z_hem": z_hem, "shell": shell, "leg_r": leg_r})
                else:
                    specs = [s for s in specs if s["kind"] != "skirt"]
                    notes.append("skirt found but too few chain positions")
        else:
            notes.append("no skirt between the legs")
    return {"chains": specs, "notes": notes}


# ---------------------------------------------------------------- build

def build(body, specs):
    """Create chain bones (parented to head/hips) and blend their weights in. Returns chain records."""
    arm = body.arm
    inv = arm.matrix_world.inverted()
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    chains = []
    counters = {}
    for spec in specs:
        idx = counters.get(spec["kind"], 0)
        counters[spec["kind"]] = idx + 1
        names = []
        parent = eb[spec["parent"]]
        joints = [inv @ j for j in spec["joints"]]
        for j in range(len(joints) - 1):
            b = eb.new(f"{PREFIX}{spec['kind']}{idx}_{j}")
            b.head, b.tail = joints[j], joints[j + 1]
            b.parent = parent
            b.use_connect = j > 0
            b.use_deform = True
            parent = b
            names.append(b.name)
        chains.append({"kind": spec["kind"], "parent": spec["parent"], "bones": names, "spec": spec})
    bpy.ops.object.mode_set(mode="OBJECT")
    for c in chains:
        for n in c["bones"]:
            pb = arm.pose.bones[n]
            pb.rotation_mode = "QUATERNION"
    _assign_weights(body, chains)
    return chains


def _chain_weights(u, nb):
    """Linear blend along a chain of nb bones; u in [0, nb] measured in bone units."""
    w = {}
    for j in range(nb):
        d = abs(u - (j + 0.5))
        if u <= 0.5 and j == 0 or u >= nb - 0.5 and j == nb - 1:
            w[j] = 1.0
        elif d < 1.0:
            w[j] = 1.0 - d
    s = sum(w.values()) or 1.0
    return {j: x / s for j, x in w.items()}


def _polyline_param(p, joints):
    """(u in bone units, distance) of the closest point on the joint polyline."""
    best = (0.0, float("inf"))
    for j in range(len(joints) - 1):
        a, b = joints[j], joints[j + 1]
        ab = b - a
        t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-12)))
        d = (p - (a + t * ab)).length
        if d < best[1]:
            best = (j + t, d)
    return best


def _assign_weights(body, chains):
    """new = chain * alpha + original * (1 - alpha); alpha grows from the attachment to the tip."""
    plan = {}  # vertex id -> {bone_name: chain weight}, alpha
    skirts = [c for c in chains if c["kind"] == "skirt"]
    for c in chains:
        if c["kind"] != "hair":
            continue
        joints, nb = c["spec"]["joints"], len(c["bones"])
        for i in c["spec"]["verts"]:
            u, _ = _polyline_param(body.rest[i], joints)
            alpha = _smoothstep(0.0, 0.2 * nb, u)
            plan[i] = ({c["bones"][j]: w for j, w in _chain_weights(u, nb).items()}, alpha)
    if skirts:
        par = skirts[0]["spec"]["param"]
        n, center, z_top, z_hem = par["n"], par["center"], par["z_top"], par["z_hem"]
        by_k = {c["spec"]["param"]["k"]: c for c in skirts}
        ks = sorted(by_k)
        span = max(z_top - z_hem, 1e-4)
        for i in par["shell"]:
            p = body.rest[i]
            phi = (math.atan2(p.y - center.y, p.x - center.x) % (2 * math.pi)) / (2 * math.pi) * n
            k0 = max((k for k in ks if k <= phi), default=ks[-1])
            k1 = min((k for k in ks if k > phi), default=ks[0])
            gap = (k1 - k0) % n or n
            f = ((phi - k0) % n) / gap
            t = (z_top - p.z) / span
            u = max(0.0, min(2.0, 2.0 * t))
            cw = {}
            for c, wk in ((by_k[k0], 1.0 - f), (by_k[k1], f)):
                for j, w in _chain_weights(u, len(c["bones"])).items():
                    cw[c["bones"][j]] = cw.get(c["bones"][j], 0.0) + wk * w
            plan[i] = (cw, _smoothstep(0.05, 0.45, t))
    offset = 0
    for obj in body.meshes:
        groups = {vg.name: vg for vg in obj.vertex_groups}
        for c in chains:
            for n in c["bones"]:
                if n not in groups:
                    groups[n] = obj.vertex_groups.new(name=n)
        by_index = {vg.index: vg for vg in obj.vertex_groups}
        for v in obj.data.vertices:
            item = plan.get(offset + v.index)
            if not item:
                continue
            cw, alpha = item
            if alpha <= 1e-3:
                continue
            for gi, w in [(g.group, g.weight) for g in v.groups]:  # copy first: adding groups reallocates v.groups
                by_index[gi].add([v.index], w * (1.0 - alpha), "REPLACE")
            for name, w in cw.items():
                if w * alpha > 1e-4:
                    groups[name].add([v.index], w * alpha, "ADD")
        obj.data.update()
        offset += len(obj.data.vertices)


# ---------------------------------------------------------------- simulate + bake

def _capsule_push(p, a, b, r):
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-12)))
    q = a + t * ab
    d = p - q
    if d.length < r:
        if d.length < 1e-9:
            d = Vector((0.0, -1.0, 0.0))
        return q + d.normalized() * r
    return p


def _colliders(body, kind):
    """[(bone_a, bone_b, radius)] capsules evaluated per frame from pose bone heads/tails."""
    roles, out = body.roles, []
    if kind == "skirt":
        shin = []
        for i in range(len(body.rest)):
            if body.role(i) in ("leftLowerLeg", "rightLowerLeg") and body.is_core(i) and body.ambig[i] is None:
                shin.append(body.dist[i])
        shin.sort()
        r = 1.35 * (shin[len(shin) // 2] if shin else 0.04 * body.H) + 0.01 * body.H
        for s in ("left", "right"):
            for role in (f"{s}UpperLeg", f"{s}LowerLeg"):
                if roles.get(role):
                    out.append((roles[role], r))
    else:
        torso = [body.dist[i] for i in range(len(body.rest)) if body.group(i) == "torso" and body.is_core(i)]
        torso.sort()
        r = (torso[len(torso) // 2] if torso else 0.1 * body.H) * 1.05
        for role in ("spine", "chest", "upperChest"):
            if roles.get(role):
                out.append((roles[role], r))
    return out


def _is_loop(arm, act, roles):
    """A clip loops when the hips/head pose at the first and last frame match."""
    sc = bpy.context.scene
    f0, f1 = (int(round(x)) for x in act.frame_range)
    mats = []
    for f in (f0, f1):
        sc.frame_set(f)
        mats.append([arm.pose.bones[roles[r]].matrix.copy() for r in ("hips", "head") if roles.get(r) in arm.pose.bones])
    if not mats[0]:
        return False
    for m0, m1 in zip(*mats):
        if (m0.translation - m1.translation).length > 0.01 or m0.to_quaternion().rotation_difference(m1.to_quaternion()).angle > math.radians(4):
            return False
    return True


def bake(body, chains, actions, assign, fps, stiffness=1.2, drag=0.35, gravity=0.15):
    """Simulate and key every chain bone for every action. `assign(arm, act)` makes act the evaluated action.
    Physics runs in armature space; world-unit constants are divided by the armature scale."""
    arm = body.arm
    sc = bpy.context.scene
    dt = 1.0 / max(fps, 1.0)
    unit = 1.0 / max(1e-6, sum(arm.matrix_world.to_scale()) / 3.0)  # world meters -> armature units
    mods = [m for o in body.meshes for m in o.modifiers if m.type == "ARMATURE"]
    for m in mods:
        m.show_viewport = False  # skinning is not needed while simulating: much faster frame_set
    rest_local = {n: arm.data.bones[n].matrix_local.copy() for c in chains for n in c["bones"] + [c["parent"]]}
    colliders = {k: [(b, r * unit) for b, r in _colliders(body, k)] for k in ("skirt", "hair")}
    down = (arm.matrix_world.to_3x3().inverted() @ Vector((0.0, 0.0, -1.0))).normalized()  # world down in armature space
    phys = {"dt": dt, "stiff": stiffness * dt * unit, "drag": drag, "grav": down * (gravity * dt * unit)}
    keyed = 0
    try:
        for act in actions:
            assign(arm, act)
            f0, f1 = (int(round(x)) for x in act.frame_range)
            frames = list(range(f0, f1 + 1))
            loop = len(frames) > 4 and _is_loop(arm, act, body.roles)
            passes = [frames, frames] if loop else [[f0] * 20, frames]
            for c in chains:
                c["state"] = None
            for pi, seq in enumerate(passes):
                record = pi == len(passes) - 1
                for f in seq:
                    sc.frame_set(f)
                    caps = {
                        k: [(arm.pose.bones[b].head.copy(), arm.pose.bones[b].tail.copy(), r) for b, r in v if b in arm.pose.bones]
                        for k, v in colliders.items()
                    }
                    for c in chains:
                        _step_chain(arm, c, rest_local, caps[c["kind"]], phys, f, record)
                        keyed += len(c["bones"]) if record else 0
            if loop:
                _close_loop(chains, act, f0, f1)
    finally:
        for m in mods:
            m.show_viewport = True
    return keyed


def _step_chain(arm, c, rest_local, caps, phys, frame, record):
    """One VRM spring-bone step for a chain, then key each bone so it points at its simulated tail."""
    parent = arm.pose.bones[c["parent"]]
    parent_pose = parent.matrix.copy()
    parent_rest = rest_local[c["parent"]]
    P = parent_pose @ parent_rest.inverted()  # parent rest -> current (armature space)
    names = c["bones"]
    heads_rest = [rest_local[n].translation.copy() for n in names]
    tails_rest = [rest_local[n].translation + rest_local[n].to_3x3().col[1].normalized() * arm.data.bones[n].length for n in names]
    if c["state"] is None:
        cur = [P @ t for t in tails_rest]
        c["state"] = {"cur": cur, "prev": [p.copy() for p in cur]}
    st = c["state"]
    rot = P.to_3x3().normalized()
    head = P @ heads_rest[0]
    keys = []
    for j, n in enumerate(names):
        length = arm.data.bones[n].length
        rest_dir = (rot @ (tails_rest[j] - heads_rest[j]).normalized()).normalized()
        cur, prev = st["cur"][j], st["prev"][j]
        nxt = cur + (cur - prev) * (1.0 - phys["drag"]) + rest_dir * phys["stiff"] + phys["grav"]
        nxt = head + (nxt - head).normalized() * length
        for a, b, r in caps:
            nxt = _capsule_push(nxt, a, b, r)
        nxt = head + (nxt - head).normalized() * length
        st["prev"][j], st["cur"][j] = cur, nxt
        current = parent_pose @ parent_rest.inverted() @ rest_local[n]  # animated rest pose of this bone
        q = current.to_3x3().col[1].normalized().rotation_difference((nxt - head).normalized())
        M = Matrix.Translation(head) @ q.to_matrix().to_4x4() @ Matrix.Translation(-head) @ current
        keys.append((n, (current.inverted() @ M).to_quaternion()))
        rot = (q.to_matrix() @ rot).normalized()
        parent_pose, parent_rest = M, rest_local[n]
        head = nxt
    if record:
        for n, quat in keys:
            pb = arm.pose.bones[n]
            pb.rotation_quaternion = quat
            pb.keyframe_insert("rotation_quaternion", frame=frame, group=n)


def _close_loop(chains, act, f0, f1):
    """Blend the last quarter of each loop toward its first frame so the clip repeats without a pop."""
    blend = max(2, (f1 - f0) // 4)
    curves = {}
    for fc in _fcurves(act):
        curves.setdefault(fc.data_path, {})[fc.array_index] = fc
    for c in chains:
        for n in c["bones"]:
            fcs = curves.get(f'pose.bones["{n}"].rotation_quaternion') or {}
            if len(fcs) != 4:
                continue
            fcs = [fcs[i] for i in range(4)]
            start = [fc.evaluate(f0) for fc in fcs]
            for f in range(f1 - blend, f1 + 1):
                w = _smoothstep(0.0, 1.0, (f - (f1 - blend)) / blend)
                vals = [fc.evaluate(f) for fc in fcs]
                tgt = start if sum(a * b for a, b in zip(vals, start)) >= 0 else [-x for x in start]
                mixed = [a * (1 - w) + b * w for a, b in zip(vals, tgt)]
                norm = math.sqrt(sum(x * x for x in mixed)) or 1.0
                for fc, x in zip(fcs, mixed):
                    for kp in fc.keyframe_points:
                        if int(round(kp.co.x)) == f:
                            kp.co.y = kp.handle_left.y = kp.handle_right.y = x / norm
            for fc in fcs:
                fc.update()


def remove(body, chains, snapshot, actions):
    """Undo build()+bake(): restore weights, drop chain vertex groups, keys and bones."""
    from cf_humanoid import restore_weights  # sibling module, already on sys.path

    names = {n for c in chains for n in c["bones"]}
    restore_weights(body.meshes, snapshot)
    for obj in body.meshes:
        for vg in [vg for vg in obj.vertex_groups if vg.name in names]:
            obj.vertex_groups.remove(vg)
    for act in actions:
        for fc in [fc for fc in _fcurves(act) if any(f'"{n}"' in fc.data_path for n in names)]:
            try:
                act.fcurves.remove(fc)
            except Exception:
                pass
    arm = body.arm
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for n in names:
        if n in arm.data.edit_bones:
            arm.data.edit_bones.remove(arm.data.edit_bones[n])
    bpy.ops.object.mode_set(mode="OBJECT")


def _fcurves(act):
    """FCurves of an action across Blender versions (4.4+ layered actions keep a legacy view)."""
    try:
        return list(act.fcurves)
    except Exception:
        out = []
        for layer in getattr(act, "layers", []):
            for strip in layer.strips:
                for bag in getattr(strip, "channelbags", []):
                    out += list(bag.fcurves)
        return out
