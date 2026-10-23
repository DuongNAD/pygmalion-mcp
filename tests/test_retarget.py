"""Offline motion library: clips retargeted by bone role onto any rig (animate(method='blender')), honest failure
for unknown action names, and the bundled clips' licenses."""
import json
import shutil
import subprocess

import pytest

from charforge import blender, config, server, state, tools_full
from tests.bpy_fixtures import OPAQUE_NAMES, make_character



def needs_blender(fn):
    return pytest.mark.blender(pytest.mark.skipif(config.blender_exe() is None, reason="Blender not installed")(fn))


_FEET = r'''
import bpy, json, sys
P = json.loads(sys.argv[sys.argv.index("--") + 1])
sys.path.insert(0, P["scripts"])
import cf_humanoid as hum
hum.import_any(P["mesh"])
arm, meshes = hum.find_rig()
roles = hum.map_roles(arm)["roles"]
act = bpy.data.actions[P["clip"]]
ad = arm.animation_data
for t in ad.nla_tracks:
    t.mute = True
ad.action = act
if hasattr(ad, "action_slot") and ad.action_slot is None and len(act.slots):
    ad.action_slot = act.slots[0]
f0, f1 = (int(round(x)) for x in act.frame_range)
out = []
for f in range(f0, f1 + 1):
    bpy.context.scene.frame_set(f)
    l, r = (arm.matrix_world @ arm.pose.bones[roles[k]].head for k in ("leftFoot", "rightFoot"))
    out.append([l.y, r.y])
json.dump(out, open(P["out"], "w"))
'''


def foot_lead(glb, clip, tmp_path):
    """Per frame: left ankle minus right ankle along the walking axis (m); negative = left foot ahead (-Y)."""
    out = tmp_path / f"feet_{clip}.json"
    params = {"scripts": str(config.BPY_SCRIPTS), "mesh": str(glb), "clip": clip, "out": str(out)}
    proc = subprocess.run(
        [config.blender_exe(), "-b", "--factory-startup", "--python-exit-code", "1", "--python-expr", _FEET, "--", json.dumps(params)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0 and out.is_file(), proc.stderr[-1500:]
    return [ly - ry for ly, ry in json.loads(out.read_text())]


def assert_alternates(lead):
    """Each foot leads by >= 15 cm at some point and the lead changes sign at least twice per cycle."""
    assert min(lead) < -0.15 and max(lead) > 0.15, f"stride too small: {min(lead):.3f} .. {max(lead):.3f}"
    signs = [x > 0 for x in lead if abs(x) > 0.05]
    flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    assert flips >= 2, f"feet do not alternate: {flips} sign change(s)"


def rigged_project(sheet_path, monkeypatch, name, rig_file):
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")
    monkeypatch.delenv("MESHY_API_KEY", raising=False)
    pid = server.project_create(sheet_path=str(sheet_path), name=name, height_cm=170.0)["project_id"]
    pdir = state.project_dir(pid)
    (pdir / "export").mkdir(exist_ok=True)
    shutil.copy(rig_file, pdir / f"export/{pid}-rigged.glb")
    p = state.load(pid)
    p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
    p["stage"] = "rigged"
    state.save(p)
    return pid, pdir


@pytest.fixture(scope="module")
def chars(tmp_path_factory):
    if config.blender_exe() is None:
        pytest.skip("Blender not installed")
    d = tmp_path_factory.mktemp("retarget_chars")
    make_character(d / "clean.glb")  # A-pose rest (arms 40 deg from the body), Mixamo names
    make_character(d / "opaque.glb", names=OPAQUE_NAMES)  # same body, meaningless bone names
    make_character(d / "tpose.glb", arm_angle=88, names=OPAQUE_NAMES)  # T-pose rest
    return d


# ---------------------------------------------------------------- name resolution (no Blender)

def test_unknown_action_fails_honestly(sheet_path, monkeypatch):
    pid, pdir = rigged_project(sheet_path, monkeypatch, "UnknownMove", sheet_path)
    monkeypatch.setattr(blender, "run", lambda *a, **k: pytest.fail("Blender must not run for an unknown action"))
    res = tools_full.animate(project_id=pid, actions=["idle", "moonwalk"], method="blender")
    assert res["ok"] is False
    err = res["error"]
    assert err["code"] == "ANIM_FAILED"
    assert "moonwalk" in err["message"]
    assert err["fix"].startswith("Use one of: ")
    names = err["fix"][len("Use one of: "):].rstrip(".").split(", ")
    assert names[:5] == list(config.PROCEDURAL_ACTIONS)
    assert set(config.MOTION_LIBRARY) <= set(names)


def test_offline_names_resolve_procedural_first(sheet_path, monkeypatch):
    pid, _ = rigged_project(sheet_path, monkeypatch, "NameRes", sheet_path)
    p = state.load(pid)
    names = ["walk", "walk_01", "Walking", "Sit Down", "attack", "wave", "wave_hello", "jog_01"]
    plan, sources = tools_full._offline_plan(p, names)
    assert [i["name"] for i in plan] == names
    assert sources == {
        "walk": "procedural", "walk_01": "procedural", "Walking": "library:walk_casual.bvh", "Sit Down": "library:sit_down.bvh",
        "attack": "library:punch.bvh", "wave": "procedural", "wave_hello": "library:wave_hello.bvh", "jog_01": "library:jog.bvh",
    }
    assert plan[1] == {"name": "walk_01", "kind": "walk"}
    assert plan[2]["loop"] is True and plan[2]["clip"].endswith("assets/motions/walk_casual.bvh")


def test_project_motion_library_adds_clips(sheet_path, monkeypatch):
    pid, pdir = rigged_project(sheet_path, monkeypatch, "OwnClip", sheet_path)
    (pdir / "clips").mkdir()
    (pdir / "clips" / "spin.bvh").write_text("HIERARCHY\n")
    p = state.load(pid)
    p["motion_library"] = {"spin": {"file": "clips/spin.bvh", "loop": False, "aliases": ["twirl"], "trim": [0.2, 1.0]}}
    plan, sources = tools_full._offline_plan(p, ["twirl", "bow"])
    assert sources == {"twirl": "library:spin.bvh", "bow": "library:bow.bvh"}
    assert plan[0]["clip"] == (pdir / "clips" / "spin.bvh").as_posix() and plan[0]["trim"] == [0.2, 1.0]


def test_library_files_licensed_and_small():
    files = sorted(f for f in config.MOTION_DIR.iterdir() if f.name != "LICENSES.md")
    licenses = (config.MOTION_DIR / "LICENSES.md").read_text(encoding="utf-8")
    assert files and all(f"`{f.name}`" in licenses for f in files), "every bundled clip needs a LICENSES.md row"
    assert {spec["file"] for spec in config.MOTION_LIBRARY.values()} == {f.name for f in files}
    assert sum(f.stat().st_size for f in files) < 10 * 1024 * 1024
    for f in files:
        head = f.read_text(encoding="ascii").splitlines()[:3]
        assert head[0] == "HIERARCHY" and head[1] == "ROOT Hips", f.name


# ---------------------------------------------------------------- retarget onto real skinned rigs

@needs_blender
@pytest.mark.parametrize("rig", ["clean", "opaque"])
def test_walk_and_wave_retarget_onto_rig(chars, sheet_path, monkeypatch, tmp_path, rig):
    pid, pdir = rigged_project(sheet_path, monkeypatch, f"Retarget{rig}", chars / f"{rig}.glb")
    wanted = ["walk_casual", "idle", "wave_hello"]  # not alphabetical: the GLB must keep this order
    res = tools_full.animate(project_id=pid, actions=wanted, method="blender", secondary="off")
    assert res["ok"], res
    d = res["data"]
    assert d["actions"] == wanted
    assert d["sources"] == {"walk_casual": "library:walk_casual.bvh", "idle": "procedural", "wave_hello": "library:wave_hello.bvh"}
    glb = pdir / d["mesh_path"]
    assert tools_full.inspect_glb_animations(glb) == wanted
    mc = d["motion_check"]
    for clip in ("walk_casual", "wave_hello"):
        m = mc["clips"][clip]
        assert m["stretch_p99"] <= config.ANIM_CHECK["stretch_p99"], (clip, m)
        assert m["ground_cm"] <= config.ANIM_CHECK["ground_cm"], (clip, m)
        assert (pdir / d["previews"][clip]).is_file()
    assert mc["passed"] is True, mc["issues"]
    assert_alternates(foot_lead(glb, "walk_casual", tmp_path))
    assert [a["source"] for a in state.load(pid)["animations"]] == ["library:walk_casual.bvh", "procedural", "library:wave_hello.bvh"]


@needs_blender
def test_every_library_clip_retargets(chars, sheet_path, monkeypatch):
    pid, pdir = rigged_project(sheet_path, monkeypatch, "WholeLibrary", chars / "clean.glb")
    names = list(config.MOTION_LIBRARY)
    res = tools_full.animate(project_id=pid, actions=names, method="blender", secondary="off")
    assert res["ok"], res
    assert tools_full.inspect_glb_animations(pdir / res["data"]["mesh_path"]) == names
    clips = res["data"]["motion_check"]["clips"]
    assert set(clips) == set(names)
    for name, m in clips.items():
        assert m["ground_cm"] <= config.ANIM_CHECK["ground_cm"], (name, m)
        assert m["stretch_p99"] < 1.8, (name, m)  # a flipped or mis-aligned bone tears the mesh far beyond this


@needs_blender
def test_glb_clip_onto_t_pose_rig(chars, sheet_path, monkeypatch, tmp_path):
    # source: a GLB clip on the A-pose Mixamo-named body; target: T-pose rest with meaningless bone names
    clip = tmp_path / "fixture_walk.glb"
    blender.run("merge_animations.py", {"mesh": chars / "clean.glb", "out": clip, "actions": "walk", "mode": "procedural", "fps": 30})
    pid, pdir = rigged_project(sheet_path, monkeypatch, "GlbClip", chars / "tpose.glb")
    p = state.load(pid)
    p["motion_library"] = {"stroll": {"file": str(clip), "loop": True, "action": "walk"}}
    state.save(p)
    res = tools_full.animate(project_id=pid, actions=["stroll"], method="blender", secondary="off")
    assert res["ok"], res
    assert res["data"]["sources"] == {"stroll": "library:fixture_walk.glb"}
    m = res["data"]["motion_check"]["clips"]["stroll"]
    assert m["stretch_p99"] <= config.ANIM_CHECK["stretch_p99"] and m["ground_cm"] <= config.ANIM_CHECK["ground_cm"], m
    assert_alternates(foot_lead(pdir / res["data"]["mesh_path"], "stroll", tmp_path))
