"""Animation QA on real skinned characters: front/side clip previews, motion metrics, secondary motion
(spring chains for skirt and ponytail), and repaired weights carried into provider animations."""
import json
import shutil

import httpx
import pytest

from charforge import blender, config, providers, server, state, tools_full
from tests.bpy_fixtures import make_character

pytestmark = [pytest.mark.blender, pytest.mark.skipif(config.blender_exe() is None, reason="Blender not installed")]


@pytest.fixture(scope="module")
def chars(tmp_path_factory):
    d = tmp_path_factory.mktemp("anim_chars")
    make_character(d / "skirt_raw.glb", skirt=True, ponytail=True)
    # as auto_rig delivers it: bone-heat bleeds the arms into the skirt, the rig QA repairs that
    r = blender.run("rig_check.py", {"mesh": d / "skirt_raw.glb", "repair": True, "out": d / "skirt.glb", **config.RIG_CHECK})
    if not r["repaired"]:
        shutil.copy(d / "skirt_raw.glb", d / "skirt.glb")
    make_character(d / "leak.glb", leak=0.5)
    return d


def rigged_project(sheet_path, monkeypatch, name, rig_file):
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")
    pid = server.project_create(sheet_path=str(sheet_path), name=name, height_cm=170.0)["project_id"]
    pdir = state.project_dir(pid)
    (pdir / "export").mkdir(exist_ok=True)
    shutil.copy(rig_file, pdir / f"export/{pid}-rigged.glb")
    p = state.load(pid)
    p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
    p["stage"] = "rigged"
    state.save(p)
    return pid, pdir


def glb_nodes(path):
    b = path.read_bytes()
    n = int.from_bytes(b[12:16], "little")
    return [nd.get("name") for nd in json.loads(b[20:20 + n]).get("nodes", [])]


def test_blender_animate_previews_metrics_and_secondary(chars, sheet_path, monkeypatch):
    monkeypatch.delenv("MESHY_API_KEY", raising=False)
    pid, pdir = rigged_project(sheet_path, monkeypatch, "SkirtAnim", chars / "skirt.glb")
    res = tools_full.animate(project_id=pid, method="blender")
    assert res["ok"], res
    d = res["data"]
    assert d["actions"] == ["idle", "walk", "wave"]
    assert set(d["previews"]) == {"idle", "walk", "wave"}
    for path in d["previews"].values():
        assert (pdir / path).is_file()
    assert not (pdir / "renders" / "anim" / "frames").exists()  # per-frame renders are folded into the sheets
    mc = d["motion_check"]
    assert set(mc["clips"]) == {"idle", "walk", "wave"}
    for m in mc["clips"].values():
        assert 1.0 <= m["stretch_p99"] < config.ANIM_CHECK["stretch_p99"]
    assert mc["passed"] is True
    sec = d["secondary"]
    assert sec["applied"] is True
    assert sec["chains"].get("skirt", 0) >= 3 and sec["chains"].get("hair", 0) >= 1
    assert sec["stretch_after"] <= sec["stretch_before"] * 1.15 + 0.05
    nodes = glb_nodes(pdir / d["mesh_path"])
    assert any(n.startswith("CF_skirt") for n in nodes) and any(n.startswith("CF_hair") for n in nodes)
    assert tools_full.inspect_glb_animations(pdir / d["mesh_path"]) == ["idle", "walk", "wave"]
    p = state.load(pid)
    assert p["qa"]["motion"]["passed"] is True
    status = server.project_status(project_id=pid)
    assert status["data"]["qa"]["motion"]["passed"] is True


def test_secondary_off_keeps_skeleton(chars, sheet_path, monkeypatch):
    monkeypatch.delenv("MESHY_API_KEY", raising=False)
    pid, pdir = rigged_project(sheet_path, monkeypatch, "SkirtNoSec", chars / "skirt.glb")
    res = tools_full.animate(project_id=pid, actions=["walk"], method="blender", secondary="off")
    assert res["ok"], res
    assert res["data"]["secondary"] == {"applied": False}
    assert not any(n.startswith("CF_") for n in glb_nodes(pdir / res["data"]["mesh_path"]))


def test_provider_animation_gets_repaired_weights(chars, sheet_path, monkeypatch, tmp_path):
    # Meshy animates its own (leaky) copy of the rig server-side; locally we repaired the weights.
    leak = chars / "leak.glb"
    fixed = tmp_path / "fixed.glb"
    r = blender.run("rig_check.py", {"mesh": leak, "repair": True, "out": fixed, **config.RIG_CHECK})
    assert r["repaired"]
    provider_anim = tmp_path / "provider_anim.glb"
    blender.run("merge_animations.py", {"mesh": leak, "out": provider_anim, "actions": "Idle,Walking", "mode": "procedural", "fps": 30})

    monkeypatch.setenv("MESHY_API_KEY", "k")
    pid, pdir = rigged_project(sheet_path, monkeypatch, "RepairedAnim", fixed)
    p = state.load(pid)
    p["outputs"].update(rig_task_id="rig-1", rig_repaired=True)
    state.save(p)

    def handler(req: httpx.Request) -> httpx.Response:
        u = str(req.url)
        if u.endswith("/animations/library"):
            return httpx.Response(200, json=[{"action_id": 1, "name": "Idle", "key": "Idle"}, {"action_id": 2, "name": "Walking", "key": "Walking"}])
        if req.method == "POST" and u.endswith("/animations"):
            return httpx.Response(202, json={"result": "anim-1"})
        if u.endswith("/animations/anim-1"):
            return httpx.Response(200, json={"status": "SUCCEEDED", "result": {"animation_glb_url": "https://cdn.mock/a.glb"}})
        if u == "https://cdn.mock/a.glb":
            return httpx.Response(200, content=provider_anim.read_bytes())
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(providers, "get_provider", lambda name, client_=None, **kw: providers.MeshyProvider(api_key="k", client=client))
    res = tools_full.animate(project_id=pid, actions=["idle", "walk"], method="provider", secondary="off")
    assert res["ok"], res
    final = pdir / res["data"]["mesh_path"]
    assert tools_full.inspect_glb_animations(final) == ["idle", "walk"]
    check = blender.run("rig_check.py", {"mesh": final, **config.RIG_CHECK})
    assert check["passed"] is True  # the leak Meshy baked into its copy is gone from the delivered file
    assert set(res["data"]["previews"]) == {"idle", "walk"}
