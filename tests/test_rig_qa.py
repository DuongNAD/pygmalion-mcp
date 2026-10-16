"""Rig QA on real skinned characters: skeleton role mapping, pose_deformation_check, weight repair, humanoid map,
and auto_rig(method='provider') replacing a leaky provider rig with its repaired version."""
import json
import shutil

import httpx
import pytest

from charforge import blender, config, providers, server, state, tools_full
from tests.bpy_fixtures import OPAQUE_NAMES, UE_NAMES, make_character

pytestmark = [pytest.mark.blender, pytest.mark.skipif(config.blender_exe() is None, reason="Blender not installed")]


@pytest.fixture(scope="module")
def chars(tmp_path_factory):
    d = tmp_path_factory.mktemp("chars")
    specs = {
        "clean": {},
        "leak": {"leak": 0.5},
        "ue": {"names": UE_NAMES},
        "opaque_leak": {"names": OPAQUE_NAMES, "leak": 0.5},
        "armsdown": {"arm_angle": 6},
    }
    out = {}
    for name, kw in specs.items():
        make_character(d / f"{name}.glb", **kw)
        out[name] = d / f"{name}.glb"
    return out


def check(path, tmp_path, repair=False):
    args = {"mesh": path, "humanoid": tmp_path / "humanoid.json", **config.RIG_CHECK}
    if repair:
        args.update(repair=True, out=tmp_path / "fixed.glb")
    return blender.run("rig_check.py", args)


@pytest.mark.parametrize(
    "name, arm, thigh",
    [("clean", "LeftArm", "LeftUpLeg"), ("ue", "upperarm_l", "thigh_l"), ("opaque_leak", OPAQUE_NAMES["LeftArm"], OPAQUE_NAMES["LeftUpLeg"])],
)
def test_roles_come_from_structure_not_names(chars, tmp_path, name, arm, thigh):
    r = check(chars[name], tmp_path)
    assert r["mapping"]["ok"] and not r["mapping"]["missing"]
    assert r["mapping"]["roles"]["leftUpperArm"] == arm
    assert r["mapping"]["roles"]["leftUpperLeg"] == thigh
    hmap = json.loads((tmp_path / "humanoid.json").read_text())
    assert hmap["bones"]["leftUpperArm"] == arm
    assert hmap["mixamo"][arm] == "LeftArm"


def test_clean_rig_passes_without_repair(chars, tmp_path):
    r = check(chars["clean"], tmp_path, repair=True)
    assert r["passed"] is True
    assert r["repaired"] is False
    assert len([t for t in r["before"]["tests"] if "skipped" not in t]) == 6


def test_leak_is_detected_and_repaired(chars, tmp_path):
    r = check(chars["leak"], tmp_path, repair=True)
    before = {t["name"]: t for t in r["before"]["tests"]}
    assert before["left_arm_swing"]["passed"] is False
    assert before["left_arm_swing"]["regions"]["legs"]["max_m"] > 0.1  # the thigh follows the hand
    assert before["right_arm_swing"]["passed"] is True
    assert r["repaired"] is True and r["passed"] is True
    again = check(tmp_path / "fixed.glb", tmp_path)
    assert again["passed"] is True  # the exported repaired rig keeps the fix


def test_touching_arms_are_reported(chars, tmp_path):
    r = check(chars["armsdown"], tmp_path, repair=True)
    assert r["mapping"]["ok"]
    assert r["contacts"]["left_arm"]["separated"] is False
    assert any("touches the body" in i for i in r["contact_issues"])


def meshy_rig_server(rigged_bytes: bytes):
    def handler(req: httpx.Request) -> httpx.Response:
        u = str(req.url)
        if req.method == "POST" and u.endswith("/openapi/v1/rigging"):
            return httpx.Response(202, json={"result": "rig-task-1"})
        if req.method == "GET" and u.endswith("/openapi/v1/rigging/rig-task-1"):
            return httpx.Response(200, json={"status": "SUCCEEDED", "result": {"rigged_character_glb_url": "https://cdn.mock/rigged.glb"}})
        if req.method == "GET" and u == "https://cdn.mock/rigged.glb":
            return httpx.Response(200, content=rigged_bytes)
        return httpx.Response(404)

    return handler


def test_auto_rig_provider_repairs_leaky_rig(chars, sheet_path, monkeypatch):
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")
    monkeypatch.setenv("MESHY_API_KEY", "k")
    raw = chars["leak"].read_bytes()
    client = httpx.Client(transport=httpx.MockTransport(meshy_rig_server(raw)))
    monkeypatch.setattr(providers, "get_provider", lambda name, client_=None, **kw: providers.MeshyProvider(api_key="k", client=client))
    pid = server.project_create(sheet_path=str(sheet_path), name="LeakyMeshy", height_cm=170.0)["project_id"]
    pdir = state.project_dir(pid)
    shutil.copy(chars["clean"], pdir / "meshes" / "cand-1-meshy.glb")
    p = state.load(pid)
    state.register_mesh(p, "meshes/cand-1-meshy.glb", source="meshy", task_id="gen-1")
    state.save(p)

    res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/cand-1-meshy.glb", method="provider")
    assert res["ok"], res
    qa = res["data"]["rig_check"]
    assert qa["repaired"] is True and qa["passed"] is True
    assert qa["issues_before_repair"]
    assert any("weight leaks" in w for w in res["warnings"])
    p = state.load(pid)
    assert p["outputs"]["rig_repaired"] is True
    assert (pdir / p["outputs"]["rigged_provider"]).read_bytes() == raw  # the provider file is kept as delivered
    assert (pdir / p["outputs"]["rigged"]).read_bytes() != raw
    assert json.loads((pdir / p["outputs"]["humanoid"]).read_text())["bones"]["hips"] == "Hips"
    assert (pdir / p["qa"]["rig"]["report"]).is_file()
    assert res["next_action"]["tool"] == "animate"
