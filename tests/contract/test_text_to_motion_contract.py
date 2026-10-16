"""Contract tests for Meshy Text to Motion and the animate() job group that merges motions with preset clips.

Verifies (Meshy OpenAPI v1):
- POST /openapi/v1/text-to-motion {prompt <= 400, duration 2-10 in 0.5 steps, mode} -> {"result": id}
- GET /openapi/v1/text-to-motion/{id} -> result.motion_url / motion_format
- POST /openapi/v1/animations takes exactly one of action_ids | motion_task_id
- 403 (content moderation) -> ANIM_FAILED with a rephrase fix
- animate(motions=[...]) submits presets + motions, chains motion -> animation, merges into one GLB, names clips
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import httpx
import pytest

from charforge import blender, providers, server, state, tools_full
from charforge.envelope import CFError


def glb(animations: list[str], skinned: bool = True) -> bytes:
    gltf = {
        "asset": {"version": "2.0"},
        "nodes": [{"name": "Body", "mesh": 0, "skin": 0}, {"name": "Hips"}] if skinned else [{"name": "Root"}],
        "meshes": [{"primitives": []}] if skinned else [],
        "skins": [{"joints": [1]}] if skinned else [],
        "animations": [
            {"name": a, "channels": [{"sampler": 0, "target": {"node": 0, "path": "rotation"}}], "samplers": [{"input": 0, "output": 1}]}
            for a in animations
        ],
    }
    jb = json.dumps(gltf).encode()
    jb += b" " * ((4 - len(jb) % 4) % 4)
    return struct.pack("<4sII", b"glTF", 2, 20 + len(jb)) + struct.pack("<I4s", len(jb), b"JSON") + jb


class MeshyMotionServer:
    def __init__(self, motion_rounds: int = 1):
        self.posts: list[tuple[str, dict]] = []
        self.motion_rounds = motion_rounds
        self.polls: dict[str, int] = {}

    def __call__(self, req: httpx.Request) -> httpx.Response:
        u = str(req.url)
        if req.method == "POST":
            body = json.loads(req.content.decode())
            self.posts.append((u.rsplit("/openapi/v1/", 1)[-1], body))
            if u.endswith("/text-to-motion"):
                if "forbidden" in body["prompt"]:
                    return httpx.Response(403, json={"message": "moderation"})
                return httpx.Response(202, json={"result": f"t2m-{len(self.posts)}"})
            if u.endswith("/animations"):
                if "motion_task_id" in body:
                    return httpx.Response(202, json={"result": f"anim-{body['motion_task_id']}"})
                return httpx.Response(202, json={"result": "anim-presets"})
        if req.method == "GET" and u.endswith("/animations/library"):
            return httpx.Response(200, json=[
                {"action_id": 17, "name": "Idle", "key": "Idle"},
                {"action_id": 42, "name": "Walking_Woman", "key": "Walking_Woman"},
            ])
        if req.method == "GET" and "/text-to-motion/" in u:
            tid = u.rsplit("/", 1)[-1]
            self.polls[tid] = self.polls.get(tid, 0) + 1
            if self.polls[tid] < self.motion_rounds:
                return httpx.Response(200, json={"id": tid, "status": "IN_PROGRESS", "progress": 50})
            return httpx.Response(200, json={
                "id": tid, "status": "SUCCEEDED", "consumed_credits": 10,
                "result": {"motion_url": f"https://cdn.mock/{tid}.fbx", "motion_format": "fbx", "duration_ms": 4000, "mode": "prime"},
            })
        if req.method == "GET" and "/animations/anim-" in u:
            aid = u.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"id": aid, "status": "SUCCEEDED", "result": {"animation_glb_url": f"https://cdn.mock/{aid}.glb"}})
        if req.method == "GET" and u == "https://cdn.mock/anim-presets.glb":
            return httpx.Response(200, content=glb(["Idle", "Walking_Woman"]))
        if req.method == "GET" and u.startswith("https://cdn.mock/anim-t2m"):
            return httpx.Response(200, content=glb(["Armature|mixamo.com|Layer0"]))
        return httpx.Response(404, json={"message": f"unhandled {u}"})


def make_provider(srv) -> providers.MeshyProvider:
    return providers.MeshyProvider(api_key="k", client=httpx.Client(transport=httpx.MockTransport(srv)))


def test_text_to_motion_payload_and_poll():
    srv = MeshyMotionServer()
    prov = make_provider(srv)
    tid = prov.submit_text_to_motion("x" * 500, duration=3.3)
    path, body = srv.posts[0]
    assert path == "text-to-motion"
    assert len(body["prompt"]) == 400
    assert body["duration"] == 3.5  # 0.5 s steps
    assert body["mode"] == "prime"
    res = make_provider(srv).poll(tid, kind="text-to-motion")
    assert res["status"] == "done" and res["url"].endswith(".fbx") and res["format"] == "fbx"


def test_text_to_motion_duration_clamped():
    srv = MeshyMotionServer()
    prov = make_provider(srv)
    prov.submit_text_to_motion("jump", duration=30)
    prov.submit_text_to_motion("jump", duration=0.4)
    assert [b["duration"] for _, b in srv.posts] == [10.0, 2.0]


def test_text_to_motion_moderation_is_agent_fixable():
    prov = make_provider(MeshyMotionServer())
    with pytest.raises(CFError) as e:
        prov.submit_text_to_motion("forbidden move")
    assert e.value.code == "ANIM_FAILED" and "Rephrase" in e.value.fix


def test_submit_animation_with_motion_task_id_sends_only_that():
    srv = MeshyMotionServer()
    make_provider(srv).submit_animation(rig_task_id="rig-1", motion_task_id="t2m-9")
    _, body = srv.posts[0]
    assert body == {"rig_task_id": "rig-1", "motion_task_id": "t2m-9"}


@pytest.fixture
def rigged_project(tmp_path, monkeypatch):
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")
    monkeypatch.setenv("MESHY_API_KEY", "k")
    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(b"x")
    pid = server.project_create(sheet_path=str(sheet), name="MotionGroup", height_cm=165.0)["project_id"]
    pdir = state.project_dir(pid)
    (pdir / "export").mkdir(exist_ok=True)
    (pdir / f"export/{pid}-rigged.glb").write_bytes(glb([]))
    p = state.load(pid)
    p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
    p["outputs"]["rig_task_id"] = "rig-1"
    p["stage"] = "rigged"
    state.save(p)
    return pid, pdir


def _mock_post(calls):
    def run(script, args, timeout=600):
        calls.append((script, dict(args)))
        if script == "merge_animations.py":
            assert args["mode"] == "post"
            names = args["actions"].split(",")
            assert len([c for c in args["clips"].split(",") if c]) == len(names) - 2
            Path(args["out"]).write_bytes(glb(names, skinned=False))
            return {"ok": True, "actions": names, "clip_count": len(names), "changed": True, "secondary": {"applied": False, "reason": "no hanging hair or skirt found"}}
        raise AssertionError(f"unexpected script {script}")
    return run


def test_animate_with_motions_merges_presets_and_motions(rigged_project, monkeypatch):
    pid, pdir = rigged_project
    srv = MeshyMotionServer(motion_rounds=2)
    monkeypatch.setattr(providers, "get_provider", lambda name, client=None: make_provider(srv))
    calls: list = []
    monkeypatch.setattr(blender, "run", _mock_post(calls))

    res = tools_full.animate(
        project_id=pid,
        actions=["idle", "walk"],
        motions=[{"prompt": "spins once with arms out", "name": "twirl"}, {"prompt": "bows politely", "duration": 3}],
        method="provider",
    )
    assert res["ok"], res
    assert res["data"]["status"] == "running"  # motions need a second poll
    gid = res["data"]["job_id"]
    assert gid.startswith("group-")
    assert res["next_action"]["args"]["job_id"] == gid

    res2 = tools_full.animate(project_id=pid, job_id=gid)
    assert res2["ok"], res2
    assert res2["data"]["status"] == "done"
    assert res2["data"]["actions"] == ["idle", "walk", "twirl", "motion_2"]
    assert tools_full.inspect_glb_animations(pdir / res2["data"]["mesh_path"]) == ["idle", "walk", "twirl", "motion_2"]
    assert set(res2["data"]["motion_task_ids"]) == {"twirl", "motion_2"}
    assert res2["next_action"] is None

    kinds = [path for path, _ in srv.posts]
    assert kinds.count("text-to-motion") == 2
    anim_bodies = [b for path, b in srv.posts if path == "animations"]
    assert {"rig_task_id": "rig-1", "action_ids": [17, 42]} in anim_bodies
    assert sum(1 for b in anim_bodies if "motion_task_id" in b) == 2
    p = state.load(pid)
    assert p["stage"] == "animated"
    assert [a["name"] for a in p["animations"]] == ["idle", "walk", "twirl", "motion_2"]
    assert p["animations"][2]["prompt"] == "spins once with arms out"
    assert [c[0] for c in calls] == ["merge_animations.py"]


def test_refused_motion_keeps_paid_tasks_and_retry_reuses_them(rigged_project, monkeypatch):
    pid, pdir = rigged_project
    srv = MeshyMotionServer()
    monkeypatch.setattr(providers, "get_provider", lambda name, client=None: make_provider(srv))
    monkeypatch.setattr(blender, "run", _mock_post([]))

    res = tools_full.animate(
        project_id=pid, actions=["idle", "walk"],
        motions=[{"prompt": "spins once", "name": "twirl"}, {"prompt": "forbidden move", "name": "bad"}], method="provider",
    )
    assert not res["ok"] and res["error"]["code"] == "ANIM_FAILED" and "Rephrase" in res["error"]["fix"]
    groups = [j for j in state.load(pid)["jobs"].values() if j["kind"] == "animate-group"]
    assert len(groups) == 1 and groups[0]["status"] == "failed"
    assert [m["name"] for m in groups[0]["parts"]["motions"]] == ["twirl"]  # the paid motion was recorded
    assert groups[0]["parts"]["presets"] is None  # motions go first: no preset credits spent on a refused request
    t2m_posts = [b for path, b in srv.posts if path == "text-to-motion"]

    res2 = tools_full.animate(
        project_id=pid, actions=["idle", "walk"],
        motions=[{"prompt": "spins once", "name": "twirl"}, {"prompt": "bows politely", "name": "bow"}], method="provider",
    )
    assert res2["ok"], res2
    new_t2m = [b for path, b in srv.posts if path == "text-to-motion"][len(t2m_posts):]
    assert [b["prompt"] for b in new_t2m] == ["bows politely"]  # "spins once" was reused, not bought again


def test_group_done_without_url_is_agent_fixable(rigged_project, monkeypatch):
    pid, _ = rigged_project
    srv = MeshyMotionServer()
    real = srv.__call__

    def no_url(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and "/animations/anim-presets" in str(req.url):
            return httpx.Response(200, json={"status": "SUCCEEDED", "result": {}})
        return real(req)

    client = httpx.Client(transport=httpx.MockTransport(no_url))
    monkeypatch.setattr(providers, "get_provider", lambda name, client_=None, **kw: providers.MeshyProvider(api_key="k", client=client))
    res = tools_full.animate(project_id=pid, actions=["idle"], motions=[{"prompt": "jumps"}], method="provider")
    assert not res["ok"]
    assert res["error"]["code"] == "PROVIDER_FAILED" and "no download URL" in res["error"]["message"]


def test_animate_motions_require_meshy(rigged_project, monkeypatch):
    pid, _ = rigged_project
    res = tools_full.animate(project_id=pid, motions=[{"prompt": "jumps"}], method="blender")
    assert not res["ok"]
    assert res["error"]["code"] == "NO_PROVIDER"
    assert "without motions" in res["error"]["fix"]


def test_animate_motions_only_skips_default_presets(rigged_project, monkeypatch):
    pid, pdir = rigged_project
    srv = MeshyMotionServer()
    monkeypatch.setattr(providers, "get_provider", lambda name, client=None: make_provider(srv))

    def run(script, args, timeout=600):
        names = args["actions"].split(",")
        Path(args["out"]).write_bytes(glb(names, skinned=False))
        return {"ok": True, "changed": True}

    monkeypatch.setattr(blender, "run", run)
    res = tools_full.animate(project_id=pid, motions=[{"prompt": "waves both hands", "name": "cheer"}, {"prompt": "sits down", "name": "sit"}])
    assert res["ok"], res
    assert res["data"]["actions"] == ["cheer", "sit"]
    assert all("action_ids" not in b for path, b in srv.posts if path == "animations")
