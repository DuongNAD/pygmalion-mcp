"""Independent visual critic: request shape, scoring, budget, failure handling, and how it feeds the policy.
No network and no SDK needed: a fake client stands in for anthropic.Anthropic()."""
import base64
import json
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from charforge import blender, config, critic, server, state, tools_full


class FakeResponse:
    def __init__(self, answer=None, stop_reason="end_turn", text=None):
        body = text if text is not None else json.dumps(answer)
        self.content = [types.SimpleNamespace(type="thinking", thinking=""), types.SimpleNamespace(type="text", text=body)]
        self.stop_reason = stop_reason
        self.stop_details = types.SimpleNamespace(category="cyber") if stop_reason == "refusal" else None
        self.model = "claude-opus-5-5"
        self.usage = types.SimpleNamespace(input_tokens=1200, output_tokens=300)


class FakeClient:
    def __init__(self, *responses, raises=None):
        self.calls, self.responses, self.raises = [], list(responses), raises
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return self.responses.pop(0)


MESH_ANSWER = {
    "scores": {"face": 8, "texture": 7, "details": 6, "anatomy": 12, "artifacts": -1},
    "issues": ["front: left hand fused to the hip"],
    "summary": "close",
}


@pytest.fixture
def img(tmp_path):
    path = tmp_path / "compare.png"
    cv2.imwrite(str(path), np.full((1000, 3000, 4), 200, np.uint8))
    return path


@pytest.fixture
def fake(monkeypatch):
    def install(*responses, raises=None):
        client = FakeClient(*responses, raises=raises)
        monkeypatch.setattr(critic, "_client_override", client)
        return client
    monkeypatch.delenv("CHARFORGE_CRITIC", raising=False)
    return install


def test_mesh_request_shape_and_scoring(fake, img):
    client = fake(FakeResponse(MESH_ANSWER))
    p = {}
    res = critic.critique_mesh(p, img)
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["betas"] == [critic.FALLBACK_BETA] and call["fallbacks"] == "default"
    fmt = call["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["required"] == ["scores", "issues", "summary"]
    assert call["output_config"]["effort"] == "medium"
    blocks = call["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["image", "text"]
    sent = cv2.imdecode(np.frombuffer(base64.b64decode(blocks[0]["source"]["data"]), np.uint8), cv2.IMREAD_UNCHANGED)
    assert max(sent.shape[:2]) <= config.CRITIC["max_image_px"] and sent.shape[2] == 3  # downscaled, alpha flattened
    assert res["scores"]["anatomy"] == 10.0 and res["scores"]["artifacts"] == 0.0  # clamped to 0-10
    w = config.CRITIC["mesh_weights"]
    expected = (w["face"] * 8 + w["texture"] * 7 + w["details"] * 6 + w["anatomy"] * 10) / sum(w.values())
    assert res["overall"] == round(expected, 2)
    assert res["issues"] == ["front: left hand fused to the hip"]
    assert res["usage"]["input_tokens"] == 1200
    assert p["critic_calls"] == 1


def test_refusal_and_bad_answers_fail_cleanly(fake, img):
    fake(FakeResponse(MESH_ANSWER, stop_reason="refusal"))
    with pytest.raises(critic.CriticFailed, match="declined"):
        critic.critique_mesh({}, img)
    fake(FakeResponse(text="not json"))
    with pytest.raises(critic.CriticFailed, match="not valid JSON"):
        critic.critique_mesh({}, img)


def test_budget_and_modes(fake, img, monkeypatch):
    fake(FakeResponse(MESH_ANSWER))
    p = {"critic_calls": config.CRITIC["max_calls"]}
    with pytest.raises(critic.CriticUnavailable, match="budget"):
        critic.critique_mesh(p, img)
    monkeypatch.setenv("CHARFORGE_CRITIC", "off")
    with pytest.raises(critic.CriticUnavailable, match="off"):
        critic.check({})
    monkeypatch.setenv("CHARFORGE_CRITIC", "auto")
    monkeypatch.setattr(critic, "_client_override", None)
    monkeypatch.setattr(critic, "_sdk", lambda: None)
    with pytest.raises(critic.CriticUnavailable, match="charforge\\[critic\\]"):
        critic.check({})
    monkeypatch.setattr(critic, "_sdk", lambda: types.SimpleNamespace())
    monkeypatch.setattr(critic, "_has_credentials", lambda: False)
    with pytest.raises(critic.CriticUnavailable, match="credentials"):
        critic.check({})


def test_api_errors_map_to_short_reasons(fake, img, monkeypatch):
    class APIConnectionError(Exception):
        pass

    class APIStatusError(Exception):
        def __init__(self, status_code):
            super().__init__(status_code)
            self.status_code = status_code

    class AuthenticationError(APIStatusError):
        pass

    class RateLimitError(APIStatusError):
        pass

    sdk = types.SimpleNamespace(APIConnectionError=APIConnectionError, APIStatusError=APIStatusError,
                                AuthenticationError=AuthenticationError, RateLimitError=RateLimitError)
    monkeypatch.setattr(critic, "_sdk", lambda: sdk)
    for exc, text in ((RateLimitError(429), "rate limit"), (AuthenticationError(401), "credentials"),
                      (APIStatusError(500), "error 500"), (APIConnectionError(), "network")):
        fake(raises=exc)
        with pytest.raises(critic.CriticFailed, match=text):
            critic.critique_mesh({}, img)


def test_motion_clips_mapped_by_name_then_position(fake, img):
    fake(FakeResponse({
        "clips": [
            {"name": "walk", "scores": {"pose": 9, "clipping": 3, "deformation": 7, "secondary": 10}, "issues": ["frame 3, side: knee bends backwards"]},
            {"name": "Clip 2", "scores": {"pose": 8, "clipping": 8, "deformation": 8, "secondary": 8}, "issues": []},
        ],
        "summary": "ok",
    }))
    res = critic.critique_motion({}, {"walk": img, "wave": img})
    assert set(res["clips"]) == {"walk", "wave"}
    assert res["clips"]["walk"]["overall"] == 7.25 and res["clips"]["wave"]["overall"] == 8.0


# ---------------------------------------------------------------- policy

def _p(tmp_path, monkeypatch):
    monkeypatch.setenv("CHARFORGE_LEVEL", "2")
    sheet = tmp_path / "s.png"
    sheet.write_bytes(b"x")
    p = state.load(server.project_create(sheet_path=str(sheet), name="Rank", height_cm=165.0)["project_id"])
    p["stage"] = "views_ready"
    for n in (1, 2, 3):
        state.register_mesh(p, f"meshes/cand-{n}-meshy.glb", source="meshy", task_id=f"t{n}")
    return p


def rec(n, total, critic_overall=None, integrity_ok=True):
    r = {"mesh": f"meshes/cand-{n}-meshy.glb", "total": total, "passed": True, "iou": {}, "integrity": {"ok": integrity_ok}}
    if critic_overall is not None:
        r["critic"] = {"overall": critic_overall}
    return r


def test_critic_breaks_ties_only_beyond_min_delta(tmp_path, monkeypatch):
    p = _p(tmp_path, monkeypatch)
    state.record_score(p, rec(1, 0.90, critic_overall=6.0))
    state.record_score(p, rec(2, 0.85, critic_overall=6.5))  # +0.5 < min_delta: silhouette total decides
    assert p["best"]["mesh"].endswith("cand-1-meshy.glb")
    state.record_score(p, rec(3, 0.84, critic_overall=7.5))  # +1.5 >= min_delta: the critic decides
    assert p["best"]["mesh"].endswith("cand-3-meshy.glb") and p["best"]["critic_overall"] == 7.5


def test_detached_parts_lose_to_sound_geometry(tmp_path, monkeypatch):
    p = _p(tmp_path, monkeypatch)
    state.record_score(p, rec(1, 0.95, integrity_ok=False))
    state.record_score(p, rec(2, 0.82))
    assert p["best"]["mesh"].endswith("cand-2-meshy.glb") and p["best"]["integrity_ok"] is True
    assert p["stage"] == "passed"  # integrity ranks meshes; it never triggers paid regeneration


# ---------------------------------------------------------------- score_mesh / animate wiring

def test_score_mesh_reports_integrity_and_critic(sheet_path, monkeypatch, fake):
    monkeypatch.setenv("CHARFORGE_LEVEL", "2")
    client = fake(FakeResponse({**MESH_ANSWER, "scores": {"face": 3, "texture": 4, "details": 5, "anatomy": 5, "artifacts": 6}}))
    pid = server.project_create(sheet_path=str(sheet_path), name="CriticScore", height_cm=165.0)["project_id"]
    server.sheet_split(project_id=pid)
    pdir = state.project_dir(pid)
    (pdir / "meshes" / "cand-1-meshy.glb").write_bytes(b"glb")
    p = state.load(pid)
    state.register_mesh(p, "meshes/cand-1-meshy.glb", source="meshy", task_id="t1")
    state.save(p)
    modes = []

    def run(script, args, timeout=600):
        out = Path(args["out"])
        out.mkdir(parents=True, exist_ok=True)
        modes.append(args["mode"])
        files = {}
        for v in ("front", "side", "back"):
            src = pdir / "views" / (f"{v}_mask.png" if args["mode"] == "mask" else f"{v}.png")
            dst = out / f"{v}_{args['mode']}.png"
            dst.write_bytes(src.read_bytes())
            files[v] = str(dst)
        res = {"ok": True, "files": files, "dims_m": [0.5, 0.3, 1.65], "tris": 20000}
        if args.get("integrity"):
            assert args["gap_frac"] == config.INTEGRITY["gap_frac"]
            res["integrity"] = {"components": 3, "debris_parts": 1, "debris_pct": 0.05, "detached_parts": 1, "detached_pct": 2.1,
                                "vertices": 9000, "edges": 27000, "boundary_edges": 0, "holes": 0, "non_manifold_edges": 0,
                                "faces": 18000, "self_intersecting_faces": 10, "interpenetrating_faces": 40, "degenerate_faces": 0}
        return res

    monkeypatch.setattr(blender, "run", run)
    res = server.score_mesh(project_id=pid, mesh_path="meshes/cand-1-meshy.glb")
    assert res["ok"], res
    d = res["data"]
    assert d["passed"] is True
    assert d["integrity"]["ok"] is False and d["integrity"]["detached_parts"] == 1
    assert any("detached part" in w for w in res["warnings"]) and any("debris" in w for w in res["warnings"])
    assert modes == ["mask", "color"]  # the critic gets a color render, after the mask scoring
    assert d["critic"]["overall"] < config.CRITIC["warn_below"] and d["critic"]["compare"].endswith("compare_color.png")
    assert any(w.startswith("Visual critic:") for w in res["warnings"])
    assert len(client.calls) == 1
    p = state.load(pid)
    assert p["critic_calls"] == 1 and p["best"]["integrity_ok"] is False
    assert server.project_status(project_id=pid)["data"]["qa"]["critic_calls"] == 1


def test_score_mesh_without_critic_sdk_is_silent(sheet_path, monkeypatch):
    monkeypatch.setenv("CHARFORGE_LEVEL", "2")
    monkeypatch.delenv("CHARFORGE_CRITIC", raising=False)
    monkeypatch.setattr(critic, "_client_override", None)
    monkeypatch.setattr(critic, "_sdk", lambda: None)
    pid = server.project_create(sheet_path=str(sheet_path), name="NoCritic", height_cm=165.0)["project_id"]
    server.sheet_split(project_id=pid)
    pdir = state.project_dir(pid)
    (pdir / "meshes" / "cand-1-meshy.glb").write_bytes(b"glb")
    p = state.load(pid)
    state.register_mesh(p, "meshes/cand-1-meshy.glb", source="meshy", task_id="t1")
    state.save(p)

    def run(script, args, timeout=600):
        assert args["mode"] == "mask", "no color render when the critic cannot run"
        out = Path(args["out"])
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        for v in ("front", "side", "back"):
            dst = out / f"{v}_mask.png"
            dst.write_bytes((pdir / "views" / f"{v}_mask.png").read_bytes())
            files[v] = str(dst)
        return {"ok": True, "files": files, "dims_m": [0.5, 0.3, 1.65], "tris": 20000}

    monkeypatch.setattr(blender, "run", run)
    res = server.score_mesh(project_id=pid, mesh_path="meshes/cand-1-meshy.glb")
    assert res["ok"] and res["data"]["critic"]["skipped"].startswith("critic needs the anthropic SDK")
    assert not any("critic" in w.lower() for w in res["warnings"])


def test_motion_critic_warns_on_low_clips(fake, img):
    fake(FakeResponse({
        "clips": [
            {"name": "walk", "scores": {"pose": 4, "clipping": 3, "deformation": 4, "secondary": 5}, "issues": ["frame 2, front: arms pass through the skirt"]},
            {"name": "idle", "scores": {"pose": 9, "clipping": 9, "deformation": 9, "secondary": 9}, "issues": []},
        ],
        "summary": "walk has clipping",
    }))
    pdir = img.parent
    res, warnings = tools_full._motion_critic({}, pdir, {"walk": img.name, "idle": img.name})
    assert res["clips"]["walk"]["overall"] == 4.0
    assert warnings == ["Visual critic: walk 4.0/10; frame 2, front: arms pass through the skirt"]
