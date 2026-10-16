"""Independent visual critic: request shape, scoring, budget, failure handling, backend choice, and how it feeds
the policy. No network: a fake client stands in for anthropic.Anthropic(), and the real google-genai SDK talks to an
httpx mock transport (those tests skip when google-genai is not installed)."""
import base64
import json
import types
from pathlib import Path

import cv2
import httpx
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
        monkeypatch.setattr(critic, "_client_override", {"claude": client})
        return client
    monkeypatch.delenv("CHARFORGE_CRITIC", raising=False)
    monkeypatch.setenv("CHARFORGE_CRITIC_BACKEND", "claude")
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
    assert res["usage"]["input_tokens"] == 1200 and res["usage"]["backend"] == "claude"
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
    monkeypatch.setattr(critic, "_client_override", {})
    monkeypatch.setattr(critic, "_sdk", lambda b: None)
    with pytest.raises(critic.CriticUnavailable, match="charforge\\[critic-claude\\]"):
        critic.check({})
    monkeypatch.setattr(critic, "_sdk", lambda b: types.SimpleNamespace())
    monkeypatch.setattr(critic, "_has_credentials", lambda b: False)
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
    monkeypatch.setattr(critic, "_sdk", lambda b: sdk)
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


# ---------------------------------------------------------------- Gemini backend (real SDK, mock HTTP)

def interaction(answer=None, status="completed", text=None, errors=None):
    """An Interactions API reply as the Gemini API sends it."""
    body = json.dumps(answer) if text is None else text
    reply = {
        "id": "int-1", "object": "interaction", "status": status, "model": "gemini-4-argon",
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": body}]}] if body else [],
        "usage": {"total_input_tokens": 2100, "total_output_tokens": 180, "total_thought_tokens": 900, "total_tokens": 3180},
    }
    if errors:
        reply["errors"] = errors
    return reply


def api_error(code, status, message):
    return httpx.Response(code, json={"error": {"code": code, "message": message, "status": status}})


@pytest.fixture
def gemini(monkeypatch):
    genai = pytest.importorskip("google.genai")
    if not hasattr(genai.Client, "interactions"):
        pytest.skip("google-genai without the Interactions API")
    from google.genai import types as gtypes
    monkeypatch.delenv("CHARFORGE_CRITIC", raising=False)
    monkeypatch.setenv("CHARFORGE_CRITIC_BACKEND", "gemini")

    def install(*replies):
        seen, queue = [], list(replies)

        def handler(request):
            seen.append(request)
            reply = queue.pop(0) if len(queue) > 1 else queue[0]  # SDK retries see the last reply again
            if reply == "network":
                raise httpx.ConnectError("unreachable", request=request)
            return reply if isinstance(reply, httpx.Response) else httpx.Response(200, json=reply)

        http = gtypes.HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
                                  retry_options=gtypes.HttpRetryOptions(attempts=1))
        monkeypatch.setattr(critic, "_client_override", {"gemini": genai.Client(api_key="test-key", http_options=http)})
        return seen
    return install


def test_gemini_request_shape_and_scoring(gemini, img):
    seen = gemini(interaction(MESH_ANSWER))
    p = {}
    res = critic.critique_mesh(p, img)
    req = seen[0]
    assert req.url.path.endswith("/interactions") and req.headers["x-goog-api-key"] == "test-key"
    body = json.loads(req.content)
    assert body["model"] == config.CRITIC["models"]["gemini"] == "gemini-4-argon"
    assert body["store"] is False  # the interaction is not kept server side
    assert body["response_format"] == {"type": "text", "mime_type": "application/json", "schema": critic.MESH_SCHEMA}
    assert body["generation_config"] == {"thinking_level": "medium", "max_output_tokens": 16000}
    assert body["input"][0]["type"] == "user_input"
    parts = body["input"][0]["content"]
    assert [x["type"] for x in parts] == ["text", "image"] and parts[0]["text"] == critic.MESH_PROMPT
    assert parts[1]["mime_type"] == "image/png"
    sent = cv2.imdecode(np.frombuffer(base64.b64decode(parts[1]["data"]), np.uint8), cv2.IMREAD_UNCHANGED)
    assert max(sent.shape[:2]) <= config.CRITIC["max_image_px"] and sent.shape[2] == 3  # downscaled, alpha flattened
    assert res["scores"]["anatomy"] == 10.0 and res["issues"] == ["front: left hand fused to the hip"]
    assert res["usage"] == {"backend": "gemini", "model": "gemini-4-argon", "input_tokens": 2100,
                            "output_tokens": 180, "thought_tokens": 900}
    assert p["critic_calls"] == 1


def test_gemini_motion_sends_one_image_per_clip(gemini, img):
    seen = gemini(interaction({
        "clips": [{"name": n, "scores": {"pose": 8, "clipping": 8, "deformation": 8, "secondary": 8}, "issues": []} for n in ("walk", "wave")],
        "summary": "ok",
    }))
    res = critic.critique_motion({}, {"walk": img, "wave": img})
    body = json.loads(seen[0].content)
    parts = body["input"][0]["content"]
    assert [x["type"] for x in parts] == ["text", "image", "image"] and parts[0]["text"].endswith("walk, wave.")
    assert body["response_format"]["schema"] == critic.MOTION_SCHEMA
    assert set(res["clips"]) == {"walk", "wave"} and res["clips"]["walk"]["overall"] == 8.0


def test_project_can_pick_another_gemini_model(gemini, img):
    seen = gemini(interaction(MESH_ANSWER))
    critic.critique_mesh({"critic_params": {"models": {"gemini": "gemini-3.8-flash"}, "effort": "high"}}, img)
    body = json.loads(seen[0].content)
    assert body["model"] == "gemini-3.8-flash" and body["generation_config"]["thinking_level"] == "high"


@pytest.mark.parametrize("reply, reason", [
    (interaction(status="incomplete", text=""), r"did not finish \(status: incomplete\)"),
    (interaction(status="failed", text="", errors=[{"code": "blocked", "message": "image blocked"}]), "status: failed; image blocked"),
    (interaction(text="not json"), "not valid JSON"),
    (api_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."), "key was rejected"),
    (api_error(404, "NOT_FOUND", "models/gemini-4-argon is not found"), "gemini-4-argon is not available"),
    (api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded"), "rate limit or quota"),
    (api_error(500, "INTERNAL", "boom"), "Gemini API error 500"),
    ("network", "network error"),
])
def test_gemini_failures_map_to_short_reasons(gemini, img, reply, reason):
    gemini(reply)
    p = {}
    with pytest.raises(critic.CriticFailed, match=reason):
        critic.critique_mesh(p, img)
    assert p["critic_calls"] == 1  # the call was made, so it counts against the budget


def test_backend_choice(monkeypatch):
    monkeypatch.delenv("CHARFORGE_CRITIC", raising=False)
    monkeypatch.delenv("CHARFORGE_CRITIC_BACKEND", raising=False)
    keys = {"gemini": False, "claude": False}
    installed = {"gemini": True, "claude": True}
    monkeypatch.setattr(critic, "_client_override", {})
    monkeypatch.setattr(critic, "_sdk", lambda b: types.SimpleNamespace() if installed[b] else None)
    monkeypatch.setattr(critic, "_has_credentials", lambda b: keys[b])
    with pytest.raises(critic.CriticUnavailable, match="GEMINI_API_KEY"):  # SDKs installed, no key: say which key
        critic.check({})
    keys["claude"] = True
    assert critic.check({}) == "claude"
    keys["gemini"] = True
    assert critic.check({}) == "gemini"  # both ready: Gemini comes first
    monkeypatch.setenv("CHARFORGE_CRITIC_BACKEND", "claude")
    assert critic.check({}) == "claude"
    monkeypatch.setenv("CHARFORGE_CRITIC_BACKEND", "gemini")
    keys["gemini"] = False
    with pytest.raises(critic.CriticUnavailable, match="GEMINI_API_KEY"):
        critic.check({})
    monkeypatch.setenv("CHARFORGE_CRITIC", "on")  # on: try even without a key
    assert critic.check({}) == "gemini"
    installed["gemini"] = False
    with pytest.raises(critic.CriticUnavailable, match="google-genai"):
        critic.check({})
    monkeypatch.delenv("CHARFORGE_CRITIC_BACKEND")
    assert critic.check({}) == "claude"  # auto + on: the first backend whose SDK is installed
    installed["claude"] = False
    with pytest.raises(critic.CriticUnavailable, match="vision model"):
        critic.check({})
    s = critic.status()
    assert s["backend"] is None and s["reason"].startswith("critic needs a vision model")
    assert s["keys"] == {"gemini": False, "claude": True} and s["sdks"] == {"gemini": None, "claude": None}


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
    monkeypatch.delenv("CHARFORGE_CRITIC_BACKEND", raising=False)
    monkeypatch.setattr(critic, "_client_override", {})
    monkeypatch.setattr(critic, "_sdk", lambda b: None)
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
    assert res["ok"] and res["data"]["critic"]["skipped"].startswith("critic needs a vision model")
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
