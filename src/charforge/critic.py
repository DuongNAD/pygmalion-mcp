"""Independent visual critic: a vision model grades renders against the reference with a fixed rubric.

The driving agent never grades its own output (the "maker" must not be its own judge). The critic only sees
images and the rubric below, never the agent's reasoning. It is advisory: silhouette scores still decide
`passed`; the critic breaks ties between passed provider meshes and flags animation problems.

Backends (config.CRITIC["backend"], env CHARFORGE_CRITIC_BACKEND; auto = the first ready one in BACKENDS order):
- gemini: Gemini through the official `google-genai` SDK, Interactions API (pip install "charforge[critic]", GEMINI_API_KEY).
- claude: Claude through the official `anthropic` SDK (pip install "charforge[critic-claude]", ANTHROPIC_API_KEY).
Mode (config.CRITIC["mode"], env CHARFORGE_CRITIC): auto = on when a backend SDK and its key are present,
on = always try, off = never. Every call is counted against a per-project budget (CRITIC["max_calls"]).
"""
from __future__ import annotations

import base64
import importlib.metadata as md
import json
import os
from pathlib import Path
from typing import Any

import cv2

from . import config

BACKENDS = ("gemini", "claude")  # auto picks the first ready backend, in this order
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # Claude: server-side model fallbacks
GEMINI_LEVELS = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}  # effort -> thinking_level
_DIST = {"gemini": "google-genai", "claude": "anthropic"}
_INSTALL = {
    "gemini": 'critic needs the google-genai SDK (>= 2.18): pip install "charforge[critic]"',
    "claude": 'critic needs the anthropic SDK: pip install "charforge[critic-claude]"',
}
_NO_KEY = {
    "gemini": "critic needs a Gemini API key (GEMINI_API_KEY)",
    "claude": "critic needs Claude credentials (ANTHROPIC_API_KEY or `ant auth login`)",
}

MESH_PROMPT = """You are a strict QA reviewer for 3D game characters.
The image has one row per view (front, side, back). In each row the LEFT picture is the 2D reference
drawing of the character and the RIGHT picture is a flat-lit render of the generated 3D model.
Grade how faithfully the 3D model reproduces the reference. Score each criterion from 0 to 10
(10 = matches the reference, 5 = recognizable with clear defects, 0 = wrong or broken):
- face: face shape and features (eyes, brows, mouth, nose) match; no melted or smeared face.
- texture: colors and patterns on clothes, skin and hair match and sit in the right places; no smears,
  visible seams, baked shadows or wrong colors.
- details: hair shape, accessories, clothing layers and small details are present and placed correctly.
- anatomy: right number and shape of limbs, hands and fingers; nothing extra, missing or fused together.
- artifacts: clean geometry: no holes, floating fragments, spikes, melted or doubled surfaces.
Judge only the character; ignore lighting, background and small differences in pose.
List at most 6 concrete issues, each naming the body part and the view, e.g. "front: left hand fused to the hip".
"""

MOTION_PROMPT = """You are a strict QA reviewer for 3D character animation.
Each image is one animation clip: its name is written at the top, the rows are front and side views,
the columns are frames in time order. Grade every clip. Score each criterion from 0 to 10
(10 = no problem, 5 = noticeable problem, 0 = broken):
- pose: poses look natural, balanced and anatomically possible.
- clipping: no limbs, hair or clothes passing through the body or through each other.
- deformation: no stretched, collapsed or twisted joints; hands and arms do not stick to the body;
  no tearing or ballooning of the surface.
- secondary: hair, skirt and cloth move plausibly with the body (10 if the character has none).
List at most 4 concrete issues per clip, naming the frame column and body part, e.g.
"frame 3, side: right forearm sinks into the thigh".
"""

MESH_KEYS = ("face", "texture", "details", "anatomy", "artifacts")
MOTION_KEYS = ("pose", "clipping", "deformation", "secondary")


def _scores_schema(keys: tuple[str, ...]) -> dict:
    return {
        "type": "object",
        "properties": {k: {"type": "integer"} for k in keys},
        "required": list(keys),
        "additionalProperties": False,
    }


MESH_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": _scores_schema(MESH_KEYS),
        "issues": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["scores", "issues", "summary"],
    "additionalProperties": False,
}

MOTION_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "scores": _scores_schema(MOTION_KEYS),
                    "issues": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "scores", "issues"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["clips", "summary"],
    "additionalProperties": False,
}


class CriticUnavailable(Exception):
    """The critic cannot run here (off, no SDK, no credentials, budget spent); the reason is the message."""


class CriticFailed(Exception):
    """A critic call was made and failed (refusal, API error, unreadable answer)."""


_client_override: dict[str, Any] = {}  # tests inject fake or mock-transport clients here, per backend


def _sdk(backend: str):
    try:
        if backend == "gemini":
            from google import genai  # optional dependency
            return genai if hasattr(genai.Client, "interactions") else None  # Interactions API: google-genai >= 2.18
        import anthropic  # optional dependency
        return anthropic
    except ImportError:
        return None


def settings(p: dict | None = None) -> dict:
    s = {**config.CRITIC, **((p or {}).get("critic_params") or {})}
    env = (os.environ.get("CHARFORGE_CRITIC") or "").strip().lower()
    if env in ("auto", "on", "off"):
        s["mode"] = env
    env = (os.environ.get("CHARFORGE_CRITIC_BACKEND") or "").strip().lower()
    if env in ("auto", *BACKENDS):
        s["backend"] = env
    return s


def _model(s: dict, backend: str) -> str:
    return (s.get("models") or {}).get(backend) or config.CRITIC["models"][backend]


def _has_credentials(backend: str) -> bool:
    if backend == "gemini":
        return any(os.environ.get(k) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    if any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")):
        return True
    return (Path.home() / ".config" / "anthropic").is_dir()  # profile from `ant auth login`


def check(p: dict | None = None) -> str:
    """The backend the next critic call uses; raises CriticUnavailable with the reason when no call may be made."""
    s = settings(p)
    if s["mode"] == "off":
        raise CriticUnavailable("critic is off (CHARFORGE_CRITIC=off)")
    if p is not None and (p.get("critic_calls") or 0) >= int(s["max_calls"]):
        raise CriticUnavailable(f"critic budget spent ({s['max_calls']} calls for this project)")
    wanted = [s["backend"]] if s["backend"] in BACKENDS else list(BACKENDS)
    for b in wanted:
        if b in _client_override or (_sdk(b) is not None and (s["mode"] == "on" or _has_credentials(b))):
            return b
    installed = [b for b in wanted if _sdk(b) is not None]
    if installed:  # the SDK is there, the key is what is missing
        raise CriticUnavailable(_NO_KEY[installed[0]])
    if len(wanted) == 1:
        raise CriticUnavailable(_INSTALL[wanted[0]])
    raise CriticUnavailable('critic needs a vision model: GEMINI_API_KEY + pip install "charforge[critic]", '
                            'or ANTHROPIC_API_KEY + pip install "charforge[critic-claude]"')


def status() -> dict:
    """Environment summary for `python -m charforge --check`."""
    s = settings()
    try:
        backend, reason = check(None), None
    except CriticUnavailable as e:
        backend, reason = None, str(e)
    sdks = {}
    for b in BACKENDS:
        try:
            sdks[b] = md.version(_DIST[b]) if _sdk(b) is not None else None
        except md.PackageNotFoundError:
            sdks[b] = None
    return {
        "mode": s["mode"],
        "backend": backend,
        "model": _model(s, backend) if backend else None,
        "reason": reason,
        "sdks": sdks,
        "keys": {b: _has_credentials(b) for b in BACKENDS},
    }


def _png_b64(path: Path, max_px: int) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise CriticFailed(f"cannot read image {path}")
    if img.ndim == 3 and img.shape[2] == 4:  # composite transparent renders on white
        a = img[..., 3:] / 255.0
        img = (img[..., :3] * a + 255 * (1 - a)).astype("uint8")
    h, w = img.shape[:2]
    scale = min(1.0, max_px / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise CriticFailed(f"cannot encode image {path}")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def _ask(p: dict | None, images: list[Path], prompt: str, schema: dict) -> tuple[dict, dict]:
    """One structured critic call. Returns (answer, usage). Counts against the project budget."""
    backend = check(p)
    s = settings(p)
    data = [_png_b64(Path(i), int(s["max_image_px"])) for i in images]
    if p is not None:
        p["critic_calls"] = (p.get("critic_calls") or 0) + 1
    ask = _ask_gemini if backend == "gemini" else _ask_claude
    text, usage = ask(s, _model(s, backend), data, prompt, schema)
    try:
        answer = json.loads(text)
    except json.JSONDecodeError:
        raise CriticFailed("the critic answer is not valid JSON") from None
    if not isinstance(answer, dict):
        raise CriticFailed("the critic answer is not a JSON object")
    return answer, {"backend": backend, **usage}


def _ask_gemini(s: dict, model: str, data: list[str], prompt: str, schema: dict) -> tuple[str, dict]:
    client = _client_override.get("gemini")
    if client is None:
        try:
            client = _sdk("gemini").Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        except ValueError:
            raise CriticFailed("no Gemini API key (set GEMINI_API_KEY)") from None
    content = [{"type": "text", "text": prompt}] + [{"type": "image", "data": d, "mime_type": "image/png"} for d in data]
    try:
        it = client.interactions.create(
            model=model,
            input=content,
            response_format={"type": "text", "mime_type": "application/json", "schema": schema},
            generation_config={"thinking_level": GEMINI_LEVELS.get(s["effort"], "medium"), "max_output_tokens": 16000},
            store=False,  # stateless call: the interaction is not kept for later retrieval
            timeout=float(s["timeout_s"]),
        )
    except _gemini_error_types() as e:  # typed SDK errors only; anything else is a bug and should surface
        raise CriticFailed(_gemini_error(e, model)) from None
    state = str(getattr(it, "status", "") or "")
    if state != "completed":
        errs = "; ".join(str(getattr(e, "message", e)) for e in (getattr(it, "errors", None) or [])[:2])
        raise CriticFailed(f"the critic did not finish (status: {state or 'unknown'}{'; ' + errs if errs else ''})")
    text = getattr(it, "output_text", None)
    if not text:
        raise CriticFailed("the critic returned no answer (status: completed)")
    u = getattr(it, "usage", None)
    return text, {
        "model": getattr(it, "model", None) or model,
        "input_tokens": getattr(u, "total_input_tokens", None),
        "output_tokens": getattr(u, "total_output_tokens", None),
        "thought_tokens": getattr(u, "total_thought_tokens", None),
    }


def _gemini_error_types() -> tuple[type[BaseException], ...]:
    try:
        from google.genai import errors
    except ImportError:
        return ()
    found: list[type[BaseException]] = [errors.APIError]
    try:  # Interactions API errors have no public import path in google-genai 2.x
        from google.genai._gaos.lib.compat_errors import GeminiNextGenAPIClientError
        found.append(GeminiNextGenAPIClientError)
    except ImportError:
        pass
    return tuple(found)


def _gemini_error(e: Exception, model: str) -> str:
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    if code == 401 or "API key" in str(getattr(e, "message", None) or e):
        return "Gemini API key was rejected (check GEMINI_API_KEY)"
    if code in (403, 404):
        return f"Gemini model {model} is not available to this key (set CRITIC models.gemini in config.py)"
    if code == 429:
        return "Gemini API rate limit or quota reached; critic skipped"
    if isinstance(code, int):
        return f"Gemini API error {code}"
    return "network error reaching the Gemini API"


def _ask_claude(s: dict, model: str, data: list[str], prompt: str, schema: dict) -> tuple[str, dict]:
    sdk = _sdk("claude")
    client = _client_override.get("claude") or sdk.Anthropic()
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": d}} for d in data]
    content.append({"type": "text", "text": prompt})
    errors = (sdk.APIConnectionError, sdk.APIStatusError) if sdk is not None else ()
    try:
        resp = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            output_config={"effort": s["effort"], "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": content}],
        )
    except errors as e:  # typed SDK errors only; anything else is a bug and should surface
        raise CriticFailed(_claude_error(sdk, e)) from None
    if resp.stop_reason == "refusal":
        cat = getattr(getattr(resp, "stop_details", None), "category", None)
        raise CriticFailed(f"the critic model declined to review (category: {cat})")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        raise CriticFailed(f"the critic returned no answer (stop_reason: {resp.stop_reason})")
    usage = getattr(resp, "usage", None)
    return text, {
        "model": getattr(resp, "model", model),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


def _claude_error(sdk, e: Exception) -> str:
    if isinstance(e, sdk.AuthenticationError):
        return "Claude credentials were rejected (check ANTHROPIC_API_KEY)"
    if isinstance(e, sdk.RateLimitError):
        return "Claude API rate limit reached; critic skipped"
    if isinstance(e, sdk.APIStatusError):
        return f"Claude API error {e.status_code}"
    return "network error reaching the Claude API"


def _clamp(v: Any) -> float:
    try:
        return max(0.0, min(10.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def critique_mesh(p: dict | None, compare_image: Path) -> dict:
    """Grade a reference|render comparison sheet. Returns {overall, scores, issues, summary, usage}."""
    answer, usage = _ask(p, [compare_image], MESH_PROMPT, MESH_SCHEMA)
    w = settings(p)["mesh_weights"]
    scores = {k: _clamp((answer.get("scores") or {}).get(k)) for k in MESH_KEYS}
    overall = sum(w[k] * scores[k] for k in MESH_KEYS) / max(1e-9, sum(w.values()))
    return {
        "overall": round(overall, 2),
        "scores": scores,
        "issues": [str(i) for i in (answer.get("issues") or [])][:6],
        "summary": str(answer.get("summary") or ""),
        "usage": usage,
    }


def critique_motion(p: dict | None, sheets: dict[str, Path]) -> dict:
    """Grade clip contact sheets. Returns {clips: {name: {overall, scores, issues}}, summary, usage}."""
    names = list(sheets)
    answer, usage = _ask(p, [sheets[n] for n in names], MOTION_PROMPT + f"\nThe clips are, in order: {', '.join(names)}.", MOTION_SCHEMA)
    clips = {}
    for i, c in enumerate(answer.get("clips") or []):
        name = c.get("name") if c.get("name") in sheets else (names[i] if i < len(names) else None)
        if name is None:
            continue
        scores = {k: _clamp((c.get("scores") or {}).get(k)) for k in MOTION_KEYS}
        clips[name] = {
            "overall": round(sum(scores.values()) / len(MOTION_KEYS), 2),
            "scores": scores,
            "issues": [str(x) for x in (c.get("issues") or [])][:4],
        }
    return {"clips": clips, "summary": str(answer.get("summary") or ""), "usage": usage}
