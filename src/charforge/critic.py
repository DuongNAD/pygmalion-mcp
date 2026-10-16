"""Independent visual critic: a vision model grades renders against the reference with a fixed rubric.

The driving agent never grades its own output (the "maker" must not be its own judge). The critic only sees
images and the rubric below, never the agent's reasoning. It is advisory: silhouette scores still decide
`passed`; the critic breaks ties between passed provider meshes and flags animation problems.

Backend: Claude via the official `anthropic` SDK (optional dependency: pip install "charforge[critic]").
Mode (config.CRITIC["mode"], env CHARFORGE_CRITIC): auto = on when the SDK and credentials are present,
on = always try, off = never. Every call is counted against a per-project budget (CRITIC["max_calls"]).
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import cv2

from . import config

FALLBACK_BETA = "server-side-fallback-2026-07-01"

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


_client_override: Any = None  # tests inject a fake client here


def _sdk():
    try:
        import anthropic  # optional dependency
    except ImportError:
        return None
    return anthropic


def settings(p: dict | None = None) -> dict:
    s = {**config.CRITIC, **((p or {}).get("critic_params") or {})}
    env = (os.environ.get("CHARFORGE_CRITIC") or "").strip().lower()
    if env in ("auto", "on", "off"):
        s["mode"] = env
    return s


def _has_credentials() -> bool:
    if any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")):
        return True
    return (Path.home() / ".config" / "anthropic").is_dir()  # profile from `ant auth login`


def check(p: dict | None = None) -> None:
    """Raise CriticUnavailable with the reason when no critic call may be made for this project."""
    s = settings(p)
    if s["mode"] == "off":
        raise CriticUnavailable("critic is off (CHARFORGE_CRITIC=off)")
    if p is not None and (p.get("critic_calls") or 0) >= int(s["max_calls"]):
        raise CriticUnavailable(f"critic budget spent ({s['max_calls']} calls for this project)")
    if _client_override is not None:
        return
    if _sdk() is None:
        raise CriticUnavailable('critic needs the anthropic SDK: pip install "charforge[critic]"')
    if s["mode"] == "auto" and not _has_credentials():
        raise CriticUnavailable("critic needs Claude credentials (ANTHROPIC_API_KEY or `ant auth login`)")


def _png_block(path: Path, max_px: int) -> dict:
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
    data = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _ask(p: dict | None, images: list[Path], prompt: str, schema: dict) -> tuple[dict, dict]:
    """One structured critic call. Returns (answer, usage). Counts against the project budget."""
    check(p)
    s = settings(p)
    sdk = _sdk()
    client = _client_override if _client_override is not None else sdk.Anthropic()
    content = [_png_block(Path(i), int(s["max_image_px"])) for i in images] + [{"type": "text", "text": prompt}]
    if p is not None:
        p["critic_calls"] = (p.get("critic_calls") or 0) + 1
    errors = (sdk.APIConnectionError, sdk.APIStatusError) if sdk is not None else ()
    try:
        resp = client.beta.messages.create(
            model=s["model"],
            max_tokens=16000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            output_config={"effort": s["effort"], "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": content}],
        )
    except errors as e:  # typed SDK errors only; anything else is a bug and should surface
        raise CriticFailed(_api_error(sdk, e)) from None
    if resp.stop_reason == "refusal":
        cat = getattr(getattr(resp, "stop_details", None), "category", None)
        raise CriticFailed(f"the critic model declined to review (category: {cat})")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        raise CriticFailed(f"the critic returned no answer (stop_reason: {resp.stop_reason})")
    try:
        answer = json.loads(text)
    except json.JSONDecodeError:
        raise CriticFailed("the critic answer is not valid JSON") from None
    usage = getattr(resp, "usage", None)
    return answer, {
        "model": getattr(resp, "model", s["model"]),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


def _api_error(sdk, e: Exception) -> str:
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
