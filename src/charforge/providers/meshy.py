"""Meshy cloud generation, rigging, and animation provider.

Conforms to official Meshy OpenAPI v1 specifications:
- POST /openapi/v1/multi-image-to-3d (multi-view with A-pose, meshy-7.1, remesh + PBR texture)
- POST /openapi/v1/image-to-3d (single-view fallback)
- POST /openapi/v1/rigging (humanoid rigging -> result.rigged_character_glb_url)
- GET  /openapi/v1/animations/library (animation catalog with non-contiguous action_id)
- POST /openapi/v1/animations (multi-action animation -> result.animation_glb_url; exactly one of
  action_id | action_ids (1-10) | motion_task_id)
- POST /openapi/v1/text-to-motion (prompt <= 400 chars, duration 2-10 s in 0.5 steps, mode prime|swift ->
  result.motion_url; the clip lives 3 days and is applied with POST /animations {motion_task_id})
"""
from __future__ import annotations

import base64
import difflib
import json
import re
import time
from pathlib import Path
from typing import Any

from .base import Provider
from .. import config
from ..envelope import CFError


def _norm_action(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


class MeshyProvider(Provider):
    name = "meshy"

    def __init__(self, api_key: str | None = None, client: Any = None):
        super().__init__(api_key=api_key, client=client)
        # In-memory cache only; durable kind in project.json["jobs"] is the source of truth
        self._job_types: dict[str, str] = {}
        self.last_submitted_kind: str | None = None

    def submit_multiview(
        self,
        views: dict[str, Path | str],
        pose_mode: str = "a-pose",
        seed: int | None = None,
        target_polycount: int | None = None,
    ) -> str:
        """Submit multi-view generation task to POST /openapi/v1/multi-image-to-3d.

        Note: Official Meshy multi-image-to-3d API does not support a `seed` parameter;
        regenerating a new candidate means submitting a new job.
        """
        key = self._require_key()
        image_urls: list[str] = []

        # Front view MUST be first (Meshy 7.1 treats first image as front view)
        for v in ("front", "side", "back"):
            p = views.get(v)
            if p and Path(p).is_file():
                b64 = base64.b64encode(Path(p).read_bytes()).decode("ascii")
                image_urls.append(f"data:image/png;base64,{b64}")

        # Any additional view paths (up to 4 total per Meshy spec)
        for k, p in views.items():
            if k not in ("front", "side", "back") and p and Path(p).is_file() and len(image_urls) < 4:
                b64 = base64.b64encode(Path(p).read_bytes()).decode("ascii")
                image_urls.append(f"data:image/png;base64,{b64}")

        if not image_urls:
            raise CFError(
                "PROVIDER_FAILED",
                "At least one valid image view required for Meshy multi-view.",
                "Verify sheet_split output.",
            )

        polycount = max(100, min(300000, int(target_polycount or config.MESHY_TARGET_POLYCOUNT)))
        payload: dict[str, Any] = {
            "image_urls": image_urls,
            "ai_model": config.MESHY_AI_MODEL,
            "pose_mode": pose_mode,
            "should_remesh": True,
            "topology": "triangle",
            "target_polycount": polycount,
            "should_texture": True,
            "enable_pbr": True,
            "target_formats": ["glb"],
        }
        if config.MESHY_TEXTURE_FROM_VIEWS:
            payload["texture_image_urls"] = [image_urls[0]]

        res = self.client.post(
            "https://api.meshy.ai/openapi/v1/multi-image-to-3d",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        self._check_status(res)
        task_id = res.json().get("result")
        if not task_id:
            raise CFError(
                "PROVIDER_FAILED",
                f"Meshy response missing result: {res.text}",
                "Retry or use provider='blockout'",
            )
        task_id_str = str(task_id)
        self._job_types[task_id_str] = "multi-image-to-3d"
        self.last_submitted_kind = "multi-image-to-3d"
        return task_id_str

    def submit(
        self,
        views: dict[str, Path],
        seed: int | None = None,
        target_polycount: int | None = None,
    ) -> str:
        """Submit 3D generation task (multi-view if 2+ views, else single image).

        Note: Meshy image-to-3d / multi-image-to-3d endpoints do not take a `seed` parameter;
        regenerating requires submitting a new job.
        """
        key = self._require_key()
        valid_views = [v for v in ("front", "side", "back") if views.get(v) and Path(views[v]).is_file()]

        if len(valid_views) > 1:
            return self.submit_multiview(
                views,
                pose_mode="a-pose",
                seed=seed,
                target_polycount=target_polycount,
            )

        front_p = views.get("front")
        if not front_p or not Path(front_p).is_file():
            raise CFError("PROVIDER_FAILED", "Front view required for Meshy.", "Verify sheet_split output.")

        b64 = base64.b64encode(Path(front_p).read_bytes()).decode("ascii")
        polycount = max(100, min(300000, int(target_polycount or config.MESHY_TARGET_POLYCOUNT)))
        payload: dict[str, Any] = {
            "image_url": f"data:image/png;base64,{b64}",
            "ai_model": config.MESHY_AI_MODEL,
            "pose_mode": "a-pose",
            "should_remesh": True,
            "topology": "triangle",
            "target_polycount": polycount,
            "should_texture": True,
            "enable_pbr": True,
            "target_formats": ["glb"],
        }

        res = self.client.post(
            "https://api.meshy.ai/openapi/v1/image-to-3d",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        self._check_status(res)
        result_id = res.json().get("result")
        if not result_id:
            raise CFError(
                "PROVIDER_FAILED",
                f"Meshy response missing result: {res.text}",
                "Retry or use provider='blockout'",
            )
        result_id_str = str(result_id)
        self._job_types[result_id_str] = "image-to-3d"
        self.last_submitted_kind = "image-to-3d"
        return result_id_str

    def submit_rig(
        self,
        input_task_id: str | None = None,
        height_meters: float = 1.7,
        model_url: str | None = None,
        texture_image_url: str | None = None,
    ) -> str:
        """Submit automated rigging task to POST /openapi/v1/rigging."""
        key = self._require_key()
        payload: dict[str, Any] = {
            "height_meters": float(height_meters),
        }
        if input_task_id:
            payload["input_task_id"] = input_task_id
        elif model_url:
            payload["model_url"] = model_url
        else:
            raise CFError(
                "PROVIDER_FAILED",
                "Either input_task_id or model_url is required for Meshy rigging.",
                "Use method='blender'",
            )
        if texture_image_url:
            payload["texture_image_url"] = texture_image_url

        res = self.client.post(
            "https://api.meshy.ai/openapi/v1/rigging",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        self._check_status(res)
        rig_id = res.json().get("result")
        if not rig_id:
            raise CFError("PROVIDER_FAILED", f"Meshy rigging response missing result: {res.text}", "Use method='blender'")
        rig_id_str = str(rig_id)
        self._job_types[rig_id_str] = "rigging"
        self.last_submitted_kind = "rigging"
        return rig_id_str

    def fetch_animation_library(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        """Fetch Meshy animation library via GET /openapi/v1/animations/library with local TTL cache."""
        cache_dir = config.home() / "_cache"
        cache_file = cache_dir / "meshy_animation_library.json"

        if not force_refresh and cache_file.is_file():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                fetched_at = float(cached.get("fetched_at", 0.0))
                items = cached.get("items")
                if (
                    isinstance(items, list)
                    and items
                    and (time.time() - fetched_at) < config.MESHY_LIBRARY_TTL_SECONDS
                ):
                    return items
            except Exception:
                pass

        key = self._require_key()
        res = self.client.get(
            "https://api.meshy.ai/openapi/v1/animations/library",
            headers={"Authorization": f"Bearer {key}"},
        )
        self._check_status(res)
        data = res.json()
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("result") or data.get("items") or data.get("data") or []
        else:
            items = []

        if not isinstance(items, list) or not items:
            raise CFError(
                "PROVIDER_FAILED",
                "Meshy animation library returned no animation items.",
                "Use method='blender'",
            )

        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps({"fetched_at": time.time(), "items": items}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

        return items

    def resolve_action_ids(
        self,
        actions: list[str],
        library: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Resolve human action names (e.g. 'idle', 'walk', 'wave') to Meshy library action_ids."""
        lib = library if library is not None else self.fetch_animation_library()
        resolved: list[int] = []

        for act in actions:
            base_act = re.sub(r"_\d+$", "", act.strip())
            candidates: list[str] = []
            for raw in (act, base_act):
                n = _norm_action(raw)
                if n and n not in candidates:
                    candidates.append(n)
            for alias in (
                config.MESHY_ACTION_ALIASES.get(act.strip().lower(), [])
                + config.MESHY_ACTION_ALIASES.get(base_act.lower(), [])
            ):
                n = _norm_action(alias)
                if n and n not in candidates:
                    candidates.append(n)

            matched_id: int | None = None

            # Pass 1: exact normalized match on key or name
            for cand in candidates:
                for item in lib:
                    k_norm = _norm_action(str(item.get("key", "")))
                    n_norm = _norm_action(str(item.get("name", "")))
                    if cand and (cand == k_norm or cand == n_norm):
                        matched_id = int(item["action_id"])
                        break
                if matched_id is not None:
                    break

            # Pass 2: token or substring match on key or name
            if matched_id is None:
                for cand in candidates:
                    for item in lib:
                        k_norm = _norm_action(str(item.get("key", "")))
                        n_norm = _norm_action(str(item.get("name", "")))
                        tokens = set(k_norm.split("_")) | set(n_norm.split("_"))
                        if cand and (cand in tokens or cand in k_norm or cand in n_norm):
                            matched_id = int(item["action_id"])
                            break
                    if matched_id is not None:
                        break

            if matched_id is None:
                avail_names = [
                    str(item.get("name") or item.get("key"))
                    for item in lib
                    if item.get("name") or item.get("key")
                ]
                close = difflib.get_close_matches(base_act, avail_names, n=5, cutoff=0.3)
                if not close:
                    close = avail_names[:5]
                suggestions = ", ".join(close) if close else "idle, walk, wave"
                raise CFError(
                    "ANIM_FAILED",
                    f"Action {act!r} not found in Meshy animation library.",
                    f"Use valid action names from the library, e.g.: {suggestions}",
                )

            resolved.append(matched_id)

        return resolved

    def submit_text_to_motion(self, prompt: str, duration: float | None = None, mode: str | None = None) -> str:
        """Submit POST /openapi/v1/text-to-motion; returns the task id to apply with submit_animation(motion_task_id=...)."""
        key = self._require_key()
        text = (prompt or "").strip()
        if not text:
            raise CFError("ANIM_FAILED", "Text to Motion prompt is empty.", "Describe the motion in a short sentence.")
        dur = float(duration if duration is not None else config.MESHY_MOTION_DURATION)
        dur = max(2.0, min(10.0, round(dur * 2) / 2))  # 2-10 s in 0.5 s steps
        payload = {"prompt": text[:400], "duration": dur, "mode": mode or config.MESHY_MOTION_MODE}
        res = self.client.post(
            "https://api.meshy.ai/openapi/v1/text-to-motion",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        if res.status_code == 403:
            raise CFError(
                "ANIM_FAILED",
                f"Meshy Text to Motion rejected the prompt (content moderation): {res.text}",
                "Rephrase the motion prompt, or drop it from motions.",
            )
        try:
            self._check_status(res)
        except CFError as e:  # the generic fix talks about generate_mesh; give the animate-side one
            raise CFError(e.code, e.message, "Drop motions (preset actions still work) or retry animate later.") from None
        task_id = res.json().get("result")
        if not task_id:
            raise CFError("PROVIDER_FAILED", f"Meshy Text to Motion response missing result: {res.text}", "Drop motions and use preset actions.")
        task_id_str = str(task_id)
        self._job_types[task_id_str] = "text-to-motion"
        self.last_submitted_kind = "text-to-motion"
        return task_id_str

    def poll_text_to_motion(self, task_id: str) -> dict:
        """Poll GET /openapi/v1/text-to-motion/{task_id}; reads result.motion_url / motion_format."""
        key = self._require_key()
        res = self.client.get(
            f"https://api.meshy.ai/openapi/v1/text-to-motion/{task_id}",
            headers={"Authorization": f"Bearer {key}"},
        )
        self._check_status(res)
        data = res.json()
        status = data.get("status")
        if status in ("PENDING", "IN_PROGRESS"):
            return {"status": "running", "progress": data.get("progress", 0)}
        if status == "SUCCEEDED":
            res_obj = data.get("result") or {}
            return {
                "status": "done",
                "url": res_obj.get("motion_url"),
                "format": res_obj.get("motion_format"),
                "duration_ms": res_obj.get("duration_ms"),
                "consumed_credits": data.get("consumed_credits"),
            }
        if status in ("FAILED", "CANCELED", "EXPIRED"):
            err = (data.get("task_error") or {}).get("message") or data.get("message") or f"Text to Motion task {status.lower()}"
            return {"status": "failed", "error": err}
        return {"status": "running"}

    def submit_animation(
        self,
        rig_task_id: str,
        action_ids: list[int] | None = None,
        action_id: int | None = None,
        motion_task_id: str | None = None,
    ) -> str:
        """Submit POST /openapi/v1/animations with exactly one of action_ids (1-10), action_id or motion_task_id."""
        key = self._require_key()
        payload: dict[str, Any] = {"rig_task_id": rig_task_id}
        if motion_task_id:
            payload["motion_task_id"] = str(motion_task_id)
        else:
            if action_ids is not None:
                # Meshy requires 1..10 unique action_id values
                unique_ids = list(dict.fromkeys(int(x) for x in action_ids))
            elif action_id is not None:
                unique_ids = [int(action_id)]
            else:
                raise CFError(
                    "PROVIDER_FAILED",
                    "Meshy submit_animation requires action_ids.",
                    "Pass 1 to 10 action names to animate.",
                )

            if not (1 <= len(unique_ids) <= 10):
                raise CFError(
                    "ANIM_FAILED",
                    f"Meshy animation supports 1 to 10 unique action_ids per task, got {len(unique_ids)}.",
                    "Pass between 1 and 10 actions.",
                )
            payload["action_ids"] = unique_ids
        res = self.client.post(
            "https://api.meshy.ai/openapi/v1/animations",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        self._check_status(res)
        anim_id = res.json().get("result")
        if not anim_id:
            raise CFError(
                "PROVIDER_FAILED",
                f"Meshy animation response missing result: {res.text}",
                "Check rig_task_id or action_ids",
            )
        anim_id_str = str(anim_id)
        self._job_types[anim_id_str] = "animations"
        self.last_submitted_kind = "animations"
        return anim_id_str

    def poll_rig(self, rig_task_id: str) -> dict:
        """Poll rigging task status via GET /openapi/v1/rigging/{task_id}.

        Reads result.rigged_character_glb_url and result.basic_animations (never model_urls).
        """
        key = self._require_key()
        res = self.client.get(
            f"https://api.meshy.ai/openapi/v1/rigging/{rig_task_id}",
            headers={"Authorization": f"Bearer {key}"},
        )
        self._check_status(res)
        data = res.json()
        status = data.get("status")
        if status in ("PENDING", "IN_PROGRESS"):
            return {"status": "running", "progress": data.get("progress", 0)}
        if status == "SUCCEEDED":
            res_obj = data.get("result") or {}
            return {
                "status": "done",
                "url": res_obj.get("rigged_character_glb_url"),
                "fbx_url": res_obj.get("rigged_character_fbx_url"),
                "basic_animations": res_obj.get("basic_animations") or {},
                "consumed_credits": data.get("consumed_credits"),
            }
        if status in ("FAILED", "CANCELED", "EXPIRED"):
            err = (data.get("task_error") or {}).get("message") or data.get("message") or f"Rigging task {status.lower()}"
            return {"status": "failed", "error": err}
        return {"status": "running"}

    def poll_animation(self, anim_task_id: str) -> dict:
        """Poll animation task status via GET /openapi/v1/animations/{task_id}.

        Reads result.animation_glb_url (never model_urls).
        """
        key = self._require_key()
        res = self.client.get(
            f"https://api.meshy.ai/openapi/v1/animations/{anim_task_id}",
            headers={"Authorization": f"Bearer {key}"},
        )
        self._check_status(res)
        data = res.json()
        status = data.get("status")
        if status in ("PENDING", "IN_PROGRESS"):
            return {"status": "running", "progress": data.get("progress", 0)}
        if status == "SUCCEEDED":
            res_obj = data.get("result") or {}
            return {
                "status": "done",
                "url": res_obj.get("animation_glb_url"),
                "fbx_url": res_obj.get("animation_fbx_url"),
                "consumed_credits": data.get("consumed_credits"),
            }
        if status in ("FAILED", "CANCELED", "EXPIRED"):
            err = (data.get("task_error") or {}).get("message") or data.get("message") or f"Animation task {status.lower()}"
            return {"status": "failed", "error": err}
        return {"status": "running"}

    def poll(self, job_id: str, kind: str | None = None) -> dict:
        """Poll task status across generation, rigging, or animation using durable `kind`."""
        effective_kind = kind or self._job_types.get(job_id)

        if effective_kind == "rigging":
            return self.poll_rig(job_id)
        if effective_kind == "animations":
            return self.poll_animation(job_id)
        if effective_kind == "text-to-motion":
            return self.poll_text_to_motion(job_id)

        key = self._require_key()

        if effective_kind == "image-to-3d":
            res = self.client.get(
                f"https://api.meshy.ai/openapi/v1/image-to-3d/{job_id}",
                headers={"Authorization": f"Bearer {key}"},
            )
        elif effective_kind == "multi-image-to-3d":
            res = self.client.get(
                f"https://api.meshy.ai/openapi/v1/multi-image-to-3d/{job_id}",
                headers={"Authorization": f"Bearer {key}"},
            )
        else:
            # Fallback if kind was omitted in a direct unit call: check multi-image first, then single-image
            res = self.client.get(
                f"https://api.meshy.ai/openapi/v1/multi-image-to-3d/{job_id}",
                headers={"Authorization": f"Bearer {key}"},
            )
            if res.status_code == 404:
                res = self.client.get(
                    f"https://api.meshy.ai/openapi/v1/image-to-3d/{job_id}",
                    headers={"Authorization": f"Bearer {key}"},
                )

        self._check_status(res)
        data = res.json()
        status = data.get("status")
        if status in ("PENDING", "IN_PROGRESS"):
            return {"status": "running", "progress": data.get("progress", 0)}
        if status == "SUCCEEDED":
            model_urls = data.get("model_urls") or {}
            return {
                "status": "done",
                "url": model_urls.get("glb"),
                "consumed_credits": data.get("consumed_credits"),
            }
        if status in ("FAILED", "CANCELED", "EXPIRED"):
            err = (data.get("task_error") or {}).get("message") or data.get("message") or f"Task {status.lower()}"
            return {"status": "failed", "error": err}
        return {"status": "running"}
