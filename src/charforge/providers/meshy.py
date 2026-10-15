"""Meshy cloud generation, rigging, and animation provider.

Conforms to official Meshy OpenAPI specifications:
- POST /openapi/v1/multi-image-to-3d (multi-view with A-pose)
- POST /openapi/v1/image-to-3d (single-view fallback)
- POST /openapi/v1/rigging (humanoid rigging)
- POST /openapi/v1/animations (character animation actions)
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from .base import Provider
from ..envelope import CFError


class MeshyProvider(Provider):
    name = "meshy"

    def __init__(self, api_key: str | None = None, client: Any = None):
        super().__init__(api_key=api_key, client=client)
        self._job_types: dict[str, str] = {}

    def submit_multiview(
        self,
        views: dict[str, Path | str],
        pose_mode: str = "a-pose",
        seed: int | None = None,
    ) -> str:
        """Submit multi-view generation task to POST /openapi/v1/multi-image-to-3d."""
        key = self._require_key()
        image_urls: list[str] = []

        # Standard canonical order if present
        for v in ("front", "side", "back"):
            p = views.get(v)
            if p and Path(p).is_file():
                b64 = base64.b64encode(Path(p).read_bytes()).decode("ascii")
                image_urls.append(f"data:image/png;base64,{b64}")

        # Any additional view paths
        for k, p in views.items():
            if k not in ("front", "side", "back") and p and Path(p).is_file():
                b64 = base64.b64encode(Path(p).read_bytes()).decode("ascii")
                image_urls.append(f"data:image/png;base64,{b64}")

        if not image_urls:
            raise CFError(
                "PROVIDER_FAILED",
                "At least one valid image view required for Meshy multi-view.",
                "Verify sheet_split output.",
            )

        payload: dict[str, Any] = {
            "image_urls": image_urls,
            "ai_model": "latest",
            "pose_mode": pose_mode,
            "enable_pbr": True,
        }
        if seed is not None:
            payload["seed"] = seed

        res = self.client.post(
            "https://api.meshy.ai/openapi/v1/multi-image-to-3d",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        self._check_status(res)
        task_id = res.json().get("result")
        if not task_id:
            raise CFError("PROVIDER_FAILED", f"Meshy response missing result: {res.text}", "Retry or use provider='blockout'")
        self._job_types[task_id] = "multi-image-to-3d"
        return str(task_id)

    def submit(self, views: dict[str, Path], seed: int | None = None) -> str:
        """Submit 3D generation task (multi-view if 2+ views, else single image)."""
        key = self._require_key()
        valid_views = [v for v in ("front", "side", "back") if views.get(v) and Path(views[v]).is_file()]

        if len(valid_views) > 1:
            return self.submit_multiview(views, pose_mode="a-pose", seed=seed)

        front_p = views.get("front")
        if not front_p or not Path(front_p).is_file():
            raise CFError("PROVIDER_FAILED", "Front view required for Meshy.", "Verify sheet_split output.")

        b64 = base64.b64encode(Path(front_p).read_bytes()).decode("ascii")
        payload: dict[str, Any] = {
            "image_url": f"data:image/png;base64,{b64}",
            "enable_pbr": True,
            "ai_model": "latest",
        }
        if seed is not None:
            payload["seed"] = seed

        res = self.client.post(
            "https://api.meshy.ai/openapi/v1/image-to-3d",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        self._check_status(res)
        result_id = res.json().get("result")
        if not result_id:
            raise CFError("PROVIDER_FAILED", f"Meshy response missing result: {res.text}", "Retry or use provider='blockout'")
        self._job_types[result_id] = "image-to-3d"
        return str(result_id)

    def submit_rig(self, input_task_id: str, height_meters: float = 1.7) -> str:
        """Submit automated rigging task to POST /openapi/v1/rigging."""
        key = self._require_key()
        payload = {
            "input_task_id": input_task_id,
            "height_meters": float(height_meters),
        }
        res = self.client.post(
            "https://api.meshy.ai/openapi/v1/rigging",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        self._check_status(res)
        rig_id = res.json().get("result")
        if not rig_id:
            raise CFError("PROVIDER_FAILED", f"Meshy rigging response missing result: {res.text}", "Use method='blender'")
        self._job_types[rig_id] = "rigging"
        return str(rig_id)

    def submit_animation(self, rig_task_id: str, action_id: int) -> str:
        """Submit character animation task to POST /openapi/v1/animations."""
        key = self._require_key()
        payload = {
            "rig_task_id": rig_task_id,
            "action_id": int(action_id),
        }
        res = self.client.post(
            "https://api.meshy.ai/openapi/v1/animations",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        self._check_status(res)
        anim_id = res.json().get("result")
        if not anim_id:
            raise CFError("PROVIDER_FAILED", f"Meshy animation response missing result: {res.text}", "Check rig_task_id or action_id")
        self._job_types[anim_id] = "animations"
        return str(anim_id)

    def poll_rig(self, rig_task_id: str) -> dict:
        """Poll rigging task status via GET /openapi/v1/rigging/{task_id}."""
        key = self._require_key()
        res = self.client.get(
            f"https://api.meshy.ai/openapi/v1/rigging/{rig_task_id}",
            headers={"Authorization": f"Bearer {key}"},
        )
        self._check_status(res)
        data = res.json()
        status = data.get("status")
        if status in ("PENDING", "IN_PROGRESS"):
            return {"status": "running"}
        if status == "SUCCEEDED":
            model_urls = data.get("model_urls") or {}
            return {"status": "done", "url": model_urls.get("glb")}
        if status in ("FAILED", "EXPIRED"):
            err = (data.get("task_error") or {}).get("message") or data.get("message", "Rigging task failed")
            return {"status": "failed", "error": err}
        return {"status": "running"}

    def poll_animation(self, anim_task_id: str) -> dict:
        """Poll animation task status via GET /openapi/v1/animations/{task_id}."""
        key = self._require_key()
        res = self.client.get(
            f"https://api.meshy.ai/openapi/v1/animations/{anim_task_id}",
            headers={"Authorization": f"Bearer {key}"},
        )
        self._check_status(res)
        data = res.json()
        status = data.get("status")
        if status in ("PENDING", "IN_PROGRESS"):
            return {"status": "running"}
        if status == "SUCCEEDED":
            model_urls = data.get("model_urls") or {}
            return {"status": "done", "url": model_urls.get("glb")}
        if status in ("FAILED", "EXPIRED"):
            err = (data.get("task_error") or {}).get("message") or data.get("message", "Animation task failed")
            return {"status": "failed", "error": err}
        return {"status": "running"}

    def poll(self, job_id: str) -> dict:
        """Poll task status across generation, rigging, or animation."""
        if job_id.startswith("rig_") or self._job_types.get(job_id) == "rigging":
            return self.poll_rig(job_id)
        if job_id.startswith("anim_") or self._job_types.get(job_id) == "animations":
            return self.poll_animation(job_id)

        key = self._require_key()
        jtype = self._job_types.get(job_id)

        if jtype == "image-to-3d":
            res = self.client.get(
                f"https://api.meshy.ai/openapi/v1/image-to-3d/{job_id}",
                headers={"Authorization": f"Bearer {key}"},
            )
        elif jtype == "multi-image-to-3d":
            res = self.client.get(
                f"https://api.meshy.ai/openapi/v1/multi-image-to-3d/{job_id}",
                headers={"Authorization": f"Bearer {key}"},
            )
        else:
            # Fallback check multi-image first; if 404, check single image
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
            return {"status": "running"}
        if status == "SUCCEEDED":
            model_urls = data.get("model_urls") or {}
            return {"status": "done", "url": model_urls.get("glb")}
        if status in ("FAILED", "EXPIRED"):
            err = (data.get("task_error") or {}).get("message") or data.get("message", "Task failed")
            return {"status": "failed", "error": err}
        return {"status": "running"}
