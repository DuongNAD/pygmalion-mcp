"""Meshy cloud generation provider."""
from __future__ import annotations

import base64
from pathlib import Path

from .base import Provider
from ..envelope import CFError


class MeshyProvider(Provider):
    name = "meshy"

    def submit(self, views: dict[str, Path], seed: int | None = None) -> str:
        key = self._require_key()
        front_p = views.get("front")
        if not front_p or not Path(front_p).is_file():
            raise CFError("PROVIDER_FAILED", "Front view required for Meshy.", "Verify sheet_split output.")

        b64 = base64.b64encode(Path(front_p).read_bytes()).decode("ascii")
        payload = {
            "image_url": f"data:image/png;base64,{b64}",
            "enable_pbr": True,
            "ai_model": "meshy-4",
        }
        if seed is not None:
            payload["seed"] = seed

        res = self.client.post(
            "https://api.meshy.ai/v1/image-to-3d",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        self._check_status(res)
        result_id = res.json().get("result")
        if not result_id:
            raise CFError("PROVIDER_FAILED", f"Meshy response missing result: {res.text}", "Retry or use provider='blockout'")
        return result_id

    def poll(self, job_id: str) -> dict:
        key = self._require_key()
        res = self.client.get(
            f"https://api.meshy.ai/v1/image-to-3d/{job_id}",
            headers={"Authorization": f"Bearer {key}"},
        )
        self._check_status(res)
        data = res.json()
        status = data.get("status")
        if status in ("PENDING", "IN_PROGRESS"):
            return {"status": "running"}
        if status == "SUCCEEDED":
            model_urls = data.get("model_urls", {})
            return {"status": "done", "url": model_urls.get("glb")}
        if status in ("FAILED", "EXPIRED"):
            err = data.get("task_error", {}).get("message", "Task failed")
            return {"status": "failed", "error": err}
        return {"status": "running"}
