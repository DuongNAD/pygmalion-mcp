"""Tripo3D cloud generation provider."""
from __future__ import annotations

import base64
from pathlib import Path

from .base import Provider
from ..envelope import CFError


class TripoProvider(Provider):
    name = "tripo"

    def submit(self, views: dict[str, Path], seed: int | None = None) -> str:
        key = self._require_key()
        files_payload = []
        for v in ("front", "side", "back"):
            p = views.get(v)
            if p and Path(p).is_file():
                b64 = base64.b64encode(Path(p).read_bytes()).decode("ascii")
                files_payload.append({"type": "png", "data": f"data:image/png;base64,{b64}"})

        payload = {
            "type": "multiview_to_model",
            "files": files_payload,
        }
        if seed is not None:
            payload["seed"] = seed

        res = self.client.post(
            "https://api.tripo3d.ai/v2/openapi/task",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        self._check_status(res)
        data = res.json().get("data", {})
        task_id = data.get("task_id")
        if not task_id:
            raise CFError("PROVIDER_FAILED", f"Tripo response missing task_id: {res.text}", "Retry or use provider='blockout'")
        return task_id

    def poll(self, job_id: str) -> dict:
        key = self._require_key()
        res = self.client.get(
            f"https://api.tripo3d.ai/v2/openapi/task/{job_id}",
            headers={"Authorization": f"Bearer {key}"},
        )
        self._check_status(res)
        data = res.json().get("data", {})
        status = data.get("status")
        if status in ("queued", "running", "processing"):
            return {"status": "running"}
        if status == "success":
            output = data.get("output", {})
            url = output.get("pbr_model") or output.get("model")
            return {"status": "done", "url": url}
        if status == "failed":
            return {"status": "failed", "error": data.get("error", "Task failed")}
        return {"status": "running"}
