"""Tencent Hunyuan3D via FAL cloud generation provider."""
from __future__ import annotations

import base64
from pathlib import Path

from .base import Provider
from ..envelope import CFError


class HunyuanProvider(Provider):
    name = "hunyuan"

    def submit(self, views: dict[str, Path], seed: int | None = None) -> str:
        key = self._require_key()
        front_p = views.get("front")
        if not front_p or not Path(front_p).is_file():
            raise CFError("PROVIDER_FAILED", "Front view required for Hunyuan.", "Verify sheet_split output.")

        b64 = base64.b64encode(Path(front_p).read_bytes()).decode("ascii")
        payload = {
            "image_url": f"data:image/png;base64,{b64}",
        }
        if seed is not None:
            payload["seed"] = seed

        res = self.client.post(
            "https://queue.fal.run/fal-ai/hunyuan3d/v2",
            headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
            json=payload,
        )
        self._check_status(res)
        data = res.json()
        req_id = data.get("request_id")
        if not req_id:
            raise CFError("PROVIDER_FAILED", f"Hunyuan response missing request_id: {res.text}", "Retry or use provider='blockout'")
        return req_id

    def poll(self, job_id: str) -> dict:
        key = self._require_key()
        res = self.client.get(
            f"https://queue.fal.run/fal-ai/hunyuan3d/requests/{job_id}/status",
            headers={"Authorization": f"Key {key}"},
        )
        self._check_status(res)
        data = res.json()
        status = data.get("status")
        if status in ("IN_QUEUE", "IN_PROGRESS"):
            return {"status": "running"}
        if status == "COMPLETED":
            # Fetch result payload
            res_result = self.client.get(
                f"https://queue.fal.run/fal-ai/hunyuan3d/requests/{job_id}",
                headers={"Authorization": f"Key {key}"},
            )
            self._check_status(res_result)
            res_data = res_result.json()
            mesh_url = (res_data.get("response", {}) or {}).get("model_mesh", {}).get("url") or res_data.get("model_mesh", {}).get("url")
            return {"status": "done", "url": mesh_url}
        if status == "FAILED":
            return {"status": "failed", "error": data.get("error", "Task failed")}
        return {"status": "running"}
