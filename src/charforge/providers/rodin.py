"""Deemos Rodin cloud generation provider."""
from __future__ import annotations

import base64
from pathlib import Path

from .base import Provider
from ..envelope import CFError


class RodinProvider(Provider):
    name = "rodin"

    def submit(self, views: dict[str, Path], seed: int | None = None) -> str:
        key = self._require_key()
        images = []
        for v in ("front", "side", "back"):
            p = views.get(v)
            if p and Path(p).is_file():
                b64 = base64.b64encode(Path(p).read_bytes()).decode("ascii")
                images.append({"view": v, "image": f"data:image/png;base64,{b64}"})

        payload = {"images": images}
        if seed is not None:
            payload["seed"] = seed

        res = self.client.post(
            "https://hyperhuman.deemos.com/api/v2/rodin",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        self._check_status(res)
        data = res.json()
        task_id = data.get("uuid") or data.get("subscription_id") or data.get("task_id")
        if not task_id:
            raise CFError("PROVIDER_FAILED", f"Rodin response missing task id: {res.text}", "Retry or use provider='blockout'")
        return str(task_id)

    def poll(self, job_id: str) -> dict:
        key = self._require_key()
        res = self.client.get(
            f"https://hyperhuman.deemos.com/api/v2/rodin/{job_id}",
            headers={"Authorization": f"Bearer {key}"},
        )
        self._check_status(res)
        data = res.json()
        status = str(data.get("status", "")).lower()
        if status in ("pending", "processing", "running"):
            return {"status": "running"}
        if status in ("success", "done", "completed"):
            url = data.get("download_url") or data.get("model_url") or (data.get("output", {}) or {}).get("glb")
            return {"status": "done", "url": url}
        if status in ("failed", "error"):
            return {"status": "failed", "error": data.get("error", "Task failed")}
        return {"status": "running"}
