"""Deemos / Hyper3D Rodin cloud generation provider v2.

Conforms to official Rodin OpenAPI specifications:
- POST /api/v2/rodin (multipart/form-data submission returning task_uuid and subscription_key)
- POST /api/v2/status (polling via subscription_key)
- POST /api/v2/download (signed GLB download URL retrieval via task_uuid)
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from .base import Provider
from ..envelope import CFError


class RodinProvider(Provider):
    name = "rodin"

    def submit_multipart(
        self,
        images: list[tuple[str, tuple[str, io.BytesIO | bytes, str]]],
        seed: int | None = None,
    ) -> str:
        """Submit multipart/form-data to POST /api/v2/rodin and return compound job_id."""
        key = self._require_key()
        res = self.client.post(
            "https://api.hyper3d.com/api/v2/rodin",
            headers={"Authorization": f"Bearer {key}"},
            files=images,
        )
        self._check_status(res)
        data = res.json()
        task_uuid = data.get("uuid") or data.get("task_id")
        if not task_uuid:
            raise CFError(
                "PROVIDER_FAILED",
                f"Rodin response missing task uuid: {res.text}",
                "Retry or use provider='blockout'",
            )

        jobs = data.get("jobs", {})
        sub_key = jobs.get("subscription_key") if isinstance(jobs, dict) else None

        if sub_key:
            return f"{task_uuid}:{sub_key}"
        return str(task_uuid)

    def submit(self, views: dict[str, Path], seed: int | None = None) -> str:
        """Submit 3D generation task using multipart form images."""
        key = self._require_key()
        images: list[tuple[str, tuple[str, io.BytesIO, str]]] = []

        for v in ("front", "side", "back"):
            p = views.get(v)
            if p and Path(p).is_file():
                b = Path(p).read_bytes()
                images.append(("images", (f"{v}.png", io.BytesIO(b), "image/png")))

        if not images:
            for k, p in views.items():
                if p and Path(p).is_file():
                    b = Path(p).read_bytes()
                    images.append(("images", (f"{k}.png", io.BytesIO(b), "image/png")))

        if not images:
            raise CFError(
                "PROVIDER_FAILED",
                "No valid views provided for Rodin.",
                "Verify sheet_split output.",
            )

        return self.submit_multipart(images, seed=seed)

    def get_download_url(self, task_uuid: str) -> str:
        """Retrieve signed GLB download URL from POST /api/v2/download."""
        key = self._require_key()
        res = self.client.post(
            "https://api.hyper3d.com/api/v2/download",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"task_uuid": task_uuid},
        )
        self._check_status(res)
        data = res.json()
        url = data.get("url")
        if not url:
            raise CFError(
                "PROVIDER_FAILED",
                f"Rodin download retrieval response missing url: {res.text}",
                "Retry or use provider='blockout'",
            )
        return str(url)

    def poll(self, job_id: str) -> dict:
        """Poll task status via POST /api/v2/status and retrieve download URL when complete."""
        key = self._require_key()
        if ":" in job_id:
            task_uuid, sub_key = job_id.split(":", 1)
        else:
            task_uuid, sub_key = job_id, job_id

        res = self.client.post(
            "https://api.hyper3d.com/api/v2/status",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"subscription_key": sub_key},
        )
        if res.status_code == 404:
            # Fallback for legacy endpoints
            res = self.client.get(
                f"https://hyperhuman.deemos.com/api/v2/rodin/{task_uuid}",
                headers={"Authorization": f"Bearer {key}"},
            )
            if res.status_code == 404:
                res = self.client.get(
                    f"https://api.hyper3d.com/api/v2/rodin/{task_uuid}",
                    headers={"Authorization": f"Bearer {key}"},
                )

        self._check_status(res)
        data = res.json()
        status = str(data.get("status", "")).lower()

        if status in ("pending", "processing", "running"):
            return {"status": "running"}
        if status in ("success", "done", "completed"):
            url = data.get("download_url") or data.get("model_url") or (data.get("output", {}) or {}).get("glb")
            if not url:
                url = self.get_download_url(task_uuid)
            return {"status": "done", "url": url}
        if status in ("failed", "error"):
            return {"status": "failed", "error": data.get("error", "Task failed")}
        return {"status": "running"}
