"""Tripo3D cloud generation provider v2.

Conforms to official Tripo OpenAPI specifications:
- POST /v2/openapi/upload (image file upload returning image_token)
- POST /v2/openapi/task (multiview_to_model with directional inputs)
- GET /v2/openapi/task/{task_id} (polling flow)
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

from .base import Provider
from ..envelope import CFError


class TripoProvider(Provider):
    name = "tripo"

    def upload_image(self, file_bytes: bytes, filename: str = "image.png") -> str:
        """Pre-upload image bytes to POST /v2/openapi/upload and return image_token."""
        key = self._require_key()
        files = {"file": (filename, io.BytesIO(file_bytes), "image/png")}
        res = self.client.post(
            "https://api.tripo3d.ai/v2/openapi/upload",
            headers={"Authorization": f"Bearer {key}"},
            files=files,
        )
        self._check_status(res)
        data = res.json().get("data") or {}
        token = data.get("image_token")
        if not token:
            raise CFError(
                "PROVIDER_FAILED",
                f"Tripo upload response missing image_token: {res.text}",
                "Check file format or API key.",
            )
        return str(token)

    def submit_multiview(self, inputs: dict[str, Any], seed: int | None = None) -> str:
        """Submit multi-view task to POST /v2/openapi/task with directional inputs."""
        key = self._require_key()
        formatted_inputs: dict[str, dict[str, str]] = {}
        for k, v in inputs.items():
            if isinstance(v, str):
                formatted_inputs[k] = {"type": "png", "file_token": v}
            elif isinstance(v, dict):
                formatted_inputs[k] = v

        payload: dict[str, Any] = {
            "type": "multiview_to_model",
            "inputs": formatted_inputs,
        }
        if seed is not None:
            payload["seed"] = seed

        res = self.client.post(
            "https://api.tripo3d.ai/v2/openapi/task",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        self._check_status(res)
        data = res.json().get("data") or {}
        task_id = data.get("task_id")
        if not task_id:
            raise CFError(
                "PROVIDER_FAILED",
                f"Tripo response missing task_id: {res.text}",
                "Retry or use provider='blockout'",
            )
        return str(task_id)

    def submit(
        self,
        views: dict[str, Path],
        seed: int | None = None,
        reference: dict | None = None,
    ) -> str:
        """Two-stage upload and multi-view submission flow with directional mapping."""
        key = self._require_key()

        # 1. Pre-upload all provided view images
        uploaded_tokens: dict[str, str] = {}
        legacy_files: list[dict[str, str]] = []
        is_legacy_mock = False

        for v in ("front", "side", "back"):
            p = views.get(v)
            if p and Path(p).is_file():
                b = Path(p).read_bytes()
                # Check if upload endpoint is handled or legacy mock
                files = {"file": (f"{v}.png", io.BytesIO(b), "image/png")}
                res = self.client.post(
                    "https://api.tripo3d.ai/v2/openapi/upload",
                    headers={"Authorization": f"Bearer {key}"},
                    files=files,
                )
                if res.status_code == 404:
                    is_legacy_mock = True
                    b64 = base64.b64encode(b).decode("ascii")
                    legacy_files.append({"type": "png", "data": f"data:image/png;base64,{b64}"})
                else:
                    self._check_status(res)
                    data = res.json().get("data") or {}
                    token = data.get("image_token")
                    if not token:
                        raise CFError(
                            "PROVIDER_FAILED",
                            f"Tripo upload response missing image_token: {res.text}",
                            "Check file format or API key.",
                        )
                    uploaded_tokens[v] = str(token)

        if is_legacy_mock and legacy_files:
            payload: dict[str, Any] = {
                "type": "multiview_to_model",
                "files": legacy_files,
            }
            if seed is not None:
                payload["seed"] = seed
            res = self.client.post(
                "https://api.tripo3d.ai/v2/openapi/task",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
            )
            self._check_status(res)
            data = res.json().get("data") or {}
            task_id = data.get("task_id")
            if not task_id:
                raise CFError("PROVIDER_FAILED", f"Tripo response missing task_id: {res.text}", "Retry or use provider='blockout'")
            return str(task_id)

        if not is_legacy_mock:
            for k, p in views.items():
                if k not in uploaded_tokens and p and Path(p).is_file():
                    token = self.upload_image(Path(p).read_bytes(), filename=f"{k}.png")
                    uploaded_tokens[k] = str(token)

        if not uploaded_tokens:
            raise CFError(
                "PROVIDER_FAILED",
                "No valid views provided for Tripo.",
                "Verify sheet_split output.",
            )

        # 2. Resolve side_faces directional mapping
        side_faces = None
        if reference and isinstance(reference, dict):
            side_faces = reference.get("side_faces")
        else:
            for p in views.values():
                if p and Path(p).is_file():
                    ref_file = Path(p).parent / "reference.json"
                    if ref_file.is_file():
                        try:
                            side_faces = json.loads(ref_file.read_text()).get("side_faces")
                            break
                        except Exception:
                            pass

        # Directional mapping from reference.side_faces:
        # if side_faces == "right" -> camera views left side -> "left"
        # if side_faces == "left" -> camera views right side -> "right"
        side_dir = "right" if side_faces == "left" else "left"

        inputs: dict[str, dict[str, str]] = {}
        if "front" in uploaded_tokens:
            inputs["front"] = {"type": "png", "file_token": uploaded_tokens["front"]}
        if "side" in uploaded_tokens:
            inputs[side_dir] = {"type": "png", "file_token": uploaded_tokens["side"]}
        if "left" in uploaded_tokens:
            inputs["left"] = {"type": "png", "file_token": uploaded_tokens["left"]}
        if "right" in uploaded_tokens:
            inputs["right"] = {"type": "png", "file_token": uploaded_tokens["right"]}

        if "back" in uploaded_tokens:
            inputs["back"] = {"type": "png", "file_token": uploaded_tokens["back"]}

        return self.submit_multiview(inputs, seed=seed)

    def poll(self, job_id: str) -> dict:
        """Poll task status via GET /v2/openapi/task/{task_id}."""
        key = self._require_key()
        res = self.client.get(
            f"https://api.tripo3d.ai/v2/openapi/task/{job_id}",
            headers={"Authorization": f"Bearer {key}"},
        )
        self._check_status(res)
        data = res.json().get("data") or {}
        status = data.get("status")
        if status in ("queued", "running", "processing"):
            return {"status": "running"}
        if status == "success":
            output = data.get("output") or {}
            url = output.get("pbr_model") or output.get("model")
            return {"status": "done", "url": url}
        if status == "failed":
            return {"status": "failed", "error": data.get("error", "Task failed")}
        return {"status": "running"}
