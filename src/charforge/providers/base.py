"""Base class for multi-view 3D cloud generation providers."""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from .. import config
from ..envelope import CFError


class Provider(ABC):
    """Abstract base class for cloud 3D generation providers."""

    name: str

    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None):
        env_var = config.PROVIDER_KEYS.get(self.name)
        self.api_key = api_key or (os.environ.get(env_var) if env_var else None)
        self.client = client or httpx.Client(timeout=60.0)

    def _require_key(self) -> str:
        if not self.api_key:
            raise CFError(
                "NO_PROVIDER",
                f"No API key found for provider {self.name!r}.",
                "Call generate_mesh with provider='auto' or 'blockout'.",
            )
        return self.api_key

    def _check_status(self, response: httpx.Response) -> None:
        """Map HTTP error codes to uniform CFError."""
        if response.status_code in (401, 402, 429):
            raise CFError(
                "PROVIDER_FAILED",
                f"{self.name} returned HTTP {response.status_code}: {response.text}",
                "Credits exhausted or provider error: call generate_mesh with provider='blockout'",
            )
        if response.status_code >= 400:
            raise CFError(
                "PROVIDER_FAILED",
                f"{self.name} returned HTTP {response.status_code}: {response.text}",
                "Check provider settings or call generate_mesh with provider='blockout'",
            )

    @abstractmethod
    def submit(self, views: dict[str, Path], seed: int | None = None) -> str:
        """Submit multi-view 3D generation task. Returns job ID."""
        ...

    @abstractmethod
    def poll(self, job_id: str) -> dict:
        """Poll task status. Returns {"status": "running"|"done"|"failed", "url"?: str, "error"?: str}."""
        ...

    def download(self, url: str, dst: Path) -> Path:
        """Download resulting 3D model (GLB) to dst."""
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.client.stream("GET", url) as res:
                self._check_status(res)
                with open(dst, "wb") as f:
                    for chunk in res.iter_bytes(chunk_size=65536):
                        f.write(chunk)
        except httpx.HTTPError as e:
            raise CFError(
                "PROVIDER_FAILED",
                f"Failed to download model from {url}: {e}",
                "Call generate_mesh with provider='blockout'",
            ) from e
        return dst
