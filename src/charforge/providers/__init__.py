"""Multi-view 3D cloud generation provider registry."""
from __future__ import annotations

from typing import Type

import httpx

from .base import Provider
from .hunyuan import HunyuanProvider
from .meshy import MeshyProvider
from .rodin import RodinProvider
from .tripo import TripoProvider
from ..envelope import CFError

PROVIDERS: dict[str, Type[Provider]] = {
    "tripo": TripoProvider,
    "meshy": MeshyProvider,
    "rodin": RodinProvider,
    "hunyuan": HunyuanProvider,
}


def get_provider(name: str, client: httpx.Client | None = None) -> Provider:
    """Retrieve provider instance by name."""
    cls = PROVIDERS.get(name)
    if not cls:
        raise CFError(
            "NO_PROVIDER",
            f"Unknown provider {name!r}. Available: {list(PROVIDERS.keys()) + ['blockout']}",
            "Call generate_mesh with provider='auto' or 'blockout'.",
        )
    return cls(client=client)


__all__ = [
    "Provider",
    "TripoProvider",
    "MeshyProvider",
    "RodinProvider",
    "HunyuanProvider",
    "PROVIDERS",
    "get_provider",
]
