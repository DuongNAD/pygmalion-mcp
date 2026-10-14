"""Settings, paths and external tool discovery. Single source of truth for tunables."""
from __future__ import annotations

import importlib.metadata as md
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = Path(__file__).resolve().parent
BPY_SCRIPTS = PKG_DIR / "bpy_scripts"


def _load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            if val.strip():
                os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

# --- scoring / loop policy (per-project overrides live in project.json) ---
THRESHOLDS = {"front": 0.90, "side": 0.85, "back": 0.85}  # min silhouette IoU per view to pass
VIEW_WEIGHTS = {"front": 0.5, "side": 0.3, "back": 0.2}  # front is the canonical view
MAX_ITERATIONS = 8  # score_mesh calls in the search phase before forced stop
PLATEAU_ROUNDS = 3  # stop when the last N scores improve best total by < PLATEAU_EPS
PLATEAU_EPS = 0.005
MAX_CANDIDATES = 3  # generated meshes per project (L2)
REGENERATE_BELOW = 0.75  # best total under this -> new candidate instead of mesh_fit (L2)
CANON_SIZE = 512  # canonical mask canvas, px
PROFILE_BINS = 64  # horizontal bands in width profiles

PROVIDER_KEYS = {"tripo": "TRIPO_API_KEY", "meshy": "MESHY_API_KEY", "rodin": "RODIN_API_KEY", "hunyuan": "FAL_KEY"}


def home() -> Path:
    path = Path(os.environ.get("CHARFORGE_HOME") or REPO_ROOT / "projects").expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def level() -> int:
    return max(1, min(3, int(os.environ.get("CHARFORGE_LEVEL") or 1)))


def blender_exe() -> str | None:
    for cand in (os.environ.get("BLENDER_PATH"), shutil.which("blender"), "/Applications/Blender.app/Contents/MacOS/Blender"):
        if cand and Path(cand).is_file():
            return cand
    return None


def providers_available() -> list[str]:
    return [name for name, key in PROVIDER_KEYS.items() if os.environ.get(key)]


def env_report() -> dict:
    exe, version = blender_exe(), None
    if exe:
        try:
            version = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60).stdout.splitlines()[0]
        except Exception as e:  # report, never crash the check
            version = f"error: {e}"
    deps = {}
    for dist in ("mcp", "numpy", "opencv-python", "scikit-image", "pillow", "pydantic"):
        try:
            deps[dist] = md.version(dist)
        except md.PackageNotFoundError:
            deps[dist] = None
    return {
        "ok": bool(exe) and all(deps.values()),
        "blender": exe,
        "blender_version": version,
        "home": str(home()),
        "level": level(),
        "providers": providers_available(),
        "deps": deps,
    }
