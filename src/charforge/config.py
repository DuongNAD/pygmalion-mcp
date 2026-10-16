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

# --- pose gaps (sheet_split / score_mesh): arms and legs must be visibly apart for clean skin weights ---
POSE_GAP_MIN = 0.012  # min gap, as a fraction of figure height, between forearm/hand and body (and between shins)
POSE_ARM_BAND_MIN = 0.25  # share of the forearm band (55-80% of the arm length) that must show the gap
POSE_LEG_BAND_MIN = 0.5  # share of the shin band (72-92% height) that must show two separate legs
POSE_RETRY = 1  # extra provider candidates when the mesh fuses arms/legs the sheet keeps apart (0 = off)

# --- mesh integrity (score_mesh): numbers before eyes; a passed mesh without detached parts ranks first ---
INTEGRITY = {
    "gap_frac": 0.006,  # a piece farther than this x mesh height from the body floats
    "debris_share": 0.002,  # floating pieces below this share of the vertices are debris, larger ones detached parts
    "self_intersect_max": 0.03,  # share of faces crossing other faces before it counts as a defect
    "non_manifold_max": 0.005,  # share of edges shared by 3+ faces before it counts as a defect
    "holes_max": 30,  # open boundary loops reported as an issue above this
}

# --- visual critic (advisory): an independent vision model grades renders against the sheet with a fixed rubric.
# It never gates `passed`; it breaks ties between passed provider meshes and flags animation problems.
CRITIC = {
    "mode": "auto",  # auto = on when a backend SDK and its API key exist | on | off  (env CHARFORGE_CRITIC)
    "backend": "auto",  # auto = first ready of gemini, claude | gemini | claude  (env CHARFORGE_CRITIC_BACKEND)
    "models": {"gemini": "gemini-4-argon", "claude": "claude-opus-5-5"},
    "effort": "medium",  # low | medium | high: Claude effort, Gemini thinking_level
    "timeout_s": 300,  # Gemini request timeout
    "max_calls": 6,  # per project
    "min_delta": 1.0,  # critic points (0-10) that outweigh silhouette total between two passed meshes
    "warn_below": 6.0,  # overall or clip score below this -> warning
    "max_image_px": 1568,  # images are downscaled to this long edge before upload
    "mesh_weights": {"face": 0.25, "texture": 0.2, "details": 0.15, "anatomy": 0.25, "artifacts": 0.15},
}

PROVIDER_KEYS = {"tripo": "TRIPO_API_KEY", "meshy": "MESHY_API_KEY", "rodin": "RODIN_API_KEY", "hunyuan": "FAL_KEY"}
PROVIDER_SOURCES = {"meshy", "tripo", "rodin", "hunyuan"}
PROVIDER_ACCEPT_TOTAL = 0.80  # min weighted silhouette total for cloud provider meshes
MAX_PROVIDER_TRIS = 300000  # max face/triangle budget accepted by Meshy rigging

# --- Meshy API tunables ---
MESHY_AI_MODEL = "meshy-7.1"
MESHY_TARGET_POLYCOUNT = 30000
MESHY_TEXTURE_FROM_VIEWS = True
MESHY_LIBRARY_TTL_SECONDS = 7 * 24 * 3600  # 7 days cache TTL for /openapi/v1/animations/library
MESHY_ACTION_ALIASES: dict[str, list[str]] = {
    "idle": ["Idle", "Breathing_Idle", "Happy_Idle", "Standing_Idle", "Alert_Idle"],
    "walk": ["Walking", "Casual_Walk", "Walk", "Walk_Forward", "Walking_Woman"],
    "wave": ["Waving", "Wave", "Waving_Gesture", "Hello", "Standing_Greeting"],
    "run": ["Running", "Run", "Run_Forward", "Fast_Run"],
    "jump": ["Jump", "Jumping", "Happy_Jump"],
    "attack": ["Attack", "Double_Combo_Attack", "Slash", "Punch"],
    "dance": ["Dancing", "Hip_Hop_Dancing", "Samba_Dancing", "Dance"],
}
MESHY_MOTION_MODE = "prime"  # Text to Motion: prime (10 credits, FBX) | swift (3 credits, BVH)
MESHY_MOTION_DURATION = 4.0  # default seconds per Text to Motion clip (2-10, 0.5 steps)

# --- L3 quality gates (per-project override: project.json "rig_check_thresholds" / "anim_check_thresholds") ---
RIG_CHECK = {
    "angle": 40.0,  # deg each upper arm (forward, up) and thigh (forward) is swung in the deformation check
    "tol_m": 0.005,  # a vertex that must stay still "leaks" when it moves more than this (x height/1.7 m)
    "max_ratio": 0.01,  # fail when more than 1% of a still region leaks
    "hard_m": 0.03,  # fail when any still vertex moves more than 3 cm
    "follow_min": 0.6,  # fail when a swung limb follows its bone less than 60% (missing weights)
    "gap_min_m": 0.01,  # forearm/hand-body and shin-shin gap below which parts count as touching
}
ANIM_PREVIEW = {"frames": 6, "res": 320, "views": "front,side"}  # renders per clip for visual review
ANIM_CHECK = {
    "stretch_p99": 1.6,  # 99th percentile edge stretch (posed/rest) above this = visible tearing/ballooning
    "ground_cm": 3.0,  # lowest vertex sinking below the rest ground by more than this = feet through the floor
}
SECONDARY = {  # spring chains baked into every clip for hanging hair and skirts (animate(secondary="auto"))
    "stiffness": 1.2,  # pull back toward the animated rest direction (per second)
    "drag": 0.35,  # velocity damping per frame (0..1)
    "gravity": 0.15,  # extra sag (m/s^2 scale, rest shape already includes gravity)
    "skirt_chains": 8,  # chains around the hips
    "max_hair_chains": 4,
}
PIPELINE_POLL_SECONDS = 5.0  # pipeline_run wait between polls of a running cloud job (avoid HTTP 429)


def home() -> Path:
    path = Path(os.environ.get("CHARFORGE_HOME") or REPO_ROOT / "projects").expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def level() -> int:
    return max(1, min(3, int(os.environ.get("CHARFORGE_LEVEL") or 1)))


def blender_exe() -> str | None:
    cands = [
        os.environ.get("BLENDER_PATH"),
        shutil.which("blender"),
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    win_blender_dir = Path("C:/Program Files/Blender Foundation")
    if win_blender_dir.is_dir():
        found = sorted(win_blender_dir.glob("Blender */blender.exe"), reverse=True)
        cands.extend(str(p) for p in found)

    for cand in cands:
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
    from . import critic  # local import: critic imports config

    return {
        "ok": bool(exe) and all(deps.values()),
        "blender": exe,
        "blender_version": version,
        "home": str(home()),
        "level": level(),
        "providers": providers_available(),
        "deps": deps,
        "critic": critic.status(),
    }
