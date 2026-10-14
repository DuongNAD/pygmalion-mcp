"""Project state and loop policy. <home>/<id>/project.json is the single source of truth.

Layout of a project dir:
  project.json   state (see docs/ARCHITECTURE.md §5)   calls.jsonl  one line per tool call
  sheet.<ext>    original sheet                        views/       front|side|back.png, *_mask.png, preview.jpg
  meshes/        candidates and outputs                renders/<mesh-stem>/  masks, colors, diffs, compare.png
"""
from __future__ import annotations

import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

from . import config
from .envelope import CFError, action

SCHEMA_VERSION = 1
VIEWS = ("front", "side", "back")
MESH_EXTS = {".glb", ".fbx", ".obj", ".ply", ".stl"}
SHEET_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SEARCH_STAGES = {"views_ready", "scored", "passed", "stopped"}  # stages where score_mesh drives the policy
_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "char"


def project_dir(pid: str | None, must_exist: bool = True) -> Path:
    if not pid or not _ID_RE.fullmatch(pid):
        raise CFError("BAD_PROJECT_ID", f"Invalid project_id {pid!r}.", "Use the project_id returned by project_create.")
    pdir = config.home() / pid
    if must_exist and not (pdir / "project.json").is_file():
        raise CFError("PROJECT_NOT_FOUND", f"No project {pid!r} in {config.home()}.", "Call project_create first, or check the id with project_status.")
    return pdir


def load(pid: str) -> dict:
    return json.loads((project_dir(pid) / "project.json").read_text())


def save(p: dict) -> None:
    pdir = project_dir(p["id"], must_exist=False)
    p["updated_at"] = _now()
    tmp = pdir / "project.json.tmp"
    tmp.write_text(json.dumps(p, indent=2, ensure_ascii=False))
    tmp.replace(pdir / "project.json")  # atomic


def create(sheet_path: str, name: str, height_cm: float) -> dict:
    src = Path(sheet_path).expanduser()
    if not src.is_file():
        raise CFError("SHEET_NOT_FOUND", f"Sheet not found: {sheet_path}", "Pass an absolute path to an existing sheet image.")
    ext = src.suffix.lower()
    if ext not in SHEET_EXTS:
        raise CFError("SHEET_FORMAT", f"Unsupported sheet format {ext!r}.", "Use a .png, .jpg or .webp image.")
    base = slugify(name)
    pid, n = base, 2
    while (config.home() / pid).exists():
        pid, n = f"{base}-{n}", n + 1
    pdir = config.home() / pid
    for sub in ("views", "meshes", "renders"):
        (pdir / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, pdir / f"sheet{ext}")
    p = {
        "schema_version": SCHEMA_VERSION, "id": pid, "name": name, "height_cm": height_cm,
        "created_at": _now(), "updated_at": _now(), "sheet": f"sheet{ext}", "stage": "created",
        "views": {}, "reference": {}, "thresholds": dict(config.THRESHOLDS), "weights": dict(config.VIEW_WEIGHTS),
        "meshes": [], "scores": [], "best": None, "jobs": {}, "outputs": {}, "stop_reason": None,
    }
    save(p)
    return p


def rel(p: dict, path: Path | str) -> str:
    return Path(path).resolve().relative_to(project_dir(p["id"]).resolve()).as_posix()


def mesh_path(p: dict, arg: str) -> tuple[Path, str]:
    """Resolve a mesh argument to (absolute path inside the project, project-relative path).
    External files are copied into meshes/ so a project is self-contained."""
    pdir = project_dir(p["id"]).resolve()
    src = Path(arg).expanduser()
    src = (src if src.is_absolute() else pdir / src).resolve()
    if not src.is_file():
        raise CFError("MESH_NOT_FOUND", f"Mesh not found: {arg}", "Use a path from project_status data.meshes, or an absolute path to an existing file.")
    ext = src.suffix.lower()
    if ext not in MESH_EXTS:
        raise CFError("MESH_FORMAT", f"Unsupported mesh format {ext!r}.", "Use .glb, .fbx, .obj, .ply or .stl (save multi-file .gltf as .glb).")
    if pdir not in src.parents:
        dst, n = pdir / "meshes" / src.name, 2
        while dst.exists():
            dst, n = pdir / "meshes" / f"{src.stem}-{n}{ext}", n + 1
        shutil.copy2(src, dst)
        src = dst
    rpath = src.relative_to(pdir).as_posix()
    register_mesh(p, rpath, "external")
    return src, rpath


def register_mesh(p: dict, rpath: str, source: str, **meta) -> None:
    if not any(m["path"] == rpath for m in p["meshes"]):
        p["meshes"].append({"path": rpath, "source": source, "rotate_z": 0, "created_at": _now(), **meta})


def render_dir(p: dict, rpath: str) -> Path:
    out = project_dir(p["id"]) / "renders" / Path(rpath).stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def record_score(p: dict, rec: dict) -> None:
    """Append a score ({mesh, total, passed, iou, ...}); in search stages update best/stage/stop_reason."""
    p["scores"].append({**rec, "at": _now()})
    if p["stage"] not in SEARCH_STAGES:
        return  # post-search check (e.g. after optimize): record only
    if not p["best"] or rec["total"] > p["best"]["total"]:
        p["best"] = {"mesh": rec["mesh"], "total": rec["total"], "passed": rec["passed"]}
    totals = [s["total"] for s in p["scores"]]
    k = config.PLATEAU_ROUNDS
    plateau = len(totals) > k and max(totals[-k:]) - max(totals[:-k]) < config.PLATEAU_EPS
    if p["best"]["passed"]:
        p["stage"], p["stop_reason"] = "passed", None
    elif len(totals) >= config.MAX_ITERATIONS:
        p["stage"], p["stop_reason"] = "stopped", "budget"
    elif plateau:
        p["stage"], p["stop_reason"] = "stopped", "plateau"
    else:
        p["stage"] = "scored"


def suggest_next(p: dict) -> dict | None:
    """Default next step for the agent (the policy table in docs/ARCHITECTURE.md §6). None = finished."""
    lvl, pid, st = config.level(), p["id"], p["stage"]
    best = (p.get("best") or {}).get("mesh")
    if st == "created":
        return action("sheet_split", "Extract the front/side/back reference views first.", project_id=pid)
    if st == "views_ready":
        if lvl >= 2:
            return action("generate_mesh", "Create the first candidate mesh.", project_id=pid, provider="auto")
        return action("score_mesh", "Level 1 has no generator: obtain a mesh file, then score it.", project_id=pid, mesh_path="<your mesh>")
    if st == "scored":
        if lvl < 2:
            return action("score_mesh", "Fix the mesh using data.hints, then score the new file.", project_id=pid, mesh_path="<new mesh>")
        n_cand = sum(m["source"] not in ("external", "fit") for m in p["meshes"])
        if p["best"]["total"] < config.REGENERATE_BELOW and n_cand < config.MAX_CANDIDATES:
            return action("generate_mesh", "Best candidate is far off; try a new candidate.", project_id=pid, provider="auto")
        return action("mesh_fit", "Fit the best candidate's proportions to the reference, then re-score.", project_id=pid, mesh_path=best)
    if st in ("passed", "stopped"):
        if lvl >= 2:
            return action("mesh_optimize", "Search finished; clean up the best mesh.", project_id=pid, mesh_path=best)
        return action("render_views", "Level 1 done: render color previews of the best mesh.", project_id=pid, mesh_path=best, mode="color") if best else None
    if st == "optimized":
        return action("mesh_export", "Export the optimized mesh.", project_id=pid, mesh_path=p["outputs"].get("optimized"), format="glb")
    if st == "exported" and lvl >= 3:
        return action("auto_rig", "Add a humanoid rig.", project_id=pid, mesh_path=p["outputs"].get("glb"))
    return None


def stats(pid: str) -> dict:
    """Driver-model quality metrics from calls.jsonl (see docs/ARCHITECTURE.md §10)."""
    log = project_dir(pid) / "calls.jsonl"
    recs = [json.loads(line) for line in log.read_text().splitlines()] if log.is_file() else []
    errors = [r for r in recs if not r["ok"]]
    p = load(pid)
    return {
        "calls": len(recs),
        "errors": len(errors),
        "error_rate": round(len(errors) / len(recs), 3) if recs else 0.0,
        "error_codes": dict(Counter(r["code"] for r in errors)),
        "iterations": len(p["scores"]),
        "best_total": (p["best"] or {}).get("total"),
        "passed": bool((p["best"] or {}).get("passed")),
    }
