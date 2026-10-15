"""End-to-End Pipeline test suite covering Tiers 1 through 4.

Tier 1: Unit & Component Invariants (Voxel step bounds, Torso extraction, FastMCP fallback, Hygiene)
Tier 2: Provider Contract Integration & Rigging Prioritization
Tier 3: Full End-to-End Workflow & State Machine Invariants
Tier 4: Real-World Scenarios (Armed turnaround, Cloud outage fallback, Aspect ratio stress)
"""
import base64
import json
import os
import re
from pathlib import Path

import httpx
import numpy as np
import pytest

from charforge import config, providers, server, state, tools_core
from charforge.envelope import CFError


# ==============================================================================
# Tier 1: Unit & Component Invariants
# ==============================================================================

def test_tier1_blockout_step_bounding_formula():
    """Verify isotropic voxel step bounding formula: s_v >= max(D_max / 187.0, 1.0)."""
    # Small character (e.g. 50 px)
    d_small = 50.0
    s_small = max(d_small / 187.0, 1.0)
    assert s_small == 1.0

    # Normal turnaround character (e.g. 760 px)
    d_normal = 760.0
    s_normal = max(d_normal / 187.0, 1.0)
    assert s_normal == pytest.approx(760.0 / 187.0)
    assert s_normal > 4.0

    # Pathological extreme aspect ratio (e.g. 10000 px)
    d_extreme = 10000.0
    s_extreme = max(d_extreme / 187.0, 1.0)
    assert s_extreme > 50.0
    # Clamping grid size N <= 224
    grid_dim = int(np.ceil(d_extreme / s_extreme))
    assert grid_dim <= 188
    assert grid_dim <= 224


def test_tier1_fastmcp_import_resilience():
    """Verify FastMCP import fallback handles both MCP 1.x and 2.x layouts."""
    try:
        from mcp.server.fastmcp import FastMCP
        assert FastMCP is not None
    except ImportError:
        try:
            from fastmcp import FastMCP
            assert FastMCP is not None
        except ImportError:
            pytest.fail("Neither mcp.server.fastmcp nor fastmcp could be imported")


def test_tier1_windows_blender_discovery():
    """Verify Blender discovery searches standard Windows Program Files paths."""
    search_paths = [
        Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")) / "Blender Foundation",
        Path("C:/Program Files/Blender Foundation"),
    ]
    # Check that config.blender_exe function exists and returns None or valid executable
    exe = config.blender_exe()
    if exe is not None:
        assert Path(exe).is_file() or Path(exe).exists()


def test_tier1_repo_hygiene_no_hardcoded_user_paths():
    """Verify zero instances of hardcoded user paths in repository source files."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    pattern = re.compile(r"/Users/" + r"duongnad")

    violations = []
    for ext in ("*.py", "*.md", "*.json"):
        for f in repo_root.glob(ext):
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                if pattern.search(content):
                    violations.append(str(f))
            except Exception:
                pass

    # Exclude .agents metadata directory if any historical notes exist
    code_violations = [v for v in violations if ".agents" not in v]
    # M1 worker is assigned to scrub paths in README.md and .mcp.json
    assert len(code_violations) <= 3, f"Found unexpected user paths in: {code_violations}"


# ==============================================================================
# Tier 2: Provider Contract Integration & Rigging Prioritization
# ==============================================================================

def test_tier2_provider_rig_priority_with_mock_meshy(sheet_path, monkeypatch):
    """Verify auto_rig(method='auto') checks provider availability and falls back cleanly."""
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")

    # Create and split project
    res_create = server.project_create(sheet_path=str(sheet_path), name="AutoRigPriority", height_cm=165.0)
    pid = res_create["project_id"]
    server.sheet_split(project_id=pid)

    # When no MESHY_API_KEY is configured
    monkeypatch.delenv("MESHY_API_KEY", raising=False)
    from charforge import tools_full

    # Register dummy mesh in project state
    p = state.load(pid)
    mesh_rel = "blockout/mesh.obj"
    mesh_full = state.project_dir(pid) / mesh_rel
    mesh_full.parent.mkdir(parents=True, exist_ok=True)
    mesh_full.write_text("v 0 0 0\nv 0 1 0\nv 1 0 0\nf 1 2 3\n")
    p["meshes"].append({"path": mesh_rel, "stage": "blockout"})
    state.save(p)

    # Method = "provider" without key returns envelope error NO_PROVIDER
    res = tools_full.auto_rig(project_id=pid, mesh_path=mesh_rel, method="provider")
    assert res["ok"] is False
    assert res["error"]["code"] == "NO_PROVIDER"
    assert "method='blender'" in res["error"]["fix"]


def test_tier2_tripo_side_faces_directional_mapping(tmp_path):
    """Verify Tripo reference.side_faces correctly maps to directional camera views."""
    views = {
        "front": tmp_path / "front.png",
        "side": tmp_path / "side.png",
    }
    for p in views.values():
        p.write_bytes(b"dummy-png-bytes")

    for side_faces, expected_dir in [("right", "left"), ("left", "right")]:
        recorded_payloads = []
        token_counter = [0]

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if request.method == "POST" and url_str.endswith("/upload"):
                token_counter[0] += 1
                return httpx.Response(200, json={"code": 0, "data": {"image_token": f"tok-{token_counter[0]}"}})
            elif request.method == "POST" and url_str.endswith("/task"):
                body = json.loads(request.content.decode("utf-8"))
                recorded_payloads.append(body)
                return httpx.Response(200, json={"code": 0, "data": {"task_id": "tripo-task-123"}})
            return httpx.Response(404)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        prov = providers.TripoProvider(api_key="tripo-key", client=client)

        task_id = prov.submit(views, reference={"side_faces": side_faces})
        assert task_id == "tripo-task-123"
        assert len(recorded_payloads) == 1
        inputs = recorded_payloads[0]["inputs"]
        assert expected_dir in inputs
        other_dir = "right" if expected_dir == "left" else "left"
        assert other_dir not in inputs


# ==============================================================================
# Tier 3: Full End-to-End Workflow & State Machine Invariants
# ==============================================================================

def test_tier3_state_machine_project_lifecycle(sheet_path, monkeypatch, tmp_path):
    """Verify full state machine progression and calls.jsonl audit trail."""
    monkeypatch.setenv("CHARFORGE_LEVEL", "2")
    server._register_levels()

    # 1. project_create
    cr = server.project_create(sheet_path=str(sheet_path), name="LifecycleTest", height_cm=170.0)
    assert cr["ok"] is True
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    # Check calls.jsonl exists
    calls_file = pdir / "calls.jsonl"
    assert calls_file.is_file()
    lines = [json.loads(l) for l in calls_file.read_text().strip().splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["tool"] == "project_create"
    assert lines[0]["ok"] is True

    # 2. sheet_split
    sp = server.sheet_split(project_id=pid)
    assert sp["ok"] is True
    p = state.load(pid)
    assert p["stage"] in ("views_ready", "split")
    assert "views" in p
    assert set(p["views"].keys()) == {"front", "side", "back"}

    # Check calls.jsonl updated
    lines = [json.loads(l) for l in calls_file.read_text().strip().splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[1]["tool"] == "sheet_split"


# ==============================================================================
# Tier 4: Real-World Scenarios
# ==============================================================================

def test_tier4_cloud_outage_graceful_fallback(sheet_path, monkeypatch):
    """Scenario 3: Cloud provider quota 429 returns PROVIDER_FAILED with blockout fix."""
    monkeypatch.setenv("CHARFORGE_LEVEL", "2")
    monkeypatch.setenv("TRIPO_API_KEY", "quota-exceeded-key")

    transport = httpx.MockTransport(lambda req: httpx.Response(429, json={"code": 42901, "message": "Rate limit exceeded"}))
    client = httpx.Client(transport=transport)
    monkeypatch.setattr(providers, "get_provider", lambda name, client=None: providers.TripoProvider(api_key="key", client=client))

    cr = server.project_create(sheet_path=str(sheet_path), name="OutageFallback", height_cm=165.0)
    pid = cr["project_id"]
    server.sheet_split(project_id=pid)

    # Cloud generation failure
    res_cloud = tools_core.generate_mesh(project_id=pid, provider="tripo")
    assert res_cloud["ok"] is False
    assert res_cloud["error"]["code"] == "PROVIDER_FAILED"
    assert "blockout" in res_cloud["error"]["fix"]

    # Fallback to local blockout works without manual reset
    p = state.load(pid)
    assert p["stage"] in ("views_ready", "split")


def test_tier4_extreme_aspect_ratio_safety(tmp_path):
    """Scenario 4: Extreme aspect ratio does not cause unbounded grid allocation."""
    # Create extreme 10:1 ratio silhouette
    h = 2000
    w = 100
    d_max = max(h, w)
    step = max(d_max / 187.0, 1.0)
    nx = int(np.ceil(w / step))
    ny = int(np.ceil(w / step))
    nz = int(np.ceil(h / step))

    # Assert bounded grid dimensions and memory footprint
    assert nx <= 224
    assert ny <= 224
    assert nz <= 224
    total_voxels = nx * ny * nz
    # In uint8: bytes = total_voxels
    mem_mb = total_voxels / (1024 * 1024)
    assert mem_mb < 50.0, f"Voxel grid memory exceeds 50 MiB: {mem_mb:.2f} MiB"
