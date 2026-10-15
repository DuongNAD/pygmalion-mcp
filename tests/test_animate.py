"""Comprehensive test suite for R2 Character Animation Tool 'animate' and Multi-Action GLB Merging.

Tests:
- inspect_glb_animations chunk 0 parser (valid, corrupt, edge cases)
- Action name deduplication
- Error handling (missing rigged mesh, missing API key, missing rig task ID)
- animate in method='provider' with MockTransport and multi-clip merging
- animate in method='blender' with headless Blender procedural animation
- Full pipeline_run through animate
"""
import json
import os
import struct
from pathlib import Path
from typing import Any

import httpx
import pytest

from charforge import blender, config, providers, server, state, tools_full
from charforge.envelope import CFError
from charforge.tools_full import inspect_glb_animations


def create_synthetic_glb(actions: list[str]) -> bytes:
    """Build a valid binary glTF (GLB) containing specified animation action tracks."""
    gltf_dict = {
        "asset": {"version": "2.0", "generator": "CharForge Test"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "RootNode"}],
        "animations": [
            {
                "name": name,
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
                "samplers": [{"input": 0, "interpolation": "LINEAR", "output": 1}],
            }
            for name in actions
        ],
    }
    json_bytes = json.dumps(gltf_dict).encode("utf-8")
    padding = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * padding

    chunk_length = len(json_bytes)
    chunk_type = b"JSON"
    header_length = 12
    total_length = header_length + 8 + chunk_length

    glb_header = struct.pack("<4sII", b"glTF", 2, total_length)
    chunk_header = struct.pack("<I4s", chunk_length, chunk_type)
    return glb_header + chunk_header + json_bytes


def create_synthetic_rigged_glb(path: Path) -> None:
    """Create a minimal GLB with Mixamo armature bones using Blender."""
    exe = config.blender_exe()
    if not exe:
        pytest.skip("Blender executable not available.")

    script = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
arm_data = bpy.data.armatures.new("Armature")
arm_obj = bpy.data.objects.new("Armature", arm_data)
bpy.context.scene.collection.objects.link(arm_obj)
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.mode_set(mode="EDIT")
bones = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"
]
for name in bones:
    b = arm_data.edit_bones.new(name)
    b.head = (0, 0, 0)
    b.tail = (0, 0, 0.1)
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.export_scene.gltf(filepath=r"{str(path)}", export_format="GLB")
"""
    import subprocess
    subprocess.run([exe, "-b", "--factory-startup", "--python-expr", script], check=True, capture_output=True)


# ==============================================================================
# 1. Chunk 0 Inspector Unit Tests
# ==============================================================================

def test_inspect_glb_animations_valid(tmp_path):
    """Verify inspect_glb_animations correctly parses single and multi-action GLBs."""
    actions = ["idle", "walk", "wave"]
    glb_data = create_synthetic_glb(actions)
    glb_file = tmp_path / "test.glb"
    glb_file.write_bytes(glb_data)

    # Test from bytes
    extracted_from_bytes = inspect_glb_animations(glb_data)
    assert extracted_from_bytes == actions

    # Test from Path
    extracted_from_path = inspect_glb_animations(glb_file)
    assert extracted_from_path == actions


def test_inspect_glb_animations_corrupt_header(tmp_path):
    """Verify ValueError is raised on corrupt or malformed GLB headers."""
    # Too small (< 20 bytes)
    with pytest.raises(ValueError, match="too small"):
        inspect_glb_animations(b"glTF\x02\x00")

    # Invalid magic
    with pytest.raises(ValueError, match="Invalid GLB header"):
        inspect_glb_animations(b"RIFF\x02\x00\x00\x00\x20\x00\x00\x00JSONdata")

    # Invalid chunk 0 type (e.g. BIN\x00 instead of JSON)
    bad_chunk_type = struct.pack("<4sII", b"glTF", 2, 28) + struct.pack("<I4s", 8, b"BIN\x00") + b"12345678"
    with pytest.raises(ValueError, match="Expected JSON chunk 0"):
        inspect_glb_animations(bad_chunk_type)


def test_inspect_glb_animations_empty_and_unnamed():
    """Verify behavior on GLB with empty animations list or unnamed action tracks."""
    # Empty animations
    gltf_empty = {"asset": {"version": "2.0"}, "animations": []}
    json_bytes = json.dumps(gltf_empty).encode("utf-8")
    glb_empty = struct.pack("<4sII", b"glTF", 2, 20 + len(json_bytes)) + struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    assert inspect_glb_animations(glb_empty) == []

    # Unnamed animation falls back to action_{i}
    gltf_unnamed = {"asset": {"version": "2.0"}, "animations": [{}]}
    json_bytes2 = json.dumps(gltf_unnamed).encode("utf-8")
    glb_unnamed = struct.pack("<4sII", b"glTF", 2, 20 + len(json_bytes2)) + struct.pack("<I4s", len(json_bytes2), b"JSON") + json_bytes2
    assert inspect_glb_animations(glb_unnamed) == ["action_0"]


# ==============================================================================
# 2. Action Name Deduplication Tests
# ==============================================================================

def test_animate_action_name_deduplication(tmp_path, monkeypatch):
    """Verify that duplicate action names in requests are automatically deduplicated."""
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")

    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(b"dummy")
    cr = server.project_create(sheet_path=str(sheet), name="DedupTest", height_cm=165.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    # Set up dummy rigged GLB
    rig_path = pdir / f"export/{pid}-rigged.glb"
    rig_path.parent.mkdir(parents=True, exist_ok=True)
    rig_path.write_bytes(b"dummy-glb")

    p = state.load(pid)
    p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
    state.save(p)

    recorded_args = []
    def mock_blender_run(script_name, args):
        recorded_args.append(args)
        actions = args["actions"].split(",")
        out_p = Path(args["out"])
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(create_synthetic_glb(actions))
        return {"ok": True, "actions": actions, "clip_count": len(actions), "mesh_path": str(out_p)}

    monkeypatch.setattr(blender, "run", mock_blender_run)

    # Duplicate names in request
    res = tools_full.animate(
        project_id=pid,
        actions=["walk", "walk", "walk", "idle", "walk"],
        method="blender",
    )
    assert res["ok"] is True
    expected = ["walk", "walk_01", "walk_02", "idle", "walk_03"]
    assert res["data"]["actions"] == expected
    assert res["data"]["clip_count"] == 5
    assert recorded_args[0]["actions"] == ",".join(expected)


# ==============================================================================
# 3. Animate Error Handling Tests
# ==============================================================================

def test_animate_missing_rigged_mesh(tmp_path, monkeypatch):
    """Verify CFError(NO_RIGGED_MESH) when project has no rigged output and no mesh_path."""
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")
    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(b"dummy")
    cr = server.project_create(sheet_path=str(sheet), name="NoRigMesh", height_cm=165.0)
    pid = cr["project_id"]

    res = tools_full.animate(project_id=pid)
    assert res["ok"] is False
    assert res["error"]["code"] == "NO_RIGGED_MESH"
    assert "Run auto_rig first" in res["error"]["fix"]


def test_animate_provider_missing_key(tmp_path, monkeypatch):
    """Verify CFError(NO_PROVIDER) when method='provider' but MESHY_API_KEY is unset."""
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")
    monkeypatch.delenv("MESHY_API_KEY", raising=False)

    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(b"dummy")
    cr = server.project_create(sheet_path=str(sheet), name="NoKeyAnim", height_cm=165.0)
    pid = cr["project_id"]

    p = state.load(pid)
    rig_path = state.project_dir(pid) / f"export/{pid}-rigged.glb"
    rig_path.parent.mkdir(parents=True, exist_ok=True)
    rig_path.write_bytes(b"dummy-glb")
    p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
    state.save(p)

    res = tools_full.animate(project_id=pid, method="provider")
    assert res["ok"] is False
    assert res["error"]["code"] == "NO_PROVIDER"
    assert "No provider API key" in res["error"]["message"]


def test_animate_provider_missing_rig_task_id(tmp_path, monkeypatch):
    """Verify CFError(NO_PROVIDER) when method='provider' with key but no rig_task_id."""
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")
    monkeypatch.setenv("MESHY_API_KEY", "valid-key")

    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(b"dummy")
    cr = server.project_create(sheet_path=str(sheet), name="NoTaskIdAnim", height_cm=165.0)
    pid = cr["project_id"]

    p = state.load(pid)
    rig_path = state.project_dir(pid) / f"export/{pid}-rigged.glb"
    rig_path.parent.mkdir(parents=True, exist_ok=True)
    rig_path.write_bytes(b"dummy-glb")
    p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
    state.save(p)

    res = tools_full.animate(project_id=pid, method="provider")
    assert res["ok"] is False
    assert res["error"]["code"] == "NO_PROVIDER"
    assert "No provider rig task ID" in res["error"]["message"]


# ==============================================================================
# 4. Provider Mode Integration & Multi-Clip Download / Merge
# ==============================================================================

def test_animate_method_provider_multi_clip_mock_transport(tmp_path, monkeypatch):
    """Verify animate(method='provider') multi-clip orchestration with MockTransport."""
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")
    monkeypatch.setenv("MESHY_API_KEY", "test-meshy-key")

    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(b"dummy")
    cr = server.project_create(sheet_path=str(sheet), name="ProviderAnim", height_cm=165.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    rig_rpath = f"export/{pid}-rigged.glb"
    rig_path = pdir / rig_rpath
    rig_path.parent.mkdir(parents=True, exist_ok=True)
    rig_path.write_bytes(create_synthetic_glb(["tpose"]))

    p = state.load(pid)
    p["outputs"]["rigged"] = rig_rpath
    p["rig_task_id"] = "rig_task_provider_123"
    state.save(p)

    # Setup mock transport for Meshy animations endpoint
    action_clips = {
        1: create_synthetic_glb(["idle"]),
        2: create_synthetic_glb(["walk"]),
        3: create_synthetic_glb(["wave"]),
    }
    submitted_tasks = []

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if request.method == "POST" and "/animations" in url_str:
            body = json.loads(request.content.decode("utf-8"))
            act_id = body.get("action_id", 1)
            task_id = f"anim_task_{act_id}"
            submitted_tasks.append((body.get("rig_task_id"), act_id))
            return httpx.Response(200, json={"result": task_id})
        elif request.method == "GET" and "/animations/anim_task_" in url_str:
            act_id = int(url_str.split("_")[-1])
            return httpx.Response(200, json={"status": "SUCCEEDED", "model_urls": {"glb": f"https://mock.cdn/anim_{act_id}.glb"}})
        elif request.method == "GET" and "mock.cdn/anim_" in url_str:
            act_id = int(url_str.split("/")[-1].replace(".glb", "").replace("anim_", ""))
            return httpx.Response(200, content=action_clips[act_id])
        return httpx.Response(404)

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(providers, "get_provider", lambda name, client=None: providers.MeshyProvider(api_key="test-meshy-key", client=client or mock_client))

    # Mock blender.run for merge_animations.py
    blender_runs = []
    def mock_blender_run(script_name, args):
        blender_runs.append((script_name, args))
        out_p = Path(args["out"])
        out_p.parent.mkdir(parents=True, exist_ok=True)
        requested_acts = args["actions"].split(",")
        out_p.write_bytes(create_synthetic_glb(requested_acts))
        return {"ok": True, "actions": requested_acts, "clip_count": len(requested_acts), "mesh_path": str(out_p)}

    monkeypatch.setattr(blender, "run", mock_blender_run)

    # Execute animate
    res = tools_full.animate(
        project_id=pid,
        actions=["idle", "walk", "wave"],
        method="provider",
    )

    assert res["ok"] is True
    assert res["data"]["method"] == "provider"
    assert res["data"]["actions"] == ["idle", "walk", "wave"]
    assert res["data"]["clip_count"] == 3
    assert len(submitted_tasks) == 3
    assert len(blender_runs) == 1
    assert blender_runs[0][1]["mode"] == "merge"

    # Verify state updated
    p_after = state.load(pid)
    assert p_after["stage"] == "animated"
    assert p_after["outputs"]["animated"] == f"export/{pid}-animated.glb"
    assert len(p_after["animations"]) == 3
    assert [a["name"] for a in p_after["animations"]] == ["idle", "walk", "wave"]
    assert res["next_action"] is None


# ==============================================================================
# 5. Blender Offline Headless Mode Verification
# ==============================================================================

@pytest.mark.blender
def test_animate_method_blender_headless(tmp_path, monkeypatch):
    """Verify animate(method='blender') with real headless Blender execution.

    Generates genuine procedural keyframes for idle, walk, wave and verifies
    exact presence and count in the exported GLB chunk 0.
    """
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")

    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(b"dummy")
    cr = server.project_create(sheet_path=str(sheet), name="BlenderAnimReal", height_cm=165.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    # Create real rigged GLB with Mixamo bones
    rig_path = pdir / f"export/{pid}-rigged.glb"
    rig_path.parent.mkdir(parents=True, exist_ok=True)
    create_synthetic_rigged_glb(rig_path)
    assert rig_path.is_file() and rig_path.stat().st_size > 0

    p = state.load(pid)
    p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
    p["stage"] = "rigged"
    state.save(p)

    # Execute animate via real Blender
    requested_actions = ["idle", "walk", "wave"]
    res = tools_full.animate(
        project_id=pid,
        actions=requested_actions,
        method="blender",
    )

    assert res["ok"] is True, f"animate failed: {res}"
    assert res["data"]["method"] == "blender"
    assert res["data"]["clip_count"] == 3
    assert res["data"]["actions"] == requested_actions
    assert res["next_action"] is None

    # Inspect the exported GLB binary
    out_glb_path = pdir / res["data"]["mesh_path"]
    assert out_glb_path.is_file()
    assert out_glb_path.stat().st_size > 0

    extracted_actions = inspect_glb_animations(out_glb_path)
    assert len(extracted_actions) == 3
    for act in requested_actions:
        assert act in extracted_actions, f"Expected action '{act}' in GLB animations {extracted_actions}"

    # Verify project state persistence
    p_final = state.load(pid)
    assert p_final["stage"] == "animated"
    assert p_final["outputs"]["animated"] == f"export/{pid}-animated.glb"
    assert [a["name"] for a in p_final["animations"]] == requested_actions


# ==============================================================================
# 6. Pipeline Run Integration Test
# ==============================================================================

def test_pipeline_run_integration_with_animate(tmp_path, monkeypatch):
    """Verify autonomous pipeline_run handles animate: bool = True and step formatting."""
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")

    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(b"dummy")
    cr = server.project_create(sheet_path=str(sheet), name="PipeAnim", height_cm=165.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    # Setup project directly at rigged stage to test the animation transition
    rig_rpath = f"export/{pid}-rigged.glb"
    rig_path = pdir / rig_rpath
    rig_path.parent.mkdir(parents=True, exist_ok=True)
    rig_path.write_bytes(b"dummy-glb")

    p = state.load(pid)
    p["outputs"]["glb"] = "meshes/model.glb"
    p["outputs"]["rigged"] = rig_rpath
    p["stage"] = "rigged"
    state.save(p)

    # Next action from suggest_next should be animate
    nxt = state.suggest_next(p)
    assert nxt is not None
    assert nxt["tool"] == "animate"

    # Mock blender.run for animate
    def mock_blender_run(script_name, args):
        out_p = Path(args["out"])
        out_p.parent.mkdir(parents=True, exist_ok=True)
        acts = args["actions"].split(",")
        out_p.write_bytes(create_synthetic_glb(acts))
        return {"ok": True, "actions": acts, "clip_count": len(acts), "mesh_path": str(out_p)}

    monkeypatch.setattr(blender, "run", mock_blender_run)

    # Run pipeline with animate=True
    res = tools_full.pipeline_run(project_id=pid, max_seconds=240, rig=True, animate=True)

    assert res["ok"] is True
    assert res["data"]["status"] == "done"
    assert res["data"]["stage"] == "animated"
    assert res["next_action"] is None

    # Verify step summary
    steps = res["data"]["steps"]
    assert len(steps) == 1
    assert steps[0]["tool"] == "animate"
    assert "Animated model (3 clips):" in steps[0]["summary"]

    # Verify idempotency on completed project
    res_idempotent = tools_full.pipeline_run(project_id=pid, max_seconds=240, rig=True, animate=True)
    assert res_idempotent["ok"] is True
    assert res_idempotent["data"]["status"] == "done"
    assert len(res_idempotent["data"]["steps"]) == 0
    assert res_idempotent["next_action"] is None
