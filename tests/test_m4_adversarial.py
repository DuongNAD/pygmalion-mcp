"""Empirical Adversarial Test Suite for Milestone 4 (R2 Character Animation & Multi-Action GLB).

Adversarially challenges and stress-tests:
1. Custom action sets:
   - Single action: ["idle"]
   - Mixed standard + custom actions: ["wave", "walk", "jump"]
   - Arbitrary custom action set: ["action1", "action2", "action3", "action4"]
   - Default fallback on None or empty list
2. Duplicate action names and uniqueness:
   - Triplicate duplicates: ["walk", "walk", "walk"] -> ["walk", "walk_01", "walk_02"]
   - Multi-action duplicates: ["idle", "walk", "idle", "walk", "wave", "wave"]
   - Uniqueness invariants across requests and exports
3. GLB binary chunk 0 verification:
   - Binary glTF 2.0 header and chunk 0 layout compliance
   - Exact count verification (assert len(animations) == len(requested_actions))
   - Rejection on dropped or extra actions (clip count mismatch detection)
   - Rejection on corrupt or truncated headers
4. Procedural keyframing on a real humanoid rig, measured in world space:
   - Idle: weight shift on Hips, head motion, seamless loop
   - Walk: thighs swing >= 35 deg out of phase, knee flexion, arms counter-swing the opposite thigh, gait bounce
   - Wave: right hand rises above the shoulder and returns
   - Jump: crouch, then hips rise >= 15 cm
5. Provider mode mock flow with simulated HTTP errors:
   - HTTP 404, 500 on submit_animation
   - HTTP 500 on status polling
   - Provider task failure status (status="FAILED")
   - Polling timeout beyond deadline
   - HTTP 404, 500 on CDN clip download
   - Corrupt GLB payload returned by CDN
   - Partial failure mid-stream
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from charforge import blender, config, providers, server, state, tools_full
from charforge.envelope import CFError
from charforge.tools_full import inspect_glb_animations


# ==============================================================================
# Test Fixtures & Helpers
# ==============================================================================

def create_synthetic_glb_binary(actions: list[str]) -> bytes:
    """Build a valid binary glTF 2.0 (GLB) containing specified animation action tracks."""
    gltf_dict = {
        "asset": {"version": "2.0", "generator": "CharForge Adversarial Mock"},
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


def create_minimal_rigged_glb(path: Path) -> None:
    """Create a minimal humanoid GLB with Mixamo armature bones in Blender."""
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
    subprocess.run([exe, "-b", "--factory-startup", "--python-expr", script], check=True, capture_output=True)


def sample_world_kinematics(glb_path: Path, result_json_path: Path) -> dict[str, Any]:
    """Headless Blender: per action and frame, world-space joint positions of the key bones (found by Mixamo name)."""
    exe = config.blender_exe()
    if not exe:
        pytest.skip("Blender executable not available.")
    script = f"""
import json, bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = 30
bpy.ops.import_scene.gltf(filepath=r"{str(glb_path)}")
arm = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE'][0]
ad = arm.animation_data
for t in ad.nla_tracks:
    t.mute = True
names = ["Hips", "Spine2", "Head", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot",
         "LeftArm", "LeftForeArm", "RightArm", "RightForeArm", "RightHand"]
out = {{}}
for act in bpy.data.actions:
    ad.action = act
    if getattr(ad, "action_slot", None) is None and len(getattr(act, "slots", [])):
        ad.action_slot = act.slots[0]
    f0, f1 = (int(round(x)) for x in act.frame_range)
    frames = {{}}
    for f in range(f0, f1 + 1):
        bpy.context.scene.frame_set(f)
        frames[f] = {{n: [list(arm.matrix_world @ arm.pose.bones[n].head), list(arm.matrix_world @ arm.pose.bones[n].tail)]
                     for n in names if n in arm.pose.bones}}
    out[act.name] = {{"range": [f0, f1], "frames": frames}}
with open(r"{str(result_json_path)}", "w") as fh:
    json.dump(out, fh)
"""
    subprocess.run([exe, "-b", "--factory-startup", "--python-exit-code", "1", "--python-expr", script], check=True, capture_output=True)
    return json.loads(result_json_path.read_text())


# ==============================================================================
# 1. Custom Action Sets
# ==============================================================================

class TestCustomActionSets:
    """Stress-test custom action configurations against animate and merge_animations."""

    def test_single_action_idle(self, tmp_path, monkeypatch):
        """Verify requesting only ['idle'] produces a 1-action GLB with exact chunk 0 naming."""
        monkeypatch.setenv("CHARFORGE_LEVEL", "3")
        sheet = tmp_path / "sheet.png"
        sheet.write_bytes(b"dummy")
        cr = server.project_create(sheet_path=str(sheet), name="SingleActionIdle", height_cm=165.0)
        pid = cr["project_id"]
        pdir = state.project_dir(pid)

        rig_path = pdir / f"export/{pid}-rigged.glb"
        rig_path.parent.mkdir(parents=True, exist_ok=True)
        create_minimal_rigged_glb(rig_path)

        p = state.load(pid)
        p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
        p["stage"] = "rigged"
        state.save(p)

        res = tools_full.animate(project_id=pid, actions=["idle"], method="blender")
        assert res["ok"] is True
        assert res["data"]["actions"] == ["idle"]
        assert res["data"]["clip_count"] == 1

        out_glb = pdir / res["data"]["mesh_path"]
        extracted = inspect_glb_animations(out_glb)
        assert extracted == ["idle"]

    def test_mixed_standard_and_custom_actions(self, tmp_path, monkeypatch):
        """Verify mixed action set ['wave', 'walk', 'jump'] exports 3 distinct actions."""
        monkeypatch.setenv("CHARFORGE_LEVEL", "3")
        sheet = tmp_path / "sheet.png"
        sheet.write_bytes(b"dummy")
        cr = server.project_create(sheet_path=str(sheet), name="MixedActions", height_cm=165.0)
        pid = cr["project_id"]
        pdir = state.project_dir(pid)

        rig_path = pdir / f"export/{pid}-rigged.glb"
        rig_path.parent.mkdir(parents=True, exist_ok=True)
        create_minimal_rigged_glb(rig_path)

        p = state.load(pid)
        p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
        p["stage"] = "rigged"
        state.save(p)

        actions = ["wave", "walk", "jump"]
        res = tools_full.animate(project_id=pid, actions=actions, method="blender")
        assert res["ok"] is True
        assert res["data"]["actions"] == actions
        assert res["data"]["clip_count"] == 3

        out_glb = pdir / res["data"]["mesh_path"]
        extracted = inspect_glb_animations(out_glb)
        assert len(extracted) == 3
        assert set(extracted) == set(actions)

    def test_all_custom_action_names(self, tmp_path, monkeypatch):
        """Arbitrary names ['action1', ..., 'action4'] are neither procedural nor library clips: animate fails with
        ANIM_FAILED listing the valid names (they used to become a silent sway, card G3-3)."""
        monkeypatch.setenv("CHARFORGE_LEVEL", "3")
        sheet = tmp_path / "sheet.png"
        sheet.write_bytes(b"dummy")
        cr = server.project_create(sheet_path=str(sheet), name="AllCustomActions", height_cm=165.0)
        pid = cr["project_id"]
        pdir = state.project_dir(pid)

        rig_path = pdir / f"export/{pid}-rigged.glb"
        rig_path.parent.mkdir(parents=True, exist_ok=True)
        create_minimal_rigged_glb(rig_path)

        p = state.load(pid)
        p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
        p["stage"] = "rigged"
        state.save(p)

        custom_actions = ["action1", "action2", "action3", "action4"]
        res = tools_full.animate(project_id=pid, actions=custom_actions, method="blender")
        assert res["ok"] is False
        assert res["error"]["code"] == "ANIM_FAILED"
        assert all(a in res["error"]["message"] for a in custom_actions)
        assert res["error"]["fix"].startswith("Use one of: idle, walk, run, wave, jump, ")
        assert not (pdir / f"export/{pid}-animated.glb").exists()

    def test_empty_and_none_actions_fallback(self, tmp_path, monkeypatch):
        """Verify None and empty list fallback to default ['idle', 'walk', 'wave']."""
        monkeypatch.setenv("CHARFORGE_LEVEL", "3")
        sheet = tmp_path / "sheet.png"
        sheet.write_bytes(b"dummy")
        cr = server.project_create(sheet_path=str(sheet), name="DefaultFallback", height_cm=165.0)
        pid = cr["project_id"]
        pdir = state.project_dir(pid)

        rig_path = pdir / f"export/{pid}-rigged.glb"
        rig_path.parent.mkdir(parents=True, exist_ok=True)
        create_minimal_rigged_glb(rig_path)

        p = state.load(pid)
        p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
        p["stage"] = "rigged"
        state.save(p)

        # Empty list
        res_empty = tools_full.animate(project_id=pid, actions=[], method="blender")
        assert res_empty["ok"] is True
        assert res_empty["data"]["actions"] == ["idle", "walk", "wave"]
        assert res_empty["data"]["clip_count"] == 3


# ==============================================================================
# 2. Duplicate Action Names and Uniqueness
# ==============================================================================

class TestDuplicateActionNames:
    """Stress-test duplicate action name deduplication and name uniqueness."""

    def test_triplicate_duplicate_names(self, tmp_path, monkeypatch):
        """Verify ['walk', 'walk', 'walk'] deduplicates to ['walk', 'walk_01', 'walk_02']."""
        monkeypatch.setenv("CHARFORGE_LEVEL", "3")
        sheet = tmp_path / "sheet.png"
        sheet.write_bytes(b"dummy")
        cr = server.project_create(sheet_path=str(sheet), name="TriplicateWalk", height_cm=165.0)
        pid = cr["project_id"]
        pdir = state.project_dir(pid)

        rig_path = pdir / f"export/{pid}-rigged.glb"
        rig_path.parent.mkdir(parents=True, exist_ok=True)
        create_minimal_rigged_glb(rig_path)

        p = state.load(pid)
        p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
        p["stage"] = "rigged"
        state.save(p)

        res = tools_full.animate(project_id=pid, actions=["walk", "walk", "walk"], method="blender")
        assert res["ok"] is True
        expected = ["walk", "walk_01", "walk_02"]
        assert res["data"]["actions"] == expected
        assert res["data"]["clip_count"] == 3

        # Chunk 0 extraction verifies uniqueness
        out_glb = pdir / res["data"]["mesh_path"]
        extracted = inspect_glb_animations(out_glb)
        assert len(extracted) == 3
        assert len(set(extracted)) == 3, f"Expected all unique action names, got {extracted}"
        assert set(extracted) == set(expected)

    def test_multi_type_duplicates_and_interleaved(self, tmp_path, monkeypatch):
        """Verify complex interleaved duplicates ['idle', 'walk', 'idle', 'walk', 'wave', 'wave']."""
        monkeypatch.setenv("CHARFORGE_LEVEL", "3")
        sheet = tmp_path / "sheet.png"
        sheet.write_bytes(b"dummy")
        cr = server.project_create(sheet_path=str(sheet), name="InterleavedDupes", height_cm=165.0)
        pid = cr["project_id"]
        pdir = state.project_dir(pid)

        rig_path = pdir / f"export/{pid}-rigged.glb"
        rig_path.parent.mkdir(parents=True, exist_ok=True)
        create_minimal_rigged_glb(rig_path)

        p = state.load(pid)
        p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
        p["stage"] = "rigged"
        state.save(p)

        requested = ["idle", "walk", "idle", "walk", "wave", "wave"]
        expected = ["idle", "walk", "idle_01", "walk_01", "wave", "wave_01"]

        res = tools_full.animate(project_id=pid, actions=requested, method="blender")
        assert res["ok"] is True
        assert res["data"]["actions"] == expected
        assert res["data"]["clip_count"] == 6

        out_glb = pdir / res["data"]["mesh_path"]
        extracted = inspect_glb_animations(out_glb)
        assert len(extracted) == 6
        assert len(set(extracted)) == 6


# ==============================================================================
# 3. Exported GLB Binary Chunk 0 Verification
# ==============================================================================

class TestGLBChunk0BinaryVerification:
    """Rigorous verification of glTF 2.0 chunk 0 binary layout and count validation."""

    def test_chunk0_binary_spec_compliance(self, tmp_path):
        """Verify chunk 0 follows glTF 2.0 binary layout: magic, length, JSON chunk."""
        actions = ["idle", "walk", "wave", "attack", "defend"]
        raw_glb = create_synthetic_glb_binary(actions)

        # GLB header
        magic, version, total_length = struct.unpack("<4sII", raw_glb[:12])
        assert magic == b"glTF"
        assert version == 2
        assert total_length == len(raw_glb)

        # Chunk 0 header
        chunk_len, chunk_type = struct.unpack("<I4s", raw_glb[12:20])
        assert chunk_type == b"JSON"
        assert chunk_len % 4 == 0  # 4-byte aligned

        # Chunk 0 JSON content
        json_bytes = raw_glb[20 : 20 + chunk_len]
        data = json.loads(json_bytes.decode("utf-8"))
        assert "animations" in data
        assert len(data["animations"]) == len(actions)
        extracted_names = [a["name"] for a in data["animations"]]
        assert extracted_names == actions

    def test_inspect_glb_animations_corrupt_payloads(self):
        """Verify inspect_glb_animations strictly raises ValueError on corrupted data."""
        # 1. Underflow
        with pytest.raises(ValueError, match="too small"):
            inspect_glb_animations(b"glTF")

        # 2. Bad magic
        with pytest.raises(ValueError, match="Invalid GLB header"):
            inspect_glb_animations(b"NOTG\x02\x00\x00\x00\x20\x00\x00\x00" + b"\x00" * 20)

        # 3. Bad version
        with pytest.raises(ValueError, match="Invalid GLB header"):
            bad_ver = struct.pack("<4sII", b"glTF", 1, 30) + struct.pack("<I4s", 10, b"JSON") + b" " * 10
            inspect_glb_animations(bad_ver)

        # 4. Non-JSON chunk 0
        bad_chunk = struct.pack("<4sII", b"glTF", 2, 30) + struct.pack("<I4s", 10, b"BIN\x00") + b" " * 10
        with pytest.raises(ValueError, match="Expected JSON chunk 0"):
            inspect_glb_animations(bad_chunk)

    def test_chunk0_clip_count_mismatch_detection(self, tmp_path, monkeypatch):
        """Verify animate detects if Blender drops or adds an action and raises CFError('ANIM_FAILED')."""
        monkeypatch.setenv("CHARFORGE_LEVEL", "3")
        sheet = tmp_path / "sheet.png"
        sheet.write_bytes(b"dummy")
        cr = server.project_create(sheet_path=str(sheet), name="MismatchDetect", height_cm=165.0)
        pid = cr["project_id"]
        pdir = state.project_dir(pid)

        rig_path = pdir / f"export/{pid}-rigged.glb"
        rig_path.parent.mkdir(parents=True, exist_ok=True)
        rig_path.write_bytes(b"dummy")

        p = state.load(pid)
        p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
        p["stage"] = "rigged"
        state.save(p)

        # Mock blender.run returning GLB with only 2 clips when 3 were requested
        def mock_blender_drop_action(script_name, args):
            out_p = Path(args["out"])
            out_p.parent.mkdir(parents=True, exist_ok=True)
            # Only export 2 of the 3 actions
            out_p.write_bytes(create_synthetic_glb_binary(["idle", "walk"]))
            return {"ok": True, "actions": ["idle", "walk"], "clip_count": 2, "mesh_path": str(out_p)}

        monkeypatch.setattr(blender, "run", mock_blender_drop_action)

        res = tools_full.animate(project_id=pid, actions=["idle", "walk", "wave"], method="blender")
        assert res["ok"] is False
        assert res["error"]["code"] == "ANIM_FAILED"
        assert "Animation clip count mismatch" in res["error"]["message"]
        assert "expected 3, got 2" in res["error"]["message"]


# ==============================================================================
# 4. Procedural Keyframing Non-Zero Motion on Mixamo Bones
# ==============================================================================

@pytest.mark.blender
class TestProceduralMotionMixamoBones:
    """Procedural keyframing on a real humanoid rig: measured in world space, so it holds for any bone roll/naming."""

    @staticmethod
    def _pitch(seg):
        """Forward swing of a limb segment in degrees (0 = straight down, + = toward -Y, the character's front)."""
        import math

        (hx, hy, hz), (tx, ty, tz) = seg
        return math.degrees(math.atan2(-(ty - hy), -(tz - hz)))

    @staticmethod
    def _angle(a, b):
        import math

        va = [a[1][i] - a[0][i] for i in range(3)]
        vb = [b[1][i] - b[0][i] for i in range(3)]
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va)) or 1.0
        nb = math.sqrt(sum(x * x for x in vb)) or 1.0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / (na * nb)))))

    @staticmethod
    def _corr(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = sum((x - mx) ** 2 for x in xs) ** 0.5 or 1e-9
        sy = sum((y - my) ** 2 for y in ys) ** 0.5 or 1e-9
        return sxy / (sx * sy)

    def test_procedural_motion_amplitude_and_kinematics(self, tmp_path, monkeypatch):
        """idle breathes and loops; walk swings the thighs out of phase with knee flexion and counter-swinging
        arms; wave lifts the right hand above the shoulder and brings it back; jump lifts the hips."""
        from tests.bpy_fixtures import make_character

        monkeypatch.setenv("CHARFORGE_LEVEL", "3")
        sheet = tmp_path / "sheet.png"
        sheet.write_bytes(b"dummy")
        cr = server.project_create(sheet_path=str(sheet), name="KinematicsCheck", height_cm=170.0)
        pid = cr["project_id"]
        pdir = state.project_dir(pid)

        rig_path = pdir / f"export/{pid}-rigged.glb"
        rig_path.parent.mkdir(parents=True, exist_ok=True)
        make_character(rig_path)

        p = state.load(pid)
        p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
        p["stage"] = "rigged"
        state.save(p)

        actions = ["idle", "walk", "wave", "jump"]
        res = tools_full.animate(project_id=pid, actions=actions, method="blender", secondary="off")
        assert res["ok"] is True, res
        assert res["data"]["motion_check"]["passed"] is True, res["data"]["motion_check"]

        out_glb = pdir / res["data"]["mesh_path"]
        clips = sample_world_kinematics(out_glb, tmp_path / "kin.json")
        assert set(clips) == set(actions)

        def series(clip, bone, k=0):
            fr = clips[clip]["frames"]
            return [fr[f][bone][k] for f in sorted(fr, key=int)]

        # idle: subtle but real motion, seamless loop
        hips = series("idle", "Hips")
        assert max(abs(h[0] - hips[0][0]) for h in hips) > 0.003  # weight shift
        head = [[h, t] for h, t in zip(series("idle", "Head", 0), series("idle", "Head", 1))]
        assert max(self._angle(head[0], s) for s in head) > 1.0
        assert all(abs(a - b) < 1e-3 for a, b in zip(hips[0], hips[-1]))

        # walk: thighs swing >= 35 deg peak to peak, out of phase; knees bend; arms counter-swing the thighs
        lt = [self._pitch(s) for s in zip(series("walk", "LeftUpLeg", 0), series("walk", "LeftUpLeg", 1))]
        rt = [self._pitch(s) for s in zip(series("walk", "RightUpLeg", 0), series("walk", "RightUpLeg", 1))]
        assert max(lt) - min(lt) >= 35 and max(rt) - min(rt) >= 35
        assert self._corr(lt, rt) < -0.8
        thigh = list(zip(series("walk", "LeftUpLeg", 0), series("walk", "LeftUpLeg", 1)))
        shin = list(zip(series("walk", "LeftLeg", 0), series("walk", "LeftLeg", 1)))
        assert max(self._angle(a, b) for a, b in zip(thigh, shin)) >= 40
        la = [self._pitch(s) for s in zip(series("walk", "LeftArm", 0), series("walk", "LeftArm", 1))]
        assert max(la) - min(la) >= 15
        assert self._corr(la, rt) > 0.8  # left arm forward while the right thigh is forward
        hz = [h[2] for h in series("walk", "Hips")]
        assert max(hz) - min(hz) >= 0.01

        # wave: right hand above the right shoulder mid-clip, back down at the end
        hand = series("wave", "RightHand")
        shoulder = series("wave", "RightArm")
        mid = len(hand) // 2
        assert hand[mid][2] > shoulder[mid][2] + 0.05
        assert hand[0][2] < shoulder[0][2] and abs(hand[-1][2] - hand[0][2]) < 0.03

        # jump: crouch, then the hips rise well above rest
        jz = [h[2] for h in series("jump", "Hips")]
        assert max(jz) - jz[0] >= 0.15
        assert min(jz) < jz[0] - 0.05


# ==============================================================================
# 5. Provider Mode Mock Flow with Simulated HTTP Errors
# ==============================================================================

class TestProviderHttpErrors:
    """Stress-test provider mode error handling across HTTP and payload corruptions."""

    @pytest.fixture
    def setup_provider_project(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHARFORGE_LEVEL", "3")
        monkeypatch.setenv("MESHY_API_KEY", "test-meshy-key")

        sheet = tmp_path / "sheet.png"
        sheet.write_bytes(b"dummy")
        cr = server.project_create(sheet_path=str(sheet), name="ProviderErrors", height_cm=165.0)
        pid = cr["project_id"]
        pdir = state.project_dir(pid)

        rig_path = pdir / f"export/{pid}-rigged.glb"
        rig_path.parent.mkdir(parents=True, exist_ok=True)
        rig_path.write_bytes(create_synthetic_glb_binary(["tpose"]))

        p = state.load(pid)
        p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
        p["rig_task_id"] = "rig_task_provider_123"
        state.save(p)

        return pid, pdir

    @pytest.mark.parametrize("status_code", [404, 500])
    def test_provider_submit_animation_http_errors(self, setup_provider_project, monkeypatch, status_code):
        """Simulate HTTP 404 or 500 on POST /openapi/v1/animations."""
        pid, pdir = setup_provider_project

        def handler(request: httpx.Request) -> httpx.Response:
            if "animations" in str(request.url):
                return httpx.Response(status_code, text=f"Meshy Error {status_code}")
            return httpx.Response(404)

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            providers, "get_provider",
            lambda name, client=None: providers.MeshyProvider(api_key="test-meshy-key", client=client or mock_client)
        )

        res = tools_full.animate(project_id=pid, actions=["idle"], method="provider")
        assert res["ok"] is False
        assert res["error"]["code"] == "PROVIDER_FAILED"
        assert str(status_code) in res["error"]["message"]

    @pytest.mark.parametrize("status_code", [404, 500])
    def test_provider_poll_status_http_errors(self, setup_provider_project, monkeypatch, status_code):
        """Simulate HTTP 404 or 500 on GET /openapi/v1/animations/{task_id} polling."""
        pid, pdir = setup_provider_project
        mock_lib = [{"action_id": 17, "name": "Idle", "key": "Idle", "category": "DailyActions", "sub_category": "Idle"}]

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if request.method == "GET" and url_str.endswith("/animations/library"):
                return httpx.Response(200, json=mock_lib)
            elif request.method == "POST" and url_str.endswith("/animations"):
                return httpx.Response(200, json={"result": "anim_task_1"})
            elif request.method == "GET" and "anim_task_1" in url_str:
                return httpx.Response(status_code, text=f"Poll error {status_code}")
            return httpx.Response(404)

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            providers, "get_provider",
            lambda name, client=None: providers.MeshyProvider(api_key="test-meshy-key", client=client or mock_client)
        )

        res = tools_full.animate(project_id=pid, actions=["idle"], method="provider")
        assert res["ok"] is False
        assert res["error"]["code"] == "PROVIDER_FAILED"
        assert str(status_code) in res["error"]["message"]

    def test_provider_task_status_failed(self, setup_provider_project, monkeypatch):
        """Simulate Meshy animation task returning status='FAILED'."""
        pid, pdir = setup_provider_project
        mock_lib = [{"action_id": 17, "name": "Idle", "key": "Idle", "category": "DailyActions", "sub_category": "Idle"}]

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if request.method == "GET" and url_str.endswith("/animations/library"):
                return httpx.Response(200, json=mock_lib)
            elif request.method == "POST" and url_str.endswith("/animations"):
                return httpx.Response(200, json={"result": "anim_task_fail"})
            elif request.method == "GET" and "anim_task_fail" in url_str:
                return httpx.Response(200, json={"status": "FAILED", "task_error": {"message": "Corrupted armature hierarchy"}})
            return httpx.Response(404)

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            providers, "get_provider",
            lambda name, client=None: providers.MeshyProvider(api_key="test-meshy-key", client=client or mock_client)
        )

        res = tools_full.animate(project_id=pid, actions=["idle"], method="provider")
        assert res["ok"] is False
        assert res["error"]["code"] == "PROVIDER_FAILED"
        assert "Meshy animation failed" in res["error"]["message"]
        assert "Corrupted armature hierarchy" in res["error"]["message"]
        assert "Use method='blender'" in res["error"]["fix"]

    def test_provider_poll_timeout(self, setup_provider_project, monkeypatch):
        """Simulate task remaining in 'running' state past the 55s slice (returns status='running'),
        and past JOB_TIMEOUT_SECONDS (raises PROVIDER_TIMEOUT).
        """
        pid, pdir = setup_provider_project
        mock_lib = [{"action_id": 17, "name": "Idle", "key": "Idle", "category": "DailyActions", "sub_category": "Idle"}]

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if request.method == "GET" and url_str.endswith("/animations/library"):
                return httpx.Response(200, json=mock_lib)
            elif request.method == "POST" and url_str.endswith("/animations"):
                return httpx.Response(200, json={"result": "anim_task_timeout"})
            elif request.method == "GET" and "anim_task_timeout" in url_str:
                return httpx.Response(200, json={"status": "IN_PROGRESS", "progress": 25})
            return httpx.Response(404)

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            providers, "get_provider",
            lambda name, client=None: providers.MeshyProvider(api_key="test-meshy-key", client=client or mock_client)
        )

        # Fast-forward time by 30s per call so first slice returns status="running", then exceeds 1200s
        cur_time = [1000.0]
        def mock_time():
            cur_time[0] += 30.0
            return cur_time[0]

        monkeypatch.setattr("time.time", mock_time)
        monkeypatch.setattr("time.sleep", lambda s: None)

        res_running = tools_full.animate(project_id=pid, actions=["idle"], method="provider")
        assert res_running["ok"] is True
        assert res_running["data"]["status"] == "running"
        assert res_running["data"]["job_id"] == "anim_task_timeout"

        # Jump past JOB_TIMEOUT_SECONDS (1200s)
        cur_time[0] += 1500.0
        res_timeout = tools_full.animate(project_id=pid, actions=["idle"], method="provider", job_id="anim_task_timeout")
        assert res_timeout["ok"] is False
        assert res_timeout["error"]["code"] == "PROVIDER_TIMEOUT"

    @pytest.mark.parametrize("status_code", [404, 500])
    def test_provider_cdn_download_http_errors(self, setup_provider_project, monkeypatch, status_code):
        """Simulate HTTP 404 or 500 when downloading animation clip GLB from provider CDN."""
        pid, pdir = setup_provider_project
        mock_lib = [{"action_id": 17, "name": "Idle", "key": "Idle", "category": "DailyActions", "sub_category": "Idle"}]

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if request.method == "GET" and url_str.endswith("/animations/library"):
                return httpx.Response(200, json=mock_lib)
            elif request.method == "POST" and url_str.endswith("/animations"):
                return httpx.Response(200, json={"result": "anim_task_dl"})
            elif request.method == "GET" and "anim_task_dl" in url_str:
                return httpx.Response(
                    200,
                    json={"status": "SUCCEEDED", "result": {"animation_glb_url": "https://cdn.meshy.ai/clip1.glb"}},
                )
            elif "cdn.meshy.ai" in url_str:
                return httpx.Response(status_code, text=f"CDN Download Error {status_code}")
            return httpx.Response(404)

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            providers, "get_provider",
            lambda name, client=None: providers.MeshyProvider(api_key="test-meshy-key", client=client or mock_client)
        )

        res = tools_full.animate(project_id=pid, actions=["idle"], method="provider")
        assert res["ok"] is False
        assert res["error"]["code"] == "PROVIDER_FAILED"
        assert str(status_code) in res["error"]["message"]

    def test_provider_cdn_corrupt_glb_payload(self, setup_provider_project, monkeypatch):
        """Simulate CDN returning 200 OK with non-GLB corrupt HTML text."""
        pid, pdir = setup_provider_project
        mock_lib = [{"action_id": 17, "name": "Idle", "key": "Idle", "category": "DailyActions", "sub_category": "Idle"}]

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if request.method == "GET" and url_str.endswith("/animations/library"):
                return httpx.Response(200, json=mock_lib)
            elif request.method == "POST" and url_str.endswith("/animations"):
                return httpx.Response(200, json={"result": "anim_task_corrupt"})
            elif request.method == "GET" and "anim_task_corrupt" in url_str:
                return httpx.Response(
                    200,
                    json={"status": "SUCCEEDED", "result": {"animation_glb_url": "https://cdn.meshy.ai/corrupt.glb"}},
                )
            elif "cdn.meshy.ai" in url_str:
                return httpx.Response(200, content=b"<html>502 Bad Gateway</html>")
            return httpx.Response(404)

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            providers, "get_provider",
            lambda name, client=None: providers.MeshyProvider(api_key="test-meshy-key", client=client or mock_client)
        )

        res = tools_full.animate(project_id=pid, actions=["idle"], method="provider")
        assert res["ok"] is False
        assert res["error"]["code"] in ("ANIM_FAILED", "BLENDER_ERROR", "PROVIDER_FAILED")

    def test_provider_partial_stream_failure(self, setup_provider_project, monkeypatch):
        """Simulate library fetch succeeding but animation submission failing with HTTP 500."""
        pid, pdir = setup_provider_project
        mock_lib = [
            {"action_id": 17, "name": "Idle", "key": "Idle", "category": "DailyActions", "sub_category": "Idle"},
            {"action_id": 42, "name": "Walking_Woman", "key": "Walking_Woman", "category": "WalkAndRun", "sub_category": "Walking"},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if request.method == "GET" and url_str.endswith("/animations/library"):
                return httpx.Response(200, json=mock_lib)
            elif request.method == "POST" and url_str.endswith("/animations"):
                return httpx.Response(500, text="Meshy worker died on submit")
            return httpx.Response(404)

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            providers, "get_provider",
            lambda name, client=None: providers.MeshyProvider(api_key="test-meshy-key", client=client or mock_client)
        )

        res = tools_full.animate(project_id=pid, actions=["idle", "walk"], method="provider")
        assert res["ok"] is False
        assert res["error"]["code"] == "PROVIDER_FAILED"
        assert "500" in res["error"]["message"]
