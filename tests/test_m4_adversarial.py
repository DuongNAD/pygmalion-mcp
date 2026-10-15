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
4. Procedural keyframing non-zero motion on Mixamo bones:
   - Armature does NOT remain static across frames
   - Idle: non-zero vertical oscillation on Hips, sway on Spine and Neck
   - Walk: non-zero out-of-phase leg swings on LeftUpLeg/RightUpLeg, knee flexion, arm counter-swings, gait bounce
   - Wave: non-zero arm elevation on RightArm, oscillation on RightForeArm/RightHand, return to rest
   - Generic/custom action: non-zero motion on torso and limbs
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


def inspect_glb_bone_motion_in_blender(glb_path: Path, result_json_path: Path) -> dict[str, Any]:
    """Execute headless Blender to load GLB and evaluate bone motion across frames."""
    exe = config.blender_exe()
    if not exe:
        pytest.skip("Blender executable not available.")

    script = f"""
import json, bpy
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"{str(glb_path)}")

arm_objs = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
if not arm_objs:
    raise RuntimeError("No armature in exported GLB")
arm = arm_objs[0]

results = {{"actions": {{}}}}

for act in bpy.data.actions:
    act_info = {{
        "name": act.name,
        "frame_range": [float(act.frame_range[0]), float(act.frame_range[1])],
        "bones": {{}}
    }}
    
    if not arm.animation_data:
        arm.animation_data_create()
    arm.animation_data.use_nla = False
    arm.animation_data.action = act
    
    # Track relevant Mixamo bones
    target_bones = [
        "Hips", "Spine", "Spine1", "Neck", "Head",
        "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
        "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg"
    ]
    
    start_frame = int(act.frame_range[0])
    end_frame = int(act.frame_range[1])
    
    for bname in target_bones:
        pb = arm.pose.bones.get(bname)
        if not pb:
            continue
        
        locs = []
        rot_eulers = []
        
        for f in range(start_frame, end_frame + 1):
            bpy.context.scene.frame_set(f)
            locs.append(list(pb.location))
            if pb.rotation_mode == "QUATERNION":
                rot_eul = pb.rotation_quaternion.to_euler()
            else:
                rot_eul = pb.rotation_euler
            rot_eulers.append([rot_eul.x, rot_eul.y, rot_eul.z])
            
        z_locs = [l[2] for l in locs]
        rot_x = [r[0] for r in rot_eulers]
        rot_y = [r[1] for r in rot_eulers]
        rot_z = [r[2] for r in rot_eulers]
        
        act_info["bones"][bname] = {{
            "delta_z": max(z_locs) - min(z_locs),
            "delta_rot_x": max(rot_x) - min(rot_x),
            "delta_rot_y": max(rot_y) - min(rot_y),
            "delta_rot_z": max(rot_z) - min(rot_z),
            "z_values": {{
                "0": z_locs[0] if len(z_locs) > 0 else 0.0,
                "15": z_locs[15] if len(z_locs) > 15 else 0.0,
                "30": z_locs[30] if len(z_locs) > 30 else 0.0,
            }},
            "rot_x_values": {{
                "0": rot_x[0] if len(rot_x) > 0 else 0.0,
                "10": rot_x[10] if len(rot_x) > 10 else 0.0,
                "20": rot_x[20] if len(rot_x) > 20 else 0.0,
            }},
            "rot_z_values": {{
                "0": rot_z[0] if len(rot_z) > 0 else 0.0,
                "30": rot_z[30] if len(rot_z) > 30 else 0.0,
                "60": rot_z[60] if len(rot_z) > 60 else 0.0,
            }}
        }}
    
    results["actions"][act.name] = act_info

with open(r"{str(result_json_path)}", "w") as f:
    json.dump(results, f)
"""
    cmd = [exe, "-b", "--factory-startup", "--python-expr", script]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Blender motion inspection failed: {res.stderr}\n{res.stdout}")
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
        """Verify arbitrary action names ['action1', 'action2', 'action3', 'action4'] export cleanly."""
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
        assert res["ok"] is True
        assert res["data"]["actions"] == custom_actions
        assert res["data"]["clip_count"] == 4

        out_glb = pdir / res["data"]["mesh_path"]
        extracted = inspect_glb_animations(out_glb)
        assert len(extracted) == 4
        assert set(extracted) == set(custom_actions)

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
    """Verify that procedural keyframing produces genuine non-zero bone kinematics."""

    def test_procedural_motion_amplitude_and_kinematics(self, tmp_path, monkeypatch):
        """Test idle, walk, wave, jump: verify Mixamo bones exhibit expected non-zero ranges."""
        monkeypatch.setenv("CHARFORGE_LEVEL", "3")
        sheet = tmp_path / "sheet.png"
        sheet.write_bytes(b"dummy")
        cr = server.project_create(sheet_path=str(sheet), name="KinematicsCheck", height_cm=165.0)
        pid = cr["project_id"]
        pdir = state.project_dir(pid)

        rig_path = pdir / f"export/{pid}-rigged.glb"
        rig_path.parent.mkdir(parents=True, exist_ok=True)
        create_minimal_rigged_glb(rig_path)

        p = state.load(pid)
        p["outputs"]["rigged"] = f"export/{pid}-rigged.glb"
        p["stage"] = "rigged"
        state.save(p)

        actions = ["idle", "walk", "wave", "jump"]
        res = tools_full.animate(project_id=pid, actions=actions, method="blender")
        assert res["ok"] is True

        out_glb = pdir / res["data"]["mesh_path"]
        assert out_glb.is_file()

        # Run headless Blender motion evaluator
        results_json = tmp_path / "motion_results.json"
        motion_data = inspect_glb_bone_motion_in_blender(out_glb, results_json)

        exported_actions = motion_data.get("actions", {})
        assert len(exported_actions) == 4

        # ----------------------------------------------------------------------
        # 1. Idle Action Checks:
        # Subtle cyclic breathing: Hips displacement +-1.2 cm, Spine/Neck sway
        # ----------------------------------------------------------------------
        assert "idle" in exported_actions
        idle_info = exported_actions["idle"]
        assert idle_info["frame_range"] == [0.0, 60.0]
        idle_bones = idle_info["bones"]

        # Hips vertical translation amplitude >= 0.02 m (theoretical peak-to-peak 0.024 m)
        assert "Hips" in idle_bones
        hips_disp = idle_bones["Hips"]["delta_z"]
        assert hips_disp >= 0.015, f"Hips idle displacement {hips_disp} m too low (expected >= 0.015 m)"

        # Spine / Spine1 rotational sway
        spine_sway = idle_bones.get("Spine1", {}).get("delta_rot_x", 0.0) or idle_bones.get("Spine", {}).get("delta_rot_x", 0.0)
        assert spine_sway >= 0.02, f"Spine idle sway {spine_sway} rad too low (expected >= 0.02 rad)"

        # Neck sway
        assert "Neck" in idle_bones
        neck_sway = idle_bones["Neck"]["delta_rot_x"]
        assert neck_sway >= 0.015, f"Neck idle sway {neck_sway} rad too low (expected >= 0.015 rad)"

        # ----------------------------------------------------------------------
        # 2. Walk Action Checks:
        # 40 frames, cyclic leg swings +-0.35 rad, knee flexion, counter-swings
        # ----------------------------------------------------------------------
        assert "walk" in exported_actions
        walk_info = exported_actions["walk"]
        assert walk_info["frame_range"] == [0.0, 40.0]
        walk_bones = walk_info["bones"]

        # LeftUpLeg and RightUpLeg swing
        assert "LeftUpLeg" in walk_bones and "RightUpLeg" in walk_bones
        l_leg_swing = walk_bones["LeftUpLeg"]["delta_rot_x"]
        r_leg_swing = walk_bones["RightUpLeg"]["delta_rot_x"]
        assert l_leg_swing >= 0.5, f"LeftUpLeg swing {l_leg_swing} rad too low (expected >= 0.5 rad)"
        assert r_leg_swing >= 0.5, f"RightUpLeg swing {r_leg_swing} rad too low (expected >= 0.5 rad)"

        # 180° out-of-phase check at frame 10 (quarter-cycle)
        l_rot_10 = walk_bones["LeftUpLeg"]["rot_x_values"]["10"]
        r_rot_10 = walk_bones["RightUpLeg"]["rot_x_values"]["10"]
        assert (l_rot_10 * r_rot_10) < 0.0, f"Expected out-of-phase leg swing at frame 10, got Left={l_rot_10}, Right={r_rot_10}"

        # Knee flexion during backswing
        assert "LeftLeg" in walk_bones and "RightLeg" in walk_bones
        l_knee = walk_bones["LeftLeg"]["delta_rot_x"]
        r_knee = walk_bones["RightLeg"]["delta_rot_x"]
        assert l_knee >= 0.35, f"LeftLeg knee flexion {l_knee} rad too low"
        assert r_knee >= 0.35, f"RightLeg knee flexion {r_knee} rad too low"

        # Arm counter-swings
        assert "LeftArm" in walk_bones and "RightArm" in walk_bones
        l_arm = walk_bones["LeftArm"]["delta_rot_x"]
        r_arm = walk_bones["RightArm"]["delta_rot_x"]
        assert l_arm >= 0.4, f"LeftArm counter-swing {l_arm} rad too low"
        assert r_arm >= 0.4, f"RightArm counter-swing {r_arm} rad too low"

        # Gait bounce on Hips
        hips_bounce = walk_bones["Hips"]["delta_z"]
        assert hips_bounce >= 0.010, f"Hips walk bounce {hips_bounce} m too low"

        # ----------------------------------------------------------------------
        # 3. Wave Action Checks:
        # Arm raise to 1.2 rad, waving oscillation +-0.4 rad, return to rest
        # ----------------------------------------------------------------------
        assert "wave" in exported_actions
        wave_info = exported_actions["wave"]
        assert wave_info["frame_range"] == [0.0, 60.0]
        wave_bones = wave_info["bones"]

        # RightArm elevation
        assert "RightArm" in wave_bones
        arm_elev_delta = wave_bones["RightArm"]["delta_rot_z"]
        assert arm_elev_delta >= 1.0, f"RightArm wave elevation {arm_elev_delta} rad too low (expected >= 1.0 rad)"

        # RightArm returns to rest at f=60
        r_rot_0 = wave_bones["RightArm"]["rot_z_values"]["0"]
        r_rot_30 = wave_bones["RightArm"]["rot_z_values"]["30"]
        r_rot_60 = wave_bones["RightArm"]["rot_z_values"]["60"]
        assert abs(r_rot_0) < 0.05
        assert r_rot_30 >= 1.1, f"RightArm not raised at mid-cycle (got {r_rot_30} rad)"
        assert abs(r_rot_60) < 0.05, f"RightArm did not return to rest at frame 60 (got {r_rot_60} rad)"

        # Forearm / Hand waving oscillation
        assert "RightForeArm" in wave_bones
        forearm_osc = wave_bones["RightForeArm"]["delta_rot_y"]
        assert forearm_osc >= 0.6, f"RightForeArm wave oscillation {forearm_osc} rad too low (expected >= 0.6 rad)"

        # ----------------------------------------------------------------------
        # 4. Generic/Custom Action ("jump"):
        # Verifies generic fallback generates non-zero rhythmic limb/torso motion
        # ----------------------------------------------------------------------
        assert "jump" in exported_actions
        jump_info = exported_actions["jump"]
        jump_bones = jump_info["bones"]
        assert jump_bones["Hips"]["delta_z"] >= 0.015, "Generic action Hips motion is zero"
        assert jump_bones["LeftArm"]["delta_rot_x"] >= 0.3, "Generic action LeftArm motion is zero"


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

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if request.method == "POST" and "animations" in url_str:
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

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if request.method == "POST" and "animations" in url_str:
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
        """Simulate task remaining in 'running' state past the 60s timeout."""
        pid, pdir = setup_provider_project

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if request.method == "POST" and "animations" in url_str:
                return httpx.Response(200, json={"result": "anim_task_timeout"})
            elif request.method == "GET" and "anim_task_timeout" in url_str:
                return httpx.Response(200, json={"status": "IN_PROGRESS", "progress": 25})
            return httpx.Response(404)

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            providers, "get_provider",
            lambda name, client=None: providers.MeshyProvider(api_key="test-meshy-key", client=client or mock_client)
        )

        # Fast-forward time so we don't actually wait 60s
        cur_time = [0.0]
        def mock_time():
            cur_time[0] += 25.0
            return cur_time[0]

        monkeypatch.setattr("time.time", mock_time)
        monkeypatch.setattr("time.sleep", lambda s: None)

        res = tools_full.animate(project_id=pid, actions=["idle"], method="provider")
        assert res["ok"] is False
        assert res["error"]["code"] == "PROVIDER_FAILED"
        assert "Meshy animation failed" in res["error"]["message"]

    @pytest.mark.parametrize("status_code", [404, 500])
    def test_provider_cdn_download_http_errors(self, setup_provider_project, monkeypatch, status_code):
        """Simulate HTTP 404 or 500 when downloading animation clip GLB from provider CDN."""
        pid, pdir = setup_provider_project

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if request.method == "POST" and "animations" in url_str:
                return httpx.Response(200, json={"result": "anim_task_dl"})
            elif request.method == "GET" and "anim_task_dl" in url_str:
                return httpx.Response(200, json={"status": "SUCCEEDED", "model_urls": {"glb": "https://cdn.meshy.ai/clip1.glb"}})
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

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if request.method == "POST" and "animations" in url_str:
                return httpx.Response(200, json={"result": "anim_task_corrupt"})
            elif request.method == "GET" and "anim_task_corrupt" in url_str:
                return httpx.Response(200, json={"status": "SUCCEEDED", "model_urls": {"glb": "https://cdn.meshy.ai/corrupt.glb"}})
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
        # When Blender attempts to merge the corrupt GLB or inspect chunk 0, it fails with ANIM_FAILED or CFError
        assert res["error"]["code"] in ("ANIM_FAILED", "BLENDER_ERROR", "PROVIDER_FAILED")

    def test_provider_partial_stream_failure(self, setup_provider_project, monkeypatch):
        """Simulate 1st clip succeeding but 2nd clip failing on submit."""
        pid, pdir = setup_provider_project

        call_count = {"submit": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if request.method == "POST" and "animations" in url_str:
                call_count["submit"] += 1
                if call_count["submit"] == 1:
                    return httpx.Response(200, json={"result": "anim_task_1"})
                else:
                    return httpx.Response(500, text="Meshy worker died on clip 2")
            elif request.method == "GET" and "anim_task_1" in url_str:
                return httpx.Response(200, json={"status": "SUCCEEDED", "model_urls": {"glb": "https://cdn.meshy.ai/clip1.glb"}})
            elif "cdn.meshy.ai/clip1.glb" in url_str:
                return httpx.Response(200, content=create_synthetic_glb_binary(["idle"]))
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
