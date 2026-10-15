"""Empirical Verification and Adversarial Stress Test Suite for Milestone 5 Deliverables.

Validates 3D and animation deliverables generated for project 'aria':
1. projects/aria/export/aria-animated.glb:
   - Binary glTF 2.0 chunk 0 structure: magic b"glTF", version 2, valid JSON chunk.
   - Animation actions: exactly 3 actions ('idle', 'walk', 'wave').
   - Geometry complexity: meshes and accessors with vertex and triangle count > 10,000.
2. projects/aria/export/aria-rigged.glb:
   - Headless Blender inspection.
   - Armature contains exactly the 22 Mixamo standard humanoid bones.
   - All bone lengths > 0.
   - Exact anatomical parent-child hierarchy linkages.
   - Skinned mesh verification: vertex groups correspond to bone names.
3. assets/aria_preview.gif:
   - Frame count >= 24.
   - Infinite loop flag: info['loop'] == 0.
   - Frame-by-frame PIL difference analysis: std dev > 0 for all adjacent frame pairs.
   - Non-static character motion and valid visual dimensions.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageSequence
import pytest

from charforge import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANIMATED_GLB_PATH = PROJECT_ROOT / "projects" / "aria" / "export" / "aria-animated.glb"
RIGGED_GLB_PATH = PROJECT_ROOT / "projects" / "aria" / "export" / "aria-rigged.glb"
PREVIEW_GIF_PATH = PROJECT_ROOT / "assets" / "aria_preview.gif"
TURNAROUND_PNG_PATH = PROJECT_ROOT / "assets" / "aria_turnaround.png"
RENDER_PNG_PATH = PROJECT_ROOT / "assets" / "aria_render.png"

EXPECTED_MIXAMO_BONES = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
]

EXPECTED_BONE_HIERARCHY = {
    "Hips": None,
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Head": "Neck",
    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToeBase": "LeftFoot",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToeBase": "RightFoot",
}


def parse_glb_chunk0(glb_path: Path) -> tuple[dict[str, Any], int, int]:
    """Parse binary glTF 2.0 file header and extract chunk 0 JSON."""
    assert glb_path.is_file(), f"File does not exist: {glb_path}"
    raw = glb_path.read_bytes()
    assert len(raw) >= 20, f"GLB file too small: {len(raw)} bytes"

    magic, version, total_length = struct.unpack_from("<4sII", raw, 0)
    assert magic == b"glTF", f"Invalid GLB magic: {magic!r}, expected b'glTF'"
    assert version == 2, f"Unsupported glTF version: {version}, expected 2"
    assert total_length == len(raw), f"Header length mismatch: declared {total_length}, actual {len(raw)}"

    chunk_length, chunk_type = struct.unpack_from("<II", raw, 12)
    assert chunk_type == 0x4E4F534A, f"Chunk 0 is not JSON (0x4E4F534A): {chunk_type:#x}"
    assert 20 + chunk_length <= total_length, f"Chunk 0 exceeds file bounds: 20 + {chunk_length} > {total_length}"

    json_bytes = raw[20 : 20 + chunk_length]
    gltf_json = json.loads(json_bytes.decode("utf-8"))
    return gltf_json, total_length, chunk_length


# ==============================================================================
# 1. Binary glTF Chunk 0 Verification for aria-animated.glb
# ==============================================================================


def test_animated_glb_binary_header_and_chunk0():
    """Verify aria-animated.glb magic, version, header length, and JSON chunk parsing."""
    gltf_json, total_len, chunk_len = parse_glb_chunk0(ANIMATED_GLB_PATH)
    assert total_len > 1_000_000, f"Expected non-trivial GLB file size, got {total_len} bytes"
    assert "asset" in gltf_json
    assert gltf_json["asset"].get("version") == "2.0"


def test_animated_glb_actions():
    """Assert exactly 3 actions ('idle', 'walk', 'wave') exist in aria-animated.glb."""
    gltf_json, _, _ = parse_glb_chunk0(ANIMATED_GLB_PATH)
    animations = gltf_json.get("animations", [])
    action_names = [anim.get("name") for anim in animations]

    assert len(animations) == 3, f"Expected exactly 3 animation clips, got {len(animations)}: {action_names}"
    assert set(action_names) == {"idle", "walk", "wave"}, f"Unexpected action names: {action_names}"

    # Verify channels and samplers for each animation
    for anim in animations:
        name = anim.get("name")
        channels = anim.get("channels", [])
        samplers = anim.get("samplers", [])
        assert len(channels) > 0, f"Action {name} has no animation channels"
        assert len(samplers) > 0, f"Action {name} has no animation samplers"


def test_animated_glb_geometry_complexity():
    """Assert vertex count and triangle count > 10,000 in aria-animated.glb."""
    gltf_json, _, _ = parse_glb_chunk0(ANIMATED_GLB_PATH)
    meshes = gltf_json.get("meshes", [])
    accessors = gltf_json.get("accessors", [])

    assert len(meshes) >= 1, "No meshes found in aria-animated.glb"
    assert len(accessors) > 0, "No accessors found in aria-animated.glb"

    total_vertices = 0
    total_triangles = 0

    for mesh in meshes:
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            assert "POSITION" in attrs, "Primitive missing POSITION attribute"
            pos_acc_idx = attrs["POSITION"]
            pos_acc = accessors[pos_acc_idx]
            total_vertices += pos_acc.get("count", 0)

            if "indices" in prim:
                idx_acc_idx = prim["indices"]
                idx_acc = accessors[idx_acc_idx]
                total_triangles += idx_acc.get("count", 0) // 3
            else:
                total_triangles += pos_acc.get("count", 0) // 3

    assert total_vertices > 10_000, f"Vertex count {total_vertices} not > 10,000"
    assert total_triangles > 10_000, f"Triangle count {total_triangles} not > 10,000"


# ==============================================================================
# 2. Headless Blender Inspection for aria-rigged.glb
# ==============================================================================


def test_rigged_glb_blender_armature_and_hierarchy():
    """Inspect aria-rigged.glb in headless Blender: 22 Mixamo bones, length > 0, hierarchy."""
    blender_exe = config.blender_exe()
    assert blender_exe and Path(blender_exe).is_file(), f"Blender executable not found: {blender_exe}"

    blender_code = """
import bpy
import json
import sys
from pathlib import Path

# Clean scene
bpy.ops.wm.read_factory_settings(use_empty=True)

glb_path = sys.argv[-2]
out_json = sys.argv[-1]

try:
    bpy.ops.import_scene.gltf(filepath=glb_path)
except Exception as e:
    Path(out_json).write_text(json.dumps({"ok": False, "error": str(e)}))
    sys.exit(0)

armatures = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
if not armatures:
    Path(out_json).write_text(json.dumps({"ok": False, "error": "No armature found in GLB"}))
    sys.exit(0)

arm = armatures[0]
bones_data = {}
for b in arm.data.bones:
    parent_name = b.parent.name if b.parent else None
    bones_data[b.name] = {
        "length": float(b.length),
        "parent": parent_name,
        "head": [float(c) for c in b.head],
        "tail": [float(c) for c in b.tail],
    }

meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
mesh_info = []
for m in meshes:
    mesh_info.append({
        "name": m.name,
        "vertices": len(m.data.vertices),
        "polygons": len(m.data.polygons),
        "vertex_groups": [vg.name for vg in m.vertex_groups],
    })

res = {
    "ok": True,
    "armature_name": arm.name,
    "bone_count": len(bones_data),
    "bones": bones_data,
    "meshes": mesh_info,
}
Path(out_json).write_text(json.dumps(res))
"""

    with tempfile.TemporaryDirectory(prefix="charforge-test-") as tmpdir:
        script_file = Path(tmpdir) / "inspect_rig.py"
        out_json = Path(tmpdir) / "result.json"
        script_file.write_text(blender_code, encoding="utf-8")

        cmd = [
            blender_exe,
            "-b",
            "--factory-startup",
            "-P",
            str(script_file),
            "--",
            str(RIGGED_GLB_PATH),
            str(out_json),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert out_json.is_file(), f"Blender inspection script did not write result.json. Stdout:\n{proc.stdout}\nStderr:\n{proc.stderr}"

        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data.get("ok") is True, f"Blender inspection failed: {data}"

        bones = data["bones"]
        bone_names = list(bones.keys())

        # Check bone count and exact bone names
        assert len(bone_names) == 22, f"Expected 22 bones, found {len(bone_names)}: {bone_names}"
        assert set(bone_names) == set(EXPECTED_MIXAMO_BONES), f"Bone names mismatch: {set(bone_names) ^ set(EXPECTED_MIXAMO_BONES)}"

        # Check lengths > 0
        for name, binfo in bones.items():
            length = binfo["length"]
            assert length > 0.005, f"Bone '{name}' has invalid or zero length: {length}"

        # Check parent-child linkages
        for name, expected_parent in EXPECTED_BONE_HIERARCHY.items():
            actual_parent = bones[name]["parent"]
            assert actual_parent == expected_parent, f"Bone '{name}' has parent '{actual_parent}', expected '{expected_parent}'"

        # Check meshes & vertex groups binding
        meshes = data.get("meshes", [])
        assert len(meshes) >= 1, "No mesh found in rigged GLB"
        primary_mesh = meshes[0]
        assert primary_mesh["vertices"] > 10_000, f"Rigged mesh vertex count {primary_mesh['vertices']} not > 10,000"
        vg_names = set(primary_mesh.get("vertex_groups", []))
        # Ensure vertex groups correspond to bones
        shared_groups = vg_names.intersection(set(EXPECTED_MIXAMO_BONES))
        assert len(shared_groups) >= 20, f"Mesh vertex groups missing Mixamo bones: {set(EXPECTED_MIXAMO_BONES) - shared_groups}"


# ==============================================================================
# 3. Frame-by-Frame PIL Analysis of assets/aria_preview.gif
# ==============================================================================


def test_preview_gif_properties_and_motion():
    """Inspect assets/aria_preview.gif: frame count >= 24, loop == 0, std dev > 0 for all adjacent frames."""
    assert PREVIEW_GIF_PATH.is_file(), f"GIF file does not exist: {PREVIEW_GIF_PATH}"

    im = Image.open(PREVIEW_GIF_PATH)
    assert im.format == "GIF", f"Unexpected format: {im.format}"

    loop_count = im.info.get("loop")
    assert loop_count == 0, f"Expected infinite loop (loop=0), got {loop_count}"

    frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(im)]
    num_frames = len(frames)
    assert num_frames >= 24, f"Expected >= 24 frames, got {num_frames}"

    width, height = frames[0].size
    assert width >= 256 and height >= 256, f"Frame resolution {width}x{height} too small"

    diff_stds = []
    diff_means = []

    for i in range(num_frames - 1):
        arr_curr = np.array(frames[i], dtype=np.float32)
        arr_next = np.array(frames[i + 1], dtype=np.float32)
        diff = np.abs(arr_next - arr_curr)

        std_val = float(np.std(diff))
        mean_val = float(np.mean(diff))

        diff_stds.append(std_val)
        diff_means.append(mean_val)

        assert std_val > 0.0, f"Frame {i} to {i+1} has zero standard deviation (identical static frames)"
        assert mean_val > 0.0, f"Frame {i} to {i+1} has zero mean pixel difference"

    # Also test wrap-around difference (last frame to first frame) for loop continuity
    arr_last = np.array(frames[-1], dtype=np.float32)
    arr_first = np.array(frames[0], dtype=np.float32)
    wrap_diff = np.abs(arr_first - arr_last)
    assert float(np.std(wrap_diff)) > 0.0, "Last frame identical to first frame (stagnant wrap)"

    # Validate motion energy across sequence
    min_std = min(diff_stds)
    max_std = max(diff_stds)
    avg_std = sum(diff_stds) / len(diff_stds)

    assert min_std > 5.0, f"Minimum inter-frame std dev {min_std:.2f} is too low for active turntable motion"
    assert avg_std > 15.0, f"Average inter-frame std dev {avg_std:.2f} indicates insufficient motion"


# ==============================================================================
# 4. Supplementary Visual and Project Deliverable Integrity Checks
# ==============================================================================


def test_supplementary_visual_deliverables():
    """Verify accompanying deliverables aria_turnaround.png, aria_render.png, and aria.glb."""
    assert TURNAROUND_PNG_PATH.is_file(), f"Missing turnaround image: {TURNAROUND_PNG_PATH}"
    t_im = Image.open(TURNAROUND_PNG_PATH)
    assert t_im.size[0] >= 1024 and t_im.size[1] >= 512, f"Turnaround resolution {t_im.size} too small"

    assert RENDER_PNG_PATH.is_file(), f"Missing multi-view render image: {RENDER_PNG_PATH}"
    r_im = Image.open(RENDER_PNG_PATH)
    assert r_im.size[0] >= 1024 and r_im.size[1] >= 256, f"Render sheet resolution {r_im.size} too small"

    base_glb = PROJECT_ROOT / "projects" / "aria" / "export" / "aria.glb"
    assert base_glb.is_file(), f"Missing base export aria.glb: {base_glb}"
    base_json, _, _ = parse_glb_chunk0(base_glb)
    assert len(base_json.get("meshes", [])) >= 1, "Base GLB contains no meshes"


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
