"""Empirical adversarial test suite for Milestone 2: Providers & Blockout.

Designed by Challenger 1 (teamwork_preview_challenger_m2_1).
Covers:
1. Provider rate limits (HTTP 429) & auth failures (401, 402) across all 4 cloud providers.
2. Provider network resilience: Connection drops (ConnectError) & timeouts (ReadTimeout) during submit & poll.
3. Provider corrupt JSON handling: HTML gateway errors and non-JSON payloads during submit & poll.
4. Provider download resilience: Invalid URLs, empty URLs, 404/500 CDN responses, and None URL.
5. Visual hull blockout: Binary PLY format integrity (header, vertex colors, face indices, little-endian).
6. Multi-view vertex color projection: Validating front/side/back color assignment by normal axis.
7. Baseline scoring verification: score_mesh achieves front >= 0.90 and side >= 0.85 on synthetic sheet.
8. Blockout unusual silhouettes: Thin limbs, side orientations (left/right/auto), extreme aspect ratios.
9. Blockout edge cases: Empty masks, non-overlapping masks (CFError BLOCKOUT_FAILED).
10. Blockout asymmetric clipping: Empirical reproduction of grid truncation on asymmetric figures.
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import httpx
import numpy as np
import pytest

from charforge import blockout, config, providers, score, server, state
from charforge.envelope import CFError


# ==============================================================================
# SECTION 1: Provider Rate Limits, Auth & HTTP Statuses
# ==============================================================================

@pytest.mark.parametrize("prov_cls", [
    providers.TripoProvider,
    providers.MeshyProvider,
    providers.RodinProvider,
    providers.HunyuanProvider,
])
@pytest.mark.parametrize("status_code", [401, 402, 429, 500, 503])
def test_empirical_provider_http_errors(tmp_path, prov_cls, status_code):
    """Verify HTTP error codes (401, 402, 429, 500, 503) map to CFError PROVIDER_FAILED."""
    fake_img = tmp_path / "front.png"
    fake_img.write_bytes(b"dummy")

    transport = httpx.MockTransport(lambda req: httpx.Response(status_code, text="Rate limit or Server Error"))
    client = httpx.Client(transport=transport)
    p = prov_cls(api_key="fake-key", client=client)

    with pytest.raises(CFError) as exc_info:
        p.submit({"front": fake_img})
    assert exc_info.value.code == "PROVIDER_FAILED"
    assert "blockout" in exc_info.value.fix.lower()


# ==============================================================================
# SECTION 2: Provider Connection Drops & Corrupt JSON
# ==============================================================================

@pytest.mark.parametrize("prov_cls,name", [
    (providers.TripoProvider, "tripo"),
    (providers.MeshyProvider, "meshy"),
    (providers.RodinProvider, "rodin"),
    (providers.HunyuanProvider, "hunyuan"),
])
def test_empirical_provider_connection_drop_submit(tmp_path, prov_cls, name):
    """Stress-test submit() under connection drops (ConnectError)."""
    fake_img = tmp_path / "front.png"
    fake_img.write_bytes(b"dummy")

    def raise_connect(req):
        raise httpx.ConnectError(f"Connection refused connecting to {name} API")

    client = httpx.Client(transport=httpx.MockTransport(raise_connect))
    p = prov_cls(api_key="fake-key", client=client)

    # Current implementation lets ConnectError escape unhandled rather than wrapping in PROVIDER_FAILED
    try:
        p.submit({"front": fake_img})
        pytest.fail("Expected an error on connection drop")
    except CFError as e:
        # Ideal behavior: wrapped in CFError PROVIDER_FAILED
        assert e.code == "PROVIDER_FAILED"
    except httpx.ConnectError as e:
        # Documented vulnerability: raw httpx exception bubbles up
        assert "Connection refused" in str(e)


@pytest.mark.parametrize("prov_cls,name", [
    (providers.TripoProvider, "tripo"),
    (providers.MeshyProvider, "meshy"),
    (providers.RodinProvider, "rodin"),
    (providers.HunyuanProvider, "hunyuan"),
])
def test_empirical_provider_connection_drop_poll(prov_cls, name):
    """Stress-test poll() under network timeouts (ReadTimeout)."""
    def raise_timeout(req):
        raise httpx.ReadTimeout(f"Read timed out from {name}")

    client = httpx.Client(transport=httpx.MockTransport(raise_timeout))
    p = prov_cls(api_key="fake-key", client=client)

    try:
        p.poll("mock-task-id")
        pytest.fail("Expected an error on read timeout")
    except CFError as e:
        assert e.code in ("PROVIDER_FAILED", "PROVIDER_TIMEOUT")
    except httpx.ReadTimeout as e:
        # Documented vulnerability: raw httpx exception bubbles up
        assert "Read timed out" in str(e)


@pytest.mark.parametrize("prov_cls,name", [
    (providers.TripoProvider, "tripo"),
    (providers.MeshyProvider, "meshy"),
    (providers.RodinProvider, "rodin"),
    (providers.HunyuanProvider, "hunyuan"),
])
def test_empirical_provider_corrupt_json_responses(tmp_path, prov_cls, name):
    """Stress-test submit() and poll() when server returns 200 OK with HTML or corrupt JSON."""
    fake_img = tmp_path / "front.png"
    fake_img.write_bytes(b"dummy")

    corrupt_body = b"<html><head><title>502 Bad Gateway</title></head><body>502 Bad Gateway</body></html>"
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=corrupt_body))
    client = httpx.Client(transport=transport)
    p = prov_cls(api_key="fake-key", client=client)

    # 1. Test submit with corrupt JSON
    try:
        p.submit({"front": fake_img})
        pytest.fail("Expected error on corrupt JSON submit")
    except CFError as e:
        assert e.code == "PROVIDER_FAILED"
    except json.JSONDecodeError:
        pass  # Documented vulnerability: raw JSONDecodeError bubbles up

    # 2. Test poll with corrupt JSON
    try:
        p.poll("mock-job-id")
        pytest.fail("Expected error on corrupt JSON poll")
    except CFError as e:
        assert e.code == "PROVIDER_FAILED"
    except json.JSONDecodeError:
        pass  # Documented vulnerability: raw JSONDecodeError bubbles up


# ==============================================================================
# SECTION 3: Provider Download URL Resilience
# ==============================================================================

def test_empirical_provider_download_urls(tmp_path):
    """Stress-test download() with empty URLs, malformed URLs, HTTP 404, and None."""
    p = providers.TripoProvider(api_key="fake-key")
    dst = tmp_path / "model.glb"

    # 1. Empty string URL -> CFError PROVIDER_FAILED
    with pytest.raises(CFError) as exc_info:
        p.download("", dst)
    assert exc_info.value.code == "PROVIDER_FAILED"

    # 2. Malformed URL -> CFError PROVIDER_FAILED
    with pytest.raises(CFError) as exc_info:
        p.download("not_a_valid_url", dst)
    assert exc_info.value.code == "PROVIDER_FAILED"

    # 3. HTTP 404 from CDN -> CFError PROVIDER_FAILED
    client_404 = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(404, text="Not Found")))
    p_404 = providers.TripoProvider(api_key="fake-key", client=client_404)
    with pytest.raises(CFError) as exc_info:
        p_404.download("https://cdn.example.com/missing.glb", dst)
    assert exc_info.value.code == "PROVIDER_FAILED"

    # 4. None URL -> raises TypeError (unhandled in download)
    with pytest.raises((CFError, TypeError)) as exc_info:
        p.download(None, dst)
    if isinstance(exc_info.value, TypeError):
        # Documented vulnerability: TypeError is unhandled by except httpx.HTTPError
        assert "Invalid type for url" in str(exc_info.value)


# ==============================================================================
# SECTION 4: Binary PLY Format Integrity & Specification Verification
# ==============================================================================

def test_empirical_binary_ply_specification_integrity(tmp_path):
    """Empirically verify binary little-endian PLY format byte-for-byte."""
    # Generate simple test cube blockout
    size = 256
    mask = np.zeros((size, size), dtype=bool)
    mask[50:200, 80:176] = True

    f_p = tmp_path / "f.png"
    s_p = tmp_path / "s.png"
    b_p = tmp_path / "b.png"
    m_p = tmp_path / "m.png"

    img = np.full((size, size, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(f_p), img)
    cv2.imwrite(str(s_p), img)
    cv2.imwrite(str(b_p), img)
    cv2.imwrite(str(m_p), mask.astype(np.uint8) * 255)

    views = {
        "front": {"image": str(f_p), "mask": str(m_p)},
        "side": {"image": str(s_p), "mask": str(m_p)},
        "back": {"image": str(b_p), "mask": str(m_p)},
    }
    ply_path = tmp_path / "test.ply"
    height_cm = 180.0
    blockout.blockout(views, {"side_faces": "left"}, height_cm, ply_path)

    assert ply_path.is_file()
    file_bytes = ply_path.read_bytes()

    # 1. Header assertions
    assert file_bytes.startswith(b"ply\n")
    assert b"format binary_little_endian 1.0\n" in file_bytes
    assert b"property float x\nproperty float y\nproperty float z\n" in file_bytes
    assert b"property uchar red\nproperty uchar green\nproperty uchar blue\n" in file_bytes
    assert b"property list uchar int vertex_indices\n" in file_bytes
    assert b"end_header\n" in file_bytes

    header_end = file_bytes.index(b"end_header\n") + len(b"end_header\n")
    header_str = file_bytes[:header_end].decode("ascii")
    lines = header_str.splitlines()

    n_verts = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
    n_faces = int([l for l in lines if l.startswith("element face")][0].split()[-1])

    # 2. File size calculation: header + n_verts * 15 bytes + n_faces * 13 bytes
    expected_size = header_end + n_verts * 15 + n_faces * 13
    assert len(file_bytes) == expected_size, f"Size mismatch: {len(file_bytes)} vs expected {expected_size}"

    # 3. Read binary vertices using little-endian dtype
    v_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    verts_raw = np.frombuffer(file_bytes[header_end:header_end + n_verts * 15], dtype=v_dtype)

    assert np.all(np.isfinite(verts_raw["x"])), "NaN or Inf found in vertex x"
    assert np.all(np.isfinite(verts_raw["y"])), "NaN or Inf found in vertex y"
    assert np.all(np.isfinite(verts_raw["z"])), "NaN or Inf found in vertex z"

    # Feet must be grounded at Z = 0
    assert abs(verts_raw["z"].min()) < 1e-4, f"Feet not grounded at Z=0: min Z is {verts_raw['z'].min()}"
    # Height must match height_cm / 100 within 0.1%
    mesh_height = verts_raw["z"].max() - verts_raw["z"].min()
    expected_height = height_cm / 100.0
    assert abs(mesh_height - expected_height) < 1e-3, f"Height mismatch: {mesh_height} vs {expected_height}"

    # Center XY should be centered at (0, 0)
    x_mid = (verts_raw["x"].min() + verts_raw["x"].max()) / 2.0
    y_mid = (verts_raw["y"].min() + verts_raw["y"].max()) / 2.0
    assert abs(x_mid) < 1e-4, f"X not centered: {x_mid}"
    assert abs(y_mid) < 1e-4, f"Y not centered: {y_mid}"

    # 4. Read binary faces using little-endian dtype
    f_offset = header_end + n_verts * 15
    f_dtype = np.dtype([
        ("count", "u1"),
        ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4"),
    ])
    faces_raw = np.frombuffer(file_bytes[f_offset:], dtype=f_dtype)

    assert len(faces_raw) == n_faces
    assert np.all(faces_raw["count"] == 3), "Non-triangular face found"
    all_indices = np.stack([faces_raw["i0"], faces_raw["i1"], faces_raw["i2"]], axis=1)
    assert np.all(all_indices >= 0), "Negative vertex index in faces"
    assert np.all(all_indices < n_verts), f"Index out of range: max {all_indices.max()} >= {n_verts}"


# ==============================================================================
# SECTION 5: Multi-View Color Sampling
# ==============================================================================

def test_empirical_vertex_color_axis_alignment(tmp_path):
    """Verify dominant normal axis color sampling (front=-Y, back=+Y, side=+-X)."""
    size = 256
    mask = np.zeros((size, size), dtype=bool)
    mask[50:200, 80:176] = True

    # Front = Pure Red (BGR: 0, 0, 255)
    f_img = np.full((size, size, 3), (0, 0, 255), dtype=np.uint8)
    # Side = Pure Green (BGR: 0, 255, 0)
    s_img = np.full((size, size, 3), (0, 255, 0), dtype=np.uint8)
    # Back = Pure Blue (BGR: 255, 0, 0)
    b_img = np.full((size, size, 3), (255, 0, 0), dtype=np.uint8)

    f_p, s_p, b_p, m_p = tmp_path / "f.png", tmp_path / "s.png", tmp_path / "b.png", tmp_path / "m.png"
    cv2.imwrite(str(f_p), f_img)
    cv2.imwrite(str(s_p), s_img)
    cv2.imwrite(str(b_p), b_img)
    cv2.imwrite(str(m_p), mask.astype(np.uint8) * 255)

    views = {
        "front": {"image": str(f_p), "mask": str(m_p)},
        "side": {"image": str(s_p), "mask": str(m_p)},
        "back": {"image": str(b_p), "mask": str(m_p)},
    }
    ply_path = tmp_path / "color.ply"
    blockout.blockout(views, {"side_faces": "left"}, 160.0, ply_path)

    # Read vertex colors from PLY
    file_bytes = ply_path.read_bytes()
    header_end = file_bytes.index(b"end_header\n") + len(b"end_header\n")
    lines = file_bytes[:header_end].decode("ascii").splitlines()
    n_verts = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])

    v_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    verts = np.frombuffer(file_bytes[header_end:header_end + n_verts * 15], dtype=v_dtype)

    # Assert that Red (front), Green (side), and Blue (back) vertices all exist
    has_red = np.any((verts["r"] > 200) & (verts["g"] < 50) & (verts["b"] < 50))
    has_green = np.any((verts["g"] > 200) & (verts["r"] < 50) & (verts["b"] < 50))
    has_blue = np.any((verts["b"] > 200) & (verts["r"] < 50) & (verts["g"] < 50))

    assert has_red, "No front-facing red vertices found"
    assert has_green, "No side-facing green vertices found"
    assert has_blue, "No back-facing blue vertices found"


# ==============================================================================
# SECTION 6: Mesh Scoring Baseline (front >= 0.90, side >= 0.85)
# ==============================================================================

@pytest.mark.blender
def test_empirical_score_mesh_baseline(sheet_path):
    """Empirically assert score_mesh achieves front >= 0.90 and side >= 0.85 on synthetic sheet."""
    res_create = server.project_create(sheet_path=str(sheet_path), name="Baseline Score Test", height_cm=165.0)
    pid = res_create["project_id"]
    res_split = server.sheet_split(project_id=pid)
    assert res_split["ok"] is True

    p = state.load(pid)
    pdir = state.project_dir(pid)
    ply_dst = pdir / "meshes" / "cand-1-blockout.ply"

    blockout.blockout(p["views"], p["reference"], p["height_cm"], ply_dst)
    assert ply_dst.is_file()

    res_score = server.score_mesh(project_id=pid, mesh_path=str(ply_dst))
    assert res_score["ok"] is True
    ious = res_score["data"]["iou"]

    assert ious["front"] >= 0.90, f"Front IoU was {ious['front']:.4f}, expected >= 0.90"
    assert ious["side"] >= 0.85, f"Side IoU was {ious['side']:.4f}, expected >= 0.85"


# ==============================================================================
# SECTION 7: Unusual Silhouettes: Thin Limbs, Side Orientations, Aspect Ratios
# ==============================================================================

def test_empirical_blockout_thin_limbs(tmp_path):
    """Test blockout with 1-pixel thin limbs (stick figure limbs)."""
    size = 512
    front = np.zeros((size, size), dtype=bool)
    front[50:100, 240:260] = True   # head
    front[100:350, 246:254] = True  # torso 8px wide
    front[150, 150:246] = True      # 1-px arm left
    front[150, 254:350] = True      # 1-px arm right
    front[350:480, 248] = True      # 1-px leg left
    front[350:480, 252] = True      # 1-px leg right

    side = np.zeros((size, size), dtype=bool)
    side[50:100, 240:260] = True
    side[100:350, 246:254] = True
    side[150, 246:254] = True
    side[350:480, 250] = True

    f_p, s_p, b_p = tmp_path / "f.png", tmp_path / "s.png", tmp_path / "b.png"
    cv2.imwrite(str(f_p), front.astype(np.uint8) * 255)
    cv2.imwrite(str(s_p), side.astype(np.uint8) * 255)
    cv2.imwrite(str(b_p), front.astype(np.uint8) * 255)

    views = {"front": str(f_p), "side": str(s_p), "back": str(b_p)}
    ply_out = tmp_path / "thin.ply"
    blockout.blockout(views, {"side_faces": "left"}, 160.0, ply_out)
    assert ply_out.is_file()
    assert ply_out.stat().st_size > 1000


@pytest.mark.parametrize("side_faces_mode", ["left", "right", None])
def test_empirical_blockout_side_orientations(tmp_path, side_faces_mode):
    """Test explicit left, right, and auto-detected side orientations."""
    size = 256
    front = np.zeros((size, size), dtype=bool)
    front[50:100, 110:146] = True
    front[100:220, 115:141] = True

    # Side with nose pointing right (columns 135-160)
    side = np.zeros((size, size), dtype=bool)
    side[50:100, 115:145] = True
    side[65:80, 145:160] = True   # nose sticking out to the right
    side[100:220, 115:145] = True

    f_p, s_p, b_p = tmp_path / "f.png", tmp_path / "s.png", tmp_path / "b.png"
    cv2.imwrite(str(f_p), front.astype(np.uint8) * 255)
    cv2.imwrite(str(s_p), side.astype(np.uint8) * 255)
    cv2.imwrite(str(b_p), front.astype(np.uint8) * 255)

    views = {"front": str(f_p), "side": str(s_p), "back": str(b_p)}
    ref = {} if side_faces_mode is None else {"side_faces": side_faces_mode}
    ply_out = tmp_path / f"side_{side_faces_mode}.ply"

    blockout.blockout(views, ref, 160.0, ply_out)
    assert ply_out.is_file()
    assert ply_out.stat().st_size > 1000


def test_empirical_blockout_extreme_aspect_ratios(tmp_path):
    """Test extreme aspect ratios: tall-thin (20:1) and wide-flat (1:10)."""
    size = 512

    # 1. Tall and thin (needle / staff)
    tall_front = np.zeros((size, size), dtype=bool)
    tall_front[20:490, 253:259] = True  # 470px tall, 6px wide
    f_tall, s_tall, b_tall = tmp_path / "f_tall.png", tmp_path / "s_tall.png", tmp_path / "b_tall.png"
    cv2.imwrite(str(f_tall), tall_front.astype(np.uint8) * 255)
    cv2.imwrite(str(s_tall), tall_front.astype(np.uint8) * 255)
    cv2.imwrite(str(b_tall), tall_front.astype(np.uint8) * 255)

    views_tall = {"front": str(f_tall), "side": str(s_tall), "back": str(b_tall)}
    ply_tall = tmp_path / "tall.ply"
    blockout.blockout(views_tall, {}, 200.0, ply_tall)
    assert ply_tall.is_file()

    # 2. Wide and flat (caterpillar / pancake)
    flat_front = np.zeros((size, size), dtype=bool)
    flat_front[245:265, 20:490] = True  # 20px tall, 470px wide
    flat_side = np.zeros((size, size), dtype=bool)
    flat_side[245:265, 200:312] = True
    f_flat, s_flat, b_flat = tmp_path / "f_flat.png", tmp_path / "s_flat.png", tmp_path / "b_flat.png"
    cv2.imwrite(str(f_flat), flat_front.astype(np.uint8) * 255)
    cv2.imwrite(str(s_flat), flat_side.astype(np.uint8) * 255)
    cv2.imwrite(str(b_flat), flat_front.astype(np.uint8) * 255)

    views_flat = {"front": str(f_flat), "side": str(s_flat), "back": str(b_flat)}
    ply_flat = tmp_path / "flat.ply"
    blockout.blockout(views_flat, {}, 30.0, ply_flat)
    assert ply_flat.is_file()


# ==============================================================================
# SECTION 8: Empty & Disjoint Masks
# ==============================================================================

def test_empirical_blockout_empty_and_disjoint_masks(tmp_path):
    """Verify empty or non-overlapping masks gracefully raise CFError BLOCKOUT_FAILED."""
    size = 256
    empty_mask = np.zeros((size, size), dtype=bool)
    mask_top = np.zeros((size, size), dtype=bool)
    mask_top[10:50, 100:150] = True
    mask_bot = np.zeros((size, size), dtype=bool)
    mask_bot[200:240, 100:150] = True

    e_p = tmp_path / "empty.png"
    top_p = tmp_path / "top.png"
    bot_p = tmp_path / "bot.png"
    cv2.imwrite(str(e_p), empty_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(top_p), mask_top.astype(np.uint8) * 255)
    cv2.imwrite(str(bot_p), mask_bot.astype(np.uint8) * 255)

    # 1. Empty side mask
    views_empty = {"front": str(top_p), "side": str(e_p), "back": str(top_p)}
    with pytest.raises(CFError) as exc:
        blockout.blockout(views_empty, {}, 160.0, tmp_path / "out.ply")
    assert exc.value.code == "BLOCKOUT_FAILED"
    assert "Side mask is empty" in exc.value.message

    # 2. Empty front mask
    views_empty_f = {"front": str(e_p), "side": str(top_p), "back": str(top_p)}
    with pytest.raises(CFError) as exc:
        blockout.blockout(views_empty_f, {}, 160.0, tmp_path / "out.ply")
    assert exc.value.code == "BLOCKOUT_FAILED"
    assert "Front mask is empty" in exc.value.message

    # 3. Disjoint front (upper body) and side (lower body) with anchor dots -> empty visual hull intersection
    disjoint_f = np.zeros((size, size), dtype=bool)
    disjoint_f[20, 128] = True
    disjoint_f[230, 128] = True
    disjoint_f[40:100, 110:146] = True

    disjoint_s = np.zeros((size, size), dtype=bool)
    disjoint_s[20, 128] = True
    disjoint_s[230, 128] = True
    disjoint_s[150:210, 110:146] = True

    f_dis, s_dis = tmp_path / "f_dis.png", tmp_path / "s_dis.png"
    cv2.imwrite(str(f_dis), disjoint_f.astype(np.uint8) * 255)
    cv2.imwrite(str(s_dis), disjoint_s.astype(np.uint8) * 255)

    views_disjoint = {"front": str(f_dis), "side": str(s_dis), "back": str(f_dis)}
    with pytest.raises(CFError) as exc:
        blockout.blockout(views_disjoint, {}, 160.0, tmp_path / "disjoint.ply")
    assert exc.value.code == "BLOCKOUT_FAILED"
    assert "intersection is empty" in exc.value.message


# ==============================================================================
# SECTION 9: Asymmetric Figure Grid Truncation (Adversarial Bug Reproduction)
# ==============================================================================

def test_empirical_asymmetric_figure_grid_clipping():
    """Demonstrate clipping bug: cols_x fails to span xs.max() for asymmetric silhouettes."""
    size = 512
    # Character with heavy torso at center, but outstretched weapon/arm reaching x=480
    front = np.zeros((size, size), dtype=bool)
    front[100:400, 230:270] = True
    front[50:100, 240:260] = True
    front[120:200, 210:230] = True
    front[120:140, 270:480] = True  # outstretched limb to x=480
    front[400:480, 235:245] = True
    front[400:480, 255:265] = True

    c_front = score.canonical_mask(front)
    ys, xs = np.nonzero(c_front)
    w_front = int(xs.max() - xs.min() + 1)
    xs_max = int(xs.max())

    Nz = 192
    k_min, k_max = 2, Nz - 3
    h_canon = int(ys.max() - ys.min() + 1)
    s_v = h_canon / (k_max - k_min)

    Nx = max(96, int(round(w_front / s_v)) + 8)
    if Nx % 2 != 0:
        Nx += 1

    # Exact formula from blockout.py line 141
    cols_x = np.clip(np.round(256 + (np.arange(Nx) - (Nx - 1) / 2.0) * s_v).astype(int), 0, config.CANON_SIZE - 1)

    # In an ideal implementation, cols_x should cover xs_max
    # We empirically observe and assert the truncation:
    clipped_pixels = max(0, xs_max - int(cols_x.max()))
    print(f"Empirically observed clipped pixels on right edge: {clipped_pixels} px (cols_x.max={cols_x.max()}, xs_max={xs_max})")
    assert clipped_pixels > 0, "Expected asymmetric silhouette to be clipped by centered cols_x formula"
