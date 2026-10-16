"""Contract tests for Meshy 3D generation, rigging, and animation APIs.

Verifies compliance with official Meshy OpenAPI v1 specifications:
- POST /openapi/v1/multi-image-to-3d (front-first data URIs, ai_model='meshy-7.1', pose_mode='a-pose',
  should_remesh=True, topology='triangle', target_polycount=30000, texture_image_urls, no seed)
- POST /openapi/v1/image-to-3d (single-image fallback, ai_model='meshy-7.1')
- POST /openapi/v1/rigging & GET /openapi/v1/rigging/{id} (reads result.rigged_character_glb_url &
  result.basic_animations, never reads model_urls, HTTP 400 fallback from input_task_id to model_url)
- GET /openapi/v1/animations/library & POST /openapi/v1/animations & GET /openapi/v1/animations/{id}
  (non-contiguous action_ids list in a single request, reads result.animation_glb_url, never reads model_urls)
- Durable kind-based polling across fresh MeshyProvider instances with UUID task IDs (no rig_/anim_ prefix)
- CANCELED status handling across generation, rigging, and animation
- Full derived_from lineage tracking from root Meshy candidate through fit/optimize/export to auto_rig
"""
from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any

import httpx
import pytest

from charforge import blender, config, providers, server, state, tools_core, tools_full
from charforge.envelope import CFError


def make_glb_bytes(bones: int = 24, animations: list[str] | None = None) -> bytes:
    """Create a valid glTF 2.0 binary (GLB) with specified joint count in skins and animation tracks."""
    nodes = [{"name": f"Joint_{i}"} for i in range(bones)]
    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "Meshy Contract Mock"},
        "scenes": [{"nodes": [0]}],
        "nodes": nodes if nodes else [{"name": "Root"}],
    }
    if bones > 0:
        gltf["skins"] = [{"joints": list(range(bones)), "name": "Armature"}]
    if animations is not None:
        gltf["animations"] = [
            {
                "name": name,
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
                "samplers": [{"input": 0, "interpolation": "LINEAR", "output": 1}],
            }
            for name in animations
        ]
    json_bytes = json.dumps(gltf).encode("utf-8")
    pad = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * pad
    header = struct.pack("<4sII", b"glTF", 2, 20 + len(json_bytes))
    chunk0 = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    return header + chunk0


DUMMY_GLB = make_glb_bytes(bones=0)
RIGGED_GLB_26_BONES = make_glb_bytes(bones=26)
ANIMATED_GLB_3_CLIPS = make_glb_bytes(bones=26, animations=["Idle_01", "Walking_Woman", "Waving_Gesture"])

MOCK_ANIMATION_LIBRARY = [
    {
        "action_id": 17,
        "name": "Idle",
        "key": "Idle",
        "category": "DailyActions",
        "sub_category": "Idle",
        "preview_url": "https://assets.meshy.ai/previews/17.gif",
    },
    {
        "action_id": 42,
        "name": "Walking_Woman",
        "key": "Walking_Woman",
        "category": "WalkAndRun",
        "sub_category": "Walking",
        "preview_url": "https://assets.meshy.ai/previews/42.gif",
    },
    {
        "action_id": 108,
        "name": "Waving_Gesture",
        "key": "Waving_Gesture",
        "category": "BodyMovements",
        "sub_category": "Greeting",
        "preview_url": "https://assets.meshy.ai/previews/108.gif",
    },
    {
        "action_id": 215,
        "name": "Running",
        "key": "Running",
        "category": "WalkAndRun",
        "sub_category": "Running",
        "preview_url": "https://assets.meshy.ai/previews/215.gif",
    },
]


class MeshyContractServer:
    """Mock transport server enforcing real Meshy OpenAPI v1 contract specifications."""

    def __init__(self, api_key: str = "meshy-contract-key"):
        self.api_key = api_key
        self.gen_tasks: dict[str, dict[str, Any]] = {}
        self.rig_tasks: dict[str, dict[str, Any]] = {}
        self.anim_tasks: dict[str, dict[str, Any]] = {}
        self.recorded_requests: list[httpx.Request] = []
        self.reject_rig_input_task_id: bool = False

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.recorded_requests.append(request)
        url_str = str(request.url)

        # Asset downloads from CDN do not require Meshy Bearer header
        if request.method == "GET" and url_str.endswith(".glb"):
            if "rigged" in url_str:
                return httpx.Response(200, content=RIGGED_GLB_26_BONES)
            if "anim" in url_str:
                return httpx.Response(200, content=ANIMATED_GLB_3_CLIPS)
            return httpx.Response(200, content=DUMMY_GLB)

        # 1. Check Authentication
        auth_hdr = request.headers.get("Authorization")
        if not auth_hdr or auth_hdr != f"Bearer {self.api_key}":
            return httpx.Response(401, json={"message": "Unauthorized"})

        # 2. Multi-image to 3D Generation: POST /openapi/v1/multi-image-to-3d
        if request.method == "POST" and url_str.endswith("/openapi/v1/multi-image-to-3d"):
            body = json.loads(request.content.decode("utf-8"))

            assert "image_urls" in body and isinstance(body["image_urls"], list)
            assert 1 <= len(body["image_urls"]) <= 4
            for u in body["image_urls"]:
                assert u.startswith("data:image/png;base64,")

            assert body.get("pose_mode") == "a-pose"
            assert body.get("ai_model") == config.MESHY_AI_MODEL == "meshy-7.1"
            assert body.get("should_remesh") is True
            assert body.get("topology") == "triangle"
            assert isinstance(body.get("target_polycount"), int)
            assert body.get("should_texture") is True
            assert body.get("enable_pbr") is True
            assert body.get("target_formats") == ["glb"]
            assert "seed" not in body, "Meshy multi-image-to-3d does not support 'seed'"

            # Realistic UUID task ID (NO prefix!)
            task_id = "0193abcd-1111-4000-8000-000000000001"
            self.gen_tasks[task_id] = {
                "id": task_id,
                "status": "SUCCEEDED",
                "progress": 100,
                "model_urls": {"glb": "https://assets.meshy.ai/multi/model.glb"},
            }
            return httpx.Response(202, json={"result": task_id})

        # 3. Single-image to 3D Generation: POST /openapi/v1/image-to-3d
        if request.method == "POST" and url_str.endswith("/openapi/v1/image-to-3d"):
            body = json.loads(request.content.decode("utf-8"))
            assert "image_url" in body and body["image_url"].startswith("data:image/png;base64,")
            assert body.get("ai_model") == config.MESHY_AI_MODEL == "meshy-7.1"
            assert body.get("pose_mode") == "a-pose"
            assert body.get("should_remesh") is True
            assert body.get("topology") == "triangle"
            assert body.get("enable_pbr") is True
            assert "seed" not in body

            task_id = "0193efgh-2222-4000-8000-000000000002"
            self.gen_tasks[task_id] = {
                "id": task_id,
                "status": "SUCCEEDED",
                "progress": 100,
                "model_urls": {"glb": "https://assets.meshy.ai/single/model.glb"},
            }
            return httpx.Response(202, json={"result": task_id})

        # 4. Rigging: POST /openapi/v1/rigging
        if request.method == "POST" and url_str.endswith("/openapi/v1/rigging"):
            body = json.loads(request.content.decode("utf-8"))
            assert ("input_task_id" in body) ^ ("model_url" in body), "Exactly one of input_task_id or model_url required"
            assert "height_meters" in body
            assert 0.5 <= body["height_meters"] <= 3.0

            if "input_task_id" in body and self.reject_rig_input_task_id:
                return httpx.Response(400, json={"message": "Pose estimation failed on un-normalized task mesh"})

            # Realistic UUID task ID (NO 'rig_' prefix!)
            rig_task_id = "01937777-3333-4000-8000-000000000003"
            self.rig_tasks[rig_task_id] = {
                "id": rig_task_id,
                "status": "SUCCEEDED",
                "progress": 100,
                # Official Meshy rigging response uses result.rigged_character_glb_url (NO model_urls!)
                "result": {
                    "rigged_character_glb_url": "https://assets.meshy.ai/rigged/character.glb",
                    "rigged_character_fbx_url": "https://assets.meshy.ai/rigged/character.fbx",
                    "basic_animations": {
                        "walking_glb_url": "https://assets.meshy.ai/rigged/walking.glb",
                        "running_glb_url": "https://assets.meshy.ai/rigged/running.glb",
                    },
                },
            }
            return httpx.Response(202, json={"result": rig_task_id})

        # 5. Animation Library: GET /openapi/v1/animations/library
        if request.method == "GET" and url_str.endswith("/openapi/v1/animations/library"):
            return httpx.Response(200, json=MOCK_ANIMATION_LIBRARY)

        # 6. Animation Creation: POST /openapi/v1/animations
        if request.method == "POST" and url_str.endswith("/openapi/v1/animations"):
            body = json.loads(request.content.decode("utf-8"))
            assert "rig_task_id" in body, "Body must contain 'rig_task_id'"
            assert "action_ids" in body, "Body must contain 'action_ids' (list), not legacy 'action_id'"
            assert "action_id" not in body, "Legacy singular 'action_id' must not be sent"
            action_ids = body["action_ids"]
            assert isinstance(action_ids, list) and 1 <= len(action_ids) <= 10
            assert len(set(action_ids)) == len(action_ids), "action_ids must be unique per Meshy spec"

            # Realistic UUID task ID (NO 'anim_' prefix!)
            anim_task_id = "01938888-4444-4000-8000-000000000004"
            self.anim_tasks[anim_task_id] = {
                "id": anim_task_id,
                "status": "SUCCEEDED",
                "progress": 100,
                # Official Meshy animation response uses result.animation_glb_url (NO model_urls!)
                "result": {
                    "animation_glb_url": "https://assets.meshy.ai/anim/merged_actions.glb",
                    "animation_fbx_url": "https://assets.meshy.ai/anim/merged_actions.fbx",
                },
            }
            return httpx.Response(202, json={"result": anim_task_id})

        # 7. Polling Multi-Image: GET /openapi/v1/multi-image-to-3d/{task_id}
        if request.method == "GET" and "/openapi/v1/multi-image-to-3d/" in url_str:
            task_id = url_str.split("/")[-1]
            if task_id in self.gen_tasks:
                return httpx.Response(200, json=self.gen_tasks[task_id])
            return httpx.Response(404, json={"message": "Gen task not found"})

        # 8. Polling Single-Image: GET /openapi/v1/image-to-3d/{task_id}
        if request.method == "GET" and "/openapi/v1/image-to-3d/" in url_str:
            task_id = url_str.split("/")[-1]
            if task_id in self.gen_tasks:
                return httpx.Response(200, json=self.gen_tasks[task_id])
            return httpx.Response(404, json={"message": "Single-image gen task not found"})

        # 9. Polling Rigging: GET /openapi/v1/rigging/{task_id}
        if request.method == "GET" and "/openapi/v1/rigging/" in url_str:
            task_id = url_str.split("/")[-1]
            if task_id in self.rig_tasks:
                return httpx.Response(200, json=self.rig_tasks[task_id])
            return httpx.Response(404, json={"message": "Rigging task not found"})

        # 10. Polling Animation: GET /openapi/v1/animations/{task_id}
        if request.method == "GET" and "/openapi/v1/animations/" in url_str:
            task_id = url_str.split("/")[-1]
            if task_id in self.anim_tasks:
                return httpx.Response(200, json=self.anim_tasks[task_id])
            return httpx.Response(404, json={"message": "Animation task not found"})

        return httpx.Response(404, json={"message": f"Not found: {url_str}"})


@pytest.fixture
def meshy_contract():
    srv = MeshyContractServer()
    transport = httpx.MockTransport(srv.handle_request)
    client = httpx.Client(transport=transport)
    prov = providers.MeshyProvider(api_key=srv.api_key, client=client)
    return srv, prov


def test_meshy_multi_image_contract_spec(meshy_contract, tmp_path):
    """Verify POST /openapi/v1/multi-image-to-3d payload: front first, meshy-7.1, remesh, texture_image_urls, no seed."""
    srv, prov = meshy_contract

    # Provide views in non-front-first dict order to verify front is placed at index 0
    views = {}
    for name in ("back", "side", "front"):
        p = tmp_path / f"{name}.png"
        p.write_bytes(f"image-{name}-bytes".encode())
        views[name] = p

    task_id = prov.submit_multiview(views, pose_mode="a-pose", target_polycount=25000)
    assert task_id == "0193abcd-1111-4000-8000-000000000001"

    req = next(r for r in srv.recorded_requests if "/multi-image-to-3d" in str(r.url))
    body = json.loads(req.content.decode("utf-8"))
    expected_front_uri = f"data:image/png;base64,{base64.b64encode(b'image-front-bytes').decode('ascii')}"
    assert body["image_urls"][0] == expected_front_uri, "Front view must always be first in image_urls"
    assert len(body["image_urls"]) == 3
    assert body["ai_model"] == "meshy-7.1"
    assert body["pose_mode"] == "a-pose"
    assert body["should_remesh"] is True
    assert body["topology"] == "triangle"
    assert body["target_polycount"] == 25000
    assert body["should_texture"] is True
    assert body["enable_pbr"] is True
    assert body["target_formats"] == ["glb"]
    assert body["texture_image_urls"] == [expected_front_uri]
    assert "seed" not in body


def test_meshy_rigging_and_animation_never_read_model_urls(meshy_contract):
    """Contract requirement §1.7.1: Rigging and animation responses without model_urls succeed,
    and even if bogus model_urls is present, result.rigged_character_glb_url / result.animation_glb_url is used.
    """
    srv, prov = meshy_contract

    rig_id = prov.submit_rig(input_task_id="0193abcd-1111-4000-8000-000000000001", height_meters=1.70)
    # Inject a poison model_urls to prove poll_rig never reads it
    srv.rig_tasks[rig_id]["model_urls"] = {"glb": "https://poison.invalid/wrong.glb"}
    poll_rig = prov.poll_rig(rig_id)
    assert poll_rig["status"] == "done"
    assert poll_rig["url"] == "https://assets.meshy.ai/rigged/character.glb"
    assert "walking_glb_url" in poll_rig["basic_animations"]

    anim_id = prov.submit_animation(rig_task_id=rig_id, action_ids=[17, 42, 108])
    srv.anim_tasks[anim_id]["model_urls"] = {"glb": "https://poison.invalid/wrong_anim.glb"}
    poll_anim = prov.poll_animation(anim_id)
    assert poll_anim["status"] == "done"
    assert poll_anim["url"] == "https://assets.meshy.ai/anim/merged_actions.glb"


def test_meshy_stateless_polling_with_fresh_provider_instance(meshy_contract):
    """Contract requirement §1.7.2: Polling with a brand-new MeshyProvider instance (empty _job_types)
    and UUID task_ids (no rig_/anim_ prefix) hits the exact endpoint via durable `kind`.
    """
    srv, prov = meshy_contract
    gen_id = "0193abcd-1111-4000-8000-000000000001"
    srv.gen_tasks[gen_id] = {
        "id": gen_id,
        "status": "SUCCEEDED",
        "model_urls": {"glb": "https://assets.meshy.ai/multi/model.glb"},
    }
    rig_id = prov.submit_rig(input_task_id=gen_id, height_meters=1.65)
    anim_id = prov.submit_animation(rig_task_id=rig_id, action_ids=[17, 42])

    # Create a brand-new MeshyProvider instance (simulating a subsequent MCP tool call)
    fresh_client = httpx.Client(transport=httpx.MockTransport(srv.handle_request))
    fresh_prov = providers.MeshyProvider(api_key=srv.api_key, client=fresh_client)

    assert fresh_prov.poll(gen_id, kind="multi-image-to-3d")["url"] == "https://assets.meshy.ai/multi/model.glb"
    assert fresh_prov.poll(rig_id, kind="rigging")["url"] == "https://assets.meshy.ai/rigged/character.glb"
    assert fresh_prov.poll(anim_id, kind="animations")["url"] == "https://assets.meshy.ai/anim/merged_actions.glb"


@pytest.mark.parametrize("kind", ["multi-image-to-3d", "image-to-3d", "rigging", "animations"])
def test_meshy_canceled_status_maps_to_failed(meshy_contract, kind):
    """Contract requirement §1.7.3: Status CANCELED maps to status='failed' with task_error.message."""
    srv, prov = meshy_contract
    task_id = "01939999-5555-4000-8000-000000000005"
    payload = {
        "id": task_id,
        "status": "CANCELED",
        "task_error": {"message": f"Task {kind} was canceled by user"},
    }
    if kind in ("multi-image-to-3d", "image-to-3d"):
        srv.gen_tasks[task_id] = payload
    elif kind == "rigging":
        srv.rig_tasks[task_id] = payload
    else:
        srv.anim_tasks[task_id] = payload

    fresh_prov = providers.MeshyProvider(
        api_key=srv.api_key,
        client=httpx.Client(transport=httpx.MockTransport(srv.handle_request)),
    )
    res = fresh_prov.poll(task_id, kind=kind)
    assert res["status"] == "failed"
    assert f"Task {kind} was canceled by user" in res["error"]


def test_meshy_lineage_derived_from_through_export_to_auto_rig(sheet_path, meshy_contract, monkeypatch, tmp_path):
    """Contract requirement §1.7.4: export/<id>.glb traces back through derived_from to the root Meshy task_id,
    so auto_rig(method='auto') uses Meshy rigging and counts real bones from GLB skins.
    """
    srv, _ = meshy_contract
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")
    monkeypatch.setenv("MESHY_API_KEY", srv.api_key)
    server._register_levels()

    # Always return a fresh MeshyProvider instance per call, just like production
    monkeypatch.setattr(
        providers,
        "get_provider",
        lambda name, client=None: providers.MeshyProvider(
            api_key=srv.api_key,
            client=client or httpx.Client(transport=httpx.MockTransport(srv.handle_request)),
        ),
    )

    # Mock blender.run for fit.py (--rigid-only) and export.py
    def mock_blender_run(script_name: str, args: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
        if script_name == "fit.py":
            out_p = Path(args["out"])
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_bytes(DUMMY_GLB)
            return {"ok": True, "rigid_only": bool(args.get("rigid_only"))}
        if script_name == "export.py":
            out_dir = Path(args["out_dir"])
            stem = args["stem"]
            out_p = out_dir / f"{stem}.glb"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_bytes(DUMMY_GLB)
            return {"ok": True, "files": {"glb": str(out_p)}, "size_mb": 0.1, "tris": 28500}
        raise AssertionError(f"Unexpected Blender script call: {script_name}")

    monkeypatch.setattr(blender, "run", mock_blender_run)

    cr = server.project_create(sheet_path=str(sheet_path), name="LineageTest", height_cm=168.0)
    pid = cr["project_id"]
    server.sheet_split(project_id=pid)

    # 1. generate_mesh(provider="meshy")
    gen_res = tools_core.generate_mesh(project_id=pid, provider="meshy")
    assert gen_res["ok"] is True
    cand_path = gen_res["data"]["mesh_path"]
    assert cand_path == "meshes/cand-1-meshy.glb"

    # 2. mesh_export on cand-1-meshy.glb -> auto-normalizes rigidly and writes export/<pid>.glb
    exp_res = tools_core.mesh_export(project_id=pid, mesh_path=cand_path, format="glb")
    assert exp_res["ok"] is True
    export_glb = exp_res["data"]["files"]["glb"]
    assert export_glb == f"export/{pid}.glb"

    # Verify lineage: export/<pid>.glb -> meshes/cand-1-meshy-norm.glb -> meshes/cand-1-meshy.glb (task_id)
    p = state.load(pid)
    assert state.find_provider_task_id(p, export_glb, provider="meshy") == "0193abcd-1111-4000-8000-000000000001"
    assert state.is_provider_mesh(p, export_glb) is True

    # 3. auto_rig(method="auto") with mesh_path=export/<pid>.glb must use Meshy provider!
    rig_res = tools_full.auto_rig(project_id=pid, mesh_path=export_glb, method="auto")
    assert rig_res["ok"] is True
    assert rig_res["data"]["status"] == "done"
    assert rig_res["data"]["rig_source"] == "provider"
    assert rig_res["data"]["bones"] == 26  # Real joint count from RIGGED_GLB_26_BONES skins!
    assert "walking_glb_url" in rig_res["data"]["basic_animations"]


def test_meshy_rig_http_400_fallback_to_model_url(sheet_path, meshy_contract, monkeypatch):
    """Verify §1.2: If POST /openapi/v1/rigging with input_task_id returns HTTP 400,
    auto_rig automatically retries once with model_url (data URI of the normalized GLB).
    """
    srv, _ = meshy_contract
    srv.reject_rig_input_task_id = True

    monkeypatch.setenv("CHARFORGE_LEVEL", "3")
    monkeypatch.setenv("MESHY_API_KEY", srv.api_key)
    server._register_levels()
    monkeypatch.setattr(
        providers,
        "get_provider",
        lambda name, client=None: providers.MeshyProvider(
            api_key=srv.api_key,
            client=client or httpx.Client(transport=httpx.MockTransport(srv.handle_request)),
        ),
    )

    cr = server.project_create(sheet_path=str(sheet_path), name="RigFallback400", height_cm=170.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    mesh_rel = "meshes/cand-1-meshy.glb"
    (pdir / mesh_rel).parent.mkdir(parents=True, exist_ok=True)
    (pdir / mesh_rel).write_bytes(DUMMY_GLB)

    p = state.load(pid)
    state.register_mesh(p, mesh_rel, source="meshy", task_id="0193abcd-1111-4000-8000-000000000001")
    state.save(p)

    rig_res = tools_full.auto_rig(project_id=pid, mesh_path=mesh_rel, method="provider")
    assert rig_res["ok"] is True
    assert rig_res["data"]["rig_source"] == "provider"
    assert rig_res["data"]["rig_input_mode"] == "model_url"

    rig_posts = [r for r in srv.recorded_requests if r.method == "POST" and "/openapi/v1/rigging" in str(r.url)]
    assert len(rig_posts) == 2
    first_body = json.loads(rig_posts[0].content.decode("utf-8"))
    second_body = json.loads(rig_posts[1].content.decode("utf-8"))
    assert "input_task_id" in first_body
    assert "model_url" in second_body and second_body["model_url"].startswith("data:model/gltf-binary;base64,")


def test_meshy_animation_library_resolution_and_unknown_action_error(meshy_contract, tmp_path):
    """Verify §1.3: Animation library caching, alias resolution, and helpful CFError on unknown action."""
    srv, prov = meshy_contract
    library = prov.fetch_animation_library(force_refresh=True)
    assert len(library) == 4

    ids = prov.resolve_action_ids(["idle", "walk", "wave"], library=library)
    assert ids == [17, 42, 108]

    with pytest.raises(CFError) as exc_info:
        prov.resolve_action_ids(["backflip_tornado"], library=library)
    assert exc_info.value.code == "ANIM_FAILED"
    assert "Use valid action names from the library" in exc_info.value.fix


def test_meshy_contract_unauthorized_rejection(tmp_path):
    """Verify that requests with invalid API keys are rejected with HTTP 401."""
    srv = MeshyContractServer(api_key="valid-secret")
    transport = httpx.MockTransport(srv.handle_request)
    client = httpx.Client(transport=transport)
    bad_provider = providers.MeshyProvider(api_key="wrong-secret", client=client)

    img = tmp_path / "front.png"
    img.write_bytes(b"content")

    with pytest.raises(CFError) as exc_info:
        bad_provider.submit({"front": img})
    assert exc_info.value.code == "PROVIDER_FAILED"


@pytest.mark.blender
def test_fit_rigid_only_blender_preserves_uvs_and_materials(tmp_path):
    """Verify §4.3: fit.py --rigid-only places feet at Z=0, centers XY at (0,0),
    scales uniformly to height_m, and preserves UV maps and materials without joining or lattice deformation.
    """
    import subprocess

    exe = config.blender_exe()
    if not exe:
        pytest.skip("Blender executable not available.")

    src_glb = tmp_path / "src_provider.glb"
    out_glb = tmp_path / "norm_provider.glb"
    inspect_json = tmp_path / "inspect.json"

    create_script = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(1.5, -2.0, 3.0))
obj = bpy.context.active_object
obj.name = "BodyPartA"
mat = bpy.data.materials.new(name="PBR_Mat")
mat.use_nodes = True
obj.data.materials.append(mat)
bpy.ops.export_scene.gltf(filepath=r"{str(src_glb)}", export_format="GLB")
"""
    subprocess.run([exe, "-b", "--factory-startup", "--python-expr", create_script], check=True, capture_output=True)

    res = blender.run(
        "fit.py",
        {
            "mesh": src_glb,
            "out": out_glb,
            "height_m": 1.68,
            "rotate_z": 90,
            "rigid_only": True,
        },
    )
    assert res["ok"] is True
    assert res["rigid_only"] is True
    assert abs(res["dims_m"][2] - 1.68) < 1e-3

    verify_script = f"""
import json, bpy
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"{str(out_glb)}")
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
xs, ys, zs = [], [], []
has_uv = False
has_mat = False
for m in meshes:
    if len(m.data.uv_layers) > 0:
        has_uv = True
    if len(m.data.materials) > 0:
        has_mat = True
    for v in m.data.vertices:
        w = m.matrix_world @ v.co
        xs.append(w.x); ys.append(w.y); zs.append(w.z)
info = {{
    "min_z": min(zs),
    "max_z": max(zs),
    "center_x": 0.5 * (min(xs) + max(xs)),
    "center_y": 0.5 * (min(ys) + max(ys)),
    "has_uv": has_uv,
    "has_mat": has_mat,
}}
with open(r"{str(inspect_json)}", "w") as f:
    json.dump(info, f)
"""
    subprocess.run([exe, "-b", "--factory-startup", "--python-expr", verify_script], check=True, capture_output=True)
    info = json.loads(inspect_json.read_text(encoding="utf-8"))
    assert abs(info["min_z"]) < 1e-4
    assert abs(info["max_z"] - 1.68) < 1e-3
    assert abs(info["center_x"]) < 1e-4
    assert abs(info["center_y"]) < 1e-4
    assert info["has_uv"] is True
    assert info["has_mat"] is True


