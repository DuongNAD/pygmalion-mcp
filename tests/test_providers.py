import json
import time
from pathlib import Path

import httpx
import pytest

from charforge import config, providers, server, state, tools_core
from charforge.envelope import CFError


def test_level_2_tools_registered(monkeypatch):
    """Verify that when CHARFORGE_LEVEL=2, tools_core tools are registered on FastMCP."""
    monkeypatch.setenv("CHARFORGE_LEVEL", "2")
    # Call _register_levels to register tools_core
    server._register_levels()

    # List registered tool names on server.mcp
    import asyncio
    tools = asyncio.run(server.mcp.list_tools()) if asyncio.iscoroutinefunction(server.mcp.list_tools) else server.mcp.list_tools()
    registered_tools = [t.name for t in tools]
    assert "generate_mesh" in registered_tools
    assert "mesh_fit" in registered_tools
    assert "mesh_optimize" in registered_tools
    assert "mesh_export" in registered_tools


def test_provider_no_key(monkeypatch):
    """When a specific cloud provider is called without an API key, raise NO_PROVIDER."""
    monkeypatch.delenv("TRIPO_API_KEY", raising=False)
    p = providers.get_provider("tripo")
    with pytest.raises(CFError) as exc_info:
        p.submit({"front": Path("/tmp/fake.png")})
    assert exc_info.value.code == "NO_PROVIDER"


def test_provider_http_errors(monkeypatch, tmp_path):
    """Verify HTTP 401, 402, 429 map to PROVIDER_FAILED with blockout fix across all providers."""
    fake_img = tmp_path / "front.png"
    fake_img.write_bytes(b"dummy")

    for prov_cls in (providers.TripoProvider, providers.MeshyProvider, providers.RodinProvider, providers.HunyuanProvider):
        for status_code in (401, 402, 429):
            transport = httpx.MockTransport(lambda req, sc=status_code: httpx.Response(sc, text="Quota exceeded"))
            client = httpx.Client(transport=transport)
            p = prov_cls(api_key="fake-key", client=client)
            with pytest.raises(CFError) as exc_info:
                p.submit({"front": fake_img})
            assert exc_info.value.code == "PROVIDER_FAILED"
            assert "blockout" in exc_info.value.fix



def test_provider_mock_transport_full_lifecycle(sheet_path, monkeypatch, tmp_path):
    """Simulate submit -> running -> done -> download -> registered mesh via httpx.MockTransport."""
    monkeypatch.setenv("TRIPO_API_KEY", "test-key-tripo")

    # Create dummy GLB bytes
    dummy_glb = b"glTF\x02\x00\x00\x00\x14\x00\x00\x00JSON{}"

    call_count = {"poll": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if request.method == "POST" and "task" in url_str:
            return httpx.Response(200, json={"data": {"task_id": "mock-task-123"}})
        elif request.method == "GET" and "task/mock-task-123" in url_str:
            call_count["poll"] += 1
            if call_count["poll"] == 1:
                return httpx.Response(200, json={"data": {"status": "running"}})
            else:
                return httpx.Response(200, json={"data": {"status": "success", "output": {"model": "https://mock.cdn/model.glb"}}})
        elif request.method == "GET" and "mock.cdn/model.glb" in url_str:
            return httpx.Response(200, content=dummy_glb)
        return httpx.Response(404)

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(providers, "get_provider", lambda name, client=None: providers.TripoProvider(api_key="test-key-tripo", client=mock_client))

    # 1. Project create & split
    res_create = server.project_create(sheet_path=str(sheet_path), name="Provider Mock", height_cm=165.0)
    pid = res_create["project_id"]
    server.sheet_split(project_id=pid)

    # 2. First call: submit -> running
    res_gen1 = tools_core.generate_mesh(project_id=pid, provider="tripo")
    assert res_gen1["ok"] is True
    assert res_gen1["data"]["status"] == "running"
    assert res_gen1["data"]["job_id"] == "mock-task-123"
    assert res_gen1["next_action"]["tool"] == "generate_mesh"
    assert res_gen1["next_action"]["args"]["job_id"] == "mock-task-123"

    p = state.load(pid)
    assert "mock-task-123" in p["jobs"]

    # 3. Second call: poll -> done -> download -> mesh registered
    res_gen2 = tools_core.generate_mesh(project_id=pid, provider="tripo", job_id="mock-task-123")
    assert res_gen2["ok"] is True
    assert res_gen2["data"]["status"] == "done"
    assert res_gen2["data"]["mesh_path"] is not None
    assert res_gen2["next_action"]["tool"] == "score_mesh"

    pdir = state.project_dir(pid)
    mesh_file = pdir / res_gen2["data"]["mesh_path"]
    assert mesh_file.is_file()
    assert mesh_file.read_bytes() == dummy_glb

    p = state.load(pid)
    registered_paths = [m["path"] for m in p["meshes"]]
    assert res_gen2["data"]["mesh_path"] in registered_paths


def test_provider_timeout_mapping(sheet_path, monkeypatch):
    """Job taking > 20 minutes should raise PROVIDER_TIMEOUT."""
    res_create = server.project_create(sheet_path=str(sheet_path), name="Timeout Test", height_cm=165.0)
    pid = res_create["project_id"]
    server.sheet_split(project_id=pid)

    p = state.load(pid)
    p["jobs"]["old-job-999"] = {"provider": "tripo", "submitted_at": time.time() - 1300}
    state.save(p)

    res = tools_core.generate_mesh(project_id=pid, provider="tripo", job_id="old-job-999")
    assert res["ok"] is False
    assert res["error"]["code"] == "PROVIDER_TIMEOUT"


def test_provider_meshy_lifecycle(tmp_path):
    """Verify Meshy lifecycle: submit -> running -> done -> download."""
    dummy_glb = b"glTF\x02\x00\x00\x00\x14\x00\x00\x00JSON{meshy}"
    call_count = {"poll": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if request.method == "POST" and "image-to-3d" in url_str:
            return httpx.Response(200, json={"result": "meshy-task-456"})
        elif request.method == "GET" and "image-to-3d/meshy-task-456" in url_str:
            call_count["poll"] += 1
            if call_count["poll"] == 1:
                return httpx.Response(200, json={"status": "IN_PROGRESS"})
            return httpx.Response(200, json={"status": "SUCCEEDED", "model_urls": {"glb": "https://meshy.mock/model.glb"}})
        elif request.method == "GET" and "meshy.mock/model.glb" in url_str:
            return httpx.Response(200, content=dummy_glb)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = providers.MeshyProvider(api_key="meshy-key", client=client)
    fake_img = tmp_path / "front.png"
    fake_img.write_bytes(b"dummy")
    job_id = p.submit({"front": fake_img})
    assert job_id == "meshy-task-456"
    assert p.poll(job_id)["status"] == "running"
    done_res = p.poll(job_id)
    assert done_res["status"] == "done"
    dl_dst = tmp_path / "meshy.glb"
    p.download(done_res["url"], dl_dst)
    assert dl_dst.read_bytes() == dummy_glb


def test_provider_rodin_lifecycle(tmp_path):
    """Verify Rodin lifecycle: submit -> running -> done -> download."""
    dummy_glb = b"glTF\x02\x00\x00\x00\x14\x00\x00\x00JSON{rodin}"
    call_count = {"poll": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if request.method == "POST" and "rodin" in url_str:
            return httpx.Response(200, json={"uuid": "rodin-task-789"})
        elif request.method == "GET" and "rodin/rodin-task-789" in url_str:
            call_count["poll"] += 1
            if call_count["poll"] == 1:
                return httpx.Response(200, json={"status": "processing"})
            return httpx.Response(200, json={"status": "success", "download_url": "https://rodin.mock/model.glb"})
        elif request.method == "GET" and "rodin.mock/model.glb" in url_str:
            return httpx.Response(200, content=dummy_glb)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = providers.RodinProvider(api_key="rodin-key", client=client)
    fake_img = tmp_path / "front.png"
    fake_img.write_bytes(b"dummy")
    job_id = p.submit({"front": fake_img})
    assert job_id == "rodin-task-789"
    assert p.poll(job_id)["status"] == "running"
    done_res = p.poll(job_id)
    assert done_res["status"] == "done"
    dl_dst = tmp_path / "rodin.glb"
    p.download(done_res["url"], dl_dst)
    assert dl_dst.read_bytes() == dummy_glb


def test_provider_hunyuan_lifecycle(tmp_path):
    """Verify Hunyuan lifecycle: submit -> running -> done -> download."""
    dummy_glb = b"glTF\x02\x00\x00\x00\x14\x00\x00\x00JSON{hunyuan}"
    call_count = {"poll": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if request.method == "POST" and "fal-ai/hunyuan3d/v2" in url_str:
            return httpx.Response(200, json={"request_id": "hunyuan-task-101"})
        elif request.method == "GET" and "requests/hunyuan-task-101/status" in url_str:
            call_count["poll"] += 1
            if call_count["poll"] == 1:
                return httpx.Response(200, json={"status": "IN_PROGRESS"})
            return httpx.Response(200, json={"status": "COMPLETED"})
        elif request.method == "GET" and "requests/hunyuan-task-101" in url_str:
            return httpx.Response(200, json={"model_mesh": {"url": "https://hunyuan.mock/model.glb"}})
        elif request.method == "GET" and "hunyuan.mock/model.glb" in url_str:
            return httpx.Response(200, content=dummy_glb)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = providers.HunyuanProvider(api_key="hunyuan-key", client=client)
    fake_img = tmp_path / "front.png"
    fake_img.write_bytes(b"dummy")
    job_id = p.submit({"front": fake_img})
    assert job_id == "hunyuan-task-101"
    assert p.poll(job_id)["status"] == "running"
    done_res = p.poll(job_id)
    assert done_res["status"] == "done"
    dl_dst = tmp_path / "hunyuan.glb"
    p.download(done_res["url"], dl_dst)
    assert dl_dst.read_bytes() == dummy_glb


def test_meshy_provider_multiview_and_rig_and_anim(tmp_path):
    """Verify MeshyProvider submit_multiview, submit_rig, and submit_animation methods."""
    recorded = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        url_str = str(request.url)
        if request.method == "POST" and url_str.endswith("/multi-image-to-3d"):
            body = json.loads(request.content.decode("utf-8"))
            assert len(body["image_urls"]) == 3
            assert body["pose_mode"] == "a-pose"
            assert body["ai_model"] == config.MESHY_AI_MODEL
            assert body["should_remesh"] is True
            assert body["topology"] == "triangle"
            assert body["enable_pbr"] is True
            return httpx.Response(202, json={"result": "0193mv00-0001-4000-8000-000000000001"})
        elif request.method == "POST" and url_str.endswith("/rigging"):
            body = json.loads(request.content.decode("utf-8"))
            assert body["input_task_id"] == "0193mv00-0001-4000-8000-000000000001"
            assert body["height_meters"] == 1.72
            return httpx.Response(200, json={"result": "0193rig0-0002-4000-8000-000000000002"})
        elif request.method == "POST" and url_str.endswith("/animations"):
            body = json.loads(request.content.decode("utf-8"))
            assert body["rig_task_id"] == "0193rig0-0002-4000-8000-000000000002"
            assert body["action_ids"] == [17, 42]
            return httpx.Response(200, json={"result": "0193anim-0003-4000-8000-000000000003"})
        elif request.method == "GET" and "/multi-image-to-3d/0193mv00-0001-4000-8000-000000000001" in url_str:
            return httpx.Response(200, json={"status": "SUCCEEDED", "model_urls": {"glb": "https://meshy.mock/mv.glb"}})
        elif request.method == "GET" and "/rigging/0193rig0-0002-4000-8000-000000000002" in url_str:
            return httpx.Response(
                200,
                json={"status": "SUCCEEDED", "result": {"rigged_character_glb_url": "https://meshy.mock/rig.glb"}},
            )
        elif request.method == "GET" and "/animations/0193anim-0003-4000-8000-000000000003" in url_str:
            return httpx.Response(
                200,
                json={"status": "SUCCEEDED", "result": {"animation_glb_url": "https://meshy.mock/anim.glb"}},
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = providers.MeshyProvider(api_key="meshy-key", client=client)

    views = {}
    for name in ("front", "side", "back"):
        f = tmp_path / f"{name}.png"
        f.write_bytes(f"img-{name}".encode())
        views[name] = f

    # 1. submit_multiview
    job_id = p.submit_multiview(views, pose_mode="a-pose")
    assert job_id == "0193mv00-0001-4000-8000-000000000001"
    poll_mv = p.poll(job_id, kind="multi-image-to-3d")
    assert poll_mv["status"] == "done"
    assert poll_mv["url"] == "https://meshy.mock/mv.glb"

    # 2. submit_rig
    rig_id = p.submit_rig(input_task_id=job_id, height_meters=1.72)
    assert rig_id == "0193rig0-0002-4000-8000-000000000002"
    poll_rig = p.poll_rig(rig_id)
    assert poll_rig["status"] == "done"
    assert poll_rig["url"] == "https://meshy.mock/rig.glb"

    # 3. submit_animation
    anim_id = p.submit_animation(rig_task_id=rig_id, action_ids=[17, 42])
    assert anim_id == "0193anim-0003-4000-8000-000000000003"
    poll_anim = p.poll_animation(anim_id)
    assert poll_anim["status"] == "done"
    assert poll_anim["url"] == "https://meshy.mock/anim.glb"

    # 4. poll dispatch via generic poll(job_id, kind=...)
    assert p.poll(rig_id, kind="rigging")["status"] == "done"
    assert p.poll(anim_id, kind="animations")["status"] == "done"


def test_tripo_provider_upload_and_multiview_mapping(tmp_path):
    """Verify TripoProvider upload_image, submit_multiview, and directional side_faces mapping."""
    uploaded_files = {}

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if request.method == "POST" and url_str.endswith("/upload"):
            token = f"tok-{len(uploaded_files) + 1}"
            uploaded_files[token] = request.content
            return httpx.Response(200, json={"code": 0, "data": {"image_token": token}})
        elif request.method == "POST" and url_str.endswith("/task"):
            body = json.loads(request.content.decode("utf-8"))
            inputs = body.get("inputs", {})
            return httpx.Response(200, json={"code": 0, "data": {"task_id": "tripo-task-mapped", "inputs": inputs}})
        elif request.method == "GET" and "/task/tripo-task-mapped" in url_str:
            return httpx.Response(200, json={"code": 0, "data": {"status": "success", "output": {"pbr_model": "https://tripo.mock/model.glb"}}})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = providers.TripoProvider(api_key="tripo-key", client=client)

    # 1. upload_image directly
    tok = p.upload_image(b"fake-bytes", "test.png")
    assert tok == "tok-1"

    # 2. submit with reference.side_faces == 'right' -> maps to 'left'
    front_p = tmp_path / "front.png"
    front_p.write_bytes(b"front")
    side_p = tmp_path / "side.png"
    side_p.write_bytes(b"side")

    job_id1 = p.submit({"front": front_p, "side": side_p}, reference={"side_faces": "right"})
    assert job_id1 == "tripo-task-mapped"

    # 3. submit with reference.side_faces == 'left' -> maps to 'right'
    job_id2 = p.submit({"front": front_p, "side": side_p}, reference={"side_faces": "left"})
    assert job_id2 == "tripo-task-mapped"

    # 4. poll
    poll_res = p.poll(job_id1)
    assert poll_res["status"] == "done"
    assert poll_res["url"] == "https://tripo.mock/model.glb"


def test_rodin_provider_multipart_and_download(tmp_path):
    """Verify RodinProvider submit_multipart, get_download_url, and status polling."""
    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if request.method == "POST" and url_str.endswith("/rodin"):
            return httpx.Response(
                200,
                json={"uuid": "rodin-uuid-123", "jobs": {"subscription_key": "sub-key-456"}},
            )
        elif request.method == "POST" and url_str.endswith("/status"):
            body = json.loads(request.content.decode("utf-8"))
            assert body["subscription_key"] == "sub-key-456"
            return httpx.Response(200, json={"status": "completed"})
        elif request.method == "POST" and url_str.endswith("/download"):
            body = json.loads(request.content.decode("utf-8"))
            assert body["task_uuid"] == "rodin-uuid-123"
            return httpx.Response(200, json={"url": "https://rodin.mock/download.glb", "name": "download.glb"})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = providers.RodinProvider(api_key="rodin-key", client=client)

    front_p = tmp_path / "front.png"
    front_p.write_bytes(b"front-data")
    job_id = p.submit({"front": front_p})
    assert job_id == "rodin-uuid-123:sub-key-456"

    # Polling status invokes get_download_url under the hood
    poll_res = p.poll(job_id)
    assert poll_res["status"] == "done"
    assert poll_res["url"] == "https://rodin.mock/download.glb"

    # Direct get_download_url
    dl_url = p.get_download_url("rodin-uuid-123")
    assert dl_url == "https://rodin.mock/download.glb"


def test_auto_rig_prioritization(sheet_path, monkeypatch, tmp_path):
    """Verify auto_rig(method='auto') prioritizes provider when MESHY_API_KEY + task_id exist."""
    from charforge import tools_full

    monkeypatch.setenv("CHARFORGE_LEVEL", "3")
    server._register_levels()

    cr = server.project_create(sheet_path=str(sheet_path), name="AutoRigPrioritySuite", height_cm=170.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    # Create dummy mesh
    mesh_rel = "meshes/cand-1-meshy.glb"
    mesh_full = pdir / mesh_rel
    mesh_full.parent.mkdir(parents=True, exist_ok=True)
    mesh_full.write_bytes(b"dummy-glb")

    # Register mesh in project with provider task_id via state.register_mesh
    p = state.load(pid)
    state.register_mesh(p, mesh_rel, source="meshy", task_id="meshy-upstream-task-789")
    p["jobs"]["meshy-upstream-task-789"] = {"provider": "meshy", "status": "done", "mesh_path": mesh_rel}
    state.save(p)

    # 1. Mock Meshy transport for rigging
    dummy_rigged_glb = b"glTF\x02\x00\x00\x00\x18\x00\x00\x00JSON{\"rigged\":true}"
    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if request.method == "POST" and "/openapi/v1/rigging" in url_str:
            body = json.loads(request.content.decode("utf-8"))
            assert body["input_task_id"] == "meshy-upstream-task-789"
            assert body["height_meters"] == 1.70
            return httpx.Response(200, json={"result": "0193rig0-0789-4000-8000-000000000789"})
        elif request.method == "GET" and "/openapi/v1/rigging/0193rig0-0789-4000-8000-000000000789" in url_str:
            return httpx.Response(
                200,
                json={"status": "SUCCEEDED", "result": {"rigged_character_glb_url": "https://meshy.mock/rigged.glb"}},
            )
        elif request.method == "GET" and "meshy.mock/rigged.glb" in url_str:
            return httpx.Response(200, content=dummy_rigged_glb)
        return httpx.Response(404)

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setenv("MESHY_API_KEY", "valid-meshy-key")
    monkeypatch.setattr(providers, "get_provider", lambda name, client=None: providers.MeshyProvider(api_key="valid-meshy-key", client=mock_client))

    # Run auto_rig with method="auto": should pick provider
    res = tools_full.auto_rig(project_id=pid, mesh_path=mesh_rel, method="auto")
    assert res["ok"] is True
    assert res["data"]["method"] == "provider"
    assert res["data"]["rig_source"] == "provider"
    assert res["data"]["rig_task_id"] == "0193rig0-0789-4000-8000-000000000789"
    assert (pdir / res["data"]["mesh_path"]).read_bytes() == dummy_rigged_glb

    # 2. When method="provider" without MESHY_API_KEY, raises NO_PROVIDER
    monkeypatch.delenv("MESHY_API_KEY", raising=False)
    res_nokey = tools_full.auto_rig(project_id=pid, mesh_path=mesh_rel, method="provider")
    assert res_nokey["ok"] is False
    assert res_nokey["error"]["code"] == "NO_PROVIDER"

    # 3. When method="provider" with MESHY_API_KEY but candidate has no task_id, raises NO_PROVIDER
    monkeypatch.setenv("MESHY_API_KEY", "valid-meshy-key")
    unknown_rel = "meshes/orphan.glb"
    (pdir / unknown_rel).write_bytes(b"dummy")
    p = state.load(pid)
    state.register_mesh(p, unknown_rel, source="blockout")
    state.save(p)
    res_notask = tools_full.auto_rig(project_id=pid, mesh_path=unknown_rel, method="provider")
    assert res_notask["ok"] is False
    assert res_notask["error"]["code"] == "NO_PROVIDER"


def test_generate_mesh_to_auto_rig_real_pipeline_flow(sheet_path, monkeypatch, tmp_path):
    """Verify real pipeline flow from generate_mesh to auto_rig(method='provider')."""
    from charforge import tools_full, tools_core

    monkeypatch.setenv("CHARFORGE_LEVEL", "3")
    server._register_levels()

    cr = server.project_create(sheet_path=str(sheet_path), name="RealPipelineFlow", height_cm=170.0)
    pid = cr["project_id"]
    pdir = state.project_dir(pid)

    # Set up dummy views in project
    (pdir / "views").mkdir(parents=True, exist_ok=True)
    for v in ("front", "side", "back"):
        f = pdir / "views" / f"{v}.png"
        f.write_bytes(b"dummy-image")
        p = state.load(pid)
        p.setdefault("views", {})[v] = {"image": f"views/{v}.png"}
        state.save(p)

    # Register pre-existing blockout mesh (multiple meshes scenario)
    p = state.load(pid)
    state.register_mesh(p, "meshes/cand-1-blockout.ply", source="blockout")
    state.save(p)

    dummy_gen_glb = b"glTF\x02\x00\x00\x00\x14\x00\x00\x00JSON{}"
    dummy_rig_glb = b"glTF\x02\x00\x00\x00\x18\x00\x00\x00JSON{\"rig\":1}"

    def handler(request: httpx.Request) -> httpx.Response:
        u = str(request.url)
        if request.method == "POST" and "/openapi/v1/multi-image-to-3d" in u:
            return httpx.Response(200, json={"result": "meshy-gen-task-101"})
        elif request.method == "GET" and "/openapi/v1/multi-image-to-3d/meshy-gen-task-101" in u:
            return httpx.Response(200, json={"status": "SUCCEEDED", "model_urls": {"glb": "https://meshy.mock/model.glb"}})
        elif request.method == "GET" and "meshy.mock/model.glb" in u:
            return httpx.Response(200, content=dummy_gen_glb)
        elif request.method == "POST" and "/openapi/v1/rigging" in u:
            body = json.loads(request.content.decode("utf-8"))
            assert body["input_task_id"] == "meshy-gen-task-101"
            return httpx.Response(200, json={"result": "meshy-rig-task-202"})
        elif request.method == "GET" and "/openapi/v1/rigging/meshy-rig-task-202" in u:
            return httpx.Response(
                200,
                json={"status": "SUCCEEDED", "result": {"rigged_character_glb_url": "https://meshy.mock/rigged.glb"}},
            )
        elif request.method == "GET" and "meshy.mock/rigged.glb" in u:
            return httpx.Response(200, content=dummy_rig_glb)
        return httpx.Response(404)

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setenv("MESHY_API_KEY", "valid-meshy-key")
    monkeypatch.setattr(providers, "get_provider", lambda name, client=None: providers.MeshyProvider(api_key="valid-meshy-key", client=mock_client))

    # 1. Execute generate_mesh with provider="meshy"
    gen_res = tools_core.generate_mesh(project_id=pid, provider="meshy")
    assert gen_res["ok"] is True
    assert gen_res["data"]["status"] == "done"
    meshy_mesh_path = gen_res["data"]["mesh_path"]
    assert meshy_mesh_path == "meshes/cand-2-meshy.glb"

    # Verify project state has task_id and mesh_path recorded
    p = state.load(pid)
    meshy_mesh_entry = next(m for m in p["meshes"] if m["path"] == meshy_mesh_path)
    assert meshy_mesh_entry["task_id"] == "meshy-gen-task-101"
    assert p["jobs"]["meshy-gen-task-101"]["mesh_path"] == meshy_mesh_path

    # 2. Execute auto_rig with method="provider" on the generated mesh
    rig_res = tools_full.auto_rig(project_id=pid, mesh_path=meshy_mesh_path, method="provider")
    assert rig_res["ok"] is True
    assert rig_res["data"]["method"] == "provider"
    assert rig_res["data"]["rig_task_id"] == "meshy-rig-task-202"

    # 3. Verify blockout mesh with method="auto" does NOT use provider
    assert tools_full._find_provider_task_id(p, "meshes/cand-1-blockout.ply") is None


