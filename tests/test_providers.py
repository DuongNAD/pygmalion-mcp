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
    registered_tools = [t.name for t in server.mcp._tool_manager.list_tools()]
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

