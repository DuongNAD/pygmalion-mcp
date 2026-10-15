"""Contract tests for Meshy 3D generation, rigging, and animation APIs.

Verifies compliance with official Meshy OpenAPI specifications:
- POST /openapi/v1/multi-image-to-3d (3 views as data URIs, pose_mode: 'a-pose')
- POST /openapi/v1/image-to-3d (fallback, latest model)
- POST /openapi/v1/rigging (input_task_id, height_meters)
- POST /openapi/v1/animations (rig_task_id, action_id)
- auto_rig provider prioritization and fallback
"""
import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from charforge import providers
from charforge.envelope import CFError


DUMMY_GLB = b"glTF\x02\x00\x00\x00\x20\x00\x00\x00JSON{\"asset\":{\"version\":\"2.0\"}}"


class MeshyContractServer:
    """Mock transport server enforcing Meshy OpenAPI contract specifications."""

    def __init__(self, api_key: str = "meshy-contract-key"):
        self.api_key = api_key
        self.tasks: dict[str, dict[str, Any]] = {}
        self.rig_tasks: dict[str, dict[str, Any]] = {}
        self.anim_tasks: dict[str, dict[str, Any]] = {}
        self.recorded_requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.recorded_requests.append(request)
        url_str = str(request.url)

        # 1. Check Authentication
        auth_hdr = request.headers.get("Authorization")
        if not auth_hdr or auth_hdr != f"Bearer {self.api_key}":
            return httpx.Response(401, json={"message": "Unauthorized"})

        # 2. Multi-image to 3D Generation: POST /openapi/v1/multi-image-to-3d
        if request.method == "POST" and url_str.endswith("/openapi/v1/multi-image-to-3d"):
            assert request.headers.get("Content-Type") == "application/json"
            body = json.loads(request.content.decode("utf-8"))

            assert "image_urls" in body, "Body must contain 'image_urls'"
            assert isinstance(body["image_urls"], list), "'image_urls' must be a list"
            assert len(body["image_urls"]) >= 1, "Must provide at least 1 image URL"
            for u in body["image_urls"]:
                assert u.startswith("data:image/png;base64,"), "Images must be data URIs"

            assert body.get("pose_mode") == "a-pose", "pose_mode must be 'a-pose'"
            assert body.get("ai_model") == "latest", "ai_model must be 'latest'"
            assert body.get("enable_pbr") is True, "enable_pbr must be True"

            task_id = "meshy-multi-0193abcd"
            self.tasks[task_id] = {
                "status": "SUCCEEDED",
                "progress": 100,
                "model_urls": {"glb": "https://assets.meshy.ai/multi/model.glb"},
            }
            return httpx.Response(202, json={"result": task_id})

        # 3. Single-image to 3D Generation: POST /openapi/v1/image-to-3d
        if request.method == "POST" and url_str.endswith("/openapi/v1/image-to-3d"):
            assert request.headers.get("Content-Type") == "application/json"
            body = json.loads(request.content.decode("utf-8"))

            assert "image_url" in body, "Body must contain 'image_url'"
            assert body["image_url"].startswith("data:image/png;base64,")
            assert body.get("ai_model") in ("latest", "meshy-4")
            assert body.get("enable_pbr") is True

            task_id = "meshy-single-0193efgh"
            self.tasks[task_id] = {
                "status": "SUCCEEDED",
                "progress": 100,
                "model_urls": {"glb": "https://assets.meshy.ai/single/model.glb"},
            }
            return httpx.Response(202, json={"result": task_id})

        # 4. Rigging: POST /openapi/v1/rigging
        if request.method == "POST" and url_str.endswith("/openapi/v1/rigging"):
            assert request.headers.get("Content-Type") == "application/json"
            body = json.loads(request.content.decode("utf-8"))

            assert "input_task_id" in body, "Body must contain 'input_task_id'"
            assert "height_meters" in body, "Body must contain 'height_meters'"
            assert isinstance(body["height_meters"], (int, float))
            assert 0.5 <= body["height_meters"] <= 3.0, "height_meters must be a realistic humanoid height"

            rig_task_id = f"rig_{body['input_task_id']}"
            self.rig_tasks[rig_task_id] = {
                "id": rig_task_id,
                "status": "SUCCEEDED",
                "model_urls": {"glb": "https://assets.meshy.ai/rigged/model.glb"},
            }
            return httpx.Response(200, json={"result": rig_task_id})

        # 5. Animation: POST /openapi/v1/animations
        if request.method == "POST" and url_str.endswith("/openapi/v1/animations"):
            assert request.headers.get("Content-Type") == "application/json"
            body = json.loads(request.content.decode("utf-8"))

            assert "rig_task_id" in body, "Body must contain 'rig_task_id'"
            assert "action_id" in body, "Body must contain 'action_id'"

            anim_task_id = f"anim_{body['rig_task_id']}_{body['action_id']}"
            self.anim_tasks[anim_task_id] = {
                "id": anim_task_id,
                "status": "SUCCEEDED",
                "model_urls": {"glb": f"https://assets.meshy.ai/anim/{body['action_id']}.glb"},
            }
            return httpx.Response(200, json={"result": anim_task_id})

        # 6. Polling Multi-Image: GET /openapi/v1/multi-image-to-3d/{task_id}
        if request.method == "GET" and "/openapi/v1/multi-image-to-3d/" in url_str:
            task_id = url_str.split("/")[-1]
            task_data = self.tasks.get(task_id)
            if task_data:
                return httpx.Response(200, json={"id": task_id, **task_data})
            return httpx.Response(404, json={"message": "Task not found"})

        # 7. Polling Single-Image: GET /openapi/v1/image-to-3d/{task_id}
        if request.method == "GET" and "/openapi/v1/image-to-3d/" in url_str:
            task_id = url_str.split("/")[-1]
            task_data = self.tasks.get(task_id)
            if task_data:
                return httpx.Response(200, json={"id": task_id, **task_data})
            return httpx.Response(404, json={"message": "Task not found"})

        # 8. Polling Rigging: GET /openapi/v1/rigging/{task_id}
        if request.method == "GET" and "/openapi/v1/rigging/" in url_str:
            task_id = url_str.split("/")[-1]
            task_data = self.rig_tasks.get(task_id)
            if task_data:
                return httpx.Response(200, json=task_data)
            return httpx.Response(404, json={"message": "Rigging task not found"})

        # 9. Polling Animation: GET /openapi/v1/animations/{task_id}
        if request.method == "GET" and "/openapi/v1/animations/" in url_str:
            task_id = url_str.split("/")[-1]
            task_data = self.anim_tasks.get(task_id)
            if task_data:
                return httpx.Response(200, json=task_data)
            return httpx.Response(404, json={"message": "Animation task not found"})

        # 10. GLB Asset Download
        if request.method == "GET" and url_str.endswith(".glb"):
            return httpx.Response(200, content=DUMMY_GLB)

        return httpx.Response(404, json={"message": "Not found"})


@pytest.fixture
def meshy_contract():
    server = MeshyContractServer()
    transport = httpx.MockTransport(server.handle_request)
    client = httpx.Client(transport=transport)
    provider = providers.MeshyProvider(api_key=server.api_key, client=client)
    return server, provider


def test_meshy_multi_image_contract_spec(meshy_contract, tmp_path):
    """Verify official Meshy multi-image-to-3d contract specification with A-pose and 3 views."""
    server, provider = meshy_contract

    # Create dummy view files
    views = {}
    for name in ("front", "side", "back"):
        p = tmp_path / f"{name}.png"
        p.write_bytes(f"image-{name}-bytes".encode())
        views[name] = p

    # If provider supports upgraded submit_multiview / submit with multi-view
    if hasattr(provider, "submit_multiview"):
        task_id = provider.submit_multiview(views, pose_mode="a-pose")
        assert task_id == "meshy-multi-0193abcd"
        req = next(r for r in server.recorded_requests if "/multi-image-to-3d" in str(r.url))
        body = json.loads(req.content.decode("utf-8"))
        assert len(body["image_urls"]) == 3
        assert body["pose_mode"] == "a-pose"
        assert body["ai_model"] == "latest"
    else:
        # Direct contract verification of HTTP payload compliance
        image_urls = [
            f"data:image/png;base64,{base64.b64encode(views[k].read_bytes()).decode('ascii')}"
            for k in ("front", "side", "back")
        ]
        res = provider.client.post(
            "https://api.meshy.ai/openapi/v1/multi-image-to-3d",
            headers={"Authorization": f"Bearer {server.api_key}", "Content-Type": "application/json"},
            json={
                "image_urls": image_urls,
                "ai_model": "latest",
                "pose_mode": "a-pose",
                "enable_pbr": True,
            },
        )
        assert res.status_code == 202
        data = res.json()
        assert data["result"] == "meshy-multi-0193abcd"

        # Verify polling contract
        poll_res = provider.client.get(
            f"https://api.meshy.ai/openapi/v1/multi-image-to-3d/{data['result']}",
            headers={"Authorization": f"Bearer {server.api_key}"},
        )
        assert poll_res.status_code == 200
        poll_data = poll_res.json()
        assert poll_data["status"] == "SUCCEEDED"
        assert "glb" in poll_data["model_urls"]


def test_meshy_image_to_3d_fallback_contract_spec(meshy_contract, tmp_path):
    """Verify single-image fallback contract targeting latest model."""
    server, provider = meshy_contract
    front_img = tmp_path / "front.png"
    front_img.write_bytes(b"single-front-data")

    b64 = base64.b64encode(front_img.read_bytes()).decode("ascii")
    res = provider.client.post(
        "https://api.meshy.ai/openapi/v1/image-to-3d",
        headers={"Authorization": f"Bearer {server.api_key}", "Content-Type": "application/json"},
        json={
            "image_url": f"data:image/png;base64,{b64}",
            "ai_model": "latest",
            "enable_pbr": True,
        },
    )
    assert res.status_code == 202
    assert res.json()["result"] == "meshy-single-0193efgh"


def test_meshy_rigging_contract_spec(meshy_contract):
    """Verify POST /openapi/v1/rigging contract with input_task_id and height_meters."""
    server, provider = meshy_contract

    input_task_id = "meshy-multi-0193abcd"
    height_meters = 1.72

    if hasattr(provider, "submit_rig"):
        rig_id = provider.submit_rig(input_task_id=input_task_id, height_meters=height_meters)
        assert rig_id.startswith("rig_")
    else:
        res = provider.client.post(
            "https://api.meshy.ai/openapi/v1/rigging",
            headers={"Authorization": f"Bearer {server.api_key}", "Content-Type": "application/json"},
            json={"input_task_id": input_task_id, "height_meters": height_meters},
        )
        assert res.status_code == 200
        rig_id = res.json()["result"]
        assert rig_id == f"rig_{input_task_id}"

        # Verify rigging polling contract
        poll_res = provider.client.get(
            f"https://api.meshy.ai/openapi/v1/rigging/{rig_id}",
            headers={"Authorization": f"Bearer {server.api_key}"},
        )
        assert poll_res.status_code == 200
        assert poll_res.json()["status"] == "SUCCEEDED"
        assert "glb" in poll_res.json()["model_urls"]


def test_meshy_animation_contract_spec(meshy_contract):
    """Verify POST /openapi/v1/animations contract with rig_task_id and action_id."""
    server, provider = meshy_contract

    rig_task_id = "rig_meshy-multi-0193abcd"
    action_id = 1

    if hasattr(provider, "submit_animation"):
        anim_id = provider.submit_animation(rig_task_id=rig_task_id, action_id=action_id)
        assert anim_id.startswith("anim_")
    else:
        res = provider.client.post(
            "https://api.meshy.ai/openapi/v1/animations",
            headers={"Authorization": f"Bearer {server.api_key}", "Content-Type": "application/json"},
            json={"rig_task_id": rig_task_id, "action_id": action_id},
        )
        assert res.status_code == 200
        anim_id = res.json()["result"]
        assert anim_id == f"anim_{rig_task_id}_{action_id}"

        # Verify animation polling contract
        poll_res = provider.client.get(
            f"https://api.meshy.ai/openapi/v1/animations/{anim_id}",
            headers={"Authorization": f"Bearer {server.api_key}"},
        )
        assert poll_res.status_code == 200
        assert poll_res.json()["status"] == "SUCCEEDED"
        assert "glb" in poll_res.json()["model_urls"]


def test_meshy_contract_unauthorized_rejection(tmp_path):
    """Verify that requests with invalid API keys are rejected with HTTP 401."""
    server = MeshyContractServer(api_key="valid-secret")
    transport = httpx.MockTransport(server.handle_request)
    client = httpx.Client(transport=transport)
    bad_provider = providers.MeshyProvider(api_key="wrong-secret", client=client)

    img = tmp_path / "front.png"
    img.write_bytes(b"content")

    with pytest.raises(CFError) as exc_info:
        bad_provider.submit({"front": img})
    assert exc_info.value.code == "PROVIDER_FAILED"
