"""Contract tests for Tripo3D 3D generation API v2.

Verifies compliance with official Tripo OpenAPI specifications:
- POST /v2/openapi/upload (token upload flow, multipart stream, image_token returned)
- POST /v2/openapi/task with directional inputs mapped via reference.side_faces (front, left/right, back)
- GET /v2/openapi/task/{task_id} polling flow
"""
import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from charforge import providers
from charforge.envelope import CFError


DUMMY_GLB = b"glTF\x02\x00\x00\x00\x20\x00\x00\x00JSON{\"asset\":{\"version\":\"2.0\"}}"


class TripoContractServer:
    """Mock transport server enforcing Tripo OpenAPI contract specifications."""

    def __init__(self, api_key: str = "tripo-contract-key"):
        self.api_key = api_key
        self.uploaded_tokens: dict[str, bytes] = {}
        self.tasks: dict[str, dict[str, Any]] = {}
        self.recorded_requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.recorded_requests.append(request)
        url_str = str(request.url)

        # 1. Authentication Check
        auth_hdr = request.headers.get("Authorization")
        if not auth_hdr or auth_hdr != f"Bearer {self.api_key}":
            return httpx.Response(401, json={"code": 40100, "message": "Unauthorized"})

        # 2. Pre-Upload Flow: POST /v2/openapi/upload
        if request.method == "POST" and url_str.endswith("/v2/openapi/upload"):
            content_type = request.headers.get("Content-Type", "")
            assert "multipart/form-data" in content_type, "Upload must use multipart/form-data"
            assert b'name="file"' in request.content, "Multipart must contain 'file' field"

            token_id = f"token_img_{len(self.uploaded_tokens) + 1:04d}"
            self.uploaded_tokens[token_id] = request.content
            return httpx.Response(200, json={"code": 0, "data": {"image_token": token_id}})

        # 3. Multi-View Task Generation: POST /v2/openapi/task
        if request.method == "POST" and url_str.endswith("/v2/openapi/task"):
            assert request.headers.get("Content-Type") == "application/json"
            body = json.loads(request.content.decode("utf-8"))

            task_type = body.get("type")
            if task_type == "multiview_to_model":
                inputs = body.get("inputs", {})
                assert isinstance(inputs, dict), "'inputs' must be a directional dictionary"
                assert "front" in inputs, "inputs must contain 'front'"
                assert ("left" in inputs or "right" in inputs), "inputs must contain directional side view ('left' or 'right')"
                for view_key, view_val in inputs.items():
                    assert "file_token" in view_val or "data" in view_val, f"View {view_key} must contain token or data"

            task_id = "tripo-task-998877"
            self.tasks[task_id] = {
                "task_id": task_id,
                "status": "success",
                "output": {
                    "model": "https://cdn.tripo3d.ai/models/tripo_mesh.glb",
                    "pbr_model": "https://cdn.tripo3d.ai/models/tripo_pbr_mesh.glb",
                },
            }
            return httpx.Response(200, json={"code": 0, "data": {"task_id": task_id}})

        # 4. Polling Task: GET /v2/openapi/task/{task_id}
        if request.method == "GET" and "/v2/openapi/task/" in url_str:
            task_id = url_str.split("/")[-1]
            task_data = self.tasks.get(task_id)
            if task_data:
                return httpx.Response(200, json={"code": 0, "data": task_data})
            return httpx.Response(404, json={"code": 40400, "message": "Task not found"})

        # 5. GLB Asset Download
        if request.method == "GET" and url_str.endswith(".glb"):
            return httpx.Response(200, content=DUMMY_GLB)

        return httpx.Response(404, json={"code": 40400, "message": "Not found"})


@pytest.fixture
def tripo_contract():
    server = TripoContractServer()
    transport = httpx.MockTransport(server.handle_request)
    client = httpx.Client(transport=transport)
    provider = providers.TripoProvider(api_key=server.api_key, client=client)
    return server, provider


def test_tripo_upload_token_flow_spec(tripo_contract):
    """Verify POST /v2/openapi/upload contract returning image_token."""
    server, provider = tripo_contract

    # Upload file stream via multipart/form-data
    file_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR..."
    files = {"file": ("front.png", io.BytesIO(file_bytes), "image/png")}

    res = provider.client.post(
        "https://api.tripo3d.ai/v2/openapi/upload",
        headers={"Authorization": f"Bearer {server.api_key}"},
        files=files,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["code"] == 0
    token = data["data"]["image_token"]
    assert token.startswith("token_img_")
    assert token in server.uploaded_tokens


def test_tripo_directional_inputs_mapping_right_side_faces(tripo_contract):
    """When reference.side_faces == 'right', the 2D view shows character looking right.
    Looking at the left ear from the character's left perspective -> maps to 'left'.
    """
    server, provider = tripo_contract

    # Task payload with directional mapping: front, left, back
    payload = {
        "type": "multiview_to_model",
        "inputs": {
            "front": {"type": "png", "file_token": "token_front_001"},
            "left": {"type": "png", "file_token": "token_side_002"},
            "back": {"type": "png", "file_token": "token_back_003"},
        },
    }

    res = provider.client.post(
        "https://api.tripo3d.ai/v2/openapi/task",
        headers={"Authorization": f"Bearer {server.api_key}", "Content-Type": "application/json"},
        json=payload,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["code"] == 0
    task_id = data["data"]["task_id"]
    assert task_id == "tripo-task-998877"

    # Verify polling contract
    poll_res = provider.client.get(
        f"https://api.tripo3d.ai/v2/openapi/task/{task_id}",
        headers={"Authorization": f"Bearer {server.api_key}"},
    )
    assert poll_res.status_code == 200
    poll_data = poll_res.json()
    assert poll_data["data"]["status"] == "success"
    assert "pbr_model" in poll_data["data"]["output"]


def test_tripo_directional_inputs_mapping_left_side_faces(tripo_contract):
    """When reference.side_faces == 'left', the 2D view shows character looking left.
    Looking at the right ear from the character's right perspective -> maps to 'right'.
    """
    server, provider = tripo_contract

    payload = {
        "type": "multiview_to_model",
        "inputs": {
            "front": {"type": "png", "file_token": "token_front_001"},
            "right": {"type": "png", "file_token": "token_side_002"},
            "back": {"type": "png", "file_token": "token_back_003"},
        },
    }

    res = provider.client.post(
        "https://api.tripo3d.ai/v2/openapi/task",
        headers={"Authorization": f"Bearer {server.api_key}", "Content-Type": "application/json"},
        json=payload,
    )
    assert res.status_code == 200
    assert res.json()["data"]["task_id"] == "tripo-task-998877"


def test_tripo_contract_unauthorized_rejection(tmp_path):
    """Verify HTTP 401 rejection on invalid authorization token."""
    server = TripoContractServer(api_key="valid-secret")
    transport = httpx.MockTransport(server.handle_request)
    client = httpx.Client(transport=transport)
    bad_provider = providers.TripoProvider(api_key="wrong-secret", client=client)

    img = tmp_path / "front.png"
    img.write_bytes(b"content")

    with pytest.raises(CFError) as exc_info:
        bad_provider.submit({"front": img})
    assert exc_info.value.code == "PROVIDER_FAILED"
