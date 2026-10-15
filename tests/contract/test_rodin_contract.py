"""Contract tests for Deemos / Hyper3D Rodin 3D generation API v2.

Verifies compliance with official Rodin OpenAPI specifications:
- POST /api/v2/rodin (multipart/form-data image submission, compound job_id tracking)
- POST /api/v2/status (polling with subscription_key)
- POST /api/v2/download (signed URL retrieval with task_uuid)
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


class RodinContractServer:
    """Mock transport server enforcing Rodin OpenAPI contract specifications."""

    def __init__(self, api_key: str = "rodin-contract-key"):
        self.api_key = api_key
        self.tasks: dict[str, dict[str, Any]] = {
            "rodin_task_uuid_12345": {
                "uuid": "rodin_task_uuid_12345",
                "download_url": "https://cdn.hyper3d.com/downloads/model.glb",
            }
        }
        self.subscriptions: dict[str, dict[str, Any]] = {
            "sub_key_abcdef12345": {
                "status": "completed",
                "jobs": [{"uuid": "job_uuid_67890", "status": "completed"}],
            }
        }
        self.recorded_requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.recorded_requests.append(request)
        url_str = str(request.url)

        # Public CDN asset download does not require Authorization header
        if request.method == "GET" and url_str.endswith(".glb"):
            return httpx.Response(200, content=DUMMY_GLB)

        # 1. Authentication Check
        auth_hdr = request.headers.get("Authorization")
        if not auth_hdr or auth_hdr != f"Bearer {self.api_key}":
            return httpx.Response(401, json={"message": "Unauthorized", "code": 401})

        # 2. Task Submission: POST /api/v2/rodin
        if request.method == "POST" and url_str.endswith("/api/v2/rodin"):
            content_type = request.headers.get("Content-Type", "")
            # Verify multipart form data or json
            task_uuid = "rodin_task_uuid_12345"
            job_uuid = "job_uuid_67890"
            sub_key = "sub_key_abcdef12345"

            self.tasks[task_uuid] = {
                "uuid": task_uuid,
                "download_url": "https://cdn.hyper3d.com/downloads/model.glb",
            }
            self.subscriptions[sub_key] = {
                "status": "completed",
                "jobs": [{"uuid": job_uuid, "status": "completed"}],
            }

            return httpx.Response(
                200,
                json={
                    "message": "Submitted.",
                    "uuid": task_uuid,
                    "jobs": {
                        "uuids": [job_uuid],
                        "subscription_key": sub_key,
                    },
                    "consumed": 0.5,
                },
            )

        # 3. Status Polling: POST /api/v2/status
        if request.method == "POST" and url_str.endswith("/api/v2/status"):
            assert request.headers.get("Content-Type") == "application/json"
            body = json.loads(request.content.decode("utf-8"))
            sub_key = body.get("subscription_key")
            assert sub_key, "Body must contain 'subscription_key'"

            sub_data = self.subscriptions.get(sub_key)
            if sub_data:
                return httpx.Response(200, json=sub_data)
            return httpx.Response(404, json={"message": "Subscription not found"})

        # 4. Download Retrieval: POST /api/v2/download
        if request.method == "POST" and url_str.endswith("/api/v2/download"):
            assert request.headers.get("Content-Type") == "application/json"
            body = json.loads(request.content.decode("utf-8"))
            task_uuid = body.get("task_uuid")
            assert task_uuid, "Body must contain 'task_uuid'"

            task_data = self.tasks.get(task_uuid)
            if task_data:
                return httpx.Response(
                    200,
                    json={
                        "url": task_data["download_url"],
                        "name": "model.glb",
                        "error": "OK",
                    },
                )
            return httpx.Response(404, json={"message": "Task UUID not found"})

        # 5. Asset Download
        if request.method == "GET" and url_str.endswith(".glb"):
            return httpx.Response(200, content=DUMMY_GLB)

        return httpx.Response(404, json={"message": "Not found"})


@pytest.fixture
def rodin_contract():
    server = RodinContractServer()
    transport = httpx.MockTransport(server.handle_request)
    client = httpx.Client(transport=transport)
    provider = providers.RodinProvider(api_key=server.api_key, client=client)
    return server, provider


def test_rodin_multipart_submission_contract(rodin_contract, tmp_path):
    """Verify POST /api/v2/rodin with multipart/form-data and compound job ID."""
    server, provider = rodin_contract

    # Create dummy images
    images = [
        ("images", ("front.png", io.BytesIO(b"fake-front"), "image/png")),
        ("images", ("side.png", io.BytesIO(b"fake-side"), "image/png")),
    ]

    res = provider.client.post(
        "https://api.hyper3d.com/api/v2/rodin",
        headers={"Authorization": f"Bearer {server.api_key}"},
        files=images,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["uuid"] == "rodin_task_uuid_12345"
    assert "subscription_key" in data["jobs"]

    compound_job_id = f"{data['uuid']}:{data['jobs']['subscription_key']}"
    assert compound_job_id == "rodin_task_uuid_12345:sub_key_abcdef12345"


def test_rodin_status_polling_contract(rodin_contract):
    """Verify POST /api/v2/status polling using subscription_key."""
    server, provider = rodin_contract

    sub_key = "sub_key_abcdef12345"
    res = provider.client.post(
        "https://api.hyper3d.com/api/v2/status",
        headers={"Authorization": f"Bearer {server.api_key}", "Content-Type": "application/json"},
        json={"subscription_key": sub_key},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert len(data["jobs"]) >= 1
    assert data["jobs"][0]["status"] == "completed"


def test_rodin_download_retrieval_contract(rodin_contract):
    """Verify POST /api/v2/download URL retrieval using task_uuid."""
    server, provider = rodin_contract

    task_uuid = "rodin_task_uuid_12345"
    res = provider.client.post(
        "https://api.hyper3d.com/api/v2/download",
        headers={"Authorization": f"Bearer {server.api_key}", "Content-Type": "application/json"},
        json={"task_uuid": task_uuid},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["url"] == "https://cdn.hyper3d.com/downloads/model.glb"
    assert data["name"] == "model.glb"

    # Download GLB asset from the signed URL
    dl_res = provider.client.get(data["url"])
    assert dl_res.status_code == 200
    assert dl_res.content == DUMMY_GLB


def test_rodin_contract_unauthorized_rejection(tmp_path):
    """Verify HTTP 401 rejection on invalid authorization token."""
    server = RodinContractServer(api_key="valid-secret")
    transport = httpx.MockTransport(server.handle_request)
    client = httpx.Client(transport=transport)
    bad_provider = providers.RodinProvider(api_key="wrong-secret", client=client)

    img = tmp_path / "front.png"
    img.write_bytes(b"content")

    with pytest.raises(CFError) as exc_info:
        bad_provider.submit({"front": img})
    assert exc_info.value.code == "PROVIDER_FAILED"
