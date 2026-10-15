"""Empirical Adversarial Test Suite for Milestone 3.

Adversarially challenges and stress-tests:
1. Provider error conditions (HTTP 401, 404, 500 across Meshy, Tripo, and Rodin).
2. Tripo directional orientation with missing/corrupted side_faces metadata.
3. Rodin compound job_id handling with malformed or missing subscription keys.
4. auto_rig under the full 12-permutation matrix of (method, MESHY_API_KEY, task_id).
5. Mesh attribution safety: confirming blockout-generated and external meshes are NEVER
   erroneously attributed to provider task IDs.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
import pytest
import httpx

from charforge import blender, config, providers, server, state, tools_full
from charforge.envelope import CFError


# ============================================================================
# Section 1: Provider HTTP Error Conditions (401, 404, 500)
# ============================================================================

class TestProviderHttpErrors:
    """Adversarially verify HTTP 401, 404, and 500 across all provider endpoints."""

    @pytest.mark.parametrize("status_code", [401, 404, 500])
    @pytest.mark.parametrize("endpoint_action", [
        "submit_multiview",
        "submit_single",
        "submit_rig",
        "submit_animation",
        "poll_rig",
        "poll_anim",
        "poll_gen",
        "download",
    ])
    def test_meshy_http_error_rejection(self, tmp_path, status_code, endpoint_action):
        """Meshy must cleanly reject 401, 404, 500 with CFError('PROVIDER_FAILED')."""
        fake_img = tmp_path / "front.png"
        fake_img.write_bytes(b"dummy")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, text=f"Error {status_code}")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        prov = providers.MeshyProvider(api_key="test-key", client=client)

        with pytest.raises(CFError) as exc_info:
            if endpoint_action == "submit_multiview":
                prov.submit_multiview({"front": fake_img, "side": fake_img})
            elif endpoint_action == "submit_single":
                prov.submit({"front": fake_img})
            elif endpoint_action == "submit_rig":
                prov.submit_rig(input_task_id="task-123")
            elif endpoint_action == "submit_animation":
                prov.submit_animation(rig_task_id="rig-123", action_id=1)
            elif endpoint_action == "poll_rig":
                prov.poll_rig("rig_123")
            elif endpoint_action == "poll_anim":
                prov.poll_animation("anim_123")
            elif endpoint_action == "poll_gen":
                prov.poll("gen_123")
            elif endpoint_action == "download":
                prov.download("https://meshy.mock/model.glb", tmp_path / "out.glb")

        assert exc_info.value.code == "PROVIDER_FAILED"
        assert str(status_code) in exc_info.value.message

    @pytest.mark.parametrize("status_code", [401, 404, 500])
    def test_tripo_http_error_rejection(self, tmp_path, status_code):
        """Tripo must cleanly reject 401, 404, 500 with CFError('PROVIDER_FAILED')."""
        fake_img = tmp_path / "front.png"
        fake_img.write_bytes(b"dummy")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, text=f"Tripo Error {status_code}")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        prov = providers.TripoProvider(api_key="test-key", client=client)

        # 1. upload_image
        with pytest.raises(CFError) as exc_info:
            prov.upload_image(b"fake-bytes", "test.png")
        assert exc_info.value.code == "PROVIDER_FAILED"

        # 2. submit_multiview
        with pytest.raises(CFError) as exc_info:
            prov.submit_multiview({"front": "token1", "left": "token2"})
        assert exc_info.value.code == "PROVIDER_FAILED"

        # 3. submit (when upload returns error)
        # Note: Tripo submit catches 404 on upload as legacy mock, so for 404 on upload,
        # it falls back to task submit which also returns 404!
        with pytest.raises(CFError) as exc_info:
            prov.submit({"front": fake_img})
        assert exc_info.value.code == "PROVIDER_FAILED"

        # 4. poll
        with pytest.raises(CFError) as exc_info:
            prov.poll("task-123")
        assert exc_info.value.code == "PROVIDER_FAILED"

        # 5. download
        with pytest.raises(CFError) as exc_info:
            prov.download("https://tripo.mock/model.glb", tmp_path / "out.glb")
        assert exc_info.value.code == "PROVIDER_FAILED"

    @pytest.mark.parametrize("status_code", [401, 404, 500])
    def test_rodin_http_error_rejection(self, tmp_path, status_code):
        """Rodin must cleanly reject 401, 404, 500 with CFError('PROVIDER_FAILED')."""
        fake_img = tmp_path / "front.png"
        fake_img.write_bytes(b"dummy")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, text=f"Rodin Error {status_code}")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        prov = providers.RodinProvider(api_key="test-key", client=client)

        # 1. submit_multipart
        with pytest.raises(CFError) as exc_info:
            prov.submit_multipart([("images", ("front.png", io.BytesIO(b"data"), "image/png"))])
        assert exc_info.value.code == "PROVIDER_FAILED"

        # 2. submit
        with pytest.raises(CFError) as exc_info:
            prov.submit({"front": fake_img})
        assert exc_info.value.code == "PROVIDER_FAILED"

        # 3. get_download_url
        with pytest.raises(CFError) as exc_info:
            prov.get_download_url("task-uuid-123")
        assert exc_info.value.code == "PROVIDER_FAILED"

        # 4. poll
        with pytest.raises(CFError) as exc_info:
            prov.poll("task-uuid-123:sub-key-456")
        assert exc_info.value.code == "PROVIDER_FAILED"

        # 5. download
        with pytest.raises(CFError) as exc_info:
            prov.download("https://rodin.mock/model.glb", tmp_path / "out.glb")
        assert exc_info.value.code == "PROVIDER_FAILED"


# ============================================================================
# Section 2: Tripo Directional Orientation & Corrupted Metadata
# ============================================================================

class TestTripoDirectionalMappingAndCorruption:
    """Stress test Tripo directional orientation with missing/corrupted side_faces metadata."""

    def _setup_tripo_mock(self, recorded_payloads: list):
        token_counter = [0]

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if request.method == "POST" and url_str.endswith("/upload"):
                token_counter[0] += 1
                tok = f"tok-{token_counter[0]}"
                return httpx.Response(200, json={"code": 0, "data": {"image_token": tok}})
            elif request.method == "POST" and url_str.endswith("/task"):
                body = json.loads(request.content.decode("utf-8"))
                recorded_payloads.append(body)
                return httpx.Response(200, json={"code": 0, "data": {"task_id": "tripo-task-dir"}})
            return httpx.Response(404)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        return providers.TripoProvider(api_key="tripo-key", client=client)

    def test_missing_reference_defaults_to_left(self, tmp_path):
        """When reference is None and no reference.json exists, directional mapping safely defaults to 'left'."""
        recorded = []
        prov = self._setup_tripo_mock(recorded)

        views = {
            "front": tmp_path / "front.png",
            "side": tmp_path / "side.png",
        }
        for p in views.values():
            p.write_bytes(b"image")

        prov.submit(views, reference=None)
        assert len(recorded) == 1
        inputs = recorded[0]["inputs"]
        assert "front" in inputs
        assert "left" in inputs
        assert "right" not in inputs

    def test_empty_reference_dict_defaults_to_left(self, tmp_path):
        """When reference is empty dict {}, directional mapping defaults to 'left'."""
        recorded = []
        prov = self._setup_tripo_mock(recorded)

        views = {"front": tmp_path / "front.png", "side": tmp_path / "side.png"}
        for p in views.values():
            p.write_bytes(b"image")

        prov.submit(views, reference={})
        assert len(recorded) == 1
        inputs = recorded[0]["inputs"]
        assert "left" in inputs

    def test_side_faces_right_maps_to_left(self, tmp_path):
        """reference.side_faces == 'right' (looking right -> camera sees left profile) maps to 'left'."""
        recorded = []
        prov = self._setup_tripo_mock(recorded)

        views = {"front": tmp_path / "front.png", "side": tmp_path / "side.png"}
        for p in views.values():
            p.write_bytes(b"image")

        prov.submit(views, reference={"side_faces": "right"})
        assert len(recorded) == 1
        inputs = recorded[0]["inputs"]
        assert "left" in inputs
        assert "right" not in inputs

    def test_side_faces_left_maps_to_right(self, tmp_path):
        """reference.side_faces == 'left' (looking left -> camera sees right profile) maps to 'right'."""
        recorded = []
        prov = self._setup_tripo_mock(recorded)

        views = {"front": tmp_path / "front.png", "side": tmp_path / "side.png"}
        for p in views.values():
            p.write_bytes(b"image")

        prov.submit(views, reference={"side_faces": "left"})
        assert len(recorded) == 1
        inputs = recorded[0]["inputs"]
        assert "right" in inputs
        assert "left" not in inputs

    @pytest.mark.parametrize("corrupt_val", [
        "diagonal",
        "front",
        "garbage_text",
        "",
        12345,
        None,
        ["left"],
        {"bad": True},
    ])
    def test_corrupted_side_faces_values_fallback_safely(self, tmp_path, corrupt_val):
        """Non-standard or corrupted side_faces values must not crash and fallback to 'left'."""
        recorded = []
        prov = self._setup_tripo_mock(recorded)

        views = {"front": tmp_path / "front.png", "side": tmp_path / "side.png"}
        for p in views.values():
            p.write_bytes(b"image")

        prov.submit(views, reference={"side_faces": corrupt_val})
        assert len(recorded) == 1
        inputs = recorded[0]["inputs"]
        assert "left" in inputs

    def test_corrupted_reference_json_on_disk_handled_gracefully(self, tmp_path):
        """When reference.json on disk contains truncated or corrupted JSON, it does not crash."""
        views_dir = tmp_path / "views"
        views_dir.mkdir(parents=True, exist_ok=True)
        (views_dir / "reference.json").write_text("{corrupted JSON truncated...")

        views = {
            "front": views_dir / "front.png",
            "side": views_dir / "side.png",
        }
        for p in views.values():
            p.write_bytes(b"image")

        recorded = []
        prov = self._setup_tripo_mock(recorded)

        prov.submit(views, reference=None)
        assert len(recorded) == 1
        inputs = recorded[0]["inputs"]
        assert "left" in inputs

    def test_explicit_directional_views_preserved(self, tmp_path):
        """If views already have 'left' or 'right' keys, they are preserved."""
        recorded = []
        prov = self._setup_tripo_mock(recorded)

        views = {
            "front": tmp_path / "front.png",
            "left": tmp_path / "left.png",
            "back": tmp_path / "back.png",
        }
        for p in views.values():
            p.write_bytes(b"image")

        prov.submit(views)
        assert len(recorded) == 1
        inputs = recorded[0]["inputs"]
        assert "front" in inputs
        assert "left" in inputs
        assert "back" in inputs


# ============================================================================
# Section 3: Rodin Compound job_id & Edge Cases
# ============================================================================

class TestRodinCompoundJobIdAndEdgeCases:
    """Stress test Rodin compound job_id handling with malformed or missing keys."""

    def test_valid_compound_job_id_split(self):
        """Standard task_uuid:sub_key is split and queries /status with subscription_key."""
        recorded_requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            recorded_requests.append(request)
            url_str = str(request.url)
            if request.method == "POST" and url_str.endswith("/status"):
                body = json.loads(request.content.decode("utf-8"))
                assert body["subscription_key"] == "sub-999"
                return httpx.Response(200, json={"status": "completed", "download_url": "https://rodin.mock/done.glb"})
            return httpx.Response(404)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        prov = providers.RodinProvider(api_key="rodin-key", client=client)

        res = prov.poll("uuid-123:sub-999")
        assert res["status"] == "done"
        assert res["url"] == "https://rodin.mock/done.glb"

    def test_missing_subscription_key_falls_back_to_legacy(self):
        """When job_id has no colon (e.g. legacy uuid only), fallback queries legacy GET /rodin/{uuid}."""
        called_urls = []

        def handler(request: httpx.Request) -> httpx.Response:
            called_urls.append(str(request.url))
            if request.method == "POST" and "status" in str(request.url):
                # /status returns 404 because "uuid-only" is not a valid subscription key
                return httpx.Response(404, json={"message": "Not found"})
            elif request.method == "GET" and "uuid-only" in str(request.url):
                return httpx.Response(200, json={"status": "completed", "download_url": "https://rodin.mock/legacy.glb"})
            return httpx.Response(404)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        prov = providers.RodinProvider(api_key="rodin-key", client=client)

        res = prov.poll("uuid-only")
        assert res["status"] == "done"
        assert res["url"] == "https://rodin.mock/legacy.glb"

    def test_compound_job_id_with_multiple_colons(self):
        """When job_id contains multiple colons ('uuid:sub:extra:data'), split(':', 1) prevents ValueError."""
        status_payloads = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and "status" in str(request.url):
                body = json.loads(request.content.decode("utf-8"))
                status_payloads.append(body)
                return httpx.Response(200, json={"status": "running"})
            return httpx.Response(404)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        prov = providers.RodinProvider(api_key="rodin-key", client=client)

        res = prov.poll("uuid-part:sub-key:extra:colons")
        assert res["status"] == "running"
        assert status_payloads[0]["subscription_key"] == "sub-key:extra:colons"

    def test_submit_multipart_when_jobs_field_is_malformed_or_missing(self):
        """When Rodin server returns non-dict jobs (None, list, string), does not crash with AttributeError."""
        for malformed_jobs in [None, [], "not-a-dict", {}]:
            def handler(request: httpx.Request, mj=malformed_jobs) -> httpx.Response:
                return httpx.Response(200, json={"uuid": "rodin-uuid-ok", "jobs": mj})

            client = httpx.Client(transport=httpx.MockTransport(handler))
            prov = providers.RodinProvider(api_key="rodin-key", client=client)

            job_id = prov.submit_multipart([("images", ("f.png", io.BytesIO(b"d"), "image/png"))])
            # Should safely return single uuid without crashing
            assert job_id == "rodin-uuid-ok"

    def test_submit_multipart_missing_uuid_raises_cferror(self):
        """When Rodin server returns response without uuid or task_id, raises PROVIDER_FAILED."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"message": "ok", "error": "none"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        prov = providers.RodinProvider(api_key="rodin-key", client=client)

        with pytest.raises(CFError) as exc_info:
            prov.submit_multipart([("images", ("f.png", io.BytesIO(b"d"), "image/png"))])
        assert exc_info.value.code == "PROVIDER_FAILED"


# ============================================================================
# Section 4: auto_rig 12-Permutation Matrix & Edge Cases
# ============================================================================

class TestAutoRigPermutationMatrix:
    """Systematically test all 12 permutations of (method, MESHY_API_KEY, task_id)."""

    def _setup_project_and_mesh(self, tmp_path, monkeypatch, has_task_id: bool):
        monkeypatch.setenv("CHARFORGE_LEVEL", "3")
        server._register_levels()

        sheet = tmp_path / "sheet.png"
        sheet.write_bytes(b"dummy")

        cr = server.project_create(sheet_path=str(sheet), name="MatrixTest", height_cm=175.0)
        pid = cr["project_id"]
        pdir = state.project_dir(pid)

        mesh_rel = "meshes/candidate.glb"
        mesh_full = pdir / mesh_rel
        mesh_full.parent.mkdir(parents=True, exist_ok=True)
        mesh_full.write_bytes(b"dummy-glb")

        p = state.load(pid)
        if has_task_id:
            p["meshes"].append({"path": mesh_rel, "source": "meshy", "task_id": "meshy-upstream-123"})
            p["jobs"]["meshy-upstream-123"] = {"provider": "meshy", "status": "done", "mesh_path": mesh_rel}
        else:
            p["meshes"].append({"path": mesh_rel, "source": "blockout"})
        state.save(p)

        return pid, mesh_rel, pdir

    @pytest.mark.parametrize("method,has_key,has_task,expected_behavior", [
        # method="auto" permutations (4)
        ("auto", True, True, "provider"),
        ("auto", True, False, "blender"),
        ("auto", False, True, "blender"),
        ("auto", False, False, "blender"),
        # method="blender" permutations (4)
        ("blender", True, True, "blender"),
        ("blender", True, False, "blender"),
        ("blender", False, True, "blender"),
        ("blender", False, False, "blender"),
        # method="provider" permutations (4)
        ("provider", True, True, "provider"),
        ("provider", True, False, "error_no_task"),
        ("provider", False, True, "error_no_key"),
        ("provider", False, False, "error_no_key"),
    ])
    def test_auto_rig_12_permutations(
        self, tmp_path, monkeypatch, method, has_key, has_task, expected_behavior
    ):
        """Verify exact behavior for all 12 permutations of auto_rig configuration."""
        pid, mesh_rel, pdir = self._setup_project_and_mesh(tmp_path, monkeypatch, has_task_id=has_task)

        # Set or unset MESHY_API_KEY
        if has_key:
            monkeypatch.setenv("MESHY_API_KEY", "test-meshy-key")
        else:
            monkeypatch.delenv("MESHY_API_KEY", raising=False)

        # Mock Meshy provider
        dummy_rig_glb = b"glTF\x02\x00\x00\x00\x18\x00\x00\x00JSON{\"rig\":true}"
        def meshy_handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "/rigging" in url_str and request.method == "POST":
                return httpx.Response(200, json={"result": "rig_task_999"})
            elif "/rigging/rig_task_999" in url_str:
                return httpx.Response(200, json={"status": "SUCCEEDED", "model_urls": {"glb": "https://meshy.mock/rig.glb"}})
            elif "meshy.mock/rig.glb" in url_str:
                return httpx.Response(200, content=dummy_rig_glb)
            return httpx.Response(404)

        mock_client = httpx.Client(transport=httpx.MockTransport(meshy_handler))
        monkeypatch.setattr(
            providers,
            "get_provider",
            lambda name, client=None: providers.MeshyProvider(api_key="test-meshy-key", client=mock_client),
        )

        # Mock Blender headless rig
        blender_called = []
        def mock_blender_run(script_name, args):
            blender_called.append((script_name, args))
            out_p = Path(args["out"])
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_bytes(b"blender-rigged-glb")
            return {"ok": True, "bones": 22, "unweighted_ratio": 0.0, "mean_arm_disp": 0.15, "max_thigh_disp": 0.001}

        monkeypatch.setattr(blender, "run", mock_blender_run)

        # Execute auto_rig
        if expected_behavior == "provider":
            res = tools_full.auto_rig(project_id=pid, mesh_path=mesh_rel, method=method)
            assert res["ok"] is True
            assert res["data"]["method"] == "provider"
            assert res["data"]["rig_task_id"] == "rig_task_999"
            assert len(blender_called) == 0
        elif expected_behavior == "blender":
            res = tools_full.auto_rig(project_id=pid, mesh_path=mesh_rel, method=method)
            assert res["ok"] is True
            assert res["data"]["bones"] == 22
            assert len(blender_called) == 1
        elif expected_behavior == "error_no_key":
            res = tools_full.auto_rig(project_id=pid, mesh_path=mesh_rel, method=method)
            assert res["ok"] is False
            assert res["error"]["code"] == "NO_PROVIDER"
            assert "No provider API key" in res["error"]["message"]
        elif expected_behavior == "error_no_task":
            res = tools_full.auto_rig(project_id=pid, mesh_path=mesh_rel, method=method)
            assert res["ok"] is False
            assert res["error"]["code"] == "NO_PROVIDER"
            assert "No upstream provider task ID" in res["error"]["message"]

    def test_height_cm_clamping_and_propagation(self, tmp_path, monkeypatch):
        """Verify height_cm is converted to meters and clamped between 0.5m and 3.0m."""
        pid, mesh_rel, pdir = self._setup_project_and_mesh(tmp_path, monkeypatch, has_task_id=True)
        monkeypatch.setenv("MESHY_API_KEY", "test-key")

        submitted_heights = []
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and "/rigging" in str(request.url):
                body = json.loads(request.content.decode("utf-8"))
                submitted_heights.append(body["height_meters"])
                return httpx.Response(200, json={"result": "rig_task_h"})
            elif request.method == "GET" and "/rigging/rig_task_h" in str(request.url):
                return httpx.Response(200, json={"status": "SUCCEEDED", "model_urls": {"glb": "https://meshy.mock/r.glb"}})
            elif "meshy.mock/r.glb" in str(request.url):
                return httpx.Response(200, content=b"glb")
            return httpx.Response(404)

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            providers,
            "get_provider",
            lambda name, client=None: providers.MeshyProvider(api_key="test-key", client=mock_client),
        )

        # Test height 175cm -> 1.75m
        p = state.load(pid)
        p["height_cm"] = 175.0
        state.save(p)
        tools_full.auto_rig(project_id=pid, mesh_path=mesh_rel, method="provider")
        assert submitted_heights[-1] == 1.75

        # Test height 30cm -> clamped to 0.5m
        p["height_cm"] = 30.0
        state.save(p)
        tools_full.auto_rig(project_id=pid, mesh_path=mesh_rel, method="provider")
        assert submitted_heights[-1] == 0.5

        # Test height 500cm -> clamped to 3.0m
        p["height_cm"] = 500.0
        state.save(p)
        tools_full.auto_rig(project_id=pid, mesh_path=mesh_rel, method="provider")
        assert submitted_heights[-1] == 3.0

    def test_meshy_rig_poll_failed_raises_provider_failed(self, tmp_path, monkeypatch):
        """When Meshy rigging fails during polling, raises CFError('PROVIDER_FAILED')."""
        pid, mesh_rel, pdir = self._setup_project_and_mesh(tmp_path, monkeypatch, has_task_id=True)
        monkeypatch.setenv("MESHY_API_KEY", "test-key")

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and "/rigging" in str(request.url):
                return httpx.Response(200, json={"result": "rig_task_fail"})
            elif request.method == "GET" and "/rigging/rig_task_fail" in str(request.url):
                return httpx.Response(200, json={"status": "FAILED", "task_error": {"message": "Mesh topology non-manifold"}})
            return httpx.Response(404)

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            providers,
            "get_provider",
            lambda name, client=None: providers.MeshyProvider(api_key="test-key", client=mock_client),
        )

        res = tools_full.auto_rig(project_id=pid, mesh_path=mesh_rel, method="provider")
        assert res["ok"] is False
        assert res["error"]["code"] == "PROVIDER_FAILED"
        assert "Mesh topology non-manifold" in res["error"]["message"]


# ============================================================================
# Section 5: Mesh Attribution Safety (Blockout & External Meshes)
# ============================================================================

class TestMeshAttributionSafety:
    """Confirm that blockout-generated or external meshes are NEVER erroneously attributed to provider task IDs."""

    def test_blockout_mesh_never_attributed_to_provider_task_id(self, tmp_path, monkeypatch):
        """A blockout mesh with source='blockout' must return None from _find_provider_task_id,
        even if Meshy jobs exist in project state.
        """
        p = {
            "id": "attribution_test",
            "meshes": [
                {"path": "meshes/cand-1-blockout.glb", "source": "blockout"},
            ],
            "jobs": {
                "meshy-unrelated-task-123": {"provider": "meshy", "status": "done", "mesh_path": "meshes/other.glb"},
            },
        }

        task_id = tools_full._find_provider_task_id(p, "meshes/cand-1-blockout.glb")
        assert task_id is None, "Blockout mesh must NEVER be attributed to a provider task ID"

    def test_external_mesh_never_attributed_to_provider_task_id(self, tmp_path):
        """An external mesh with source='external' must return None from _find_provider_task_id,
        even if Meshy jobs exist in project state.
        """
        p = {
            "id": "attribution_test",
            "meshes": [
                {"path": "meshes/imported_hero.glb", "source": "external"},
            ],
            "jobs": {
                "meshy-unrelated-task-456": {"provider": "meshy", "status": "done"},
            },
        }

        task_id = tools_full._find_provider_task_id(p, "meshes/imported_hero.glb")
        assert task_id is None, "External mesh must NEVER be attributed to a provider task ID"

    def test_blockout_mesh_name_containing_job_id_substring_not_misattributed(self):
        """Even if blockout mesh path contains the provider job ID substring,
        source='blockout' takes precedence and returns None.
        """
        p = {
            "id": "attribution_test",
            "meshes": [
                {"path": "meshes/meshy-task-007-blockout.glb", "source": "blockout"},
            ],
            "jobs": {
                "meshy-task-007": {"provider": "meshy", "status": "done"},
            },
        }

        task_id = tools_full._find_provider_task_id(p, "meshes/meshy-task-007-blockout.glb")
        assert task_id is None, "Blockout mesh with matching job ID in path must not be attributed"

    def test_external_mesh_name_containing_job_id_substring_not_misattributed(self):
        """Even if external mesh path contains the provider job ID substring,
        source='external' takes precedence and returns None.
        """
        p = {
            "id": "attribution_test",
            "meshes": [
                {"path": "meshes/hero-meshy-task-007.glb", "source": "external"},
            ],
            "jobs": {
                "meshy-task-007": {"provider": "meshy", "status": "done"},
            },
        }

        task_id = tools_full._find_provider_task_id(p, "meshes/hero-meshy-task-007.glb")
        assert task_id is None, "External mesh with matching job ID in path must not be attributed"

    def test_auto_rig_on_blockout_mesh_with_meshy_key_falls_back_to_blender(self, tmp_path, monkeypatch):
        """When auto_rig(method='auto') is called on a blockout mesh with MESHY_API_KEY set,
        it must cleanly fall back to Blender and NEVER call Meshy provider rigging.
        """
        monkeypatch.setenv("CHARFORGE_LEVEL", "3")
        monkeypatch.setenv("MESHY_API_KEY", "active-meshy-key")
        server._register_levels()

        sheet = tmp_path / "sheet.png"
        sheet.write_bytes(b"dummy")
        cr = server.project_create(sheet_path=str(sheet), name="BlockoutAttribution", height_cm=170.0)
        pid = cr["project_id"]
        pdir = state.project_dir(pid)

        # Register blockout mesh
        b_mesh = "meshes/cand-1-blockout.glb"
        (pdir / b_mesh).parent.mkdir(parents=True, exist_ok=True)
        (pdir / b_mesh).write_bytes(b"blockout-mesh-data")

        p = state.load(pid)
        p["meshes"].append({"path": b_mesh, "source": "blockout"})
        # Add an active meshy job to project state to tempt false attribution
        p["jobs"]["meshy-abandoned-job-888"] = {"provider": "meshy", "status": "done"}
        state.save(p)

        # Spy on Meshy provider
        meshy_called = []
        def handler(request: httpx.Request) -> httpx.Response:
            meshy_called.append(str(request.url))
            return httpx.Response(200, json={"result": "fail_should_not_call"})

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            providers,
            "get_provider",
            lambda name, client=None: providers.MeshyProvider(api_key="active-meshy-key", client=mock_client),
        )

        # Spy on Blender
        blender_called = []
        def mock_blender_run(script_name, args):
            blender_called.append(script_name)
            out_p = Path(args["out"])
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_bytes(b"blender-rigged")
            return {"ok": True, "bones": 22, "unweighted_ratio": 0.0}

        monkeypatch.setattr(blender, "run", mock_blender_run)

        # Call auto_rig with method="auto"
        res = tools_full.auto_rig(project_id=pid, mesh_path=b_mesh, method="auto")
        assert res["ok"] is True
        assert len(meshy_called) == 0, "Provider rigging must NEVER be called for blockout mesh"
        assert len(blender_called) == 1, "Blender rigging must be used as fallback"

    def test_auto_rig_on_external_mesh_with_meshy_key_falls_back_to_blender(self, tmp_path, monkeypatch):
        """When auto_rig(method='auto') is called on an external mesh with MESHY_API_KEY set,
        it must cleanly fall back to Blender and NEVER call Meshy provider rigging.
        """
        monkeypatch.setenv("CHARFORGE_LEVEL", "3")
        monkeypatch.setenv("MESHY_API_KEY", "active-meshy-key")
        server._register_levels()

        sheet = tmp_path / "sheet.png"
        sheet.write_bytes(b"dummy")
        cr = server.project_create(sheet_path=str(sheet), name="ExternalAttribution", height_cm=170.0)
        pid = cr["project_id"]
        pdir = state.project_dir(pid)

        # Register external mesh
        ext_mesh = "meshes/character_imported.glb"
        (pdir / ext_mesh).parent.mkdir(parents=True, exist_ok=True)
        (pdir / ext_mesh).write_bytes(b"imported-mesh-data")

        p = state.load(pid)
        p["meshes"].append({"path": ext_mesh, "source": "external"})
        p["jobs"]["meshy-stale-job-999"] = {"provider": "meshy", "status": "done"}
        state.save(p)

        meshy_called = []
        def handler(request: httpx.Request) -> httpx.Response:
            meshy_called.append(str(request.url))
            return httpx.Response(200, json={"result": "fail"})

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            providers,
            "get_provider",
            lambda name, client=None: providers.MeshyProvider(api_key="active-meshy-key", client=mock_client),
        )

        blender_called = []
        def mock_blender_run(script_name, args):
            blender_called.append(script_name)
            out_p = Path(args["out"])
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_bytes(b"blender-rigged")
            return {"ok": True, "bones": 22, "unweighted_ratio": 0.0}

        monkeypatch.setattr(blender, "run", mock_blender_run)

        res = tools_full.auto_rig(project_id=pid, mesh_path=ext_mesh, method="auto")
        assert res["ok"] is True
        assert len(meshy_called) == 0, "Provider rigging must NEVER be called for external mesh"
        assert len(blender_called) == 1, "Blender rigging must be used as fallback"

    def test_fitted_blockout_mesh_erroneously_attributed_to_meshy_job(self):
        """VULNERABILITY FINDING: When a blockout mesh undergoes mesh_fit, its source
        becomes 'fit'. If a meshy job exists in p['jobs'], _find_provider_task_id Step 4
        checks `if meshes[0].get('source') != 'blockout'`, which evaluates True for 'fit'!
        This erroneously attributes the blockout-derived mesh to the Meshy job ID.
        """
        p = {
            "id": "blockout_fit_test",
            "meshes": [
                {"path": "meshes/cand-1-blockout-fit.glb", "source": "fit"},
            ],
            "jobs": {
                "meshy-abandoned-task-999": {"provider": "meshy", "status": "done"},
            },
        }

        task_id = tools_full._find_provider_task_id(p, "meshes/cand-1-blockout-fit.glb")
        # Specification / Contract: blockout-derived meshes must NEVER be attributed to provider task IDs
        assert task_id is None, f"Blockout-derived mesh (source='fit') was erroneously attributed to {task_id!r}"


# ============================================================================
# Section 6: Provider Response Null-Safety Vulnerabilities
# ============================================================================

class TestProviderNullSafetyVulnerabilities:
    """Empirically test provider robustness against null values in API payloads."""

    def test_tripo_null_data_response_crashes_with_attribute_error(self):
        """VULNERABILITY FINDING: When Tripo returns code != 0 with data: null,
        upload_image and poll crash with unhandled AttributeError instead of CFError.
        """
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"code": 20001, "message": "Invalid token", "data": None})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        prov = providers.TripoProvider(api_key="tripo-key", client=client)

        # upload_image: data is None -> data.get("image_token") raises AttributeError
        with pytest.raises(CFError) as exc_info:
            prov.upload_image(b"dummy", "img.png")
        assert exc_info.value.code == "PROVIDER_FAILED"

    def test_meshy_null_task_error_crashes_with_attribute_error(self):
        """VULNERABILITY FINDING: When Meshy returns status FAILED with task_error: null,
        data.get('task_error', {}).get('message') crashes with AttributeError because
        task_error key is present with None value.
        """
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "FAILED", "task_error": None, "message": "Task failed"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        prov = providers.MeshyProvider(api_key="meshy-key", client=client)

        res = prov.poll_rig("rig_test_123")
        assert res["status"] == "failed"

