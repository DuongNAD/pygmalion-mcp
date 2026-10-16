"""Mesh integrity (render_views --integrity + score.integrity_report) on real Blender geometry."""
import pytest

from charforge import blender, config, score
from tests.bpy_fixtures import make_integrity_mesh

pytestmark = [pytest.mark.blender, pytest.mark.skipif(config.blender_exe() is None, reason="Blender not installed")]


def check(path, tmp_path, **kw):
    r = blender.run("render_views.py", {"mesh": path, "out": tmp_path / "r", "mode": "mask", "integrity": True, **kw})
    return score.integrity_report(r["integrity"])


def test_clean_body_is_one_sound_piece(tmp_path):
    rep = check(make_integrity_mesh(tmp_path / "clean.glb"), tmp_path)
    assert rep["ok"] and rep["components"] == 1  # UV-seam copies from the GLB export are welded
    assert rep["holes"] == 0 and rep["self_intersecting_faces"] == 0 and rep["issues"] == []


def test_holes_are_counted_but_do_not_fail(tmp_path):
    rep = check(make_integrity_mesh(tmp_path / "holes.glb", holes=True), tmp_path)
    assert rep["holes"] == 1 and rep["boundary_edges"] > 0 and rep["ok"]


def test_surface_folding_through_itself(tmp_path):
    rep = check(make_integrity_mesh(tmp_path / "fold.glb", intersect=True), tmp_path)
    assert rep["self_intersection_ratio"] > config.INTEGRITY["self_intersect_max"]
    assert not rep["ok"] and any("cut through" in i for i in rep["issues"])
    assert rep["detached_parts"] == 0  # the folded tube is sunk into the body: attached, not floating


def test_floating_debris_and_detached_part(tmp_path):
    rep = check(make_integrity_mesh(tmp_path / "debris.glb", debris=True), tmp_path, debris_share=0.02)
    assert rep["debris_parts"] == 1 and rep["detached_parts"] == 0 and rep["ok"]  # debris is reported, not fatal
    rep = check(make_integrity_mesh(tmp_path / "detached.glb", detached=True), tmp_path)
    assert rep["detached_parts"] == 1 and not rep["ok"] and any("detached part" in i for i in rep["issues"])
