import json
import pytest
from pathlib import Path

from charforge import blender, config, server, state, tools_core, tools_full


@pytest.fixture
def l3_setup(monkeypatch, tmp_home):
    monkeypatch.setenv("CHARFORGE_LEVEL", "3")


@pytest.mark.blender
def test_auto_rig_mixamo_and_pose(l3_setup, sheet_path):
    # 1. Create project, split sheet, generate blockout, score
    cr = server.project_create(sheet_path=sheet_path, name="RigTester", height_cm=165.0)
    pid = cr["project_id"]
    sp = server.sheet_split(project_id=pid)
    assert sp["ok"]

    gm = tools_core.generate_mesh(project_id=pid, provider="blockout")
    assert gm["ok"]
    cand_path = gm["data"]["mesh_path"]

    sc = server.score_mesh(project_id=pid, mesh_path=cand_path)
    assert sc["ok"]

    opt = tools_core.mesh_optimize(project_id=pid, mesh_path=cand_path, target_tris=15000)
    assert opt["ok"]

    exp = tools_core.mesh_export(project_id=pid, mesh_path=opt["data"]["mesh_path"], format="glb")
    assert exp["ok"]
    glb_path = exp["data"]["files"]["glb"]

    # 2. Call auto_rig
    rig_res = tools_full.auto_rig(project_id=pid, mesh_path=glb_path, method="blender")
    assert rig_res["ok"]
    assert rig_res["data"]["bones"] == 22
    assert rig_res["data"]["unweighted_ratio"] < 0.01
    assert rig_res["next_action"] is None

    p = state.load(pid)
    assert p["stage"] == "rigged"
    assert p["outputs"]["rigged"] == f"export/{pid}-rigged.glb"

    pdir = state.project_dir(pid)
    rigged_glb = pdir / p["outputs"]["rigged"]
    assert rigged_glb.is_file()
    assert rigged_glb.stat().st_size > 0

    # 3. Verify re-importing into Blender has armature and skin
    inspect_code = f"""
import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.import_scene.gltf(filepath={str(rigged_glb)!r})
arms = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
assert len(arms) > 0, "No armature object found after re-import"
bones = arms[0].data.bones
assert len(bones) == 22, f"Expected 22 bones, got {{len(bones)}}"
assert "LeftArm" in [b.name for b in bones]
"""
    exe = config.blender_exe()
    import subprocess
    proc = subprocess.run([exe, "-b", "--factory-startup", "--python-exit-code", "1", "--python-expr", inspect_code], capture_output=True, text=True)
    assert proc.returncode == 0, f"Blender inspect failed:\n{proc.stdout}\n{proc.stderr}"


def test_auto_rig_missing_mesh(l3_setup, sheet_path):
    cr = server.project_create(sheet_path=sheet_path, name="NoMesh", height_cm=165.0)
    pid = cr["project_id"]
    res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/does_not_exist.glb")
    assert not res["ok"]
    assert res["error"]["code"] == "MESH_NOT_FOUND"


def test_auto_rig_no_provider(l3_setup, sheet_path, monkeypatch):
    monkeypatch.delenv("TRIPO_API_KEY", raising=False)
    monkeypatch.delenv("MESHY_API_KEY", raising=False)
    cr = server.project_create(sheet_path=sheet_path, name="NoProv", height_cm=165.0)
    pid = cr["project_id"]
    res = tools_full.auto_rig(project_id=pid, mesh_path="meshes/dummy.glb", method="provider")
    assert not res["ok"]
    assert res["error"]["code"] == "NO_PROVIDER"
