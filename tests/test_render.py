import numpy as np
import pytest

from charforge import blender, config, score

pytestmark = [pytest.mark.blender, pytest.mark.skipif(config.blender_exe() is None, reason="Blender not installed")]

# 0.4 (x) x 1.6 (up) x 0.2 (depth) m box; OBJ is Y-up, Blender imports it Z-up.
V = [(x, y, z) for x in (-0.2, 0.2) for y in (0, 1.6) for z in (-0.1, 0.1)]
F = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]


def aspect(mask):
    ys, xs = np.nonzero(mask)
    return (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)


def test_render_box_masks(tmp_path):
    obj = tmp_path / "box.obj"
    obj.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in V) + "".join("f " + " ".join(map(str, f)) + "\n" for f in F))
    r = blender.run("render_views.py", {"mesh": obj, "out": tmp_path / "r", "mode": "mask", "res": 512})
    assert r["ok"] and r["tris"] == 12 and r["dims_m"] == pytest.approx([0.4, 0.2, 1.6], abs=1e-3)
    assert aspect(score.load_mask(r["files"]["front"])) == pytest.approx(0.25, abs=0.02)
    assert aspect(score.load_mask(r["files"]["side"])) == pytest.approx(0.125, abs=0.02)
