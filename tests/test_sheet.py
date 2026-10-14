import json

import pytest

from charforge import config, sheet


def test_split_synthetic_sheet(sheet_path, tmp_path):
    res = sheet.split_sheet(sheet_path, tmp_path / "views")
    assert len(res["candidates"]) == 3  # palette swatches ignored
    assert res["confidence"] >= 0.8
    b = res["boxes"]
    assert b["front"][0] < b["side"][0] < b["back"][0] and b["side"][2] < b["front"][2]
    for v in ("front", "side", "back"):
        assert (tmp_path / res["files"][v]["mask"]).is_file() and (tmp_path / res["files"][v]["image"]).is_file()
    assert (tmp_path / res["preview"]).is_file()


def test_split_manual_assign(sheet_path, tmp_path):
    res = sheet.split_sheet(sheet_path, tmp_path / "views", assign=[2, 1, 0])  # deliberately swap front/back
    assert res["boxes"]["front"][0] > res["boxes"]["back"][0] and res["confidence"] == 1.0


def test_reference_metrics(sheet_path, tmp_path):
    ref = sheet.split_sheet(sheet_path, tmp_path / "views")["reference"]
    assert set(ref["aspect"]) == {"front", "side", "back"} and ref["aspect"]["side"] < ref["aspect"]["front"]
    assert len(ref["profile"]["front"]) == config.PROFILE_BINS
    assert 0.05 < ref["neck_ratio"] < 0.3
    assert len(ref["palette"]) == 8 and all(c.startswith("#") and len(c) == 7 for c in ref["palette"])
    assert json.loads((tmp_path / "views" / "reference.json").read_text()) == ref
