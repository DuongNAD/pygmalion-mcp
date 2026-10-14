# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

(`AGENTS.md` và `GEMINI.md` là symlink tới file này, nên mọi coding agent đọc cùng một hướng dẫn.)

## Dự án
CharForge là MCP server (Python 3.11, FastMCP, stdio) giúp coding agent biến character turnaround sheet thành model 3D, tự động 100%. Blender headless lo render/xử lý mesh, numpy/OpenCV lo chấm điểm. Hiện đang ở Level 1: khung đã chạy, còn các stub `NotImplementedError("L1-Tn")`. Việc cần làm nằm trong [docs/ROADMAP.md](docs/ROADMAP.md).

## Lệnh
```bash
.venv/bin/python -m charforge --check                 # môi trường: Blender, deps, level, provider
.venv/bin/python -m charforge                         # chạy MCP server (stdio)
.venv/bin/python -m pytest -q                         # toàn bộ test (task chưa làm = xfail)
.venv/bin/python -m pytest tests/test_score.py::test_iou_and_profile_basics -q   # một test
.venv/bin/python -m pytest -q -m "not blender"        # bỏ qua test cần Blender
uv pip install --python .venv/bin/python --no-deps -e .   # cài lại package (pip trong venv thiếu wheel)
npx @modelcontextprotocol/inspector .venv/bin/python -m charforge   # thử tool bằng tay
```
Chạy một script Blender bằng tay: `/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python-exit-code 1 -P src/charforge/bpy_scripts/render_views.py -- --result /tmp/r.json --mesh m.glb --out /tmp/out`.

## Kiến trúc (đọc nhiều file mới thấy)
- **Hai tiến trình**: `server.py` (Python 3.11) gọi Blender qua `blender.run(script, args)` → subprocess `blender -b ... -P bpy_scripts/<x>.py -- --result <json>`. Script trong `bpy_scripts/` chạy bằng Python của Blender: chỉ import bpy/mathutils/stdlib, không import `charforge`, trả kết quả qua file JSON (không parse stdout).
- **State**: `projects/<id>/project.json` là nguồn sự thật duy nhất (`state.py`). Mỗi lần gọi tool ghi một dòng vào `calls.jsonl`, dùng để đo chất lượng model điều phối.
- **Policy tập trung**: `state.suggest_next` (stage → `next_action`) và `state.record_score` (best/passed/budget/plateau). Tool không tự quyết bước tiếp theo.
- **Envelope**: mọi tool được bọc `@mcp.tool()` + `@tool_entry` và trả `ok(...)`. Lỗi mà agent sửa được: `raise CFError(code, message, fix)`. Stub thì trả `NOT_IMPLEMENTED`.
- **Chấm điểm**: `score.canonical_mask` bỏ scale/vị trí → IoU từng view (side so cả bản lật) → `total` có trọng số → `passed` theo ngưỡng trong `config.py`.
- **Level gating**: `CHARFORGE_LEVEL` 1/2/3. Level 2 nạp `tools_core.register(mcp)`, Level 3 nạp `tools_full.register(mcp)`; thiếu file thì lùi về level thấp hơn và báo trên stderr.
- Toạ độ: Z-up, mét, nhân vật nhìn −Y. Camera: front ở −Y, side ở +X, back ở +Y.

## Quy ước
- [docs/TOOLS.md](docs/TOOLS.md) là hợp đồng: đổi tên tool, tham số hay key trong `data` thì phải sửa TOOLS.md và `src/charforge/playbook.md` trong cùng thay đổi.
- Không thêm tool chạy `bpy`/Python tùy ý: model điều phối yếu chỉ được dùng tool nghiệp vụ có schema chặt. Giữ tổng số tool khoảng 11.
- Chuỗi mà model đọc lúc chạy (docstring tool, `error.fix`, `next_action.why`, playbook, instructions) viết bằng tiếng Anh, ngắn, dạng mệnh lệnh. Tài liệu cho người đọc (README, docs/) viết bằng tiếng Việt.
- Tunable (ngưỡng, trọng số, ngân sách) chỉ để trong `config.py`, có thể ghi đè theo từng project trong `project.json`.
- Test cho phần chưa làm dùng `pytest.mark.xfail(raises=NotImplementedError)`; task xong phải bỏ marker. Test tự vẽ sheet giả trong `conftest.py`, không cần ảnh thật.
- venv dùng `--system-site-packages` (mcp, opencv, numpy, scikit-image có sẵn ở Python 3.11 hệ thống).
