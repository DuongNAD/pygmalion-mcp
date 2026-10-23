# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

(`AGENTS.md` và `GEMINI.md` là symlink tới file này, nên mọi coding agent đọc cùng một hướng dẫn.)

## Dự án
CharForge là MCP server (Python 3.11, FastMCP, stdio) giúp coding agent biến character turnaround sheet thành model 3D, tự động 100%. Blender headless lo render/xử lý mesh, numpy/OpenCV lo chấm điểm. Đầy đủ cả 3 cấp độ: Level 1 (tách sheet & chấm điểm), Level 2 (sinh mesh, fit, tối ưu, xuất GLB/FBX), Level 3 (gắn xương với `auto_rig` kèm kiểm tra biến dạng + tự sửa trọng số, sinh animation đa action với `animate` kèm Text to Motion, chuyển động phụ và preview, tự hành toàn trình với `pipeline_run`).

## Lệnh
```bash
python -m charforge --check                           # môi trường: Blender, deps, level, provider
python -m charforge                                   # chạy MCP server (stdio)
pytest -q                                             # toàn bộ test suite
pytest tests/test_animate.py tests/contract/ tests/e2e/test_pipeline_e2e.py -q
pytest tests/test_rig_qa.py tests/test_anim_qa.py -q  # QA rig/animation trên nhân vật tổng hợp (tests/bpy_fixtures.py)
pytest -q -m "not blender"                            # bỏ qua test cần Blender
uv pip install --no-deps -e .                         # cài lại package khi cần
npx @modelcontextprotocol/inspector python -m charforge   # thử tool bằng tay
```
Chạy một script Blender bằng tay: `blender -b --factory-startup --python-exit-code 1 -P src/charforge/bpy_scripts/render_views.py -- --result /tmp/r.json --mesh m.glb --out /tmp/out`.

## Kiến trúc (đọc nhiều file mới thấy)
- **Hai tiến trình**: `server.py` (Python 3.11) gọi Blender qua `blender.run(script, args)` → subprocess `blender -b ... -P bpy_scripts/<x>.py -- --result <json>`. Script trong `bpy_scripts/` chạy bằng Python của Blender: chỉ import bpy/mathutils/stdlib, không import `charforge`, trả kết quả qua file JSON (không parse stdout). Ngoại lệ duy nhất: module dùng chung cùng thư mục `cf_humanoid.py` (vai trò xương, kiểm tra biến dạng, sửa trọng số), `cf_secondary.py` (spring bone tóc/váy) và `retarget.py` (gắn clip thư viện lên rig theo vai trò xương), import qua `sys.path.insert(0, dirname(__file__))`.
- **State**: `projects/<id>/project.json` là nguồn sự thật duy nhất (`state.py`). Mỗi lần gọi tool ghi một dòng vào `calls.jsonl`, dùng để đo chất lượng model điều phối.
- **Policy tập trung**: `state.suggest_next` (stage → `next_action`: `created` → `sheet_split` → `generate_mesh` → `score_mesh` ⇄ `mesh_fit` → `mesh_optimize` → `mesh_export` → `auto_rig` → `animate` → `null`) và `state.record_score` (best/passed/budget/plateau). Tool không tự quyết bước tiếp theo.
- **Envelope**: mọi tool được bọc `@mcp.tool()` + `@tool_entry` và trả `ok(...)`. Lỗi mà agent sửa được: `raise CFError(code, message, fix)`.
- **Chấm điểm**: `score.canonical_mask` bỏ scale/vị trí → IoU từng view (side so cả bản lật) → `total` có trọng số → `passed` theo ngưỡng trong `config.py`. `score_mesh` còn đo toàn vẹn hình học (`render_views.py --integrity`) và, với mesh provider đã passed, gọi critic thị giác độc lập (`critic.py`: Gemini qua SDK `google-genai`, extra `[critic]`, hoặc Claude qua SDK `anthropic`, extra `[critic-claude]`): chỉ tư vấn, không đổi `passed`. Test critic không gọi mạng: client giả cho Claude, SDK `google-genai` thật nối vào `httpx.MockTransport` cho Gemini (`critic._client_override`); `conftest.py` tắt critic cho mọi test khác.
- **Level gating**: `CHARFORGE_LEVEL` 1/2/3. Level 2 nạp `tools_core.register(mcp)`, Level 3 nạp `tools_full.register(mcp)` (`auto_rig`, `animate`, `pipeline_run`); thiếu file thì lùi về level thấp hơn và báo trên stderr.
- **Animation & Rigging (L3)**: `auto_rig` rig bằng Meshy hoặc `rig.py` (22 xương Mixamo, medial-axis), rồi `pose_deformation_check` (vung tay/đùi 40°, đo vùng phải đứng yên) cho mọi rig; rig provider rớt thì sửa trọng số và giữ bản tốt hơn; ghi `export/<id>-humanoid.json`. `animate` ghép preset Meshy + Text to Motion (`motions`) hoặc, khi offline, procedural (tìm xương theo vai trò) và clip thư viện `assets/motions/` (BVH CC0, `retarget.py`; tên lạ báo `ANIM_FAILED`) thành một GLB, chép trọng số đã sửa, bake spring bone tóc/váy (`secondary`), render 6 frame front/side mỗi clip và đo độ giãn/xuyên sàn. Khe hở tư thế (`score.pose_gaps`) đo ở `sheet_split`/`score_mesh`.
- Toạ độ: Z-up, mét, nhân vật nhìn −Y. Camera: front ở −Y, side ở +X, back ở +Y.

## Quy ước
- [docs/TOOLS.md](docs/TOOLS.md) là hợp đồng: đổi tên tool, tham số hay key trong `data` thì phải sửa TOOLS.md và `src/charforge/playbook.md` trong cùng thay đổi.
- Không thêm tool chạy `bpy`/Python tùy ý: model điều phối yếu chỉ được dùng tool nghiệp vụ có schema chặt. Giữ tổng số tool khoảng 11.
- Chuỗi mà model đọc lúc chạy (docstring tool, `error.fix`, `next_action.why`, playbook, instructions) viết bằng tiếng Anh, ngắn, dạng mệnh lệnh. Tài liệu cho người đọc (README, docs/) viết bằng tiếng Việt.
- Tunable (ngưỡng, trọng số, ngân sách) chỉ để trong `config.py`, có thể ghi đè theo từng project trong `project.json`.
- Test cho phần chưa làm dùng `pytest.mark.xfail(raises=NotImplementedError)`; task xong phải bỏ marker. Test tự vẽ sheet giả trong `conftest.py`, không cần ảnh thật.
- venv dùng `--system-site-packages` (mcp, opencv, numpy, scikit-image có sẵn ở Python 3.11 hệ thống).
