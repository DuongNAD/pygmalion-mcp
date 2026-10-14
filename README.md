# Pygmalion MCP *(CharForge Engine)*

MCP server cho coding agent: lấy cảm hứng từ điêu khắc gia **Pygmalion** trong thần thoại Hy Lạp, biến **character turnaround sheet** 2D (front/side/back) thành **model 3D gắn xương hoàn chỉnh** theo quy trình **tự động 100%**. Agent tự sinh mesh, render lại đúng 3 góc bằng Blender headless, chấm điểm silhouette bằng số so với ảnh gốc, tự biến dạng khớp tỷ lệ, tối ưu lưới, xuất GLB/FBX và rig xương Mixamo. Model điều phối có thể là loại nhanh/rẻ (cỡ Flash), vì mọi quyết định khó nằm trong code: `next_action`, ngưỡng điểm, ngân sách vòng lặp.

| Level | Gồm | Trạng thái |
|---|---|---|
| L1 – Mắt | `project_create`, `project_status`, `sheet_split`, `render_views`, `score_mesh` | Đã hoàn thành & kiểm thử (Blender 5.2.1 + OpenCV/numpy) |
| L2 – Tay | `generate_mesh` (Tripo/Meshy/Rodin/Hunyuan hoặc visual-hull `blockout`), `mesh_fit`, `mesh_optimize`, `mesh_export` | Đã hoàn thành & kiểm thử |
| L3 – Hoàn thiện | `auto_rig` (22 xương Mixamo), `pipeline_run` (tự hành toàn trình), `scripts/driver_report.py` | Đã hoàn thành & kiểm thử (142/142 tests pass) |

## Cài đặt
Cần Python 3.11 và Blender 4.2+ (máy này: `/Applications/Blender.app`, 5.2.1).
```bash
cd /Users/duongnad/Documents/project/CharForge
python3.11 -m venv --system-site-packages .venv
uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env
.venv/bin/python -m charforge --check
```
Trong `.env`: `CHARFORGE_LEVEL` (1–3) và API key của provider (L2). Không có key thì `generate_mesh` dùng blockout.

## Gắn MCP vào agent
Server chạy qua stdio: `/Users/duongnad/Documents/project/CharForge/.venv/bin/python -m charforge`.
- **Claude Code**: đã có `.mcp.json` trong thư mục này. Mở Claude Code tại đây và duyệt server `charforge` khi được hỏi.
- **Gemini CLI**: đã có `.gemini/settings.json` trong thư mục này (timeout 10 phút).
- **Antigravity, Codex, client khác**: thêm khối sau vào cấu hình MCP của client:
```json
{"mcpServers": {"charforge": {"command": "/Users/duongnad/Documents/project/CharForge/.venv/bin/python", "args": ["-m", "charforge"]}}}
```
Kiểm tra thủ công bằng MCP Inspector: `npx @modelcontextprotocol/inspector .venv/bin/python -m charforge`.

## Dùng
Giao cho agent một câu, ví dụ:
> Dùng CharForge dựng model 3D từ `/đường/dẫn/sheet.png`, tên `aria`, cao 165 cm. Làm theo playbook `charforge_playbook` cho tới khi `next_action` là null, rồi báo file output và điểm.

Agent tự chạy `project_create → sheet_split → generate_mesh → score_mesh ⇄ mesh_fit → mesh_optimize → mesh_export → auto_rig`. Ở L3 chỉ cần gọi `pipeline_run` nhiều lần. Kết quả nằm trong `projects/<id>/` (`export/`, `renders/`, `project.json`).

## Phát triển tiếp
Tài liệu: [ARCHITECTURE](docs/ARCHITECTURE.md) (thiết kế, policy, chấm điểm) · [TOOLS](docs/TOOLS.md) (hợp đồng tool) · [ROADMAP](docs/ROADMAP.md) (task card để giao cho agent).
Test: `.venv/bin/python -m pytest -q`. Phần chưa làm hiện ra dạng `xfail`; task xong thì bỏ marker.

## Giới hạn
Chất lượng tối đa phụ thuộc provider. Tóc và váy thường bị đúc liền khối, và điểm silhouette không chấm được độ giống của mặt. Muốn đạt chất lượng như làm tay vẫn cần hair cards, cloth sim, sculpt mặt. Sheet có watermark của tác giả: dùng thương mại thì cần xin phép. Khi dùng provider, ảnh sẽ được gửi lên cloud.
