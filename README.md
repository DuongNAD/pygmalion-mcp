# Pygmalion MCP *(CharForge Engine)*

MCP server cho coding agent: lấy cảm hứng từ điêu khắc gia **Pygmalion** trong thần thoại Hy Lạp, biến **character turnaround sheet** 2D (front/side/back) thành **model 3D gắn xương và hoạt hóa hoàn chỉnh** theo quy trình **tự động 100%**. Agent tự sinh mesh, render lại đúng 3 góc bằng Blender headless, chấm điểm silhouette bằng số so với ảnh gốc, tự biến dạng khớp tỷ lệ, tối ưu lưới, xuất GLB/FBX, gắn xương Mixamo và sinh animation clips. Model điều phối có thể là loại nhanh/rẻ (cỡ Flash), vì mọi quyết định khó nằm trong code: `next_action`, ngưỡng điểm, ngân sách vòng lặp.

---

## Kết quả Mẫu Tự Hành (Autonomous Pipeline Showcase)

Toàn bộ quy trình từ ảnh turnaround 2D phẳng đến model 3D rigged & animated được điều phối tự hành qua `pipeline_run(project_id="aria", rig=True, animate=True)`:

### 1. Đầu vào 2D Turnaround Sheet (`assets/aria_turnaround.png`)
![Aria Turnaround Sheet](assets/aria_turnaround.png)
*Hình 1: Bản vẽ turnaround sheet 2D 1536×768 px của nhân vật "Aria" gồm 3 góc nhìn Front, Side (facing right) và Back ở tư thế A-pose chuẩn giải phẫu, nền trắng tuyệt đối, đạt điểm tách `sheet_split` confidence 1.0.*

### 2. Model 3D Đã Gắn Xương & Chuyển Động (`assets/aria_preview.gif`)
![Aria Animated Preview](assets/aria_preview.gif)
*Hình 2: Turntable 360° kết hợp chu kỳ bước chân (walk cycle animation) render trực tiếp từ `projects/aria/export/aria-animated.glb` bằng Blender Workbench headless.*

### 3. Ảnh Render Đa Góc Nhìn (`assets/aria_render.png`)
![Aria Multi-View Render](assets/aria_render.png)
*Hình 3: Kết quả render 4 góc nhìn chuẩn mực (Front 0°, Perspective 45°, Side 90°, Rear 180°) từ model 3D `aria.glb` với đầy đủ màu sắc đỉnh (vertex colors) từ visual-hull blockout.*

---

## Tính Năng & Cấp Độ

| Level | Gồm | Trạng thái |
|---|---|---|
| **L1 – Mắt** | `project_create`, `project_status`, `sheet_split`, `render_views`, `score_mesh` | Đã hoàn thành & kiểm thử 100% (Blender headless + OpenCV/NumPy) |
| **L2 – Tay** | `generate_mesh` (Tripo/Meshy/Rodin/Hunyuan hoặc visual-hull `blockout`), `mesh_fit`, `mesh_optimize`, `mesh_export` | Đã hoàn thành & kiểm thử 100% (Lattice fitting, decimate, bounding box safety) |
| **L3 – Hoàn thiện** | `auto_rig` (22 xương Mixamo), `animate` (hoạt hóa đa action), `pipeline_run` (tự hành toàn trình) | Đã hoàn thành & kiểm thử 100% (Medial-axis rigging, multi-clip GLB merging) |

---

## Chi Tiết Công Cụ Level 3

### `auto_rig(project_id, mesh_path=None, method="auto")`
- **Armature Mixamo 22 xương**: Tự động xác định các khớp vai, khuỷu tay, cổ tay, hông, đầu gối, mắt cá dựa trên trục trung tâm hình học (medial-axis cross-sectional centroids) với độ chính xác <= 8 cm trên cả 3 tư thế (T-pose, A-pose 45°, arms-down 80°).
- **Chống rò rỉ trọng số (Weight Leakage Prevention)**:
  * Pre-bind seam welding (`remove_doubles`) hàn kín các seam UV trước khi tính toán bone heat.
  * Bắt cảnh báo `Bone Heat Weighting: failed` từ Blender C stderr và chuyển sang thuật toán hai tầng: Geodesic BFS edge diffusion + bilateral corridor IDW.
  * Tự kiểm tra biến dạng: xoay `LeftArm` 40°, chuyển vị tay > 10 cm trong khi vùng đùi và thân dưới chuyển vị < 0.5 cm.
- **Tích hợp Cloud**: Hỗ trợ Meshy Rigging API (`POST /openapi/v1/rigging`) khi có `MESHY_API_KEY`.
- Kết quả: `export/<id>-rigged.glb` · `next_action`: `animate`.

### `animate(project_id, mesh_path=None, actions=None, method="auto")`
- **Hoạt hóa đa action & ghép file GLB**:
  * **Provider Mode**: Gọi Meshy Animation API (`POST /openapi/v1/animations`), tải các clip GLB độc lập và hợp nhất trong Blender bằng lệnh `export_animation_mode='ACTIONS'`.
  * **Blender Procedural Mode**: Sinh chuyển động theo keyframe toán học trực tiếp trên 22 xương Mixamo không cần kết nối mạng:
    - `idle`: 60 frame, nhịp thở dao động trục Z trên Hips (±1.2 cm) kèm lắc nhẹ Spine/Neck.
    - `walk`: 40 frame, chu kỳ bước chân đối xứng và vung tay nhịp nhàng ngược pha.
    - `wave`: 60 frame, nâng cánh tay và vẫy cẳng tay/bàn tay theo sóng sin.
- **Tự kiểm tra nhị phân (GLB Chunk 0 Verification)**: Phân tích trực tiếp cấu trúc nhị phân của chunk 0 JSON trong file GLB xuất ra để đảm bảo số lượng track animation và tên action khớp chính xác với yêu cầu.
- Kết quả: `export/<id>-animated.glb` chứa các animation tracks `['idle', 'walk', 'wave']`.

### `pipeline_run(project_id, max_seconds=240, rig=True, animate=False)`
- Tự động thực thi toàn bộ chuỗi công cụ theo máy trạng thái từ đầu vào sheet ảnh 2D đến thành phẩm cuối cùng mà không cần can thiệp thủ công:
  `project_create → sheet_split → generate_mesh → score_mesh ⇄ mesh_fit → mesh_optimize → mesh_export → auto_rig → animate`.

---

## Cài đặt & Kiểm Tra

Yêu cầu Python 3.11+ và Blender 4.2+ (hỗ trợ Windows, Linux, macOS).
```bash
# Cài đặt môi trường
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Tạo cấu hình môi trường
cp .env.example .env

# Kiểm tra môi trường và đường dẫn Blender
python -m charforge --check
```

Trong `.env`: cấu hình `CHARFORGE_LEVEL=3` để kích hoạt đầy đủ Level 1, 2 và 3. Khi không có cloud API keys, pipeline tự động sử dụng visual hull blockout và Blender procedural animation ngoại tuyến.

### Chạy Test Suite
```bash
# Kiểm tra bộ test hợp đồng và pipeline e2e
pytest tests/test_animate.py tests/contract/ tests/test_providers.py tests/e2e/test_pipeline_e2e.py -q
```

---

## Gắn MCP Vào Coding Agent

Server chạy qua giao thức stdio tiêu chuẩn: `python -m charforge`.
- **Claude Code**: Đã có cấu hình trong `.mcp.json`. Mở Claude Code tại thư mục này và cấp quyền chạy server `charforge`.
- **Gemini CLI**: Cấu hình trong `.gemini/settings.json`.
- **Antigravity / Cursor / Cline / Roo Code**: Thêm khối sau vào file cấu hình MCP của client:
```json
{
  "mcpServers": {
    "charforge": {
      "command": "python",
      "args": ["-m", "charforge"]
    }
  }
}
```

---

## Hướng Dẫn Sử Dụng Nhanh

Giao cho agent lời nhắc (prompt):
> Dùng CharForge dựng model 3D từ `assets/aria_turnaround.png`, tên `aria`, cao 165 cm. Chạy `pipeline_run(project_id='aria', rig=True, animate=True)` cho tới khi `next_action` là null, rồi báo các file output và animation tracks đã tạo.

Kết quả sẽ được ghi đầy đủ vào `projects/<id>/`:
- `views/reference.json`: Dữ liệu phân tích góc nhìn và thông số tỷ lệ
- `export/<id>.glb`: Model 3D tối ưu hóa lưới
- `export/<id>-rigged.glb`: Model đã gắn armature Mixamo 22 xương
- `export/<id>-animated.glb`: Model tích hợp sẵn các action tracks (`idle`, `walk`, `wave`)

---

## Tài Liệu Kỹ Thuật

- [ARCHITECTURE](docs/ARCHITECTURE.md): Kiến trúc hệ thống, máy trạng thái và công thức tính điểm silhouette
- [TOOLS](docs/TOOLS.md): Quy cách chi tiết (contracts, parameters, return envelopes, error codes) của toàn bộ 10 MCP tools
- [ROADMAP](docs/ROADMAP.md): Lịch trình phát triển và kiểm thử qua các milestone
