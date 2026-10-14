# Hợp đồng tool (MCP contract)

Phần này cố định tên tool, tham số và các key trong `data`. Đổi bất kỳ thứ gì ở đây là breaking change: phải sửa đồng thời code, file này và `playbook.md`.
Mọi tool trả envelope (ARCHITECTURE §7). Đường dẫn trong kết quả là đường dẫn tương đối project, trừ `project_dir`.
Annotation: `project_status` là read-only. Các tool khác chỉ ghi trong `projects/<id>/`. Riêng `generate_mesh` và `auto_rig(method=provider)` gửi dữ liệu ra ngoài (openWorld).

| Level | Tool | Trạng thái |
|---|---|---|
| L1 | project_create, project_status, render_views | chạy được |
| L1 | sheet_split, score_mesh | đã nối dây, còn chờ stub `sheet.py` (L1-T1/T2) và `score.compare`/`write_diffs` (L1-T3) |
| L2 | generate_mesh, mesh_fit, mesh_optimize, mesh_export | mới có spec, sẽ nằm trong `tools_core.py` |
| L3 | auto_rig, pipeline_run | mới có spec, sẽ nằm trong `tools_full.py` |

---
## L1: phần "mắt" (tách sheet, render, chấm điểm)

### project_create(sheet_path, name, height_cm=165)
- `sheet_path`: đường dẫn tuyệt đối, .png/.jpg/.webp. `name`: 1–40 ký tự, slug thành `project_id` (trùng thì thêm `-2`, `-3`…). `height_cm`: 50–300.
- data: `{project_dir, stage}` · next: `sheet_split` · lỗi: `SHEET_NOT_FOUND, SHEET_FORMAT`

### project_status(project_id)
- data: `{stage, name, height_cm, views, meshes, best, outputs, stop_reason, scores (5 lần gần nhất), stats}` · next: theo policy

### sheet_split(project_id, assign=None, boxes=None)
- Tự phát hiện các figure toàn thân và gán front/side/back. `assign=[i_front, i_side, i_back]` là chỉ số trong `candidates`, đọc từ ảnh preview có đánh số. `boxes={view: [x,y,w,h]}` theo pixel của sheet, chỉ dùng khi phát hiện hỏng hẳn.
- data: `{candidates, boxes, confidence, files: {view: {image, mask}}, preview}`. Ghi `views/reference.json` và set stage `views_ready`.
- next: confidence < 0.8 → `sheet_split(assign=…)` kèm warning; ngược lại theo policy · lỗi: `SPLIT_FAILED`

### render_views(project_id, mesh_path, mode="color"|"mask", rotate_z=0|90|180|270)
- Render ortho front/side/back bằng Blender (Workbench, đèn phẳng). Mesh ngoài project được copy vào `meshes/`.
- data: `{files: {view}, dims_m: [x,y,z], tris, compare}`. `compare` là ảnh reference | render từng hàng, chỉ có khi đã có views.
- Chỉ để xem. Mọi quyết định dùng `score_mesh`. · lỗi: `MESH_NOT_FOUND, MESH_FORMAT, BLENDER_*, RENDER_FAILED`

### score_mesh(project_id, mesh_path, rotate_z=0)
- Render mask → `score.compare` → lưu `scores[]`, cập nhật `best/stage/stop_reason` → ghi `{view}_diff.png`.
- data: `{iou: {front,side,back}, total, passed, side_mirrored, rotated_90, aspect_delta, worst_bands: [{view, from, to, region, delta}], hints, diffs, iteration, best, stage, stop_reason}`
- next: `rotated_90` → `score_mesh(rotate_z=90)`; ngược lại theo policy (ARCHITECTURE §6) · lỗi: `VIEWS_MISSING`, các lỗi mesh/Blender như trên

---
## L2: phần "tay" (model tự động đầu tiên)

### generate_mesh(project_id, provider="auto", job_id=None, seed=None)
- `provider`: `auto|tripo|meshy|rodin|hunyuan|blockout`. `auto` chọn provider có key mà project chưa dùng; hết provider thì đổi seed; không có key nào thì dùng `blockout`.
- Provider: upload 3 view (PNG nền trắng) → tạo task multi-view → poll tối đa ~60 s. Xong thì tải GLB về `meshes/cand-<n>-<provider>.glb` và `register_mesh(source=provider, seed)`. Chưa xong thì trả `status: "running"` + `job_id` (lưu ở `jobs`).
- `blockout`: chạy local, đồng bộ (L2-T2).
- data: `{status: running|done, job_id, provider, mesh_path?, seconds}` · next: running → `generate_mesh(job_id)`; done → `score_mesh(mesh_path)`
- lỗi: `NO_PROVIDER` (gọi provider chưa có key; fix: dùng `auto`/`blockout`), `PROVIDER_FAILED`, `PROVIDER_TIMEOUT` (job > 20 phút)

### mesh_fit(project_id, mesh_path)
- Python tính hệ số scale theo từng dải chiều cao: tỷ lệ width profile reference / render (front → X, side → Y), lấy từ mask của lần chấm gần nhất của mesh này, kẹp trong [0.8, 1.25], làm mượt.
- `bpy_scripts/fit.py`: áp `rotate_z`, đặt chân ở z=0 và tâm XY ở gốc, scale đều về `height_cm`, rồi lattice 2×2×16 áp các hệ số trên.
- data: `{mesh_path: "meshes/<stem>-fit<n>.glb", scale_x, scale_y}`, source=`fit` · next: `score_mesh(mesh_path)` · lỗi: `FIT_FAILED`

### mesh_optimize(project_id, mesh_path, target_tris=30000)
- `target_tris`: 2000–200000. `bpy_scripts/optimize.py`: merge by distance 1e-4, xoá loose, tính lại normal, decimate collapse tới target, smart UV project nếu chưa có UV, giữ material/texture.
- Tự kiểm tra: chấm lại. Front IoU giảm > 0.02 thì làm lại một lần với `target_tris × 1.5`. Set `outputs.optimized`, stage `optimized`.
- data: `{mesh_path, tris_before, tris_after, iou_drop}` · next: `mesh_export`

### mesh_export(project_id, mesh_path, format="glb"|"fbx"|"both")
- `bpy_scripts/export.py`: apply transform, đơn vị mét, xuất `export/<id>.glb|.fbx` (texture nhúng trong GLB). Set `outputs.glb/fbx`, stage `exported`.
- data: `{files, size_mb, tris}` · next: L3 → `auto_rig`; L2 → null

---
## L3: hoàn thiện

### auto_rig(project_id, mesh_path, method="auto"|"blender"|"provider")
- `blender`: `bpy_scripts/rig.py` dựng armature khoảng 22 xương, đặt tên kiểu Mixamo (Hips, Spine…, LeftArm, LeftUpLeg…). Vị trí khớp lấy từ bbox và `reference` (neck_ratio, profile). Parent bằng automatic weights; lỗi bone heat thì chuyển sang envelope weights.
- `provider`: dùng API rig của Tripo/Meshy nếu có key. `auto`: có key thì dùng provider, không thì `blender`.
- Tự kiểm tra: xoay LeftArm 45°, mesh phải biến dạng và tỷ lệ vertex không có weight < 1%. Xuất `export/<id>-rigged.glb`.
- data: `{mesh_path, bones, unweighted_ratio}` · next: null · lỗi: `RIG_FAILED`

### pipeline_run(project_id, max_seconds=240, rig=true)
- `max_seconds`: 60–540. Gọi lần lượt các hàm của tool L1–L3 theo đúng policy §6 cho đến khi hết thời gian, rồi lưu trạng thái và trả về.
- data: `{status: running|done|blocked, stage, steps: [{tool, ok, summary}], best, outputs}`
- next: running → `pipeline_run(project_id)`; blocked → đúng tool và args cần agent làm (ví dụ `sheet_split(assign=…)`); done → null
