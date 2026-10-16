# Hợp đồng tool (MCP contract)

Phần này cố định tên tool, tham số và các key trong `data`. Đổi bất kỳ thứ gì ở đây là breaking change: phải sửa đồng thời code, file này và `playbook.md`.
Mọi tool trả envelope (ARCHITECTURE §7). Đường dẫn trong kết quả là đường dẫn tương đối project, trừ `project_dir`.
Annotation: `project_status` là read-only. Các tool khác chỉ ghi trong `projects/<id>/`. Riêng `generate_mesh` và `auto_rig(method=provider)` gửi dữ liệu ra ngoài (openWorld).

| Level | Tool | Trạng thái |
|---|---|---|
| L1 | project_create, project_status, render_views, sheet_split, score_mesh | Đã hoàn thành & kiểm thử |
| L2 | generate_mesh, mesh_fit, mesh_optimize, mesh_export | Đã hoàn thành & kiểm thử |
| L3 | auto_rig, animate, pipeline_run | Đã hoàn thành & kiểm thử |

---
## L1: phần "mắt" (tách sheet, render, chấm điểm)

### project_create(sheet_path, name, height_cm=165)
- `sheet_path`: đường dẫn tuyệt đối, .png/.jpg/.webp. `name`: 1–40 ký tự, slug thành `project_id` (trùng thì thêm `-2`, `-3`…). `height_cm`: 50–300.
- data: `{project_dir, stage}` · next: `sheet_split` · lỗi: `SHEET_NOT_FOUND, SHEET_FORMAT`

### project_status(project_id)
- data: `{stage, name, height_cm, views, meshes, best, outputs, stop_reason, scores (5 lần gần nhất), qa: {rig, motion}, animations, stats}` · next: theo policy

### sheet_split(project_id, assign=None, boxes=None)
- Tự phát hiện các figure toàn thân và gán front/side/back. `assign=[i_front, i_side, i_back]` là chỉ số trong `candidates`, đọc từ ảnh preview có đánh số. `boxes={view: [x,y,w,h]}` theo pixel của sheet, chỉ dùng khi phát hiện hỏng hẳn.
- Đo **khe hở tư thế** trên silhouette front (`score.pose_gaps`): có nền trống giữa cẳng tay/bàn tay với thân (dải 30–62% chiều cao) và giữa hai ống chân (dải 72–92%) không. Tay/chân dính nhau là nguyên nhân chính khiến trọng số rò ("dính" khi chạy animation), nên có warning ngay từ bước này; không đổi `next_action`.
- data: `{candidates, boxes, confidence, files: {view: {image, mask}}, preview, pose: {arms: {left, right: {separated, share, gap}}, legs: {separated, share, gap}, t_pose, ok}}`. Ghi `views/reference.json` (thêm `pose`) và set stage `views_ready`.
- next: confidence < 0.8 → `sheet_split(assign=…)` kèm warning; ngược lại theo policy · lỗi: `SPLIT_FAILED`

### render_views(project_id, mesh_path, mode="color"|"mask", rotate_z=0|90|180|270)
- Render ortho front/side/back bằng Blender (Workbench, đèn phẳng). Mesh ngoài project được copy vào `meshes/`.
- data: `{files: {view}, dims_m: [x,y,z], tris, compare}`. `compare` là ảnh reference | render từng hàng, chỉ có khi đã có views.
- Chỉ để xem. Mọi quyết định dùng `score_mesh`. · lỗi: `MESH_NOT_FOUND, MESH_FORMAT, BLENDER_*, RENDER_FAILED`

### score_mesh(project_id, mesh_path, rotate_z=0)
- Render mask → `score.compare` → lưu `scores[]`, cập nhật `best/stage/stop_reason` → ghi `{view}_diff.png`.
- Với mesh từ provider (`meshy`, `tripo`, `rodin`, `hunyuan`), `passed = (total >= 0.80) and (tris <= 300000)`. Với mesh `blockout`/`external`, `passed` khi mọi view đạt ngưỡng IoU (0.90/0.85/0.85).
- Đo khe hở tư thế trên mask front của mesh (như `sheet_split`). `pose_mismatch` liệt kê phần sheet tách rời nhưng mesh lại dính (`"left arm"`, `"right arm"`, `"legs"`) kèm warning. Trong các mesh đã passed, mesh không dính được xếp trên mesh dính dù IoU thấp hơn một chút; mesh provider đã passed nhưng dính thì policy cho sinh thêm **một** ứng viên (`config.POSE_RETRY`, còn ngân sách `MAX_CANDIDATES`).
- data: `{iou: {front,side,back}, total, passed, over_poly_budget, side_mirrored, rotated_90, aspect_delta, worst_bands: [{view, from, to, region, delta}], hints, pose, pose_mismatch, diffs, iteration, best, stage, stop_reason}`
- next: `rotated_90` → `score_mesh(rotate_z=90)`; ngược lại theo policy (ARCHITECTURE §6) · lỗi: `VIEWS_MISSING`, các lỗi mesh/Blender như trên

---
## L2: phần "tay" (model tự động đầu tiên)

### generate_mesh(project_id, provider="auto", job_id=None, seed=None, target_polycount=None)
- `provider`: `auto|tripo|meshy|rodin|hunyuan|blockout`. `auto` chọn provider có key mà project chưa dùng; hết provider thì sinh ứng viên mới; không có key nào thì dùng `blockout`.
- `target_polycount`: giới hạn số đa giác mong muốn cho provider hỗ trợ remesh (mặc định `config.MESHY_TARGET_POLYCOUNT = 30000` trên Meshy 7.1 với `pose_mode="a-pose"`, `should_remesh=True`, `topology="triangle"`). Lưu ý: Meshy `/openapi/v1/multi-image-to-3d` không có tham số `seed` nên `seed` chỉ ghi vào metadata.
- Provider: gửi 3 view (PNG nền trắng, front luôn đứng đầu) → tạo task multi-view → lưu `jobs[job_id]` kèm `kind` bền vững → poll một lượt. Xong thì tải GLB về `meshes/cand-<n>-<provider>.glb` và `register_mesh(source=provider, task_id=job_id)`. Chưa xong thì trả `status: "running"` + `job_id`.
- `blockout`: chạy local, đồng bộ (L2-T2).
- data: `{status: running|done, job_id, provider, mesh_path?, seconds}` · next: running → `generate_mesh(job_id)`; done → `score_mesh(mesh_path)`
- lỗi: `NO_PROVIDER` (gọi provider chưa có key; fix: dùng `auto`/`blockout`), `PROVIDER_FAILED`, `PROVIDER_TIMEOUT` (job > 20 phút)

### mesh_fit(project_id, mesh_path, rigid_only=False)
- Dành cho mesh `blockout`/`external` (policy tự động bỏ qua `mesh_fit` với mesh provider). Nếu `rigid_only=True` (hoặc gọi trực tiếp trên mesh provider), chỉ chuẩn hóa rigid mà không áp lattice.
- Khi `rigid_only=False`: Python tính hệ số scale theo từng dải chiều cao từ width profile reference / render, kẹp trong [0.8, 1.25], làm mượt. `bpy_scripts/fit.py` áp `rotate_z`, đặt chân ở z=0 và tâm XY ở gốc, scale đều về `height_cm`, rồi lattice 2×2×16 áp các hệ số trên.
- Ghi `derived_from=mesh_path` vào `meshes[]` để bảo toàn lineage.
- data: `{mesh_path: "meshes/<stem>-fit<n>.glb", scale_x, scale_y, rigid_only}`, source=`fit` · next: `score_mesh(mesh_path)` · lỗi: `FIT_FAILED`

### mesh_optimize(project_id, mesh_path, target_tris=30000)
- Dành cho mesh `blockout`/`external` (policy tự động bỏ qua `mesh_optimize` với mesh provider vì Meshy đã remesh + bake PBR texture ở `target_polycount`).
- `target_tris`: 2000–200000. `bpy_scripts/optimize.py`: merge by distance 1e-4, xoá loose, tính lại normal, decimate collapse tới target, smart UV project nếu chưa có UV, giữ material/texture. Ghi `derived_from=mesh_path`.
- Tự kiểm tra: chấm lại. Front IoU giảm > 0.02 thì làm lại một lần với `target_tris × 1.5`. Set `outputs.optimized`, stage `optimized`.
- data: `{mesh_path, tris_before, tris_after, iou_drop}` · next: `mesh_export`

### mesh_export(project_id, mesh_path, format="glb"|"fbx"|"both")
- Nếu `mesh_path` có gốc từ provider và chưa qua chuẩn hóa rigid, tự động chạy `bpy_scripts/fit.py --rigid-only` (áp `rotate_z`, đặt chân ở z=0, tâm XY ở (0,0), scale đều về `height_cm`, không join/lattice để giữ nguyên UV/texture) trước khi xuất.
- `bpy_scripts/export.py`: apply transform, đơn vị mét, xuất `export/<id>.glb|.fbx` (texture nhúng trong GLB). Ghi `derived_from` vào `meshes[]`, set `outputs.glb/fbx`, stage `exported`.
- data: `{files, size_mb, tris}` · next: L3 → `auto_rig`; L2 → null

---
## L3: hoàn thiện

### auto_rig(project_id, mesh_path=None, method="auto"|"blender"|"provider", job_id=None)
- `blender`: `bpy_scripts/rig.py` dựng armature khoảng 22 xương, đặt tên kiểu Mixamo (Hips, Spine…, LeftArm, LeftUpLeg…). Vị trí khớp xác định bằng geometry/medial-axis centroids (sai số <= 8 cm trên cả T-pose, A-pose, arms-down). Pre-bind seam welding, bắt lỗi bone heat failure, hai tầng fallback weighting (geodesic BFS + bilateral IDW) ngăn chặn rò rỉ trọng số tay sang đùi.
- `provider`: dò ngược chuỗi `derived_from` trong `meshes[]` để tìm `input_task_id` gốc từ provider; gọi `POST /openapi/v1/rigging` (nếu Meshy trả HTTP 400 do mesh không hợp lệ với `input_task_id`, tự động gửi lại một lần bằng `model_url` data URI của file GLB đã chuẩn hóa). Mỗi lần gọi poll một lượt (`kind="rigging"` lưu trong `project.json.jobs`); nếu chưa xong trả `status="running"` + `job_id`. Khi xong tải `result.rigged_character_glb_url` về `export/<id>-rigged.glb`, lưu `outputs.rig_task_id` và `outputs.basic_animations`, và đếm số xương thật từ mảng `skins` trong GLB JSON.
- `auto`: ưu tiên `provider` khi có `MESHY_API_KEY` và mesh gốc từ Meshy (hoặc gửi qua `model_url`), ngược lại dùng `blender`.
- **Kiểm tra biến dạng (pose_deformation_check)** cho mọi rig (`bpy_scripts/rig_check.py` cho rig provider, chạy ngay trong `rig.py` cho rig Blender):
  - Nhận diện vai trò xương (hips, spine…, leftUpperArm…, rightToes) từ **cấu trúc cây xương và vị trí khớp**, tên xương chỉ để chú thích; chạy được với tên Mixamo, UE (`upperarm_l`), hay tên vô nghĩa (`Bone.008`).
  - Phân vùng cơ thể của từng đỉnh bằng khoảng cách tới đoạn xương (không dùng trọng số đang bị kiểm tra); đỉnh ở vùng giáp ranh giữa hai nhóm bị loại khỏi thống kê.
  - 6 phép thử: mỗi cánh tay vung trước 40° và giơ ngang 40°, mỗi đùi vung trước 40°. Đo vùng phải đứng yên: chân/thân dưới/đầu/tay kia khi xoay tay; ống chân kia/cẳng tay–bàn tay/thân trên khi xoay đùi. Rớt khi > `max_ratio` (1%) số đỉnh của một vùng di chuyển quá `tol_m` (0,5 cm, nhân theo chiều cao) hoặc có đỉnh di chuyển > `hard_m` (3 cm); hoặc chi được xoay chỉ theo xương < `follow_min` (60%, thiếu trọng số).
  - Đo tiếp xúc: mỗi bộ phận gồm đỉnh rõ ràng cộng dải giáp ranh của chính nó (đỉnh trùng vị trí do seam UV được hàn lại); dải đó chạm vào thân/chân là **dính**, còn lại đo khe hở bề mặt–bề mặt (`gap_min_m` 1 cm). Vai và háng (nơi các phần nối nhau theo thiết kế) được loại trừ. Giữa hai chân chỉ dùng đỉnh lõi: váy/đầm nối hai chân được báo `joined_by_cloth` và tiếp xúc đùi để "không xác định" thay vì "dính".
  - Rig provider rớt kiểm tra → **sửa trọng số**: bỏ ảnh hưởng của xương sai giải phẫu (tay không bao giờ kéo chân/thân/váy ngoài vùng hòa trộn vai; chân và tay kia không kéo tay; ống chân kia không kéo ống chân khi hai chân tách), chuẩn hóa lại, kiểm tra lại; chỉ giữ bản sửa nếu tốt hơn. Bản sửa thay `export/<id>-rigged.glb` (cùng khung xương), file gốc giữ ở `meshes/<id>-rigged-provider.glb`, set `outputs.rig_repaired`. Rig Blender chỉ báo cáo (đã có hành lang chi riêng).
  - Ghi `export/<id>-humanoid.json` (vai trò → tên xương, bảng tên Mixamo tương ứng, tư thế nghỉ, hệ trục glTF) để retarget sang engine/clip khác; báo cáo đầy đủ ở `renders/qa/rig_check.json`; tóm tắt ở `project.json.qa.rig`.
  - Thresholds trong `config.RIG_CHECK`, ghi đè theo project bằng `project.json.rig_check_thresholds`.
- data: `{status: "running"|"done", job_id?, mesh_path: "export/<id>-rigged.glb", bones: int, unweighted_ratio: float, rig_source: "provider"|"blender", rig_check: {passed: bool|null, repaired, issues, tests: {name: {passed, follow, worst_region, max_cm, leak_pct}}, contacts: {left_arm|right_arm|legs: {gap_cm, touching, joined_by_cloth?}}, skeleton_mapped, missing_roles, issues_before_repair?, report}, humanoid_map: "export/<id>-humanoid.json", basic_animations?: dict}` · warnings khi đã sửa hoặc vẫn rớt · next: running → `auto_rig(job_id=…)`; done → `animate` · lỗi: `NO_PROVIDER`, `RIG_FAILED`, `PROVIDER_FAILED`, `PROVIDER_TIMEOUT`

### animate(project_id, mesh_path=None, actions=None, motions=None, method="auto"|"blender"|"provider", secondary="auto"|"off", job_id=None)
- `project_id`: ID của project (ví dụ: 'aria').
- `mesh_path`: đường dẫn tương đối tới rigged GLB (mặc định lấy `outputs.rigged` từ `auto_rig`).
- `actions`: danh sách animation preset cần tạo (mặc định `["idle", "walk", "wave"]`; nếu chỉ truyền `motions` thì mặc định là rỗng). Trùng tên sẽ tự động deduplicate (`walk` → `walk`, `walk_01`).
- `motions`: clip riêng tạo từ mô tả văn bản bằng **Meshy Text to Motion** (cần rig Meshy): `[{prompt (≤ 400 ký tự), name? (mặc định motion_1…), duration? (2–10 s, bước 0,5, mặc định 4)}]`. Dùng thay cho việc tự viết keyframe.
- `method`: `"auto"|"blender"|"provider"`. Mặc định `"auto"`: nếu có `MESHY_API_KEY` và `outputs.rig_task_id` thì dùng `"provider"`, ngược lại dùng `"blender"`. `motions` với method khác provider → `NO_PROVIDER`.
- `secondary`: `"auto"` (mặc định) tự phát hiện tóc buông (khối do đầu điều khiển treo dưới hộp sọ, hoặc khối treo sau lưng nối với đầu) và váy (bề mặt băng qua khe giữa hai đùi), thêm chuỗi xương lò xo (`CF_hair*`, `CF_skirt*`), trộn trọng số từ gốc ra ngọn, mô phỏng kiểu VRM spring bone (quán tính + cản, kéo về hướng nghỉ, trọng lực, capsule va chạm: chân cho váy, thân cho tóc) và **bake vào từng clip** (clip lặp được chạy khởi động một vòng rồi khép vòng). Chỉ giữ khi độ giãn bề mặt không xấu đi; kèm số liệu váy xuyên chân trước/sau. `"off"` giữ nguyên khung xương. Tham số trong `config.SECONDARY`.
- `provider`: tải và cache (TTL 7 ngày tại `<CHARFORGE_HOME>/_cache/meshy_animation_library.json`) danh mục `GET /openapi/v1/animations/library`, ánh xạ tên action sang `action_id` thật qua bảng alias (`config.MESHY_ACTION_ALIASES`), rồi gửi **1 request duy nhất** `POST /openapi/v1/animations` với `{"rig_task_id": ..., "action_ids": [...]}` (tối đa 10 ID). Có `motions`: mỗi motion một `POST /openapi/v1/text-to-motion` (`mode` theo `config.MESHY_MOTION_MODE`), xong thì `POST /openapi/v1/animations {rig_task_id, motion_task_id}`; tất cả gom trong một job nhóm `kind="animate-group"` (`job_id` dạng `group-…`). Mỗi lần gọi poll một lượt; xong thì tải, đổi tên track, rồi hậu xử lý trong Blender (`merge_animations.py --mode post`): ghép clip motion vào sau clip preset, **chép trọng số đã sửa** từ `outputs.rigged` khi `outputs.rig_repaired` (Meshy hoạt hình hóa bản rig chưa sửa của họ), thêm chuyển động phụ. Không có gì cần làm thì giữ nguyên file Meshy (không xuất lại).
- `blender`: keyframe procedural trên bất kỳ rig humanoid nào (tìm xương theo vai trò, xoay quanh trục giải phẫu thế giới nên không phụ thuộc tên/roll xương): `idle` (thở, dồn trọng tâm), `walk`/`run` (chu kỳ dáng đi: hông nhún/lệch/xoay, gối gập pha vung, cổ chân, tay hạ từ A-pose về ~22° và vung ngược pha chân), `wave`, `jump`; tên khác → lắc nhẹ. 30 fps.
- **Kiểm tra clip**: `bpy_scripts/preview_clips.py` render 6 frame mỗi clip từ front và side (`config.ANIM_PREVIEW`), ghép thành `renders/anim/<clip>.png` có nhãn để người duyệt bằng mắt; đồng thời đo `stretch_p99`/`stretch_max` (tỉ lệ dài cạnh khi chạy so với tư thế nghỉ), `ground_cm` (đỉnh thấp nhất xuyên sàn), `travel_m`. Rớt khi `stretch_p99 > 1.6` hoặc `ground_cm > 3` (`config.ANIM_CHECK`, ghi đè bằng `project.json.anim_check_thresholds`).
- Tự kiểm tra: đọc trực tiếp chunk 0 JSON của file GLB xuất ra để xác minh số lượng và tên clip khớp chính xác với yêu cầu.
- data: `{status: "running"|"done", job_id?, mesh_path: "export/<id>-animated.glb", actions: [...], clip_count: int, method: str, previews: {clip: "renders/anim/<clip>.png"}, motion_check: {passed: bool|null, clips: {clip: {stretch_p99, stretch_max, ground_cm, travel_m}}, issues}, secondary: {applied, chains: {hair, skirt}, reason?, stretch_before?, stretch_after?, skirt_in_legs_before?, skirt_in_legs_after?}, action_ids?, motion_task_ids?, parts? (khi running)}` · next: running → `animate(job_id=…)`; done → `null` (hoàn tất quy trình) · lỗi: `NO_RIGGED_MESH`, `NO_PROVIDER`, `ANIM_FAILED`, `PROVIDER_FAILED`, `PROVIDER_TIMEOUT`

### pipeline_run(project_id, max_seconds=240, rig=true, animate=false)
- `max_seconds`: 60–540. Gọi lần lượt các hàm của tool L1–L3 theo đúng policy §6 cho đến khi hết thời gian, rồi lưu trạng thái và trả về. Job cloud đang chạy được poll lại sau mỗi `config.PIPELINE_POLL_SECONDS` (5 s) để tránh HTTP 429.
- `rig`: boolean (mặc định True), tự động chạy `auto_rig` sau `mesh_export`.
- `animate`: boolean (mặc định False), tự động chạy `animate` sau `auto_rig` đưa project về trạng thái `animated`.
- data: `{status: running|done|blocked, stage, steps: [{tool, ok, summary}], best, outputs}`; `summary` của `auto_rig`/`animate` ghi kết quả QA; warnings của từng bước được gom vào `warnings` (tiền tố tên tool).
- next: running → `pipeline_run(project_id, rig=…, animate=…)`; blocked → đúng tool và args cần agent làm (ví dụ `sheet_split(assign=…)`); done → null
