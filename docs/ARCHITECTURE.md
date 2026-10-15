# Kiến trúc CharForge

## 1. Mục tiêu
CharForge là một MCP server gắn vào coding agent (Claude Code, Gemini CLI, Antigravity, Codex…). Nó biến một **character turnaround sheet** (front/side/back) thành **model 3D** theo quy trình **tự động 100%**: sinh mesh → render lại đúng 3 góc → chấm điểm bằng số → tự sửa → tối ưu → xuất file → rig. Không bước nào bắt buộc có người. Khi không đạt ngưỡng, pipeline vẫn xuất mesh tốt nhất kèm `stop_reason`.

Model điều phối có thể là model nhanh/rẻ (cỡ Flash), vì mọi quyết định khó đã nằm trong code:

| Rủi ro của model yếu | Cách MCP chặn |
|---|---|
| Viết script Blender dài, dễ sai | Không có tool chạy `bpy` tùy ý; chỉ có tool nghiệp vụ với schema chặt |
| Quên bước, sai thứ tự | Mọi kết quả có `next_action`; có server `instructions`, prompt `charforge_playbook`, và `pipeline_run` (L3) |
| Đánh giá "đẹp chưa" bằng mắt | Chỉ dùng IoU/total/passed với ngưỡng cố định |
| Mất mạch khi pipeline dài | Trạng thái nằm trong `project.json`, tool nào cũng nhận `project_id` |
| Gọi sai tham số | Ít tool, kiểu `Literal`/`pattern`/min-max, lỗi trả kèm `fix` |
| Lặp vô hạn | Ngân sách `MAX_ITERATIONS=8`, dừng khi plateau, vẫn xuất best |

## 2. Chọn framework
| Khâu | Chọn | Lý do | Phương án thay thế |
|---|---|---|---|
| Điều phối | Coding agent qua MCP stdio | Agent nào cũng gắn được, không khoá vào một vendor | — |
| Sinh mesh (L2) | Cloud image-to-3D multi-view: Tripo, Meshy, Rodin, Hunyuan3D (qua fal.ai) | Chất lượng tự động cao nhất hiện có. Máy Mac không có GPU NVIDIA nên không chạy local TRELLIS/Hunyuan3D | `blockout`: visual hull từ silhouette, không cần key, chất lượng thô |
| Xử lý, render, rig, xuất | Blender 5.x headless (`bpy`) | Miễn phí, làm được mọi khâu, script hoá được, đã cài sẵn | trimesh/PyMeshLab cho thao tác nhẹ |
| Chấm điểm | numpy + OpenCV + scikit-image | Xác định, nhanh, không cần GPU | Thêm CLIP/DINO hoặc VLM judge (tương lai) |

## 3. Pipeline
```
sheet.png ─ sheet_split ─► views/{front,side,back}.png + *_mask.png + reference.json
                                   │
            (L2) generate_mesh ────┤  provider="auto": provider đầu tiên có key, nếu không có thì blockout
                                   ▼
          ┌──────────────► score_mesh  (Blender render mask 3 góc → canonical → IoU, profile, hints)
          │                        │ passed? (provider: total >= 0.80 & tris <= 300k; blockout: view IoU >= 0.90/0.85/0.85)
          │   no, còn ngân sách    │
          └── (blockout: mesh_fit / generate_mesh) | (provider: generate_mesh ứng viên mới)
                                   │ yes, hoặc stopped (budget/plateau → lấy best)
                                   ▼
   (blockout: mesh_optimize) ─► mesh_export (provider: tự chuẩn hóa rigid trước khi xuất GLB/FBX) ─► (L3) auto_rig ─► animate
(L3) pipeline_run: chạy toàn bộ vòng trên bằng code, trả kết quả theo từng lát thời gian (status running → gọi lại)
```

## 4. Mô hình tiến trình
- **MCP server**: Python 3.11, stdio, tool đồng bộ. `server.py` chỉ điều phối; logic nằm trong `state.py` (policy), `sheet.py`, `score.py`, `blender.py`.
- **Blender**: mỗi lệnh là một subprocess `blender -b --factory-startup --python-exit-code 1 -P bpy_scripts/<x>.py -- --result <json> --key value…`. Script trong `bpy_scripts/` chạy bằng Python của Blender nên **chỉ import bpy/mathutils/stdlib**, không import `charforge`. Kết quả trả qua file `--result`: `{"ok": true, …}` hoặc `{"ok": false, "code", "message", "fix"}`. Không parse stdout.
- **Provider (L2/L3)**: gọi HTTP bằng `httpx` (đi kèm `mcp`). Job dài (`generate_mesh`, `auto_rig`, `animate`) không block quá ~55 s mà trả `status: "running"` + `job_id`, lưu job kèm `kind` bền vững (`multi-image-to-3d` / `image-to-3d` / `rigging` / `animations`) vào `project.json.jobs` để gọi lại ở lượt sau mà không phụ thuộc bộ nhớ tiến trình.

## 5. Dữ liệu
```
projects/<id>/
  project.json      trạng thái (bên dưới)          calls.jsonl  mỗi lần gọi tool một dòng
  sheet.<ext>       ảnh gốc                         views/       front|side|back.png (RGBA), *_mask.png, preview.jpg, reference.json
  meshes/           ứng viên, bản fit/optimize/export          renders/<mesh-stem>/  {view}_mask|color.png, {view}_diff.png, compare_*.png
```
Các trường của `project.json` (schema_version 1):

| Trường | Ý nghĩa |
|---|---|
| `id`, `name`, `height_cm` | định danh, chiều cao thật để scale khi export |
| `stage` | `created → views_ready → scored ⇄ … → passed/stopped → optimized → exported → rigged → animated` |
| `views` | `{view: {"image", "mask"}}` (đường dẫn tương đối project) |
| `reference` | `aspect`, `profile`, `neck_ratio`, `palette` (xem L1-T2) |
| `thresholds`, `weights` | ngưỡng IoU, trọng số view (copy từ `config.py`, sửa theo từng project được) |
| `meshes[]` | `{path, source: external/blockout/meshy/tripo/…/fit/optimized/export, derived_from, task_id, rotate_z, created_at, …}` |
| `scores[]` | mỗi lần `score_mesh`: `{mesh, total, passed, iou, worst_bands, hints, …, at}` |
| `best` | `{mesh, total, passed}` của lần chấm tốt nhất trong pha tìm kiếm |
| `jobs` | job provider đang chạy: `{job_id: {provider, kind, started_at, ...}}` |
| `outputs` | `{optimized, glb, fbx, rigged, animated, rig_task_id, basic_animations}` |
| `stop_reason` | `null` / `budget` / `plateau` |

## 6. Policy: stage → next_action (`state.suggest_next`)
| stage | L1 | L2+ (mesh `blockout`/`external`) | L2+ (mesh provider: `meshy`/`tripo`/`rodin`/`hunyuan`) |
|---|---|---|---|
| created | sheet_split | sheet_split | sheet_split |
| views_ready | score_mesh (mesh có sẵn) | generate_mesh(provider="auto") | generate_mesh(provider="auto") |
| scored | score_mesh (mesh mới theo hints) | best total < 0.75 và ứng viên < 3 → generate_mesh, còn lại → mesh_fit(best) | `over_poly_budget` → generate_mesh(`target_polycount=30000`); ứng viên < 3 → generate_mesh; ngược lại → mesh_export(best) |
| passed / stopped | render_views(best, color), rồi hết | mesh_optimize(best) | **Bỏ qua `mesh_fit` & `mesh_optimize`** → đi thẳng `mesh_export(best)` để bảo toàn UV/PBR texture |
| optimized | — | mesh_export(glb) | mesh_export(glb) |
| exported | — | L3: auto_rig, L2: hết | L3: auto_rig (dò ngược chuỗi `derived_from` lấy `task_id` gốc), L2: hết |
| rigged | — | L3: animate | L3: animate (dùng `rig_task_id` của provider) |

`record_score` chỉ đổi stage khi đang ở pha tìm kiếm (`views_ready/scored/passed/stopped`). Lần chấm kiểm tra sau optimize chỉ được ghi lại.

## 7. Envelope và lỗi
Mọi tool trả JSON: `{ok, project_id, data, warnings, next_action: {tool, args, why} | null, error: {code, message, fix} | null}`.
- Lỗi agent sửa được: `raise CFError(code, message, fix)`. `fix` phải là một hành động cụ thể.
- `NotImplementedError("L1-T1")` → `NOT_IMPLEMENTED` (stub chưa làm). Exception khác → `INTERNAL` kèm `trace`.
- Mã lỗi hiện có: `BAD_PROJECT_ID, PROJECT_NOT_FOUND, SHEET_NOT_FOUND, SHEET_FORMAT, SPLIT_FAILED, VIEWS_MISSING, MESH_NOT_FOUND, MESH_FORMAT, BLENDER_NOT_FOUND, BLENDER_TIMEOUT, BLENDER_FAILED, RENDER_FAILED, NOT_IMPLEMENTED, INTERNAL`. L2/L3 thêm: `NO_PROVIDER, PROVIDER_FAILED, PROVIDER_TIMEOUT, FIT_FAILED, RIG_FAILED, NO_RIGGED_MESH, ANIM_FAILED`.

## 8. Chấm điểm (`score.py`)
1. **Canonical mask**: crop theo bbox, scale cho cao 94% của khung 512, đáy cách 3%, trọng tâm x ở giữa. Bước này bỏ scale và vị trí, nên không cần khớp camera với ảnh vẽ.
2. **IoU** từng view. Side lấy `max(IoU, IoU lật ngang)`, vì sheet có thể vẽ nghiêng trái hoặc phải.
3. `total = 0.5·front + 0.3·side + 0.2·back`.
   - Với mesh `blockout`/`external`: `passed` khi mọi view ≥ ngưỡng (0.90 / 0.85 / 0.85).
   - Với mesh từ provider (`meshy`, `tripo`, `rodin`, `hunyuan`): 3 view trên sheet 2D không đồng nhất hình học 100%, nên `passed` khi `total >= PROVIDER_ACCEPT_TOTAL` (0.80) và số mặt `tris <= MAX_PROVIDER_TRIS` (300.000).
4. **Width profile** 64 dải ngang → `worst_bands` (dải lệch nhiều nhất, có vùng head/torso/hips-skirt/legs) → `hints` dạng câu lệnh.
5. `rotated_90`: front render giống side reference hơn front reference → cảnh báo, gọi lại với `rotate_z=90`.
6. **Dừng**: passed | `MAX_ITERATIONS=8` | plateau (3 lần chấm gần nhất cải thiện best < 0.005). Khi dừng vẫn đi tiếp với best, nên lúc nào cũng có output.

## 9. Quy ước toạ độ
Z-up, đơn vị mét, nhân vật nhìn về −Y (mặc định khi Blender import glTF). Camera ortho: front ở −Y, side ở +X, back ở +Y. Sau chuẩn hóa rigid (`fit.py --rigid-only` tự chạy trong `mesh_export` cho mesh provider) hoặc `mesh_fit`: chân ở z=0, tâm XY ở gốc, chiều cao = `height_cm`, `rotate_z` đã được áp vĩnh viễn mà không làm hỏng UV/texture.

## 10. Đo chất lượng model điều phối
`calls.jsonl` ghi `{t, tool, args, ok, code, sec}` cho mỗi lần gọi. `project_status.data.stats` trả `calls, errors, error_rate, error_codes, iterations, best_total, passed`. Để so model (ví dụ Flash với model mạnh hơn): chạy N lần trên cùng sheet, so `error_rate`, số vòng đến khi passed, tỷ lệ passed, tổng thời gian (L3-T3).

## 11. Giới hạn của tự động 100%
- Chất lượng tối đa phụ thuộc provider. Tóc và váy thường bị "đúc" liền khối, không có hair cards hay cloth sim như làm tay.
- Điểm silhouette không đánh giá được mặt và biểu cảm. Muốn kiểm tra mặt thì thêm VLM judge (tùy chọn, ngoài roadmap).
- Ảnh có watermark của tác giả (ví dụ @higgs.osiris). Muốn dùng thương mại thì cần xin phép. Ảnh sẽ được gửi lên provider cloud khi dùng L2.
