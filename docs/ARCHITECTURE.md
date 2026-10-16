# Kiến trúc CharForge

## 1. Mục tiêu
CharForge là một MCP server gắn vào coding agent (Claude Code, Gemini CLI, Antigravity, Codex…). Nó biến một **character turnaround sheet** (front/side/back) thành **model 3D** theo quy trình **tự động 100%**: sinh mesh → render lại đúng 3 góc → chấm điểm bằng số → tự sửa → tối ưu → xuất file → rig. Không bước nào bắt buộc có người. Khi không đạt ngưỡng, pipeline vẫn xuất mesh tốt nhất kèm `stop_reason`.

Model điều phối có thể là model nhanh/rẻ (cỡ Flash), vì mọi quyết định khó đã nằm trong code:

| Rủi ro của model yếu | Cách MCP chặn |
|---|---|
| Viết script Blender dài, dễ sai | Không có tool chạy `bpy` tùy ý; chỉ có tool nghiệp vụ với schema chặt |
| Quên bước, sai thứ tự | Mọi kết quả có `next_action`; có server `instructions`, prompt `charforge_playbook`, và `pipeline_run` (L3) |
| Đánh giá "đẹp chưa" bằng mắt | Chỉ dùng IoU/total/passed với ngưỡng cố định; phần số không đo được (mặt, texture, độ tự nhiên chuyển động) do **critic thị giác độc lập** chấm theo rubric cố định, chỉ mang tính tư vấn |
| Mất mạch khi pipeline dài | Trạng thái nằm trong `project.json`, tool nào cũng nhận `project_id` |
| Gọi sai tham số | Ít tool, kiểu `Literal`/`pattern`/min-max, lỗi trả kèm `fix` |
| Lặp vô hạn | Ngân sách `MAX_ITERATIONS=8`, dừng khi plateau, vẫn xuất best |
| Tự viết keyframe animation (dính, giật) | Không có tool keyframe; preset Meshy, `animate(motions=[…])` (Text to Motion) hoặc procedural có sẵn |
| Không biết rig có "dính" không | `auto_rig` tự vung tay/đùi đo rò trọng số, tự sửa; `animate` render frame từng clip + đo độ giãn bề mặt |

## 2. Chọn framework
| Khâu | Chọn | Lý do | Phương án thay thế |
|---|---|---|---|
| Điều phối | Coding agent qua MCP stdio | Agent nào cũng gắn được, không khoá vào một vendor | — |
| Sinh mesh (L2) | Cloud image-to-3D multi-view: Tripo, Meshy, Rodin, Hunyuan3D (qua fal.ai) | Chất lượng tự động cao nhất hiện có. Máy Mac không có GPU NVIDIA nên không chạy local TRELLIS/Hunyuan3D | `blockout`: visual hull từ silhouette, không cần key, chất lượng thô |
| Xử lý, render, rig, xuất | Blender 5.x headless (`bpy`) | Miễn phí, làm được mọi khâu, script hoá được, đã cài sẵn | trimesh/PyMeshLab cho thao tác nhẹ |
| Chấm điểm | numpy + OpenCV + scikit-image; toàn vẹn hình học bằng bmesh/BVH trong Blender | Xác định, nhanh, không cần GPU | CLIP/DINO embedding |
| Critic thị giác (tùy chọn) | `critic.py`: Gemini `gemini-4-argon` (SDK `google-genai`, Interactions API, `store=false`) hoặc Claude `claude-opus-5-5` (SDK `anthropic`, fallback `"default"`); cả hai trả JSON theo schema | Độc lập với model điều phối, chấm mặt/texture/chuyển động mà số đo không thấy | Tắt (`CHARFORGE_CRITIC=off`) |

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

Cổng chất lượng của L3 (không thêm tool, nằm trong auto_rig/animate):
  auto_rig ─► rig_check (vung tay/đùi 40°, đo vùng phải đứng yên, đo khe hở) ─ rớt ─► sửa trọng số ─► kiểm tra lại
           └─► export/<id>-humanoid.json (bản đồ xương để retarget) + renders/qa/rig_check.json
  animate  ─► (provider) ghép clip Text to Motion + chép trọng số đã sửa + chuyển động phụ (merge_animations --mode post)
           ─► preview_clips: 6 frame × front/side mỗi clip → renders/anim/<clip>.png + stretch/ground metrics
```

## 4. Mô hình tiến trình
- **MCP server**: Python 3.11, stdio, tool đồng bộ. `server.py` chỉ điều phối; logic nằm trong `state.py` (policy), `sheet.py`, `score.py`, `blender.py`.
- **Blender**: mỗi lệnh là một subprocess `blender -b --factory-startup --python-exit-code 1 -P bpy_scripts/<x>.py -- --result <json> --key value…`. Script trong `bpy_scripts/` chạy bằng Python của Blender nên **chỉ import bpy/mathutils/stdlib**, không import `charforge`. Kết quả trả qua file `--result`: `{"ok": true, …}` hoặc `{"ok": false, "code", "message", "fix"}`. Không parse stdout.
  - Hai module dùng chung nằm cùng thư mục và được import bằng `sys.path.insert(0, dirname(__file__))`: `cf_humanoid.py` (nhận diện vai trò xương theo cấu trúc, phân vùng đỉnh theo đoạn xương, phép thử biến dạng, đo tiếp xúc, sửa trọng số, bản đồ humanoid) và `cf_secondary.py` (phát hiện tóc/váy, chuỗi spring bone, mô phỏng và bake). Chúng cũng chỉ dùng bpy/mathutils/stdlib.
  - Script L3: `rig.py` (rig offline + QA chỉ báo cáo), `rig_check.py` (QA + sửa cho rig bất kỳ), `merge_animations.py` (procedural / merge / rename / post), `preview_clips.py` (render frame từng clip + chỉ số chuyển động).
- **Provider (L2/L3)**: gọi HTTP bằng `httpx` (đi kèm `mcp`). Job dài (`generate_mesh`, `auto_rig`, `animate`) không block quá ~55 s mà trả `status: "running"` + `job_id`, lưu job kèm `kind` bền vững (`multi-image-to-3d` / `image-to-3d` / `rigging` / `animations`) vào `project.json.jobs` để gọi lại ở lượt sau mà không phụ thuộc bộ nhớ tiến trình.

## 5. Dữ liệu
```
projects/<id>/
  project.json      trạng thái (bên dưới)          calls.jsonl  mỗi lần gọi tool một dòng
  sheet.<ext>       ảnh gốc                         views/       front|side|back.png (RGBA), *_mask.png, preview.jpg, reference.json
  meshes/           ứng viên, bản fit/optimize/export, <id>-rigged-provider.glb (rig gốc trước khi sửa), <id>-anim-provider.glb, <id>-motion-*.glb
  renders/<mesh-stem>/  {view}_mask|color.png, {view}_diff.png, compare_*.png
  renders/qa/rig_check.json   báo cáo đầy đủ của kiểm tra biến dạng      renders/anim/<clip>.png   preview front+side từng clip
  export/           <id>.glb|.fbx, <id>-rigged.glb, <id>-humanoid.json, <id>-animated.glb
```
Các trường của `project.json` (schema_version 1):

| Trường | Ý nghĩa |
|---|---|
| `id`, `name`, `height_cm` | định danh, chiều cao thật để scale khi export |
| `stage` | `created → views_ready → scored ⇄ … → passed/stopped → optimized → exported → rigged → animated` |
| `views` | `{view: {"image", "mask"}}` (đường dẫn tương đối project) |
| `reference` | `aspect`, `profile`, `neck_ratio`, `palette` (xem L1-T2), `pose` (khe hở tay–thân, chân–chân trên silhouette front) |
| `thresholds`, `weights` | ngưỡng IoU, trọng số view (copy từ `config.py`, sửa theo từng project được) |
| `rig_check_thresholds`, `anim_check_thresholds`, `anim_preview`, `secondary_params`, `integrity_thresholds`, `critic_params` | (tùy chọn) ghi đè `config.RIG_CHECK`, `ANIM_CHECK`, `ANIM_PREVIEW`, `SECONDARY`, `INTEGRITY`, `CRITIC` cho riêng project |
| `critic_calls` | số lượt đã gọi critic thị giác (giới hạn `CRITIC.max_calls`) |
| `meshes[]` | `{path, source: external/blockout/meshy/tripo/…/fit/optimized/export, derived_from, task_id, rotate_z, created_at, …}` |
| `scores[]` | mỗi lần `score_mesh`: `{mesh, total, passed, iou, worst_bands, hints, pose, pose_mismatch, …, at}` |
| `best` | `{mesh, total, passed, pose_mismatch?}` của lần chấm tốt nhất trong pha tìm kiếm |
| `pose_retries` | số ứng viên đã sinh thêm vì mesh dính tay/chân mà sheet tách rời (tối đa `config.POSE_RETRY`) |
| `jobs` | job provider đang chạy: `{job_id: {provider, kind, started_at, ...}}`; `kind="animate-group"` gom preset + Text to Motion (`parts`) |
| `outputs` | `{optimized, glb, fbx, rigged, rigged_provider, rig_repaired, humanoid, animated, previews, rig_task_id, basic_animations}` |
| `qa` | `{rig: tóm tắt rig_check, motion: tóm tắt motion_check}` |
| `animations` | `[{name, action_id? \| prompt + motion_task_id}]` |
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

Xếp hạng `best`: `(passed, tư thế sạch nếu đã passed, total)`. Tư thế sạch nghĩa là mesh không dính phần mà sheet tách rời (`pose_mismatch` rỗng): tay dính thân làm rig rò trọng số, đáng giá hơn vài phần trăm IoU. Khi `best` của provider đã passed mà vẫn dính, còn ngân sách ứng viên và `pose_retries < POSE_RETRY` (1): stage về `scored` và `next_action` là `generate_mesh` với lý do nêu phần bị dính. Sheet tự nó đã dính thì không sinh thêm (sinh lại cũng vậy), chỉ cảnh báo.

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
5b. **Khe hở tư thế** (`pose_gaps`): trên mask front canonical, dải 30–62% chiều cao phải có hàng mà run ngoài cùng mảnh như cánh tay (< 12% chiều cao) cách run kế tiếp ≥ `POSE_GAP_MIN` (1,2% chiều cao) ở ít nhất `POSE_ARM_BAND_MIN` (25%) số hàng; T-pose (sải tay > 0,75 chiều cao) coi như tách. Dải 72–92% phải có hai run quanh tâm cách nhau ≥ ngưỡng ở `POSE_LEG_BAND_MIN` (50%) số hàng. Áp cho sheet (`reference.pose`) và cho mesh (`scores[].pose`); `pose_mismatch` = phần sheet tách mà mesh dính.
5c. **Toàn vẹn hình học** (`render_views.py --integrity` → `score.integrity_report`): mảnh vụn, phần bị tách khỏi thân (không cắt vào và không gần bề mặt khối chính quá `gap_frac` × chiều cao), lỗ hở, cạnh non-manifold, tam giác tự cắt trong cùng một mảnh. Kiểm bằng số trước, rồi mới tới kiểm bằng mắt. Phần bị tách hoặc tự cắt nhiều làm `integrity.ok = false` và mesh đó xếp sau mesh lành trong số các mesh đã passed.
5d. **Critic thị giác độc lập** (`critic.py`, chỉ mesh provider đã passed và các clip animation): model critic (mặc định Gemini, hoặc Claude; `CRITIC.backend` = auto chọn backend đầu tiên có SDK và API key) nhận ảnh sheet | render màu (hoặc contact sheet clip) cùng một rubric cố định, trả điểm 0–10 theo tiêu chí và lỗi cụ thể bằng structured output. Model điều phối không bao giờ tự chấm sản phẩm của mình. Critic không đổi `passed` và không kích hoạt sinh lại; nó chỉ quyết định giữa hai mesh đã passed khi chênh ≥ 1 điểm, và cảnh báo. Có ngân sách lượt gọi theo project.
6. **Dừng**: passed | `MAX_ITERATIONS=8` | plateau (3 lần chấm gần nhất cải thiện best < 0.005). Khi dừng vẫn đi tiếp với best, nên lúc nào cũng có output.

## 9. Quy ước toạ độ
Z-up, đơn vị mét, nhân vật nhìn về −Y (mặc định khi Blender import glTF). Camera ortho: front ở −Y, side ở +X, back ở +Y. Sau chuẩn hóa rigid (`fit.py --rigid-only` tự chạy trong `mesh_export` cho mesh provider) hoặc `mesh_fit`: chân ở z=0, tâm XY ở gốc, chiều cao = `height_cm`, `rotate_z` đã được áp vĩnh viễn mà không làm hỏng UV/texture.

## 10. Đo chất lượng model điều phối
`calls.jsonl` ghi `{t, tool, args, ok, code, sec}` cho mỗi lần gọi. `project_status.data.stats` trả `calls, errors, error_rate, error_codes, iterations, best_total, passed`. Để so model (ví dụ Flash với model mạnh hơn): chạy N lần trên cùng sheet, so `error_rate`, số vòng đến khi passed, tỷ lệ passed, tổng thời gian (L3-T3).

## 11. Giới hạn của tự động 100%
- Chất lượng tối đa phụ thuộc provider. Tóc và váy bị "đúc" liền khối với thân. `animate(secondary="auto")` thêm chuỗi spring bone đã bake (tóc đuôi ngựa/tóc dài sau lưng, váy quanh hông, va chạm capsule với chân và thân) nên có độ trễ và lắc theo quán tính, nhưng không phải cloth sim thật: không có nếp gấp vải, va chạm chỉ xấp xỉ, tóc hai bên mặt và áo choàng rộng có thể không được nhận ra.
- Không có rig mặt hay blendshape: mesh sinh ra thường vẽ mắt/miệng lên texture, không có hình học để tạo biểu cảm hay nhép miệng. Cần mesh có mặt mô hình hóa riêng hoặc làm tay.
- Tay sát thân hoặc hai đùi dính nhau trên mesh vẫn làm rò trọng số ở chỗ tiếp xúc; QA phát hiện, sửa phần rò "xa" và báo vùng tiếp xúc, nhưng không tách được hình học đã dính. Sheet A-pose có khe hở rõ dưới nách và giữa hai chân quan trọng ngang code (đã cảnh báo từ `sheet_split`).
- Animation gắn với khung xương của rig (Meshy hoặc rig offline). `export/<id>-humanoid.json` cho bản đồ vai trò → tên xương (kèm tên Mixamo) để retarget trong Unity/UE/Blender; tư thế nghỉ là A-pose nên công cụ retarget phải bù khác biệt với clip T-pose.
- Nhánh offline (Blender procedural) đã tách khỏi tên xương và có dáng đi đầy đủ pha, nhưng vẫn là keyframe công thức: kém tự nhiên hơn thư viện preset hay Text to Motion của Meshy.
- Điểm silhouette không đánh giá được mặt và biểu cảm. Phần này do critic thị giác (5d) chấm, nếu được bật.
- Critic thị giác là một model, không phải phép đo: điểm có thể dao động giữa các lần gọi. Vì vậy nó chỉ tư vấn và chỉ quyết định khi chênh rõ (≥ 1 điểm).
- Ảnh có watermark của tác giả (ví dụ @higgs.osiris). Muốn dùng thương mại thì cần xin phép. Ảnh sẽ được gửi lên provider cloud khi dùng L2, và lên Gemini API hoặc Claude API khi bật critic.
