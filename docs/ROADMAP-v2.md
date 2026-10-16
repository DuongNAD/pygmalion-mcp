# Roadmap v2: từ "đạt test trên mock" đến "dùng được với nhân vật thật"

Lập ngày 2026-10-17, dựa trên việc đọc toàn bộ mã nguồn, chạy lại test suite trên máy hiện tại và tra cứu tài liệu chính thức của các bên thứ ba. Roadmap cũ (`docs/ROADMAP.md`, L1 → L3) đã xong; tài liệu này là giai đoạn tiếp theo.

## 0. Cách dùng tài liệu này

- Mỗi việc là một **card** (khối code ở mục 4), viết bằng tiếng Anh theo đúng quy ước card của `docs/ROADMAP.md`: tự đủ ngữ cảnh, có `CONTEXT`, `GOAL`, `FILES`, `TASKS`, `DONE WHEN` và khối `COMMON RULES` ở cuối (gồm cả yêu cầu báo cáo `REPORT`). Bạn dán nguyên khối, không cần giải thích thêm. Mọi số liệu trong card (dòng code, giá, endpoint) đã kiểm chứng ngày 2026-10-17; agent vẫn được dặn đối chiếu lại khi làm.
- Ba cách giao việc:
  1. Agent mở trong repo (Claude Code, Gemini CLI, Codex, Antigravity): sau khi file này đã được commit, gõ `Làm card G1-4 trong docs/ROADMAP-v2.md`.
  2. Dán nguyên khối card vào khung chat của AI.
  3. AGY qua mcp-agy: dán card vào `agy_start_task`, rồi kiểm tra bằng `agy_get_diff` và `agy_run_tests` trước khi nhận.
- Mỗi card làm trên **một nhánh riêng từ `main`**. Chỉ chạy song song các card khác đợt hoặc cùng đợt nhưng khác file (cột "File chính" ở mục 3), và mỗi agent phải có worktree riêng.
- Làm xong một card: giao **prompt review** (mục 5) cho một AI khác, rồi mới merge.
- Card có nhãn **[cần bạn]** cần bạn cung cấp thứ gì đó (API key, ảnh sheet, quyết định) trước khi giao.

## 1. Hiện trạng đã kiểm chứng (2026-10-17)

**Điểm mạnh**
- Kiến trúc chặt: 12 tool nghiệp vụ, envelope thống nhất, `next_action` do `state.suggest_next` quyết định, Blender chạy qua subprocess với giao thức file JSON. Model điều phối yếu vẫn đi đúng đường.
- QA phong phú: IoU silhouette, khe hở tư thế, toàn vẹn hình học, kiểm tra biến dạng rig (vung tay/đùi 40°) kèm tự sửa trọng số, preview + `motion_check` từng clip, critic thị giác độc lập.
- Test suite chạy lại trên máy này: **431 passed, 1 skipped** (test live Meshy), 9 phút 34 giây, Python 3.11 hệ thống, Blender 4.5.4 LTS.

**Thực tế cần biết**
- PR [DuongNAD/pygmalion-mcp#1](https://github.com/DuongNAD/pygmalion-mcp/pull/1) (nhánh `fix/meshy-pipeline`, 6 commit, gồm toàn bộ L3 QA và critic) **chưa merge** vào `main`.
- **Chưa từng chạy với API thật** (PR #1 tự ghi "Chưa chạy với Meshy thật"). Mọi test provider dùng mock dựng theo spec do agent tự suy ra. Demo `aria` là blockout trên một sheet tổng hợp 10 KB; `projects/aria/project.json` đã lỗi thời (`qa: null`, `reference.pose: null`, file export bị ghi nguồn `external`).
- Máy hiện tại có **NVIDIA RTX 5060 Ti 16 GB**, i5-14600KF, 48 GB RAM. `docs/ARCHITECTURE.md` vẫn ghi "Máy Mac không có GPU NVIDIA nên không chạy local TRELLIS/Hunyuan3D".
- Không có file `.env`, nên server MCP trong repo này đang chạy **Level 1** (chỉ 5 tool).

**Lỗ hổng chính**

| # | Vấn đề | Bằng chứng | Card |
|---|---|---|---|
| 1 | Tên tiếng Việt làm hỏng `project_create` trên Windows | `project_create(name="Thủy Tiên")` → `INTERNAL UnicodeEncodeError` ở `state.save` (`write_text` dùng cp1252), để lại thư mục dở `th-y-ti-n/` | G0-1 |
| 2 | Server chạy nhờ gói không khai báo | `pyproject.toml` ghim `mcp<2` nhưng Python hệ thống có `mcp 2.2.0` (đã đổi FastMCP thành MCPServer), nên `server.py` rơi về gói bên thứ ba `fastmcp 4.0.10`; ghim không có tác dụng vì CLAUDE.md cài với `--no-deps`; mcp 2.x không còn kéo theo `httpx` mà providers cần. `.venv` (uv, Python 3.12) thiếu pytest và google-genai | G0-2 |
| 3 | Metadata tool sai | Mọi tool `openWorldHint=False`, kể cả tool gọi API ngoài (`tools_core.py:26`, `tools_full.py:38`); `auto_rig` bắt buộc `mesh_path` dù TOOLS.md ghi `=None`; `ActionName` thừa; `INSTRUCTIONS` dài 609 ký tự trong khi Codex chỉ giữ 512 | G0-2 |
| 4 | Không có CI | Không có `.github/workflows` | G0-4 |
| 5 | Blender: hai lỗi âm thầm và chưa chạy trên 5.2 LTS | `hasattr(bpy.ops.wm, "fbx_import")` luôn đúng nên nhánh dự phòng không bao giờ chạy; `cf_secondary.py:559` gọi `act.fcurves.remove` (đã bị bỏ ở Blender 5) bên trong `except: pass`. Bản LTS hiện hành là 5.2.2 | G0-5 |
| 6 | Adapter Tripo, Rodin, Hunyuan hỏng với API thật | Đối chiếu docs ngày 2026-10-17: Hunyuan gửi `image_url` trong khi field bắt buộc là `input_image_url`, và không bật texture; Rodin đọc `status`/`url` sai chỗ nên job treo tới khi hết 20 phút; Tripo gửi `inputs` dạng dict cho v2 (v2 cần 4 slot `files`), mà v2 đang bị khai tử | G1-6, G1-7, G1-8 |
| 6b | Meshy đúng endpoint nhưng lệch chi tiết | Khi rig bằng `model_url`, code gửi ảnh sheet làm texture (docs yêu cầu texture base-color đã UV); HTTP 429 (giới hạn tốc độ) bị báo là "hết credit, dùng blockout"; chỉ gửi ảnh front làm texture | G1-2 |
| 7 | `side_faces` không bao giờ được ghi | `reference_metrics` không tạo key này, nhưng `blockout.py:110` và `tripo.py:157` đọc nó → Tripo luôn gán ảnh side là "left"; `tools_core` không truyền `reference` cho provider | G1-4 |
| 8 | Không có trần chi phí | Mỗi lần generate/rig/animate/Text to Motion đều tốn credit; chỉ có `MAX_CANDIDATES` | G1-3 |
| 9 | Không có thước đo trên sheet thật | Chỉ có sheet đồ chơi (hình không có tay, `tests/conftest.py`) | G1-1, G1-5 |
| 10 | Lời gọi dài dễ bị client cắt | `blender.run` timeout 600 s mỗi lần; `pipeline_run` chỉ kiểm `max_seconds` giữa các bước; không có progress notification. Codex và Cursor cắt ở khoảng 60 s, Gemini CLI ở 600 s | G2-1 |
| 11 | State dễ hỏng | Không khóa `project.json`; SDK v2 chạy tool đồng bộ trên nhiều thread nên hai lời gọi có thể ghi đè nhau (cùng dùng một file `project.json.tmp`); không có migration schema; `calls.jsonl` trộn lời gọi của agent với lời gọi nội bộ của `pipeline_run` nên thống kê model điều phối sai | G0-2, G2-2 |
| 12 | Policy lãng phí khi không có key | best < 0.75 → `generate_mesh` lại blockout, ra mesh y hệt (blockout không có seed, `state.py:362`) | G2-2 |
| 13 | Tách sheet chỉ là heuristic đơn giản | side = box hẹp nhất, front = box trái nhất (`sheet.py:95`); sheet thật có chữ, nền xám, góc 3/4, nhiều hàng | G2-3 |
| 14 | Chỉ chấm silhouette | Màu, texture, mặt chỉ có critic (tùy chọn) xem | G3-1 |
| 15 | Nhánh offline còn thô | Blockout chỉ có màu đỉnh; animation offline có 5 kiểu, tên lạ âm thầm thành "sway" (`merge_animations.py:258`) | G3-2, G3-3 |
| 16 | Chỉ xuất GLB cho rig/animation | Không có FBX/VRM cho Unity, Unreal, VTuber | G3-4 |
| 17 | Motion QA còn thiếu | Không đo trượt chân, khe nối vòng lặp, tay xuyên thân | G3-5 |
| 18 | Chưa đo model điều phối | `docs/reports/driver-default.md` chỉ có 1 lượt chạy | G4-1 |
| 19 | Test khó bảo trì | Tên file theo milestone (`test_m3_adversarial.py`, `test_l2_challenger.py`…) | G4-2 |

## 2. Việc của bạn trước khi giao card

1. **Merge PR #1** vào `main` (đã xanh 431/432 trên máy này). Mọi nhánh mới tách từ `main` sau khi merge.
2. **Tạo `.env`** từ `.env.example`: `CHARFORGE_LEVEL=3`, `MESHY_API_KEY`, `GEMINI_API_KEY` (critic). Key tùy chọn: `FAL_KEY` (Hunyuan3D qua fal, khoảng 0,4 USD mỗi mesh, card G1-7), `TRIPO_API_KEY` (G1-6), `RODIN_API_KEY` (cần gói Business, G1-8). Không commit `.env`.
3. **Gom bộ sheet thật (golden set)** cho G1-1: 8 đến 12 ảnh mà bạn có quyền dùng (tự vẽ, đặt vẽ, hoặc giấy phép CC). Nên phủ các trường hợp: A-pose có khe tay rõ; tay buông sát thân; T-pose; váy dài; tóc dài/đuôi ngựa; chibi; sheet 4–5 góc có góc 3/4; nền xám hoặc giấy; có chữ "FRONT/SIDE/BACK"; ảnh side quay trái và quay phải.
4. **Chốt ngân sách credit** cho G1-3 và G1-5 (ví dụ: tối đa 2 lần sinh mesh, 1 lần rig, 1 lần animate mỗi nhân vật).
5. **Chốt hướng local GPU** sau khi đọc mục G3-6: có muốn chạy model sinh 3D trên RTX 5060 Ti không (tốn 30–60 GB đĩa, có ràng buộc giấy phép).

## 3. Lộ trình

| Card | Việc | Ưu tiên | Phụ thuộc | File chính | Cỡ |
|---|---|---|---|---|---|
| G0-1 | Windows + Unicode | P0 | — | state, config, envelope, blender, imageio (mới), sheet/score/blockout/critic (IO ảnh) | S |
| G0-2 | Runtime MCP, đóng gói, metadata tool | P0 | — | server, tools_core, tools_full, pipeline_run, `__main__`, pyproject, .mcp.json | M |
| G0-3 | Rà tài liệu cho đúng sự thật | P0 | G0-2 | README, CLAUDE.md, docs/* | S |
| G0-4 | CI GitHub Actions | P0 | — | .github/workflows, marker test | S |
| G0-5 | Tương thích Blender 5.2 LTS (giữ 4.5) | P1 | G0-1 | bpy_scripts/*, config (tìm Blender) | M |
| G1-1 | Golden set + benchmark | P0 | — | scripts/bench.py, bench/ | M |
| G1-2 | Meshy khớp docs + xử lý lỗi HTTP chung | P0 | G0-2 | providers/base, providers/meshy, tools_full (rig fallback) | M |
| G1-3 | Trần chi phí API trả phí | P0 | G1-2, G2-2 | config, state, tools_core, tools_full, server | M |
| G1-4 | `side_faces` cho reference | P0 | G0-1 | score, sheet, blockout, providers/tripo, tools_core | S |
| G1-5 | Chạy thật với Meshy + ghi fixture thật [cần bạn] | P0 | G1-1, G1-2, G1-3 | báo cáo, tests/contract/fixtures | M |
| G1-6 | Tripo chuyển sang API v3 | P1 | G1-2, G1-4 | providers/tripo | M |
| G1-7 | Hunyuan3D qua fal.ai (sửa + multi-view + v3.1) | P1 | G1-2, G1-4 | providers/fal (mới), providers/hunyuan | M |
| G1-8 | Rodin: sửa poll/download, chọn tier | P2 | G1-2 | providers/rodin | S |
| G2-1 | Lời gọi dài: progress, timeout, hủy | P1 | G0-2 | blender, pipeline_run, server, tools_* | M |
| G2-2 | State: khóa, migration, telemetry, policy blockout | P1 | G0-2 | state, envelope, pipeline_run, scripts/driver_report.py | M |
| G2-3 | Tách sheet bền với sheet thật | P1 | G1-1, G1-4 | sheet, test sheet | L |
| G3-1 | Chấm màu, tỉ lệ, mặt | P1 | G1-1 | score, server, state, critic, render_views | M |
| G3-2 | Chiếu texture từ sheet lên mesh không texture | P2 (P1 nếu làm G3-6) | G3-1 | bpy bake_texture (mới), tools_core | L |
| G3-3 | Thư viện chuyển động + retarget offline | P1 | — | bpy retarget (mới), merge_animations, tools_full | L |
| G3-4 | Xuất FBX/VRM cho engine | P1 | — | bpy export_rig (mới), tools_full | M |
| G3-5 | Motion QA: trượt chân, vòng lặp, xuyên thân | P1 | — | preview_clips, tools_full | M |
| G3-6 | Sinh mesh local trên GPU (Hunyuan3D-2mv) [cần bạn] | P2 | G1-2, G3-2 | workers/ (mới), local_worker, providers/local | L |
| G3-7 | Rig local (Make-It-Animatable) [cần bạn] | P2 | G3-6 | workers/mia, tools_full | M |
| G3-8 | Text to Motion local (Kimodo) [cần bạn] | P2 | G3-3, G3-6 | workers/kimodo, tools_full | L |
| G4-1 | Đo model điều phối bằng LLM thật | P1 | G0-2, G2-2 | scripts/driver_eval.py | M |
| G4-2 | Sắp xếp lại test suite | P2 | sau các card code | tests/* | M |
| G4-3 | Phát hành 0.2.0 + showcase thật | P2 | G1-5 | scripts/render_showcase.py, README, CHANGELOG | S |

**Thứ tự đề xuất theo đợt** (trong một đợt có thể chạy song song, mỗi agent một worktree):

1. Đợt 1: G0-1 ‖ G0-4 ‖ G1-1
2. Đợt 2: G0-2 ‖ G0-5
3. Đợt 3: G0-3 ‖ G1-4 ‖ G3-4
4. Đợt 4: G1-2 ‖ G2-2 ‖ G3-5
5. Đợt 5: G1-3 ‖ G1-7 ‖ G3-3
6. Đợt 6: G1-5 (cần bạn) ‖ G1-6 ‖ G2-1
7. Đợt 7: G2-3 ‖ G1-8 ‖ G3-1
8. Đợt 8: G4-1 ‖ G3-2
9. Đợt 9: G3-6 → (G3-7 ‖ G3-8), chỉ khi bạn chọn hướng local
10. Đợt 10: G4-2 → G4-3

Mốc kiểm tra: hết đợt 3 là nền ổn định (CI xanh, chạy được trên Windows, tài liệu đúng); hết đợt 6 là có số liệu thật từ Meshy và các provider khác đã khớp docs; hết đợt 10 là bản 0.2.0.

Nếu chỉ muốn làm phần cốt lõi: G0-1, G0-2, G0-4, G1-1, G1-4, G1-2, G1-3, G1-5, G2-2. Chín card này biến dự án từ "đạt test trên mock" thành "đã chạy thật, có số đo, an toàn chi phí".

## 4. Các card

### G0-1 · Windows + Unicode (P0, cỡ S)
Vì sao: người dùng đặt tên nhân vật tiếng Việt là hỏng ngay bước đầu. Phụ thuộc: không.

```text
CARD G0-1 — Windows and Unicode hardening

CONTEXT
Verified on 2026-10-17 (Windows 11, Python 3.11, locale encoding cp1252):
server.project_create(sheet_path=<png>, name="Thủy Tiên") returns INTERNAL
"UnicodeEncodeError: 'charmap' codec can't encode character '\u1ee7'", raised by state.save()
(tmp.write_text(json.dumps(p, indent=2, ensure_ascii=False)) uses the locale encoding). It leaves a
half-created folder th-y-ti-n/ (views/, meshes/, renders/, sheet.png, project.json.tmp), and the slug is
poor ("th-y-ti-n" instead of "thuy-tien"). The users are Vietnamese. cv2.imread/cv2.imwrite also fail on
Windows for non-ASCII paths (for example CHARFORGE_HOME under C:\Users\Dương\...).

GOAL
CharForge works on Windows with any locale and with non-ASCII names and paths.

FILES
src/charforge/state.py, config.py, envelope.py, blender.py, server.py, tools_full.py, providers/tripo.py,
sheet.py, score.py, blockout.py, critic.py, src/charforge/imageio.py (new), scripts/driver_report.py,
the --result writers in src/charforge/bpy_scripts/*.py, tests/test_unicode.py (new).

TASKS
1. Text IO: pass encoding="utf-8" to every read_text / write_text / open() of a text file in
   src/charforge and scripts/ (config._load_dotenv uses "utf-8-sig" so a BOM is accepted). In every
   bpy_scripts/*.py, the --result JSON file is written with encoding="utf-8".
2. state.slugify(): transliterate first (unicodedata NFKD, drop combining marks, map đ/Đ to d), then the
   existing regex and 40-char cap: "Thủy Tiên" -> "thuy-tien", "Đặng Mỹ" -> "dang-my".
3. state.create(): atomic. If anything fails after the folder exists, delete the folder and re-raise.
   A leftover folder without project.json still counts as taken (keep the -2, -3 suffix rule).
4. Image IO: src/charforge/imageio.py with imread(path, flags=cv2.IMREAD_UNCHANGED) (np.fromfile +
   cv2.imdecode; returns None when missing or unreadable, like cv2) and imwrite(path, img, params=None)
   (cv2.imencode + ndarray.tofile; returns bool). Replace every cv2.imread / cv2.imwrite in src/charforge
   (not in bpy_scripts) with these helpers.
5. blender.run(): check that non-ASCII paths reach Blender intact (subprocess list arguments) and that
   Blender can import/export from such a folder.

RULES
- ASCII names and paths must behave exactly as before (same ids, same file names).
- No new dependency.

DONE WHEN
- tests/test_unicode.py (no Blender): project_create with "Thủy Tiên" -> ok, project_id "thuy-tien",
  project_status returns the original name; an injected failure after the folder exists (monkeypatch
  shutil.copy2 to raise) leaves no folder behind; imageio round-trips a PNG under tmp_path / "ảnh thử".
- One test marked blender runs sheet_split + score_mesh on the synthetic sheet with CHARFORGE_HOME set
  to a folder named "dự án".
- `git grep -n "read_text()" -- src scripts` and a review of write_text/open calls show no text IO
  without an explicit encoding.
- Full suite green on Windows.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g0-1-unicode). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G0-2 · Runtime MCP: SDK chính thức v2, đóng gói đúng, metadata tool đúng (P0, cỡ M)
Vì sao: server đang chạy nhờ một gói bên thứ ba không khai báo, cài sạch trên máy khác có thể hỏng, và client thấy sai thông tin tool. Phụ thuộc: không (nên làm sau G0-1 vì cùng `server.py`).

```text
CARD G0-2 — MCP runtime: official SDK v2 (v1 still works), honest packaging, correct tool metadata

CONTEXT (verified on this machine and in the SDK docs, 2026-10-17)
- .mcp.json runs `python -m charforge` with the system Python 3.11, which has mcp 2.2.0 and the
  third-party fastmcp 4.0.10. In mcp 2.x, `mcp.server.fastmcp` only raises ModuleNotFoundError ("FastMCP
  was renamed to MCPServer (from mcp.server.mcpserver import MCPServer)"), so server.py, tools_core.py
  and tools_full.py silently fall back to the third-party fastmcp. It is not a declared dependency and it
  changes the wire format (every dict return becomes structuredContent with a generic outputSchema).
  pyproject pins mcp>=1.28,<2, but CLAUDE.md installs with --no-deps, so the pin never applies. The repo's
  .venv is a uv venv on Python 3.12 with mcp 1.30.0 and no pytest, while CLAUDE.md says the venv uses
  --system-site-packages on 3.11.
- Official SDK v2 (2.0.0 released 2026-07-28, latest 2.3.0; v1.30 still gets security fixes):
  `from mcp.server.mcpserver import MCPServer, Context, Image`; no compat module; `mcp.types` still works
  (ToolAnnotations); `@mcp.tool(annotations=...)` and `@mcp.prompt(name=, description=)` unchanged;
  pass `instructions=` as a keyword (positionally it becomes the title) and pass `version=`; a
  `ctx: Context` parameter is injected (works through tool_entry's functools.wraps); a bare `-> dict`
  return gives text content only, `dict[str, Any]` adds an outputSchema + structured content; sync tools
  run on worker threads, so calls can now run concurrently (state.save() always writes the same
  project.json.tmp, which collides between threads); mcp.run() unchanged; the SDK no longer reads .env or
  MCP_* env vars; mcp 2.x depends on httpx2, so `import httpx` in providers/base.py loses its transitive
  source.
- Tool metadata bugs: every tool is annotated openWorldHint=False, including generate_mesh, auto_rig,
  animate and pipeline_run (external APIs) and score_mesh / animate when the critic is on
  (server.py:34, tools_core.py:26, tools_full.py:38); the annotations are defined three times; auto_rig
  requires mesh_path although docs/TOOLS.md documents mesh_path=None; tools_full.ActionName is unused.
- server.INSTRUCTIONS is 609 characters; Codex CLI keeps only the first 512.
- There is no .env on this machine and .mcp.json sets no env, so the server runs at Level 1 (5 tools).

GOAL
CharForge runs on the official MCP SDK v2 (still importable on v1), installs with one reproducible
command that brings every real dependency, exposes Level 3 in the repo's client configs, and advertises
correct tool metadata.

FILES
src/charforge/mcpcompat.py (new), server.py, tools_core.py, tools_full.py, pipeline_run.py,
__main__.py, config.py (env_report), state.py (save() temp name only), pyproject.toml, uv.lock,
.mcp.json, .gemini/settings.json, tests/test_server_contract.py (new), CLAUDE.md (commands only),
docs/TOOLS.md (annotations line, auto_rig signature).

TASKS
1. mcpcompat.py: try `from mcp.server.mcpserver import MCPServer as Server` (mcp ≥ 2), except
   ImportError `from mcp.server.fastmcp import FastMCP as Server` (mcp 1.x). Remove the third-party
   fastmcp fallback everywhere. Construct with keywords: Server("charforge", instructions=INSTRUCTIONS,
   version=<package version>).
2. Annotations defined once: READ_ONLY, LOCAL_WRITE, OPEN_WORLD (openWorldHint=True). OPEN_WORLD on
   generate_mesh, auto_rig, animate, pipeline_run and score_mesh.
3. Return type: keep `-> dict` (JSON text content, as v1 clients saw it) for every tool unless you add a
   deliberate output schema; never send image content and structuredContent in the same result.
4. auto_rig(mesh_path=None): default to outputs.glb; without it raise CFError("MESH_NOT_FOUND", ...,
   "Run mesh_export first, or pass mesh_path."). Remove ActionName.
5. INSTRUCTIONS ≤ 512 characters, same rules in the same order of importance.
6. state.save(): a unique temp file per write (e.g. project.json.<pid>.<thread-id>.tmp) + os.replace, so
   concurrent threads cannot clobber each other's temp file (the real project lock is card G2-2).
7. pyproject: mcp>=1.28,<3; declare httpx; keep the extras; regenerate uv.lock. Document ONE dev setup in
   CLAUDE.md (for example `uv venv --python 3.11` then `uv pip install -e ".[dev,critic]"`), used for
   tests and for running the server. Make .mcp.json and .gemini/settings.json launch that environment
   (choose `uv run ...` or a documented interpreter path and say why), with env {"CHARFORGE_LEVEL": "3"}
   and a 900000 ms timeout where the format supports it.
8. `python -m charforge --check` also prints the MCP SDK and version and the tools registered at the
   configured level (count + names).
9. tests/test_server_contract.py: run the server in-process with the SDK's in-memory client (or over
   stdio as a subprocess) and assert: 12 tools at level 3 and 5 at level 1; annotations as in task 2;
   every tool's input schema has exactly the parameters docs/TOOLS.md lists; instructions ≤ 512 chars;
   prompt charforge_playbook returns playbook.md; project_status on a missing project returns the
   envelope with ok=false and code PROJECT_NOT_FOUND.
10. Run the full suite once with mcp 2.x and the contract test also in a venv with mcp 1.30.

RULES
- No behavior change in any tool except auto_rig's optional mesh_path.
- The wider documentation pass is card G0-3; here only CLAUDE.md commands and TOOLS.md lines you touch.

DONE WHEN
- tests/test_server_contract.py passes with mcp 2.x and with mcp 1.30; full suite green with mcp 2.x.
- `python -m charforge --check` shows the SDK and 12 tools when CHARFORGE_LEVEL=3.
- In a fresh venv, `pip install -e ".[dev]"` then
  `python -c "import charforge.server, charforge.providers"` works (no system site-packages).

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g0-2-mcp-runtime). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G0-3 · Tài liệu nói đúng sự thật (P0, cỡ S)
Vì sao: tài liệu đang hứa nhiều hơn thực tế và hướng dẫn cài đặt sai. Phụ thuộc: G0-2 (lệnh cài/test mới).

```text
CARD G0-3 — Make the documentation tell the truth

CONTEXT
Docs drifted from reality:
- README says "Python 3.11+ và Blender 4.2+", ARCHITECTURE says "Blender 5.x"; the suite is verified on
  Blender 4.5.4 LTS (Blender 5.x only after card G0-5).
- ARCHITECTURE §2 says the dev machine is a Mac without an NVIDIA GPU; it is now Windows 11 with an
  RTX 5060 Ti 16 GB.
- CLAUDE.md says the venv uses --system-site-packages with Python 3.11; the repo's .venv is a uv venv on
  Python 3.12 without pytest, and tests actually run with the system Python 3.11.
- docs/ROADMAP.md still reads as the active plan; README's level table says "Đã hoàn thành & kiểm thử
  100%" although no provider was ever called live.

GOAL
Every statement in README.md, CLAUDE.md, .env.example and docs/ matches the code and the environment
after card G0-2.

FILES
README.md, CLAUDE.md, .env.example, docs/ARCHITECTURE.md, docs/TOOLS.md (factual fixes only),
docs/ROADMAP.md (one header line).

TASKS
1. Versions: the Python and Blender versions actually tested (CI matrix from G0-4; Blender 5.x status
   from G0-5 when merged).
2. Setup: the install and test commands decided in G0-2; how to run at Level 3 (.env, or the env block
   in .mcp.json / .gemini/settings.json); Windows notes (PowerShell venv activation, BLENDER_PATH,
   Blender auto-discovery under C:\Program Files\Blender Foundation).
3. Honest status: in README's level table separate "kiểm thử với mock" from "đã chạy thật"; link
   docs/ROADMAP-v2.md and the reports in docs/reports/.
4. ARCHITECTURE §2: hardware note (local GPU optional, see card G3-6); re-check §11 limits.
5. docs/ROADMAP.md: one line at the top: completed, see docs/ROADMAP-v2.md.
6. Human docs stay Vietnamese; do not touch runtime strings (docstrings, error.fix, playbook).

DONE WHEN
- Following README from a clean clone on Windows reaches `python -m charforge --check` with level 3 and
  all deps OK, with no missing step (do it in a scratch folder and paste the output in the REPORT).
- `git grep -n "4.2+\|Blender 5.x\|system-site-packages"` returns only lines you meant to keep.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g0-3-docs). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G0-4 · CI GitHub Actions (P0, cỡ S)
Vì sao: 432 test nhưng chỉ chạy tay; mọi card sau cần lưới an toàn tự động. Phụ thuộc: không.

```text
CARD G0-4 — Continuous integration on GitHub Actions

CONTEXT
github.com/DuongNAD/pygmalion-mcp has no CI (only a GitGuardian check). The suite has 432 tests: about a
third need Blender (most carry the `blender` marker with a skipif, but some only call pytest.skip()
inside the test and have no marker), 1 needs live APIs (marker `live`). A full local run takes 9.5 min
on Windows with Blender 4.5.4 LTS. tests/conftest.py already forces CHARFORGE_LEVEL=1 and
CHARFORGE_CRITIC=off per test.

GOAL
Every push and pull request runs the fast suite on Linux and Windows, and the Blender suite on Linux.

FILES
.github/workflows/ci.yml (new), pyproject.toml (pytest markers only), test files (markers only),
README.md (badge + one line).

TASKS
1. Job `fast`: matrix os [ubuntu-latest, windows-latest] × python ["3.11", "3.12"]. Install with the
   command CLAUDE.md documents (if card G0-2 is not merged yet: pip install -e ".[dev]"). Run
   `python -m pytest -q -m "not blender and not live"`. The runners have no Blender, so every test that
   needs Blender must skip cleanly: find the ones that error instead and give them @pytest.mark.blender
   plus the same skipif used in tests/test_render.py.
2. Job `blender`: ubuntu-latest, python 3.11, with the Blender version as a matrix value (start with
   4.5 LTS; card G0-5 adds 5.2 LTS). Download the official Linux x64 build from download.blender.org
   (verify the exact URL, pin version and SHA-256, cache it with actions/cache),
   install the shared libraries headless Blender needs (libegl1, libgl1, libxi6, libxkbcommon0,
   libxrender1, libsm6, libxxf86vm1, mesa drivers), export BLENDER_PATH, run `python -m charforge
   --check`, then `python -m pytest -q -m "not live"`. If Workbench renders fail headless, try xvfb-run
   and write down in a YAML comment what worked.
3. Set CHARFORGE_HOME to a workspace folder; on failure upload the pytest log and renders as artifacts.
4. Triggers: push to main, pull_request, workflow_dispatch; the blender job also nightly (cron).
   One concurrency group per ref with cancel-in-progress.

RULES
- Change tests only by adding markers.
- Pushing a branch to see the workflow run is allowed only if the user said so in this session;
  otherwise validate locally (actionlint if available) and list what the maintainer must check.

DONE WHEN
- Both jobs green on a pushed branch (paste the run URL), or the workflow validated locally with the
  remaining checks listed for the maintainer.
- `python -m pytest -q -m "not blender and not live"` passes locally with BLENDER_PATH pointing to a
  non-existent file (simulates the runner).

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g0-4-ci). Small commits with conventional
  messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G0-5 · Tương thích Blender 5.2 LTS, giữ 4.5 LTS (P1, cỡ M)
Vì sao: bản LTS hiện hành là 5.2; Blender 5 đã bỏ API mà code đang dùng, và có hai lỗi âm thầm (import FBX hỏng trên 4.2–4.4 dù README hứa "4.2+", xóa fcurve không có tác dụng trên 5.x). Phụ thuộc: G0-1 (cùng các script Blender). Cần bạn cho phép tải Blender 5.2.2 bản portable (khoảng 400 MB).

```text
CARD G0-5 — Blender 5.2 LTS compatibility (keep 4.5 LTS)

CONTEXT (Blender release notes, checked 2026-10-17)
The suite is verified only on Blender 4.5.4 LTS (supported until July 2027). The current stable and LTS
release is 5.2 LTS (5.2.2, 2026-09-15, supported until July 2028); 5.3 is in beta. Changes that touch
src/charforge/bpy_scripts:
- Slotted actions (4.4) and Blender 5.0's removal of Action.fcurves, Action.groups and Action.id_root
  (id_root → slot.target_id_type; action_group= → group_name=). Assigning animation_data.action can leave
  no slot assigned: set ad.action_slot (and NlaStrip.action_slot) explicitly. Bug: cf_secondary.py
  (around line 559) calls act.fcurves.remove(fc) inside `except: pass`, so on 5.x the removal silently
  does nothing. Portable access for 4.5 and 5.x: act.layers → strips → channelbags (or
  strip.channelbag(slot, ensure=True), Action.fcurve_ensure_for_datablock()), then channelbag.fcurves.
- FBX: the C++ importer bpy.ops.wm.fbx_import exists since 4.5 and is the default in 5.0
  (import_scene.fbx is legacy); there is no C++ exporter, keep export_scene.fbx. Bug: rig.py, fit.py,
  optimize.py, export.py and cf_humanoid.py test `hasattr(bpy.ops.wm, "fbx_import")`, which is always
  True (operator lookup is lazy), so their fallback never runs on older versions; render_views.py's
  try/except AttributeError is the correct form.
- No breaking change reported for glTF import/export (export_animation_mode='ACTIONS', export_skins),
  the OBJ/PLY/STL importers or ARMATURE_AUTO. Workbench light 'FLAT' and color_type 'VERTEX' still exist.
  5.0 renamed BLENDER_EEVEE_NEXT to BLENDER_EEVEE; mathutils is float32 since 5.0; 5.1 moved to Python
  3.13; BGL is gone.

GOAL
Every bpy script works on Blender 4.5 LTS and 5.2 LTS with the same results; 4.5 is the stated minimum.

FILES
src/charforge/bpy_scripts/*.py, src/charforge/config.py (blender_exe, version report),
tests/test_blender_versions.py (new), README.md and CLAUDE.md (the minimum-version line only),
.github/workflows/ci.yml (Blender matrix value only, if card G0-4 is merged).

TASKS
1. Ask the user before downloading. Install Blender 5.2.2 portable next to 4.5 (never uninstall 4.5).
   Run the full suite with BLENDER_PATH pointing at each version and list every failure before changing
   code.
2. One shared importer helper (try the wm.* operator, except AttributeError fall back to the legacy one)
   used by every script; remove the hasattr checks.
3. Animation code (merge_animations.py, preview_clips.py, cf_secondary.py, cf_humanoid.py, rig.py):
   channelbag-based fcurve access and removal; explicit action_slot on animation_data and NLA strips; no
   bare `except: pass` around API calls (catch the specific exception and report it in the result JSON).
4. Check the deformation QA tolerances (millimetre-level vertex distances) for float32 effects; change
   them only if a test shows a difference.
5. config.blender_exe(): when several versions exist under C:\Program Files\Blender Foundation (or
   BLENDER_PATH is unset), pick the newest supported (4.5 ≤ version < 6) deterministically;
   `python -m charforge --check` prints the version in use.
6. tests/test_blender_versions.py (marker blender): a fast smoke test (render_views on the box mesh,
   rig.py on the bpy_fixtures character, merge_animations procedural + preview_clips) asserting the same
   key numbers within a tolerance on any supported version; document how to run it per version.
7. If card G0-4 is merged, add 5.2.2 to the Blender matrix of .github/workflows/ci.yml.

DONE WHEN
- Full suite green with Blender 4.5.4 and with 5.2.2 (paste both summaries).
- A test that failed before the fix proves secondary-motion fcurves are really removed on 5.2.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g0-5-blender52). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G1-1 · Golden set + benchmark (P0, cỡ M)
Vì sao: không có thước đo thì mọi nâng cấp chất lượng chỉ là cảm giác. Phụ thuộc: không (ảnh thật do bạn cung cấp, xem mục 2).

```text
CARD G1-1 — Golden set and benchmark harness

CONTEXT
CharForge has never been measured on real turnaround sheets. Tests use a toy sheet drawn in
tests/conftest.py (head, torso, skirt, legs; no arms). The only quality report is
docs/reports/driver-default.md (one run on a synthetic sheet). Later cards (sheet split, scoring,
providers, texture) need one repeatable measuring stick.

GOAL
`python scripts/bench.py` runs the pipeline on a set of sheets and writes a comparable report, so every
change can be judged by numbers before and after.

FILES
scripts/bench.py (new), scripts/make_synthetic_sheets.py (new), bench/README.md (new, Vietnamese),
bench/manifest.json (new), bench/sheets/ (synthetic PNGs only), .gitignore, tests/test_bench.py (new),
docs/reports/bench-<date>.md (generated baseline).

TASKS
1. bench/manifest.json: a list of {id, file (relative to bench/sheets/), height_cm, license, source,
   notes, expect: {views: 3, side_faces?: "left"|"right", arms_apart?: bool}, assign?: [f, s, b]}.
   bench/sheets/ is git-ignored except files whose manifest license is "synthetic", "CC0" or "CC-BY*";
   bench/runs/ is always git-ignored.
2. scripts/make_synthetic_sheets.py writes 4 deterministic synthetic sheets that, unlike conftest, have
   arms: A-pose with clear arm gaps; arms down touching the hips; T-pose; long skirt + ponytail on a
   light-gray paper background with "FRONT / SIDE / BACK" labels. cv2/numpy only; registered in the
   manifest with license "synthetic".
3. scripts/bench.py (argparse): --manifest, --only id1,id2, --level 3, --provider blockout|auto|<name>
   (default blockout), --rig/--no-rig, --animate/--no-animate, --max-calls 40, --yes-spend,
   --out docs/reports/bench-<YYYYMMDD-HHMM>.md. It sets CHARFORGE_HOME to bench/runs/<timestamp>/ and
   calls the tool functions in-process like the tests do: project_create, then pipeline_run
   (max_seconds=540) while status == "running". If pipeline_run is "blocked" on sheet_split and the
   manifest has `assign`, call sheet_split(assign=...) once and continue; otherwise record "blocked".
   Paid providers run only when --provider is not blockout AND --yes-spend is given.
4. Per sheet record: split confidence, views found, reference pose ok, side_faces (if present), best
   total and per-view IoU, passed, stop_reason, iterations, integrity ok + issue count, rig_check
   passed/repaired/issue count, motion_check passed + worst stretch_p99 and ground_cm, critic overall
   when present, credits consumed (sum of jobs[*].consumed_credits), wall time per tool (calls.jsonl),
   errors by code. Build one PNG per sheet: sheet thumbnail | color renders of the best mesh (front,
   side, back) | first clip preview (use render_views and score.contact_sheet).
5. Write bench/runs/<ts>/summary.json and the markdown report (Vietnamese headings): an aggregate table
   (pass rate, mean total, rig pass rate, motion pass rate, mean time, credits) and one row per sheet
   linking its PNG. A sheet that raises becomes a row with the error; it never stops the run.
6. bench/README.md (Vietnamese): adding a sheet, the license rule, running, comparing two reports.

RULES
- Do not change src/charforge in this card; if you hit a crash, report it with a minimal repro.
- No new dependency.

DONE WHEN
- `python scripts/make_synthetic_sheets.py` then `python scripts/bench.py --provider blockout --rig
  --animate` completes on the 4 synthetic sheets (Blender installed) and writes the report.
- tests/test_bench.py (marker blender) runs bench.py on one synthetic sheet with --no-rig --no-animate
  and checks summary.json has one row with the expected keys.
- The baseline report is committed under docs/reports/.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g1-1-bench). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G1-2 · Meshy khớp docs hiện hành + xử lý lỗi HTTP chung (P0, cỡ M)
Vì sao: Meshy là provider chính; các lệch nhỏ làm giảm chất lượng rig và báo sai cách sửa khi bị giới hạn tốc độ. Phụ thuộc: G0-2 (cùng `tools_full.py`).

```text
CARD G1-2 — Meshy: align every call with the current docs; shared HTTP error handling

CONTEXT (checked against docs.meshy.ai on 2026-10-17)
Already correct: endpoints; ai_model "meshy-7.1" (valid, "latest" = 7.1, meshy-5 retired on
2026-10-10); polling statuses and model_urls.glb; rigging result fields (rigged_character_glb_url,
basic_animations); POST /animations with action_ids (1–10 unique) or motion_task_id;
GET /animations/library (top-level array of {action_id, name, key, category, ...}); Text to Motion
(prompt ≤ 400 chars, duration 2–10 s in 0.5 steps, mode prime|swift, 403 = moderation, clip kept 3 days).
Gaps:
1. tools_full.auto_rig (model_url fallback, around lines 455-465) sends the front sheet crop as the
   rigging texture_image_url. The docs require a UV-unwrapped base-color PNG (the mesh's own texture).
2. providers/base.py _check_status turns 401, 402 and 429 into "Credits exhausted ... use blockout".
   Meshy 429 means rate limiting or a full task queue (Retry-After header, or NoMorePendingTasks) and
   should be retried; 402 means no credits; 401 a rejected key.
3. meshy.py sends only the front image in texture_image_urls; on meshy-7.1 the docs accept 1–4 views
   (front first).
4. meshy.py forces should_remesh=True, topology "triangle", target_polycount 30000. The docs say
   should_remesh=false gives the best quality and offer save_pre_remeshed_model; rigging accepts textured
   humanoids up to 300k faces.
5. "EXPIRED" is handled but is not in the status enum (harmless).
Prices (for card G1-3): multi-image 7.1 = 20 credits mesh only, 30 with 2K/4K texture, 35 with 8K, +5
for 2k geometry; rigging 5; animation 3 per action; Text to Motion prime 10 / swift 3. Limits: 20
requests/s; task queue 10 (Pro) to 100 (Ultra). Signed URLs expire; assets are kept ≤ 3 days.

GOAL
Meshy calls follow the docs, every HTTP error tells the agent the right fix, and the remesh trade-off
becomes a config choice that card G1-5 measures on real sheets.

FILES
src/charforge/providers/base.py, providers/meshy.py, tools_full.py (rigging fallback only), config.py
(MESHY_*, HTTP_RETRY), tests/contract/test_meshy_contract.py, tests/test_providers.py, docs/TOOLS.md
(error codes), docs/ARCHITECTURE.md (§7 and provider notes).

TASKS
1. base._check_status: 401 → PROVIDER_FAILED "API key rejected" (fix: "Check the key in .env, or use
   provider='blockout'."); 402 → PROVIDER_FAILED "No credits left" (fix: "Use provider='blockout' or
   method='blender'."); 429 → retry inside the call up to config.HTTP_RETRY times (honor Retry-After,
   capped at 20 s; otherwise exponential backoff), then CFError("PROVIDER_BUSY", ..., "Wait one minute,
   then call the same tool again with the same arguments."); 5xx → one retry, then PROVIDER_FAILED.
   PROVIDER_BUSY is a new code: document it, and make pipeline_run return status "running" (not
   "blocked") when it sees it.
2. Rigging with model_url: send texture_image_url only when you can extract the mesh's base-color
   texture from the GLB (read the glTF JSON like tools_full._glb_json, slice the image's bufferView from
   the BIN chunk, send it as a PNG data URI; convert JPEG to PNG with cv2 if needed). Without one, omit
   the field.
3. texture_image_urls: front, side, back (front first) when config.MESHY_TEXTURE_FROM_VIEWS.
4. config.MESHY_REMESH = {"should_remesh": True, "topology": "triangle", "target_polycount": 30000,
   "save_pre_remeshed_model": True}; today's behavior stays the default; store the pre-remeshed URL's
   existence in the mesh entry (download it only if config says so). Card G1-5 decides the default.
5. Status handling: keep EXPIRED harmlessly (comment) and make sure CANCELED is covered by tests.
6. Contract tests for each item: 429 with and without Retry-After, 402 message, 3 views in
   texture_image_urls, rigging fallback with and without a base-color texture.

DONE WHEN
- Contract tests cover every item and pass offline; no test sends real HTTP.
- `python -m pytest -q tests/contract tests/test_providers.py tests/e2e` green, then the full suite.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g1-2-meshy). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G1-3 · Trần chi phí API trả phí (P0, cỡ M)
Vì sao: agent yếu lặp lại lời gọi hoặc pipeline_run chạy lại có thể đốt credit không giới hạn. Phụ thuộc: G1-2 (cùng đường submit của provider), G2-2 (cùng `state.py`).

```text
CARD G1-3 — Spending guardrails for paid APIs

CONTEXT
Paid calls today: generate_mesh with a cloud provider; auto_rig(method="provider") (Meshy rigging, plus
one resubmit with model_url after HTTP 400); animate(method="provider") (one preset task, one Text to
Motion task per motion, one animation task per motion). The only cap is config.MAX_CANDIDATES (3
meshes). Failed or timed-out jobs, agent retries and repeated pipeline_run calls can keep spending.
Providers report `consumed_credits` on some results; it is stored in project.json
jobs[*].consumed_credits (animate groups keep a group total).

GOAL
A project can never exceed a configured number of paid submissions or credits, and both the agent and
the human can see what was spent.

FILES
src/charforge/config.py, state.py, tools_core.py, tools_full.py, server.py (project_status),
docs/TOOLS.md, docs/ARCHITECTURE.md, src/charforge/playbook.md, tests/test_budget.py (new).

TASKS
1. config.PAID_LIMITS = {"generate": 4, "rig": 2, "animate": 2, "text_to_motion": 6, "credits": None}
   per project, overridable with project.json "paid_limits" (None = no credit cap).
2. state.paid_usage(p) -> {"generate", "rig", "animate", "text_to_motion": counts, "credits": sum},
   counted from p["jobs"]: every submitted job counts, failed ones too; reused Text to Motion tasks do
   not count twice. state.check_paid(p, kind, n=1) raises CFError("BUDGET_EXCEEDED", <spent vs limit>,
   <fix>) BEFORE any HTTP request. Fix per kind — generate: "Stop generating: call mesh_export on the
   best mesh, or use provider='blockout'."; rig: "Use auto_rig(method='blender')."; animate and
   text_to_motion: "Use animate(method='blender') without motions."
3. Call check_paid right before every paid submit: tools_core.generate_mesh, tools_full.auto_rig (both
   the input_task_id submit and the model_url resubmit), tools_full.animate (preset task, each new Text
   to Motion task) and _advance_group (each motion animation task).
4. Policy: when the generate budget is spent, suggest_next behaves as if MAX_CANDIDATES were reached
   (continue with the best mesh); pipeline_run never loops on BUDGET_EXCEEDED.
5. project_status data.stats gains `paid` (usage and limits). Update TOOLS.md (project_status data,
   BUDGET_EXCEEDED on generate_mesh/auto_rig/animate), ARCHITECTURE.md §5 and §7, and add to the
   playbook rules: "BUDGET_EXCEEDED: do what error.fix says; never retry the same paid call."

DONE WHEN
- tests/test_budget.py with httpx.MockTransport: the 5th generate submit raises BUDGET_EXCEEDED and the
  transport receives no request; the rig model_url resubmit is counted; a reused Text to Motion task is
  not counted twice; project_status shows usage; with a credits limit of 10 and a mocked job reporting
  12 consumed credits, the next submit is refused.
- Existing provider, contract and e2e tests pass unchanged.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g1-3-budget). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G1-4 · Hướng nhìn của ảnh side (P0, cỡ S)
Vì sao: Tripo đang nhận sai nhãn trái/phải cho ảnh side; blockout đoán riêng một kiểu. Phụ thuộc: G0-1 (cùng file IO).

```text
CARD G1-4 — Side view facing (`side_faces`) in the reference

CONTEXT
reference.side_faces ("left" | "right") is read by blockout.py (around line 110) and
providers/tripo.py (around line 157) but never written: sheet.reference_metrics() does not produce it.
blockout.py guesses the facing with an inline heuristic (toe direction near the feet, then head
protrusion); tripo.py silently treats the side image as the "left" view, and reads
views/reference.json from disk because tools_core._submit_provider never passes the reference to
providers.

GOAL
The facing of the side view is computed once in sheet_split, stored in the reference, and used the same
way by every consumer.

FILES
src/charforge/score.py, sheet.py, blockout.py, tools_core.py (_submit_provider only),
providers/tripo.py (how the value is obtained only; the request format belongs to card G1-2),
server.py (sheet_split data), docs/ARCHITECTURE.md §5, docs/TOOLS.md (sheet_split data),
tests/test_side_faces.py (new).

TASKS
1. Move the heuristic out of blockout.py into score.side_facing(mask) -> "left" | "right" | None (on the
   canonical mask; None when the evidence is under the existing 4 px margin). Keep the numbers identical
   so blockout output does not change.
2. sheet.reference_metrics() adds "side_faces"; sheet_split returns it in data. Old projects without the
   key: blockout computes it with the helper.
3. tools_core._submit_provider passes reference=p["reference"] to providers whose submit() has a
   `reference` parameter (inspect.signature, like target_polycount). tripo.py stops reading
   views/reference.json when a reference is passed; when side_faces is None it computes it from the side
   mask instead of defaulting to "left".
4. Document the key (ARCHITECTURE §5 reference row, TOOLS.md sheet_split data).

DONE WHEN
- tests/test_side_faces.py (no Blender): the conftest sheet (side figure has a nose bump on its right)
  gives "right"; the mirrored sheet gives "left"; a symmetric blob gives None.
- Blockout tests keep their numbers (front ≥ 0.9, side ≥ 0.85): run tests/test_blockout*.py.
- Tripo contract tests pass, plus a new one proving the reference passed by generate_mesh reaches
  TripoProvider.submit.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g1-4-side-faces). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G1-5 · Chạy thật với Meshy, ghi lại response làm fixture [cần bạn] (P0, cỡ M)
Vì sao: biến mock "đoán" thành mock "ghi từ thực tế"; lần đầu biết rig Meshy có qua được QA không. Phụ thuộc: G1-1, G1-2, G1-3, và `MESHY_API_KEY` + hạn mức credit bạn chốt.

```text
CARD G1-5 — First real run with Meshy, recorded as fixtures   [needs MESHY_API_KEY and a credit ceiling]

CONTEXT
Every Meshy test uses mocks written from documentation; PR #1 says it never ran against the real API.
Only a real run answers: the real response shapes; the rigged GLB's bone names and hierarchy (does
cf_humanoid.map_roles map them?); whether rig_check passes or repairs; whether repaired weights reach the
animation GLB; how idle/walk/wave resolve in the animation library; what Text to Motion returns; real
timings and credits.

GOAL
One documented real run on 2 golden-set sheets; every finding becomes a fix or a follow-up; real
(redacted) responses become the contract-test fixtures.

FILES
src/charforge/providers/base.py (recorder), docs/reports/live-meshy-<date>.md (new),
tests/contract/fixtures/meshy/*.json (new), tests/contract/test_meshy_contract.py, small fixes anywhere
in src/charforge (each listed in the report).

TASKS
1. Preconditions: G1-1, G1-2 and G1-3 merged; .env has CHARFORGE_LEVEL=3 and MESHY_API_KEY. Ask the
   user for the credit ceiling and which 2 sheets of bench/manifest.json to use (suggest one A-pose
   with clear gaps and one with a skirt or long hair); set paid_limits accordingly.
2. Recorder: when env CHARFORGE_RECORD_DIR is set, providers/base.py writes each request (method, URL,
   JSON body with data URIs cut to 64 chars) and response (status, JSON body) to numbered files there.
   Never record headers or keys. Off by default.
3. Run `python scripts/bench.py --provider meshy --yes-spend --rig --animate --only <id1>,<id2>` with
   the recorder on. Then on one project call animate(motions=[{"prompt": "waves with both hands then
   bows", "name": "greet", "duration": 4}]) to exercise Text to Motion.
4. Inspect and write down: bone names and hierarchy of export/<id>-rigged.glb, renders/qa/rig_check.json,
   renders/anim/*.png, motion_check, critic (if GEMINI_API_KEY is set), time and credits per call.
5. Turn the recordings into fixtures (replace ids and URLs consistently) and make the Meshy contract tests
   assert against them instead of hand-written shapes.
6. Fix small in-scope bugs; list bigger ones as follow-ups with evidence.

RULES
- Never spend beyond the ceiling; stop and ask before any step that would exceed it.
- Never commit keys, signed URLs or raw recordings (bench/runs is git-ignored).

DONE WHEN
- docs/reports/live-meshy-<date>.md (Vietnamese): what ran, output paths and images, numbers (IoU,
  rig_check, motion_check, critic, time, credits), the bone list, each bug and its status.
- The Meshy contract tests use the recorded fixtures and pass offline.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g1-5-live-meshy). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G1-6 · Tripo chuyển sang API v3 (P1, cỡ M)
Vì sao: adapter hiện gửi sai định dạng cho v2, mà v2 đang bị khai tử. Phụ thuộc: G1-2 (xử lý lỗi chung), G1-4 (`side_faces`).

```text
CARD G1-6 — Tripo: move to API v3 with a correct multiview request

CONTEXT (developers.tripo3d.ai, 2026-10-17)
providers/tripo.py targets API v2 and posts `inputs` as a dict {front, left, back, right} to
POST /v2/openapi/task. The docs say:
- V2 is being shut down (no date) and expects `files` as exactly 4 ordered slots [front, left, back,
  right] ({} for a missing view; front required; at least 2 views).
- V3 (international base https://openapi.tripo3d.ai/v3/): POST /v3/generation/multiview-to-model with
  `inputs` as an array (single-key objects, or 4 positional entries); `model` is required
  (v3.1-20260211, v3.0-20250812, v2.5-20250123); base64 `data` is rejected; upload with POST /v3/files →
  data.file_token; poll GET /v3/tasks/{id} → output.model_url; statuses queued | running | success |
  failed | cancelled, errors in error_code + error_message. Options: texture, pbr, texture_quality,
  texture_version, geometry_quality, face_limit (adaptive when omitted, can exceed 1.5M faces), quad,
  model_seed, texture_seed. Images PNG/JPEG/WebP, ≥ 256 px.
- Price v3.1: 20 / 30 / 40 credits (no / standard / detailed texture), +20 detailed geometry, +5 quad;
  failed tasks are free. Errors: HTTP 429 code 1007 (rate), code 2000 (concurrency, 10 per category),
  code 2010 with HTTP 403 (out of credits). Download URLs last about 24 h.
Code issues besides the request shape: a "legacy mock" branch sends base64 when upload returns 404
(tripo.py around lines 109-142); `seed` instead of model_seed/texture_seed; no model version;
target_polycount ignored; cancelled/banned/expired treated as running. The docs never define "left":
a side image facing image-right shows the character's RIGHT side, so today's mapping (side facing right
→ "left" slot) may be mirrored.

GOAL
TripoProvider works against API v3, puts the side view in the right slot, bounds the face count, and maps
every status and error.

FILES
src/charforge/providers/tripo.py, config.py (TRIPO_*), tests/contract/test_tripo_contract.py,
tests/contract/fixtures/tripo/ (new), docs/ARCHITECTURE.md (provider notes).

TASKS
1. Rewrite for v3: upload each view (POST /v3/files, multipart) → file_token; submit
   POST /v3/generation/multiview-to-model with model = config.TRIPO_MODEL ("v3.1-20260211"), inputs in
   [front, left, back, right] order with an empty entry for the unused side slot, texture and pbr on,
   face_limit = target_polycount or config.TRIPO_FACE_LIMIT (30000), model_seed/texture_seed when a seed
   is given. Poll GET /v3/tasks/{id}: success → output.model_url; failed/cancelled → failed with
   error_code + error_message; 1007/2000 → PROVIDER_BUSY and 2010/403 → the no-credits fix (base.py from
   card G1-2).
2. Remove the legacy-mock 404 branch and the reference.json disk read (card G1-4 passes `reference`).
3. Side slot: config.TRIPO_SIDE_SLOT maps reference.side_faces ("left"/"right") to a Tripo slot. Choose
   the default from evidence — the docs' example images, the Tripo web app, or (only with the user's OK)
   one cheap live task on a deliberately asymmetric sheet — and write the evidence in the REPORT.
4. Remove the v2 code path unless the user asks to keep it.
5. Contract tests with fixtures built from the docs' example responses (cite the URL inside each
   fixture): upload, submit body (order, empty slot, model, face_limit, seeds), poll states, errors.

DONE WHEN
- Contract tests pass offline; generate_mesh(provider="tripo") against a mocked v3 server writes
  meshes/cand-N-tripo.glb end to end (test).
- The side-slot decision and its evidence are in the REPORT.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g1-6-tripo-v3). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G1-7 · Hunyuan3D qua fal.ai: sửa request, dùng multi-view và model mới (P1, cỡ M)
Vì sao: request hiện tại thiếu field bắt buộc nên hỏng ngay; mesh trả về lại không có texture. fal rẻ (khoảng 0,4 USD mỗi mesh) nên đáng làm. Phụ thuộc: G1-2, G1-4.

```text
CARD G1-7 — Hunyuan3D through fal.ai: fix the request, use multi-view and the current model

CONTEXT (fal.ai model pages, 2026-10-17)
providers/hunyuan.py posts {"image_url": <front>} to https://queue.fal.run/fal-ai/hunyuan3d/v2. That
endpoint's required field is `input_image_url`, so today's request fails; `textured_mesh` defaults to
false (untextured result; true triples the $0.16 price); only the front view is sent although these exist:
fal-ai/hunyuan3d/v2/multi-view (front, back and left image URLs required), fal-ai/hunyuan3d-v21
(output model_glb), fal-ai/hunyuan3d-v3/image-to-3d and fal-ai/hunyuan-3d/v3.1/pro/image-to-3d (optional
back/left/right views, enable_pbr, face_count; output model_glb; $0.375 + $0.15 per add-on). fal queue
protocol: the submit response returns status_url and response_url (use them; do not rebuild URLs by
hand); statuses IN_QUEUE | IN_PROGRESS | COMPLETED; a failure is COMPLETED with error / error_type (there
is no FAILED status, so hunyuan.py's FAILED branch never fires). Data URIs are accepted. fal also hosts
tripo3d/h3.1/multiview-to-3d (ordered [front, left, back, right], seeds) with the same FAL_KEY.

GOAL
provider="hunyuan" returns a textured multi-view mesh from the current model, and the fal queue protocol
lives in one reusable client.

FILES
src/charforge/providers/fal.py (new), providers/hunyuan.py, providers/__init__.py, config.py
(HUNYUAN_MODEL, fal settings), tests/contract/test_hunyuan_contract.py (new),
tests/contract/fixtures/fal/ (new), docs/ARCHITECTURE.md (provider notes).

TASKS
1. providers/fal.py: FalQueue(model_id): submit(payload) → request_id plus the returned status_url and
   response_url; status() → running | done | failed (COMPLETED with error → failed with its message);
   result() → JSON. Header "Authorization: Key <FAL_KEY>".
2. hunyuan.py: config.HUNYUAN_MODEL = "fal-ai/hunyuan-3d/v3.1/pro/image-to-3d"; front as the main image,
   back and the side view as optional views (side slot from reference.side_faces, same convention as card
   G1-6), enable_pbr true, face_count = target_polycount or 30000; mesh URL from model_glb. Support
   "fal-ai/hunyuan3d/v2/multi-view" as an alternative model id (fields per its page, textured_mesh true).
3. Persist status_url and response_url in the project.json job record so polling survives a restart
   (tools_core saves job records; pass the extra fields through).
4. Contract tests with fixtures from the fal pages (cite the URLs): submit body per model, queue states,
   COMPLETED + error, result parsing.
5. Optional, separate commit, only if small: provider "tripo-fal" for tripo3d/h3.1/multiview-to-3d on
   the same FalQueue.

DONE WHEN
- Contract tests pass offline; generate_mesh(provider="hunyuan") against a mocked fal queue writes
  meshes/cand-N-hunyuan.glb (test).

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g1-7-fal-hunyuan). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G1-8 · Rodin: sửa poll và download, chọn tier (P2, cỡ S)
Vì sao: job Rodin hiện treo "running" tới khi hết 20 phút vì đọc sai cấu trúc response. API Rodin cần gói Business nên để ưu tiên thấp. Phụ thuộc: G1-2.

```text
CARD G1-8 — Hyper3D Rodin: fix polling and download, choose the tier

CONTEXT (docs.hyper3d.ai, 2026-10-17)
providers/rodin.py submits multipart `images` only, then reads a top-level `status` from
POST /api/v2/status and a top-level `url` from POST /api/v2/download. The docs: /status returns
jobs: [{uuid, status: Waiting | Generating | Done | Failed}] — the task is done when every job is Done,
and there is no top-level status (today a job stays "running" until the 20-minute timeout); /download
returns list: [{name, url}] (HTTP 201); `tier` must be sent (a Gen-2.5 value or Gen-2), otherwise the
Regular tier (Gen-1/1.5) is used; `images` takes 1–5 files and the first drives the material; results
are kept 7 days. Gen-2.5 costs 0.5 credit (+0.5 for Extreme-High); API access needs the Business plan.
The code also sends the task uuid as subscription_key when none came back, and has undocumented GET
fallbacks (rodin.py around lines 100-118).

GOAL
provider="rodin" completes a job end to end against the documented API.

FILES
src/charforge/providers/rodin.py, config.py (RODIN_*), tests/contract/test_rodin_contract.py,
tests/contract/fixtures/rodin/ (new).

TASKS
1. Submit: images front, side, back (front first); tier = config.RODIN_TIER (a documented Gen-2.5
   value); geometry_file_format "glb"; quality / mesh options from config, aiming near the project's
   polycount target. Keep the compound job id "uuid:subscription_key"; raise PROVIDER_FAILED when
   subscription_key is missing.
2. Poll: done only when every job is Done; failed when any job is Failed; otherwise running.
3. Download: POST /download {task_uuid}; pick the .glb entry from `list`.
4. Remove the undocumented GET fallbacks and the uuid-as-subscription-key path.
5. Contract tests with fixtures from the docs' examples (cite the URLs).

DONE WHEN
- Contract tests pass offline; generate_mesh(provider="rodin") against a mocked server writes
  meshes/cand-N-rodin.glb (test).

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g1-8-rodin). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G2-1 · Lời gọi dài: progress, hủy, ngân sách thời gian hợp mọi client (P1, cỡ M)
Vì sao: Codex và Cursor cắt lời gọi sau khoảng 60 giây, Gemini CLI sau 10 phút, trong khi một bước Blender có thể chạy tới 10 phút và `pipeline_run` có thể kéo 240 + 600 giây. Phụ thuộc: G0-2 (SDK v2 có `Context`).

```text
CARD G2-1 — Long-running calls: progress, cancellation, time budgets that fit every client

CONTEXT (MCP spec 2026-07-28 and client docs, checked 2026-10-17)
- Blender runs synchronously inside tool calls: blender.run() uses subprocess.run with timeout=600 s.
  score_mesh, mesh_fit, mesh_optimize, auto_rig and animate can take minutes on real meshes.
  pipeline_run checks max_seconds (default 240, max 540) only between steps, so one call can last
  240 + 600 s.
- Client limits: Claude Code — very long tool timeout but a 30-minute idle window on stdio that progress
  notifications reset, auto-backgrounding after 2 minutes, "timeout" in .mcp.json; Gemini CLI — 600 000 ms
  by default ("timeout" in settings.json); Codex CLI — 60 s by default (tool_timeout_sec); Cursor — about
  60 s, not reset by progress; Antigravity — undocumented.
- Progress: the client opts in with _meta.progressToken; the server sends notifications/progress
  {progress (must increase), total?, message?}. With SDK v2 (card G0-2) a `ctx: Context` parameter is
  injected; a sync tool runs on a worker thread and can send progress with
  anyio.from_thread.run(ctx.report_progress, ...); without a token it is a no-op. Cancellation arrives as
  notifications/cancelled on stdio; an async tool running Blender through anyio's process API gets its
  child killed when cancelled. The MCP Tasks extension has no client support yet: do not use it.
- CharForge already has a portable job pattern: cloud jobs return status "running" + job_id and a
  next_action that polls; pipeline_run returns "running" and is called again.

GOAL
No call blocks longer than a configurable budget, long Blender work reports progress, a cancelled call
kills its Blender process, and clients with short timeouts (Codex, Cursor) can still finish the pipeline
by polling.

FILES
src/charforge/blender.py, pipeline_run.py, server.py, tools_core.py, tools_full.py, envelope.py,
config.py (CALL_BUDGET_S, PROGRESS_EVERY_S), docs/ARCHITECTURE.md (§4), docs/TOOLS.md, README.md (client
timeout settings), tests/test_long_calls.py (new).

TASKS
1. Blender runner: start Blender with a process API that can be waited on in slices (anyio process or
   Popen), call an on_tick(elapsed) callback every config.PROGRESS_EVERY_S (15 s), kill the whole process
   tree on timeout or cancellation, always clean the temp dir, keep the JSON result protocol. Prefer async
   long tools + anyio processes so MCP cancellation kills Blender; if you keep sync tools, check a cancel
   event in the wait loop.
2. Progress: the long tools take an optional `ctx: Context` (the SDK hides it from the input schema) and
   pass an on_tick that reports (elapsed, message "<tool>: <step>"); pipeline_run reports one message per
   step. No-op without ctx or token.
3. Budgets: config.CALL_BUDGET_S (default 50) is the per-call target for clients that cannot wait.
   pipeline_run gives each Blender step timeout = min(script default, remaining budget + grace), and
   returns "running" before a step whose typical duration (moving average per script, kept in
   project.json) does not fit the remaining budget.
4. Detached Blender jobs for auto_rig (rig.py) and animate (procedural + previews): when the expected
   duration exceeds CALL_BUDGET_S, start Blender detached, store it in project.json jobs with kind
   "blender", return status "running" + job_id + a next_action that calls the same tool with job_id;
   collecting the result finishes the step exactly as today. Other tools only if measurements show the
   need.
5. README (Vietnamese): client settings — Claude Code `"timeout": 900000` in .mcp.json; Gemini CLI
   `"timeout": 900000`; Codex `tool_timeout_sec = 900`; Cursor and Antigravity rely on the job pattern.
6. Tests with a fake slow bpy script (sleep loop): ticks are emitted; timeout kills the process and no
   Blender process is left; cancellation kills it; a detached job goes running → done.

DONE WHEN
- tests/test_long_calls.py passes and asserts no leftover Blender processes.
- Through the SDK's client in a test, a call with a progressToken receives progress notifications
  during score_mesh.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g2-1-long-calls). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G2-2 · State an toàn, migration, thống kê đúng, policy blockout (P1, cỡ M)
Vì sao: gọi đồng thời làm mất dữ liệu (SDK v2 chạy tool trên nhiều thread); số liệu đo model điều phối đang bị đếm sai. Phụ thuộc: G0-2.

```text
CARD G2-2 — State safety, schema migration, honest telemetry, blockout policy

CONTEXT
- A tool loads project.json at the start and saves it at the end, possibly minutes later (Blender). Two
  calls on the same project at once (an agent calling score_mesh while pipeline_run runs, or two
  clients) silently lose updates. Since card G0-2 (MCP SDK v2) sync tools run on worker threads, so a
  client's parallel tool calls really do run at the same time.
- No migration exists: SCHEMA_VERSION = 1 never changes. The committed projects/aria is stale (qa null,
  reference.pose null, export files registered with source "external").
- envelope._log_call writes one calls.jsonl line per tool call, including the calls pipeline_run makes
  internally through _dispatch_tool, so state.stats() and scripts/driver_report.py count the server's
  own calls as the driving model's calls.
- With no provider key, suggest_next ("scored", best total < REGENERATE_BELOW) asks for generate_mesh,
  which rebuilds the identical blockout (blockout has no seed).

GOAL
Concurrent calls cannot corrupt a project, old projects upgrade themselves, driver statistics count only
the driver's calls, and the offline loop never regenerates an identical blockout.

FILES
src/charforge/state.py, envelope.py, pipeline_run.py, config.py, scripts/driver_report.py,
docs/ARCHITECTURE.md (§5, §7, §10), docs/TOOLS.md (project_status stats), tests/test_state_safety.py
(new).

TASKS
1. Lock: state.project_lock(pid), an exclusive lock file projects/<id>/.lock (os.open O_CREAT|O_EXCL;
   content pid + time; stale after config.LOCK_STALE_SECONDS = 900), re-entrant within one process
   (pipeline_run calls other tools). tool_entry takes it for every tool that writes an existing project
   (all except project_create and project_status); a held lock raises CFError("PROJECT_BUSY", ...,
   "Wait for the running call to finish, then call project_status.").
2. Migration: state.load() runs migrate(p): add missing keys (jobs, outputs, qa, animations,
   pose_retries), re-label meshes under export/ whose source is "external" (exported / rigged /
   animated from the file name), set schema_version = 2, save once. Idempotent; documented.
3. Telemetry: a contextvar set by pipeline_run while it dispatches; _log_call adds "via": "agent" |
   "pipeline_run". state.stats() reports calls / errors / error_rate for via == "agent" only, plus
   internal_calls. driver_report.py uses the same rule; old lines without "via" count as agent.
4. Policy: stage "scored" with a blockout best mesh and config.providers_available() empty → mesh_fit,
   not generate_mesh.
5. Do not regenerate projects/aria here (card G4-3 does); only make sure migrate() upgrades it.

DONE WHEN
- tests/test_state_safety.py: two threads calling a mutating tool on one project → one gets
  PROJECT_BUSY and project.json stays valid; a stale lock is taken over; pipeline_run re-enters its own
  lock; migrate() on a copy of projects/aria/project.json gives schema 2 with correct sources; stats
  exclude internal calls; with no keys and a blockout best total of 0.6, next_action is mesh_fit.
- Full suite green.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g2-2-state). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G2-3 · Tách sheet bền với sheet thật (P1, cỡ L)
Vì sao: bước đầu tiên sai thì cả pipeline sai; model điều phối không nhìn ảnh được thì đang kẹt ở `assign=["<front>", …]`. Phụ thuộc: G1-1 (đo), G1-4 (cùng `sheet.py`).

```text
CARD G2-3 — Sheet splitting that survives real sheets

CONTEXT
sheet.py today: foreground_mask (median of an 8 px border ring as background, Lab distance > 18, close,
open, hole fill); detect_candidates (dilate 25x3, connected components, keep boxes with h ≥ 0.6 × tallest
and h/w ≥ 1.6); assign_views (side = narrowest box, front = leftmost of the rest, back = rightmost;
confidence 0.5 when more than 3 figures). Real sheets add paper / gradient / gray backgrounds, titles
and "FRONT/SIDE/BACK" labels, 4–5 views with 3/4 views, extra rows (expressions, close-ups, props),
figures touching each other, thin line-art with white fill, side views facing either way. When
confidence < 0.8 the agent gets next_action sheet_split(assign=["<front>", "<side>", "<back>"]) and must
read views/preview.jpg; a model without vision cannot.

GOAL
On realistic sheets the split picks the right three figures with a confidence that means something,
and when unsure it proposes a concrete assign instead of placeholders.

FILES
src/charforge/sheet.py, server.py (sheet_split next_action only), critic.py (optional view labeler),
config.py, tests/sheet_factory.py (new), tests/test_sheet_real.py (new), docs/TOOLS.md (sheet_split).

TASKS
1. tests/sheet_factory.py: synthetic sheets with ground truth (boxes + labels): backgrounds (white, light
   gray, vertical gradient, paper noise), labels and titles, 3–5 figures including a 3/4 view, a second
   row of small heads, two figures whose hair touches, line-art figures (outline + white fill), side
   facing left or right. Figures have arms (A-pose).
2. foreground_mask: background model (border ring + heavy blur for gradients); adaptive Lab threshold
   (Otsu on the distance image, clamped around the current 18); line-art interiors filled; thin text
   strokes dropped (stroke width or area/perimeter test).
3. detect_candidates: group boxes into rows by vertical overlap; keep the row with the most tall
   figures; split figures merged by touching hair at a vertical projection valley; ignore text.
4. assign_views for N ≥ 3: score candidates by width ratio, left/right mirror symmetry of the mask
   (front and back symmetric; side and 3/4 not) and a head-region cue (front shows a face, back mostly
   hair); choose the best (front, side, back) triple; confidence = margin between the best and the
   second-best triple, mapped to 0..1. Exactly 3 clean figures must give the same result as today.
5. When confidence < 0.8, next_action carries the best triple as numbers (assign=[i, j, k]) with a
   warning to check the preview. Optional: if the critic is available and has budget, ask it to label
   the numbered preview with a tiny JSON schema, and use the answer only if it is a valid triple.
6. Run scripts/bench.py (card G1-1) before and after; put both numbers in the REPORT.

DONE WHEN
- tests/test_sheet_real.py: on ≥ 30 generated sheets, ≥ 95% get the right triple (box IoU ≥ 0.9 per
  view), and every wrong one has confidence < 0.8.
- Existing sheet tests (tests/test_sheet*.py) pass unchanged.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g2-3-sheet). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G3-1 · Chấm màu, tỉ lệ và khuôn mặt (P1, cỡ M)
Vì sao: hai mesh cùng dáng nhưng sai màu hay "tan chảy" mặt đang được chấm bằng nhau. Phụ thuộc: G1-1 (đo trước/sau).

```text
CARD G3-1 — Fidelity beyond silhouettes: color, proportions, face

CONTEXT
score.compare() judges silhouettes only (IoU per view, width profiles). Two meshes with the same outline
but wrong or misplaced colors, or a melted face, get the same score. The visual critic (critic.py) sees
color but is optional, costs money and is noisy, so it only breaks ties of ≥ 1 point.
render_views.py already supports mode=color (Workbench, flat light ≈ albedo); score_mesh renders color
only for the critic, and only for passed provider meshes.

GOAL
Deterministic color and proportion numbers in score_mesh, used as an advisory signal and a tie-break,
plus a high-resolution face check by the critic.

FILES
src/charforge/score.py, server.py (score_mesh), state.py (_better), config.py (COLOR, PROPORTIONS),
critic.py (face rubric), bpy_scripts/render_views.py (resolution option), docs/TOOLS.md,
docs/ARCHITECTURE.md §8, tests/test_color_fidelity.py (new).

TASKS
1. score.color_fidelity(ref_rgba, ren_rgba) per view: canonicalize RGB with the same transform as
   canonical_mask (blockout._canonical_rgb shows how), then in Lab: (a) palette match: k-means (k=8) on
   both foregrounds, clusters matched greedily by share, mean ΔE2000; (b) band match: mean color per
   horizontal band (the 64 profile bins) on the overlap, ΔE per band, the worst 3 bands with region
   names like worst_bands. Return {delta_e, palette_score 0..1, worst_bands, hints}. Side view: same
   mirror choice as the silhouette IoU.
2. score.proportions(ref_mask, ren_mask): head height (neck_ratio), shoulder width, crotch height and leg
   length from canonical front masks, as relative deltas; hints when |delta| > 5%.
3. score_mesh renders color for every mesh when config.COLOR["enabled"] (default True; one extra
   Blender render), adds data.color and data.proportions and warnings for low values. It never
   changes `passed` in this card.
4. Ranking in state._better: passed → clean pose → integrity → critic (≥ min_delta) → color
   palette_score (both measured and difference ≥ COLOR["min_delta"]) → total.
5. Face check: crop the head (above neck_ratio, square, padded) from the sheet front and from a
   1024-px-tall front color render; send both to the critic with a short face rubric (identity, eyes,
   mouth, hairline; 0–10) as one extra call within the same budget; store data.critic.face.
6. Document the new keys and the ranking.

DONE WHEN
- tests/test_color_fidelity.py (no Blender): identical images → palette_score ≈ 1 and ΔE ≈ 0; a
  recolored band appears in worst_bands; a horizontally shifted color layout is detected.
- A Blender test: score_mesh on the synthetic-sheet blockout returns data.color with palette_score > 0.7.
- Bench report (G1-1) before/after shows the new columns.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g3-1-color). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G3-2 · Chiếu texture từ sheet lên mesh không có texture (P2; thành P1 nếu chọn G3-6) (cỡ L)
Vì sao: nhánh không cần key hiện chỉ có màu đỉnh mờ, còn mesh local của G3-6 không có texture; ảnh sheet thì sắc nét và đã khớp khung hình. Phụ thuộc: G3-1 (dùng `color_fidelity` làm cổng chất lượng).

```text
CARD G3-2 — Project the sheet onto untextured meshes as a real texture

CONTEXT
The offline path (no API key) builds a visual hull with per-vertex colors sampled from the views
(blockout.py), then mesh_optimize decimates it and smart-UV-projects it. Colors are blurry: eyes,
patterns and edges are lost. The reference images are sharp and aligned with the mesh by construction
(same canonical framing as score.canonical_mask). Card G3-6 (local GPU generation) will also produce
shape-only meshes that need the same treatment.

GOAL
Blockout-based exports (and any untextured source, e.g. "local" from card G3-6) carry a baked image
texture projected from the front, back and side views, visibly sharper than vertex colors.

FILES
src/charforge/bpy_scripts/bake_texture.py (new), tools_core.py (mesh_optimize, blockout/external
lineage only), config.py (BAKE), docs/TOOLS.md, docs/ARCHITECTURE.md, tests/test_bake.py (new).

TASKS
1. bake_texture.py --mesh --out --views-json (RGBA view paths + mask bboxes) --res 2048: make sure a UV
   map exists (smart project with island margin); one orthographic projector per view framed like
   render_views (front at −Y, side at +X, back at +Y; the side image mirrored for −X); one UV map per
   projector (UV Project); a material mixing the projected images by facing weight (normal · view
   direction, sharpened) and falling back to the existing vertex color where no view sees the surface;
   bake EMIT into the final UV image with Cycles on CPU; export GLB with the image embedded.
2. Exact mapping: crop each reference RGBA to its mask bbox and fit the projector's ortho scale and
   offset to the mesh's projected bbox in that view.
3. Run it after mesh_optimize for blockout/external lineage and for untextured sources (data gains
   `textured` and the texture path); textured provider meshes are never touched; on failure keep the
   vertex-color mesh and warn.
4. Quality gate: keep the bake only if score.color_fidelity (card G3-1) palette_score does not drop
   against the vertex-color version.

DONE WHEN
- tests/test_bake.py (marker blender): on the synthetic-sheet blockout the baked GLB has one image
  texture ≥ 1024 px and UVs, and its front color render has a lower ΔE to the reference than the
  vertex-color version.
- Bench report before/after on the synthetic sheets.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g3-2-bake). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G3-3 · Thư viện chuyển động + retarget offline (P1, cỡ L)
Vì sao: không có key Meshy thì chỉ có 5 kiểu procedural, và tên lạ âm thầm thành "lắc nhẹ". Phụ thuộc: không (cũng là nền cho G3-8).

```text
CARD G3-3 — Offline motion library: retarget clips onto any rig by bone role

CONTEXT
Without Meshy, animate(method="blender") offers five procedural clips (idle, walk, run, wave, jump); any
other name silently becomes a gentle sway (merge_animations.Animator.make: kind = ... else "sway").
cf_humanoid already maps any humanoid skeleton to roles (hips, spine, chest, ..., leftUpperArm, ...,
rightToes) from its structure, and the procedural animator rotates bones about world anatomical axes, so
a role-based retarget fits the codebase. auto_rig writes export/<id>-humanoid.json (role → bone name,
rest pose = A-pose).

GOAL
animate(actions=[...], method="blender") can play real motion clips from a small bundled library,
retargeted by role onto any rig CharForge makes or receives, and unknown names fail honestly.

FILES
src/charforge/bpy_scripts/retarget.py (new; bpy/mathutils/stdlib + cf_humanoid), merge_animations.py
(procedural mode resolves names), cf_humanoid.py (helpers only if missing), assets/motions/ (new: clip
files + LICENSES.md), config.py (MOTION_LIBRARY aliases), tools_full.py (animate offline branch),
docs/TOOLS.md, src/charforge/playbook.md, tests/test_retarget.py (new).

TASKS
1. Library: 8–12 humanoid clips whose license allows redistribution in a public repo: prefer CC0
   (check, for example, Quaternius' Universal Animation Library), CC-BY only with attribution; never
   Mixamo downloads, never non-commercial or no-derivatives data. Cover idle variants, walk, run, jump,
   wave, dance, sit down, punch/attack, bow, cheer. Record source URL, license and retrieval date per
   file in assets/motions/LICENSES.md. Keep the folder small (convert to BVH or trimmed GLB, < 10 MB
   total).
2. retarget.py: import the clip (BVH via import_anim.bvh, or GLB/FBX), map the source skeleton to roles
   with cf_humanoid.map_roles (fallback: a name table for common BVH/CMU/Mixamo names), compute a per-role
   rest alignment (source rest direction vs target rest direction of each limb segment, so A-pose and
   T-pose rests both work), copy world-space rotation deltas frame by frame onto the target pose bones,
   copy root translation scaled by the hip-height ratio, keep feet on the ground (offset by the lowest
   foot height of frame 0), bake an action at 30 fps; optional trim and loop closing for cyclic clips.
3. Offline name resolution: procedural names first, then the library via config.MOTION_LIBRARY aliases;
   an unknown name raises CFError("ANIM_FAILED", ..., "Use one of: <names>.") instead of a sway. data
   gains sources: {clip: "procedural" | "library:<file>"}.
4. Retargeted clips go through the existing preview + motion_check unchanged.

DONE WHEN
- tests/test_retarget.py (marker blender): walk and wave retargeted onto the bpy_fixtures character
  (A-pose rest) and onto its renamed-bone variant pass motion_check (stretch_p99 ≤ 1.6, ground ≤ 3 cm);
  the walk alternates the feet (measured); an unknown action raises ANIM_FAILED listing the names; clip
  names and order in the GLB match the request.
- assets/motions/LICENSES.md lists every bundled file with its license.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g3-3-retarget). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G3-4 · Xuất FBX cho Unity/Unreal, VRM tùy chọn (P1, cỡ M)
Vì sao: người dùng cuối thường đưa nhân vật vào Unity/Unreal/VTuber; GLB thôi thì phải tự chuyển và tự map xương. Phụ thuộc: không.

```text
CARD G3-4 — Engine-ready exports: FBX for Unity and Unreal, optional VRM

CONTEXT
auto_rig writes export/<id>-rigged.glb and animate writes export/<id>-animated.glb;
export/<id>-humanoid.json maps roles to bone names and to Mixamo names (rest pose A-pose). mesh_export
writes FBX only for the static mesh. Unity's Humanoid importer and Unreal's IK Retargeter work best with
FBX, Mixamo-style names, no leaf bones, and a known rest pose.

GOAL
Rigged and animated results also ship as FBX files that import cleanly into Unity and Unreal, and
optionally as VRM 1.0, without adding tools.

FILES
src/charforge/bpy_scripts/export_rig.py (new), tools_full.py (auto_rig / animate finish paths),
config.py (EXPORT), docs/TOOLS.md, README.md (Unity/Unreal import steps), tests/test_export_rig.py (new).

TASKS
1. export_rig.py --mesh <glb> --out <fbx> [--mixamo-names] [--tpose]: import the GLB; optionally rename
   bones to Mixamo names through cf_humanoid roles (write the reverse map into humanoid.json); optionally
   turn the A-pose rest into a T-pose rest (pose the arms horizontal, apply the pose as rest with the
   mesh re-bound so the bind stays consistent); export binary FBX (apply unit scale, axis settings for
   Unity/Unreal, add_leaf_bones=False, bake_anim=True with every action as a take, embedded textures).
2. auto_rig also writes export/<id>-rigged.fbx and animate export/<id>-animated.fbx, controlled by
   config.EXPORT = {"fbx": True, "mixamo_names": True, "tpose": False, "vrm": False}; failures only warn.
   outputs gains rigged_fbx / animated_fbx; data gains `files`.
3. Self-check in the same script: re-import the FBX and compare bone count, action count and names,
   height within 1% of height_cm, and the presence of every bone Unity's Humanoid requires (hips, spine,
   chest, neck, head, upper/lower arms, hands, upper/lower legs, feet).
4. Optional VRM 1.0: only when the "VRM Add-on for Blender" is installed in the user's Blender (detect
   it; never vendor it): map roles to VRM humanoid bones and export export/<id>.vrm; otherwise skip with
   an info line in data.
5. README (Vietnamese): import steps for Unity (Rig → Humanoid) and Unreal (IK Retargeter, Mixamo
   preset).

DONE WHEN
- tests/test_export_rig.py (marker blender): the bpy_fixtures character gives a rigged FBX and an
  animated FBX that re-import with the right bone count, clip names and height; Mixamo renaming covers
  all 22 roles.
- Existing auto_rig / animate tests pass unchanged.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g3-4-export). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G3-5 · Motion QA: trượt chân, khe vòng lặp, tay xuyên thân (P1, cỡ M)
Vì sao: đây là các lỗi người xem thấy đầu tiên, nhưng `motion_check` hiện chỉ đo độ giãn và xuyên sàn. Phụ thuộc: không.

```text
CARD G3-5 — Motion QA: foot sliding, loop seams, hands inside the body

CONTEXT
preview_clips.py measures stretch_p99 / stretch_max (edge length ratio), ground_cm (feet below the floor)
and travel_m (hips displacement) on 6 sampled frames per clip; tools_full._clip_qa turns them into
motion_check with thresholds from config.ANIM_CHECK. It misses the defects viewers notice first: feet
sliding while planted, a jump where a looping clip wraps, forearms or hands passing through the torso or
thighs (the "sticking" users complain about).

GOAL
motion_check reports three more numbers per clip, with thresholds in config, so bad clips are flagged
without human eyes.

FILES
src/charforge/bpy_scripts/preview_clips.py, cf_humanoid.py (helpers only), tools_full.py (_clip_qa
issue texts), config.py (ANIM_CHECK), docs/TOOLS.md (motion_check keys), tests/test_motion_qa.py (new).

TASKS
1. Sample motion densely for metrics (every frame, or every 2nd for clips over 120 frames), separate from
   the 6 rendered frames, using bone head positions (cheap) rather than the whole mesh.
2. foot_slip_cm: per foot role (leftFoot/rightFoot, toes when mapped), contact phases = frames where the
   foot is within 2 cm of its lowest height and its vertical speed is small; slip = horizontal foot
   displacement during a phase (for in-place clips, i.e. travel_m under a threshold, measure the
   deviation from a constant velocity instead); report the p95 over phases in cm.
3. loop_gap_cm: for looping clips (the logic of cf_secondary._is_loop, or names idle/walk/run), the
   largest joint position difference between the first and the last frame, in cm.
4. hand_in_body: share of sampled frames where a hand or forearm joint is inside the torso or a thigh
   capsule (capsules from the mapped Body, like cf_secondary colliders; radius from the rest mesh).
5. config.ANIM_CHECK gains foot_slip_cm (5), loop_gap_cm (2), hand_in_body (0.1); issues and warnings in
   _clip_qa; document the keys in TOOLS.md.

DONE WHEN
- tests/test_motion_qa.py (marker blender): the procedural walk on the bpy_fixtures character passes all
  three; a clip with a root jump on its last frame fails loop_gap_cm; a clip rotating the forearm into
  the torso fails hand_in_body; a clip dragging a planted foot fails foot_slip_cm.
- Existing animation QA tests pass unchanged.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g3-5-motion-qa). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G3-6 · Sinh mesh local trên GPU (Hunyuan3D-2mv) [cần bạn] (P2, cỡ L)
Vì sao: máy có RTX 5060 Ti 16 GB; sinh mesh offline không tốn credit và không gửi ảnh ra ngoài. Phụ thuộc: G1-2 (provider base), G3-2 (texture). Bạn cần đồng ý: khoảng 30–60 GB đĩa và giấy phép Tencent Hunyuan (Việt Nam được phép; cấm EU, UK, Hàn Quốc; trên 1 triệu người dùng/tháng phải xin riêng; không dùng output để huấn luyện model AI khác).

```text
CARD G3-6 — Local mesh generation on the GPU (Hunyuan3D-2mv) as provider "local"   [needs the user's go-ahead]

CONTEXT
The dev machine is Windows 11, RTX 5060 Ti 16 GB (Blackwell, sm_120), 48 GB RAM; docs still say local
models are impossible (Mac without NVIDIA). Research on 2026-10-17:
- Hunyuan3D-2mv takes front/left/back views natively; shape generation needs about 6 GB VRAM; the 2.0
  paint model needs 16 GB and textures from the front view only. Official Windows support; the community
  Hunyuan3D-2-WinPortable (cu129) supports Blackwell and builds the texture extensions; in a torch cu128
  env the custom rasterizer must be rebuilt with TORCH_CUDA_ARCH_LIST=12.0 (VS 2022 + CUDA 12.8, flag
  --allow-unsupported-compiler). PyTorch ≥ 2.7 built with CUDA ≥ 12.8 supports sm_120.
- License: Tencent Hunyuan Community License 2.0. Vietnam is allowed; the EU, UK and South Korea are
  excluded; above 1M monthly active users a separate license is needed; outputs may not be used to
  improve other AI models. Hence opt-in only.
- Permissive alternative: Pixal3D (MIT + DINOv3 license; multi-view needs camera transforms; low-VRAM
  mode; Windows wheels for Python 3.12 only).
- Texture: instead of the 16–21 GB painters, project the sheet's own views onto the shape (card G3-2).

GOAL
generate_mesh(provider="local") makes a candidate from the three views on this GPU, offline, through the
same job protocol as the cloud providers; all model code and weights live outside the charforge package.

FILES
workers/hunyuan2mv/ (new: worker.py, README.md in Vietnamese, install notes), src/charforge/
local_worker.py (new launcher), providers/local.py (new), providers/__init__.py, config.py
(LOCAL_MODELS, source sets), state.py (policy for untextured sources), docs/ARCHITECTURE.md (§2, §4, §6),
docs/TOOLS.md (provider list), tests/test_local_provider.py (new).

TASKS
1. Worker protocol, like blender.run: `<worker env python> workers/hunyuan2mv/worker.py --result <json>
   --front <png> --left <png> --back <png> --out <glb> [--seed N] [--octree 256] [--faces 30000]`
   writes {"ok": true, "mesh", "seconds", "vram_peak_mb"} or {"ok": false, "code", "message", "fix"}.
   Only the worker imports torch / hy3dgen; charforge never does.
2. Side view: feed the worker's `left` input from the side image, mirrored when reference.side_faces says
   it shows the character's right side; prove the convention on an asymmetric synthetic sheet.
3. local_worker.py: start the worker detached (subprocess.Popen, own process group, stdout/stderr to a
   log in the job folder), return a job id, poll by the result file, kill + PROVIDER_TIMEOUT after
   config.LOCAL_MODELS["mesh"]["timeout_s"]. If card G2-1 already added a detached-process helper for
   Blender jobs, reuse it instead of writing a second one.
4. providers/local.py: LocalProvider(Provider) with submit / poll / download over local_worker;
   available when config.LOCAL_MODELS["mesh"]["python"] exists (no API key); provider="auto" picks it only
   when config.LOCAL_MODELS["mesh"]["auto"] is True.
5. Policy for an untextured, dense source ("local"): passed by the provider rule (total ≥ 0.80), then
   mesh_optimize (decimate + UV), then the G3-2 bake when available (else export untextured with a
   warning), then mesh_export. Put the source in a new config set (e.g. UNTEXTURED_SOURCES) and implement
   it in state.suggest_next only.
6. workers/hunyuan2mv/README.md (Vietnamese): exact Windows + RTX 50xx install (env, torch build,
   rasterizer rebuild or the WinPortable route), disk size, license summary, smoke-test command.
7. Tests without a GPU: a fake worker (writes a box GLB) drives submit → running → done →
   meshes/cand-N-local.glb → score_mesh; one test marked `gpu` (skipped unless CHARFORGE_GPU=1) runs the
   real worker on the synthetic sheet.

DONE WHEN
- Fake-worker tests pass; on this machine the gpu test yields a mesh that passes score_mesh on the
  synthetic sheet (report IoU, time, VRAM peak).
- The README install steps were followed once from scratch (summary in the REPORT).

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g3-6-local-mesh). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G3-7 · Rig bằng model học sẵn (Make-It-Animatable) [cần bạn] (P2, cỡ M)
Vì sao: rig.py dựa trên hình học; model học từ Mixamo + VRoid thường đặt khớp tốt hơn trên nhân vật anime. QA biến dạng hiện có vẫn là cổng chặn. Phụ thuộc: G3-6 (hạ tầng worker).

```text
CARD G3-7 — Learned local rigging (Make-It-Animatable) behind the deformation QA   [needs the user's go-ahead]

CONTEXT
auto_rig has two methods: Meshy (cloud) and rig.py (Blender, 22 bones, medial-axis joints, bone heat +
fallback weighting). Every rig goes through the deformation check (cf_humanoid.run_qa); provider rigs
also get weight repair (rig_check.py --repair). Research on 2026-10-17: Make-It-Animatable (MIA) predicts
a fixed 52-bone Mixamo-like skeleton (Mixamo's 65 minus 13 leaf bones, a superset of CharForge's 22
roles) with skin weights in under a second; trained on 95 Mixamo + 35 VRoid anime characters; code MIT,
weights Apache-2.0; fails on out-of-distribution shapes, hard poses and mesh defects; does not handle
skirts or long hair; v2 uses the Hunyuan3D-2.1 ShapeVAE (its license may carry over — check); setup is
Linux-first. ComfyUI-UniRig wraps MIA using SDPA instead of flash-attn, but it is GPL-3.0: do not copy its
code.

GOAL
auto_rig(method="local") rigs with MIA in its own environment; the existing QA and weight repair decide
whether to keep the result, with rig.py as the automatic fallback.

FILES
workers/mia/ (new: worker.py, README.md in Vietnamese), src/charforge/local_worker.py (from card G3-6),
tools_full.py (auto_rig), config.py (LOCAL_MODELS["rig"]), docs/TOOLS.md (auto_rig method values),
src/charforge/playbook.md, tests/test_local_rig.py (new).

TASKS
1. Worker: input the normalized export GLB (A-pose), output a rigged GLB with MIA's skeleton and weights;
   the JSON result protocol of card G3-6.
2. auto_rig: method gains "local"; "auto" order = provider (Meshy) when available, then local when
   configured, then blender. After the worker, run rig_check.py with --repair on the result (like the
   provider path), keep the repaired rig only if better, write humanoid.json (roles come from structure,
   so the 52-bone skeleton should map; extra bones are ignored by QA).
3. If QA still fails after repair, fall back to rig.py automatically and report both results
   (rig_source, rig_check of each) in data.
4. Prove that procedural animation and the retarget of card G3-3 work on the MIA skeleton (test).
5. Tests: a fake worker (copies a pre-rigged fixture GLB) for CI; a `gpu` test on the synthetic character.

DONE WHEN
- Fake-worker tests pass; on this machine the gpu test rigs the bpy_fixtures character and rig_check
  passes (compare the numbers with rig.py in the REPORT).
- Code and weight licenses verified and written in workers/mia/README.md.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g3-7-local-rig). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G3-8 · Text to Motion local (NVIDIA Kimodo) [cần bạn] (P2, cỡ L)
Vì sao: `animate(motions=…)` hiện chỉ chạy với Meshy (10 credit mỗi clip). Kimodo cho phép thương mại và chạy được trên 16 GB khi để text encoder trên CPU. Phụ thuộc: G3-3 (retarget), G3-6 (hạ tầng worker). Tải về khoảng 17 GB (text encoder Llama-3-8B cần chấp nhận điều khoản trên Hugging Face).

```text
CARD G3-8 — Local Text to Motion (NVIDIA Kimodo) through the retarget pipeline   [needs the user's go-ahead]

CONTEXT
animate(motions=[...]) works only with Meshy (Text to Motion, 10 credits per clip in prime mode). Card
G3-3 adds a role-based retarget (bpy_scripts/retarget.py) for BVH/GLB/FBX clips. Research on 2026-10-17:
NVIDIA Kimodo (v1.2, 2026-10-15): code Apache-2.0; SOMA weights under the NVIDIA Open Model License
(commercial use allowed; the SMPL-X variant is research-only); output NPZ on a 77-joint SOMA skeleton with
BVH export; keyframe, end-effector and root-path constraints; ~17 GB VRAM with the text encoder on the GPU,
< 3 GB with TEXT_ENCODER_DEVICE=cpu; developed on Linux, Windows "should work"; the text encoder is
Llama-3-8B (gated download). Alternative: Tencent HY-Motion 1.0 Lite (SMPL-H 22 joints, Tencent license
with territory limits, 24 GB official minimum).

GOAL
animate(motions=[...]) generates custom clips locally when Meshy is not used, retargets them, and runs the
same preview and motion_check as every other clip.

FILES
workers/kimodo/ (new: worker.py, README.md in Vietnamese), tools_full.py (animate), retarget.py (from
card G3-3), config.py (LOCAL_MODELS["motion"]), docs/TOOLS.md, src/charforge/playbook.md,
tests/test_local_motion.py (new).

TASKS
1. Worker: --prompt --duration --fps 30 --out <bvh> [--seed]; SOMA skeleton exported as BVH; text
   encoder on CPU by default (config); the JSON result protocol of card G3-6.
2. animate: when motions are given and Meshy is not used, run the worker per motion as async jobs (like
   the Meshy group: status "running" between calls), retarget each BVH with retarget.py, append after the
   preset clips, then preview + motion_check. The NO_PROVIDER error for motions without Meshy remains
   only when no local motion model is configured.
3. Map SOMA's 77 joints to CharForge roles (a table in retarget.py) and test the mapping.
4. Tests: a fake worker that copies a BVH from the motion library (card G3-3) for CI; a `gpu` test that
   generates "waves with both hands" here and passes motion_check.

DONE WHEN
- Fake-worker tests pass; the gpu test produces a clip that passes motion_check (attach its
  renders/anim preview).
- License notes in workers/kimodo/README.md.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g3-8-local-motion). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G4-1 · Đo model điều phối bằng LLM thật qua MCP (P1, cỡ M)
Vì sao: lời hứa "model cỡ Flash điều phối được" chưa từng được đo. Phụ thuộc: G0-2 (client MCP), G2-2 (telemetry tách lời gọi nội bộ), `GEMINI_API_KEY`.

```text
CARD G4-1 — Measure driving models with a real LLM over MCP

CONTEXT
CharForge is designed so a cheap model (Flash class) can drive it: next_action in every result, short
error fixes, a playbook prompt, server instructions. Nobody has measured it: docs/reports/
driver-default.md has one in-process run on a synthetic sheet. docs/ARCHITECTURE.md §10 defines the
metrics (error_rate, iterations to pass, pass rate, time). Card G2-2 makes calls.jsonl tell the driver's
calls apart from pipeline_run's internal calls (field "via").

GOAL
`python scripts/driver_eval.py --model <id> --runs 3` drives the real MCP server over stdio with a real
LLM on golden-set sheets and writes docs/reports/driver-<model>.md, so docstrings, error fixes and the
playbook can be improved with evidence.

FILES
scripts/driver_eval.py (new), scripts/driver_report.py, docs/reports/, bench/README.md (one section),
pyproject.toml (optional extra "eval" if a dependency is needed).

TASKS
1. Start the server as a subprocess over stdio with the official MCP client of the SDK chosen in card
   G0-2 (ClientSession + stdio_client in v1; check the v2 client names in the SDK docs), env
   CHARFORGE_LEVEL=3 and CHARFORGE_HOME=bench/runs/eval-<ts>.
2. list_tools, the server instructions from initialize, and the playbook via
   get_prompt("charforge_playbook"); convert the tool input schemas to the LLM's function-calling format.
3. LLM backends: Gemini through google-genai (default; model id from --model; GEMINI_API_KEY), Claude
   optional through anthropic. System prompt = instructions + playbook. User prompt = the quick-start
   prompt in README.md with the sheet path, name and height.
4. Loop: model → tool calls → call_tool → result JSON back to the model (truncate `data` to 8 KB) → ...
   until next_action is null, the model stops, 40 tool calls, or 30 minutes. Two modes: --mode
   single-tools (pipeline_run hidden) and --mode pipeline.
5. Save each transcript as JSONL; aggregate with driver_report.py: error rate, deviations from
   next_action (other tool or other args), invented paths or ids, retries after the same error, pass
   rate, tool calls, time; a "top 5 failure patterns" section quoting transcripts.
6. Offline (blockout) by default; --provider auto needs --yes-spend.

DONE WHEN
- 2 models × 3 runs × 2 synthetic sheets complete and produce the report (attach it).
- The report proposes concrete wording changes for docstrings / error.fix / playbook (do not apply them
  in this card).

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g4-1-driver-eval). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G4-2 · Sắp xếp lại test suite (P2, cỡ M)
Vì sao: tên file theo milestone cũ làm khó tìm test và giấu test trùng. Phụ thuộc: làm khi không còn card code nào đang mở.

```text
CARD G4-2 — Reorganize the test suite by domain

CONTEXT
About 430 tests, ~9.5 min with Blender. Many files are named after milestones of the previous
multi-agent build (test_m3_adversarial.py, test_m4_adversarial.py, test_m5_adversarial.py,
test_l2_challenger.py, test_l2_adversarial.py, test_l3_adversarial.py, test_challenger_joint_accuracy.py,
test_deformation_leakage_challenger.py, test_empirical_m5_preview.py, ...), so finding the tests of a
module is hard and duplicates hide.

GOAL
Same coverage, organized by domain, faster, with clear markers.

FILES
tests/** and pyproject.toml (markers) only.

TASKS
1. Inventory every test: file, name, module under test, needs Blender, duration from
   `python -m pytest -q --durations=0`. Put the table in the REPORT.
2. Move tests into tests/<domain>/test_*.py (sheet, score, blockout, fit, optimize, render, integrity,
   rig, anim, providers incl. contract, pipeline, state, critic, e2e). Keep every assertion; merge only
   exact duplicates and list them.
3. Session-scoped fixtures (tmp_path_factory) for expensive Blender artifacts that tests do not mutate:
   the bpy_fixtures synthetic characters, the synthetic-sheet blockout.
4. Every test that needs Blender carries @pytest.mark.blender; add a `slow` marker for tests over 10 s.
5. Targets: `python -m pytest -q -m "not blender"` under 60 s; the full suite faster than before.

DONE WHEN
- Test count before == after (minus listed exact duplicates); all green; durations before/after in the
  REPORT. Run this card only when no other code card is open.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g4-2-tests). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

### G4-3 · Phát hành 0.2.0 với showcase thật (P2, cỡ S)
Vì sao: README đang khoe kết quả từ sheet tổng hợp, ảnh minh họa dựng bằng script không có trong repo. Phụ thuộc: G1-5 (nếu muốn showcase từ provider).

```text
CARD G4-3 — Release 0.2.0 with a reproducible showcase

CONTEXT
README shows "aria", a blockout of a 10 KB synthetic sheet; its images were produced by ad-hoc scripts
that are not in the repo. projects/aria/project.json is stale. pyproject's version is 0.1.0 and there is
no CHANGELOG.

GOAL
A reproducible showcase, an honest README status, and a ready-to-tag 0.2.0.

FILES
scripts/render_showcase.py (new) + a bpy script if needed, assets/, projects/aria/ (regenerated),
README.md, CHANGELOG.md (new, Vietnamese), pyproject.toml (version), src/charforge/__init__.py
(__version__), src/charforge/__main__.py (--version).

TASKS
1. scripts/render_showcase.py <project_id>: from a finished project write assets/<id>_turnaround.png
   (input), assets/<id>_render.png (front, 45°, side, back color renders) and assets/<id>_preview.gif
   (turntable + walk cycle, 36 frames, ≤ 3 MB) with headless Blender.
2. Regenerate projects/aria with the current offline pipeline and its images; add a provider showcase
   only if card G1-5 produced one that may be published.
3. README: what is verified live vs only with mocks; tested Blender versions; level setup; links to
   docs/reports (bench, live, driver).
4. CHANGELOG.md for 0.2.0 grouped by area; bump the version; `python -m charforge --version`.
5. Do not tag or push; give the maintainer the exact commands.

DONE WHEN
- `python scripts/render_showcase.py aria` regenerates all three images and README shows them.
- Full suite green.

COMMON RULES (CharForge)
- Read CLAUDE.md, docs/ARCHITECTURE.md and docs/TOOLS.md first. docs/TOOLS.md is a contract: if you
  change a tool name, parameter or `data` key, update docs/TOOLS.md and src/charforge/playbook.md in the
  same change.
- Tools return ok(...) or raise CFError(code, message, fix) with one concrete fix. Loop policy lives only
  in state.suggest_next / state.record_score. Tunables live only in config.py (per-project override in
  project.json). New error codes go into docs/ARCHITECTURE.md §7.
- src/charforge/bpy_scripts/*.py run inside Blender: import only bpy/mathutils/stdlib (+ cf_humanoid,
  cf_secondary); report through the --result JSON file, never stdout.
- Strings the driving model reads at runtime (docstrings, error.fix, next_action.why, playbook) are short
  imperative English. Human docs (README, docs/) are Vietnamese.
- Keep 12 tools; never add a tool that runs arbitrary bpy/Python; no new runtime dependency unless this
  card allows it.
- Tests never call paid APIs: use httpx.MockTransport. Live calls only in tests marked `live` and gated
  by CHARFORGE_LIVE=1.
- Work on a new branch from main named after the card (e.g. g4-3-release). Small commits with
  conventional messages. Do not push, merge or open a PR unless the user asks.
- Done = every DONE WHEN item + the full suite green (`python -m pytest -q`; Blender tests run when
  Blender is installed).
- End with REPORT: files changed, commands run and their results, deviations from the card, open risks.
```

## 5. Prompt review (dùng sau mỗi card)

Thay `<CARD-ID>` và `<BRANCH>` rồi giao cho một AI khác với AI đã làm card. Nên dán kèm nguyên văn card nếu file này chưa được commit.

```text
REVIEW TASK — CharForge card <CARD-ID>

You are reviewing work another agent did on branch <BRANCH> of this repo. Do not rewrite the work; find
problems and prove them.
1. Read CLAUDE.md, docs/TOOLS.md, and card <CARD-ID> in docs/ROADMAP-v2.md (or the card text below).
2. `git diff main...<BRANCH>`: list every changed file and check it is inside the card's FILES list.
3. Check each DONE WHEN item yourself: run the commands, open the files, compare numbers. Do not trust
   the other agent's report.
4. Check the common rules: TOOLS.md/playbook updated when a tool name, parameter or data key changed;
   tunables only in config.py; bpy_scripts import only bpy/mathutils/stdlib (+ cf_humanoid,
   cf_secondary); runtime strings in short imperative English; no new tool; no new dependency unless
   allowed; no test that calls a paid API without the live marker.
5. Look for regressions: run `python -m pytest -q` (full suite when Blender is installed).
6. Answer with: VERDICT (APPROVE / REQUEST_CHANGES), a table of findings (severity, file:line, problem,
   evidence, suggested fix), and the exact commands you ran with their results.
```

## 6. Prompt điều phối (khi muốn một AI chạy cả một đợt)

Thay `<N>` bằng số đợt ở mục 3. Chỉ dùng với agent có sub-agent hoặc worktree (ví dụ Claude Code, Antigravity); file này phải đã được commit để agent đọc được.

```text
ORCHESTRATION TASK — CharForge roadmap v2, wave <N>

Read docs/ROADMAP-v2.md (sections 3 and 4). Run the cards of wave <N> listed in section 3.
- One branch and one worktree per card, created from the current main.
- Give each card's text verbatim to a sub-agent (or do it yourself if you have no sub-agents). Never let
  two writers edit the same file at the same time.
- When a card reports done, review it with the REVIEW TASK prompt in section 5 using a different agent.
  Send back REQUEST_CHANGES findings to the author once; if it fails review twice, stop that card and
  report it.
- Do not merge, push or open pull requests; at the end list each branch, its review verdict, and the
  test results, so the maintainer can merge.
```

## 7. Ngoài phạm vi đợt này (đã tra cứu, chưa nên làm)

- **Rig mặt / 52 blendshape ARKit**: chưa có công cụ mở nào tự sinh blendshape cho mặt nhân vật stylized; hướng thực tế hơn là đổi texture vùng mặt cho từng biểu cảm, vì mặt anime phần lớn là vẽ trên texture.
- **Tách lớp tóc/váy để mô phỏng vải**: PartCrafter (MIT), PartPacker (giấy phép phi thương mại của NVIDIA), Hunyuan3D-Part đều cho ra khối kín, không phải tấm vải. Hữu ích hơn là nhãn phân đoạn để spring bone biết đỉnh nào thuộc tóc/váy.
- **Tripo làm provider rig/animation thứ hai**: Tripo v3 có `/v3/animations/rig` (spec mixamo) và retarget preset. Cần tách lớp trừu tượng "provider rig" vì `auto_rig` đang gắn cứng Meshy; làm sau G1-5 và G1-6.
- **Chạy server qua HTTP cho nhiều người dùng**: chưa cần khi mới dùng cá nhân qua stdio.

Xem lại các mục này sau khi có số liệu từ G1-5 và G4-1.
