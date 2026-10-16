# Roadmap và task card

Thứ tự làm: **L1-T1 → T2 → T3 → T4 → L2-T2 (blockout, không cần key) → L2-T3 → L2-T4 → L2-T5 → L2-T1 (khi có API key) → L3**.
Mỗi card có thể giao cho bất kỳ coding agent nào. Card viết bằng tiếng Anh, đủ thông tin để agent làm mà không cần đọc lịch sử chat:
- Claude Code / Gemini CLI mở trong thư mục này: gõ `Làm task L1-T1 trong docs/ROADMAP.md` (CLAUDE.md/GEMINI.md được nạp tự động).
- AGY (mcp-agy): dán nguyên phần card vào `agy_start_task`, rồi kiểm tra bằng `agy_get_diff` và `agy_run_tests`.

**Rules for every card**
- Read `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/TOOLS.md` first. Tool names, params and `data` keys in TOOLS.md are a contract; if you must change one, update TOOLS.md and `src/charforge/playbook.md` in the same change.
- Tools return `ok(...)` or raise `CFError(code, message, fix)`; `fix` is one concrete action. Policy changes go in `state.suggest_next` / `state.record_score`, never inside tools.
- `src/charforge/bpy_scripts/*.py` run inside Blender: import only bpy/mathutils/stdlib; report through the `--result` JSON file.
- No new dependencies unless the card allows it. Python 3.11, numpy 2 (no `ndarray.ptp`).
- Done means: `.venv/bin/python -m pytest -q` is green and the task's `xfail` markers are removed.

---
## L1: phần "mắt"

**L1-T1 — Sheet split**
- GOAL: implement `foreground_mask`, `detect_candidates`, `assign_views`, `split_sheet` in `src/charforge/sheet.py` exactly as their docstrings say.
- FILES: `src/charforge/sheet.py`; `tests/test_sheet.py` (only remove the `@todo1` markers).
- RULES: numpy/OpenCV only. `split_sheet` calls `reference_metrics` inside `try/except NotImplementedError` and stores `{}` until L1-T2 is done. Preview labels must be big enough to read at 1024 px wide.
- DONE WHEN: `test_split_synthetic_sheet` and `test_split_manual_assign` pass.

**L1-T2 — Reference metrics**
- GOAL: implement `reference_metrics` (docstring spec) and write `views/reference.json` from `split_sheet`.
- FILES: `src/charforge/sheet.py`; `tests/test_sheet.py` (remove `@todo2`).
- DONE WHEN: `test_reference_metrics` passes.

**L1-T3 — Compare and diff images**
- GOAL: implement `score.compare` and `score.write_diffs` (docstring spec). Reuse `canonical_mask`, `iou`, `width_profile`.
- FILES: `src/charforge/score.py`; `tests/test_score.py` (remove `@todo`).
- DONE WHEN: the 3 tests in `tests/test_score.py` that were xfail now pass.

**L1-T4 — End-to-end L1 through the MCP tools**
- GOAL: add `tests/test_e2e_l1.py` (marker `blender`). Use the synthetic sheet from `conftest.py` and a box OBJ mesh (see `tests/test_render.py`). Call the tool functions in `charforge.server` in order: `project_create → sheet_split → score_mesh → render_views`. Assert `ok`, assert that `next_action` follows ARCHITECTURE §6, and assert that diffs and compare images exist.
- FILES: `tests/test_e2e_l1.py`. Fix only wiring bugs you find in `server.py`/`state.py`.
- DONE WHEN: the test passes; `project_status` stats show 0 errors.

---
## L2: model tự động đầu tiên (`tools_core.py`, `register(mcp)` thêm 4 tool)

**L2-T1 — Providers + generate_mesh**
- GOAL: `src/charforge/providers/{__init__,base,tripo,meshy,rodin,hunyuan}.py` + `generate_mesh` in `src/charforge/tools_core.py` per TOOLS.md.
- RULES: read each provider's CURRENT official API docs before coding; do not guess endpoints. `base.Provider` exposes `submit(views: dict[str, Path], seed) -> job_id`, `poll(job_id) -> {"status": "running"|"done"|"failed", "url"?, "error"?}`, `download(url, dst)`. Keys come from env names in `config.PROVIDER_KEYS`. Use httpx with timeouts. HTTP 401/402/429 → `PROVIDER_FAILED` with a fix such as "credits exhausted: call generate_mesh with provider='blockout'". Persist jobs in `project.json.jobs`. Each call blocks at most 60 s.
- DONE WHEN: tests using `httpx.MockTransport` (no network) cover submit → running → done → file in `meshes/` → registered mesh. `CHARFORGE_LEVEL=2` lists the tool. If a key exists, one real run produces a GLB that `score_mesh` accepts.

**L2-T2 — Blockout (fallback when there is no key)**
- GOAL: `src/charforge/blockout.py` + `provider="blockout"` in `generate_mesh`. Build a visual hull: voxel grid 96×64×192 (X×Y×Z) sized from the front/side aspect ratios. A voxel survives only if it projects inside the canonical front mask (x,z), side mask (y,z) and mirrored back mask. Then `skimage.measure.marching_cubes`, 10 Laplacian smoothing passes (λ=0.5), and vertex colors sampled from the front/back/side images by the dominant normal axis. Write a binary PLY with uchar RGB. Scale to `height_cm`, Z-up, facing −Y. Assume the side view faces image-left; flip when `reference.side_faces == "right"`.
- DONE WHEN: `tests/test_blockout.py` builds the synthetic sheet into a blockout, and `score_mesh` gives front ≥ 0.9 and side ≥ 0.85.

**L2-T3 — mesh_fit**
- GOAL: `mesh_fit` per TOOLS.md. Python computes the per-band X/Y factors from the mask renders of the mesh's last score. `bpy_scripts/fit.py` only applies rotate_z, grounding, uniform height scaling and the lattice factors.
- DONE WHEN: a test takes a box mesh at the wrong proportions, runs mesh_fit and then score_mesh, and the total rises by ≥ 0.1. Output height equals `height_cm` ±1%.

**L2-T4 — mesh_optimize + mesh_export**
- GOAL: per TOOLS.md (`bpy_scripts/optimize.py`, `bpy_scripts/export.py`).
- DONE WHEN: a test on a dense mesh (about 100k tris) gives `tris_after ≤ target × 1.05` and `iou_drop ≤ 0.02`, and the exported GLB re-imports with the same tri count ±1%.

**L2-T5 — End-to-end L2**
- GOAL: `tests/test_e2e_l2.py`: `CHARFORGE_LEVEL=2`, synthetic sheet, `provider="blockout"`. Follow `next_action` from tool results only, until it is null.
- DONE WHEN: `outputs.glb` exists, stage `exported`, and it takes ≤ 12 tool calls.

---
## L3: hoàn thiện (`tools_full.py`)

**L3-T1 — auto_rig**: implement it per TOOLS.md (`bpy_scripts/rig.py`, provider rig optional). DONE WHEN: on the blockout, posing LeftArm deforms the mesh, `unweighted_ratio < 0.01`, and the rigged GLB re-imports with an armature.

**L3-T2 — pipeline_run**: implement it per TOOLS.md by calling the same functions the tools use (no duplicated logic). DONE WHEN: on the synthetic sheet at Level 3 with blockout, calling it again while `running` reaches `done` with `outputs.glb` + `outputs.rigged`, and every call stays within `max_seconds + 30`.

**L3-T3 — Driver model report**: write `scripts/driver_report.py` to aggregate `project_status` stats for `projects/*` into `docs/reports/driver-<model>.md` (error_rate, iterations to pass, pass rate, time). How to run: N=5 runs on the same sheet for each driver model (for example Gemini Flash vs a stronger model), driving each run with the playbook.

**L3-T4 — Rig/animation QA (đã làm)**: `pose_deformation_check` cho mọi rig (`bpy_scripts/rig_check.py`, `cf_humanoid.py`), tự sửa trọng số rig provider, `export/<id>-humanoid.json`; preview front/side + `motion_check` cho mọi clip (`preview_clips.py`); khe hở tư thế ở `sheet_split`/`score_mesh` và policy sinh thêm một ứng viên khi mesh dính; Text to Motion (`animate(motions=…)`); spring bone tóc/váy bake vào clip (`cf_secondary.py`); procedural theo vai trò xương. Test: `tests/test_rig_qa.py`, `tests/test_anim_qa.py`, `tests/test_pose_gaps.py`, `tests/contract/test_text_to_motion_contract.py`.

---
## Sau roadmap (tùy chọn)
VLM judge cho mặt và chi tiết; chiếu texture từ reference lên mesh (bake); rig mặt / ARKit 52 blendshapes (cần mesh có mặt mô hình hóa); tách lớp tóc và váy thành mesh riêng + cloth sim thật; retarget tự động sang skeleton Mixamo T-pose.
