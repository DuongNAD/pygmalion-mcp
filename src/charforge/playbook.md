# CharForge playbook (for the driving agent)

Goal: the best possible 3D model of the character in the sheet, with no human in the loop.
You orchestrate tools. You never write Blender code, never write animation keyframes and never edit files under projects/ by hand.

## Procedure
1. `project_create(sheet_path, name, height_cm)` -> keep `project_id`.
2. `sheet_split(project_id)`. If `data.confidence < 0.8`: open `data.preview` (numbered candidates) and call
   `sheet_split(project_id, assign=[front_idx, side_idx, back_idx])`.
   A warning that an arm or the legs touch the body is a sheet problem: keep going, mention it in the final report.
3. Get a candidate mesh:
   - Level >= 2: `generate_mesh(project_id, provider="auto")`. While `data.status == "running"`, call it again with the returned `job_id`.
   - Level 1: use the mesh file you were given.
4. `score_mesh(project_id, mesh_path)`. Read `data.passed`, `data.iou`, `data.hints`, `data.pose_mismatch`, `data.integrity`.
   If a warning says the mesh is turned 90 degrees, re-score with `rotate_z=90`.
   `data.critic` is an independent visual review (advisory). Report its issues; never re-grade renders yourself.
5. Not passed (or a passed mesh fuses arms/legs the sheet keeps apart): follow `next_action` (`generate_mesh` for provider meshes or `mesh_fit` / `generate_mesh` for blockout meshes). Score again after every change.
6. `stage` is `passed` or `stopped`: follow `next_action`. Provider meshes skip `mesh_fit` and `mesh_optimize` to preserve UVs/PBR textures and go directly to `mesh_export` -> `auto_rig` -> `animate`. Blockout meshes run `mesh_optimize` -> `mesh_export` -> `auto_rig` -> `animate`.
   While `auto_rig` or `animate` returns `data.status == "running"`, call the same tool again with the returned `job_id`.
   `stopped` still ships the best mesh found; that is expected, not a failure.
7. `auto_rig` checks how the rig deforms (`data.rig_check`) and repairs weight leaks itself. `passed=false` after repair: do not retry, report the issues (usually arms touching the body in the sheet or mesh).
8. `animate` returns `data.previews` (front+side frames per clip) and `data.motion_check`. Report failed clips; never try to fix motion by hand.
   Custom move: `animate(project_id, motions=[{"prompt": "...", "name": "...", "duration": 4}])` (needs a Meshy rig).
9. Done when `next_action` is null. Final report: output files (rigged and animated GLBs, humanoid map, preview images), best total, passed, stop_reason, rig_check and motion_check results, stats from `project_status`.

Level 3 shortcut: `pipeline_run(project_id, rig=True, animate=True)` runs steps 2-8 itself. Call it again while `data.status == "running"`.
Only intervene with single tools when it returns an error or `data.status == "blocked"`.

## Rules
- `next_action` is the default plan (`auto_rig` advances to `animate`, and `animate` finishes with `null`). Deviate only to follow `error.fix`.
- `ok == false`: do exactly what `error.fix` says. The same error code twice in a row: stop and report it.
- Decide with numbers (`passed`, `iou`, `total`, `rig_check.passed`, `motion_check.passed`). Images (preview, diffs, compare, clip previews) are for locating problems and for the human report.
- Copy paths from tool results. Never invent paths or project ids.
- Lost context: `project_status(project_id)`.
