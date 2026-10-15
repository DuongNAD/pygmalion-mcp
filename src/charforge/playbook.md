# CharForge playbook (for the driving agent)

Goal: the best possible 3D model of the character in the sheet, with no human in the loop.
You orchestrate tools. You never write Blender code and never edit files under projects/ by hand.

## Procedure
1. `project_create(sheet_path, name, height_cm)` -> keep `project_id`.
2. `sheet_split(project_id)`. If `data.confidence < 0.8`: open `data.preview` (numbered candidates) and call
   `sheet_split(project_id, assign=[front_idx, side_idx, back_idx])`.
3. Get a candidate mesh:
   - Level >= 2: `generate_mesh(project_id, provider="auto")`. While `data.status == "running"`, call it again with the returned `job_id`.
   - Level 1: use the mesh file you were given.
4. `score_mesh(project_id, mesh_path)`. Read `data.passed`, `data.iou`, `data.hints`.
   If a warning says the mesh is turned 90 degrees, re-score with `rotate_z=90`.
5. Not passed: follow `next_action` (`generate_mesh` for provider meshes or `mesh_fit` / `generate_mesh` for blockout meshes). Score again after every change.
6. `stage` is `passed` or `stopped`: follow `next_action`. Provider meshes skip `mesh_fit` and `mesh_optimize` to preserve UVs/PBR textures and go directly to `mesh_export` -> `auto_rig` -> `animate`. Blockout meshes run `mesh_optimize` -> `mesh_export` -> `auto_rig` -> `animate`.
   While `auto_rig` or `animate` returns `data.status == "running"`, call the same tool again with the returned `job_id`.
   `stopped` still ships the best mesh found; that is expected, not a failure.
7. Done when `next_action` is null. Final report: output files (including rigged and animated GLBs), best total, passed, stop_reason, stats from `project_status`.

Level 3 shortcut: `pipeline_run(project_id, rig=True, animate=True)` runs steps 2-7 itself. Call it again while `data.status == "running"`.
Only intervene with single tools when it returns an error or `data.status == "blocked"`.

## Rules
- `next_action` is the default plan (`auto_rig` advances to `animate`, and `animate` finishes with `null`). Deviate only to follow `error.fix`.
- `ok == false`: do exactly what `error.fix` says. The same error code twice in a row: stop and report it.
- Decide with numbers (`passed`, `iou`, `total`). Images (preview, diffs, compare) are only for locating problems.
- Copy paths from tool results. Never invent paths or project ids.
- Lost context: `project_status(project_id)`.
