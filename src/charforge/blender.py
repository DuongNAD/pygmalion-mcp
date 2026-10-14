"""Run a script from bpy_scripts/ inside headless Blender.

Protocol: blender -b --factory-startup --python-exit-code 1 -P <script> -- --result <json> [--key value ...]
The script writes one JSON object to --result: {"ok": true, ...} or {"ok": false, "code", "message", "fix"}.
stdout is never parsed (too noisy).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from . import config
from .envelope import CFError


def run(script: str, args: dict, timeout: int = 600) -> dict:
    exe = config.blender_exe()
    if not exe:
        raise CFError("BLENDER_NOT_FOUND", "Blender executable not found.", "Setup problem: install Blender 4.2+ or set BLENDER_PATH in .env, then stop and report.")
    with tempfile.TemporaryDirectory(prefix="charforge-") as tmp:
        result = Path(tmp) / "result.json"
        cmd = [exe, "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(config.BPY_SCRIPTS / script), "--", "--result", str(result)]
        for key, val in args.items():
            if val is None or val is False:
                continue
            cmd.append("--" + key.replace("_", "-"))
            if val is not True:
                cmd.append(str(val))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise CFError("BLENDER_TIMEOUT", f"{script} exceeded {timeout}s.", "Reduce the mesh (mesh_optimize) and retry once.") from None
        if result.is_file():
            data = json.loads(result.read_text())
            if data.get("ok"):
                return data
            raise CFError(data.get("code", "BLENDER_FAILED"), data.get("message", "script error"), data.get("fix", "Retry once with another candidate mesh."))
        tail = (proc.stdout[-1200:] + "\n" + proc.stderr[-1200:]).strip()
        raise CFError("BLENDER_FAILED", f"{script} exited {proc.returncode} without a result. Log tail:\n{tail}", "Retry once; if it fails again the mesh is likely corrupt: use another candidate.")
