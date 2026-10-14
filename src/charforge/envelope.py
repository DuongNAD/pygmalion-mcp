"""Uniform tool result envelope, agent-fixable error type and per-call logging.

Every MCP tool returns:
  {"ok": bool, "project_id": str | None, "data": {...}, "warnings": [str],
   "next_action": {"tool": str, "args": {...}, "why": str} | None,
   "error": {"code": str, "message": str, "fix": str} | None}
"""
from __future__ import annotations

import functools
import json
import time
import traceback
from typing import Any, Callable


class CFError(Exception):
    """Expected failure the agent can fix. `fix` must say exactly what to do next."""

    def __init__(self, code: str, message: str, fix: str, next_action: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.fix, self.next_action = code, message, fix, next_action


def action(tool: str, why: str, **args: Any) -> dict:
    return {"tool": tool, "args": args, "why": why}


def ok(data: dict, project_id: str | None = None, next_action: dict | None = None, warnings: list[str] | None = None) -> dict:
    return {"ok": True, "project_id": project_id, "data": data, "warnings": warnings or [], "next_action": next_action, "error": None}


def fail(code: str, message: str, fix: str, project_id: str | None = None, next_action: dict | None = None) -> dict:
    return {
        "ok": False,
        "project_id": project_id,
        "data": {},
        "warnings": [],
        "next_action": next_action,
        "error": {"code": code, "message": message, "fix": fix},
    }


def tool_entry(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Wrap a tool: exceptions -> error envelopes, one JSON line per call in <project>/calls.jsonl."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        t0, pid = time.time(), kwargs.get("project_id")
        try:
            res = fn(*args, **kwargs)
        except CFError as e:
            res = fail(e.code, e.message, e.fix, pid, e.next_action)
        except NotImplementedError as e:  # template stub, see docs/ROADMAP.md
            res = fail("NOT_IMPLEMENTED", f"Task {e} is not built yet.", "Stop and report: this needs docs/ROADMAP.md task " + str(e), pid)
        except Exception as e:  # server bug, not an agent mistake
            res = fail("INTERNAL", f"{type(e).__name__}: {e}", "Retry once; if it fails again stop and report this error verbatim.", pid)
            res["error"]["trace"] = traceback.format_exc(limit=4)[-1500:]
        _log_call(fn.__name__, kwargs, res, time.time() - t0)
        return res

    return wrapper


def _log_call(tool: str, args: dict, res: dict, seconds: float) -> None:
    from .state import project_dir  # lazy: state imports this module

    pid = res.get("project_id") or args.get("project_id")
    try:
        pdir = project_dir(pid, must_exist=True)
    except Exception:
        return  # no valid project yet (e.g. failed project_create)
    rec = {
        "t": round(time.time(), 3),
        "tool": tool,
        "args": args,
        "ok": res["ok"],
        "code": (res.get("error") or {}).get("code"),
        "sec": round(seconds, 2),
    }
    with open(pdir / "calls.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
