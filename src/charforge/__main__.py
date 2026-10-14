"""`python -m charforge` runs the MCP server over stdio. `--check` prints an environment report."""
import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser(prog="charforge")
    ap.add_argument("--check", action="store_true", help="print environment report and exit")
    if ap.parse_args().check:
        from .config import env_report

        report = env_report()
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["ok"] else 1)
    from .server import mcp

    mcp.run()


if __name__ == "__main__":
    main()
