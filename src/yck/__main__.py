"""Command line entry point: `python -m yck fetch`."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=value lines) so the FRED key never lives in code."""
    import os
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def main(argv: list[str] | None = None) -> int:
    _load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(prog="yck")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download public data and build monthly panels")
    f.add_argument("--only", nargs="*", choices=["USD", "EUR", "GBP"], help="limit to these currencies")
    f.add_argument("--refresh", action="store_true", help="re-download instead of using data/raw cache")
    f.add_argument("--config", type=Path, default=ROOT / "config" / "data_sources.yaml")
    f.add_argument("--data-dir", type=Path, default=ROOT / "data")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if a.cmd == "fetch":
        from .data.build import run
        ok = run(a.config, a.data_dir, a.only, a.refresh)
        print(f"\nReport: {a.data_dir / 'processed' / 'coverage.md'}")
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
