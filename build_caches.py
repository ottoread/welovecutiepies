#!/usr/bin/env python3
"""Build-time cache generator (runs in Dockerfile; not committed to git)."""
from __future__ import annotations

import ast
import csv
import gzip
import os
import pickle
import sys
from pathlib import Path

SQLGLOT_ROOT = Path(os.environ.get("SQLGLOT_ROOT", "/app/sqlglot")).resolve()
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SQLGLOT_ROOT))


def _load_csv_rows(file_name: str) -> list[tuple]:
    with open(file_name, "rb") as fb:
        is_gz = fb.read(2) == b"\x1f\x8b"
    fh = gzip.open(file_name, "rt", newline="") if is_gz else open(file_name, encoding="utf-8", newline="")
    with fh:
        reader = csv.reader(fh, delimiter="|")
        next(reader)
        rows = []
        ctypes = []
        convert = None
        for row in reader:
            if not ctypes:
                for v in row:
                    try:
                        ctypes.append(type(ast.literal_eval(v)))
                    except (ValueError, SyntaxError):
                        ctypes.append(str)
                if ctypes:
                    values = [
                        f"row[{i}]" if ct is str else f"types[{i}](row[{i}]) if row[{i}] else None"
                        for i, ct in enumerate(ctypes)
                    ]
                    convert = eval(f"lambda row: ({', '.join(values)},)", {"types": tuple(ctypes)})
            if convert is not None and len(row) >= len(ctypes):
                rows.append(convert(row))
            else:
                rows.append(
                    tuple(None if (ct is not str and v == "") else ct(v) for ct, v in zip(ctypes, row))
                )
    return rows


def main() -> None:
    os.chdir(SQLGLOT_ROOT)
    from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA

    dir_tpcds = FIXTURES_DIR + "/optimizer/tpc-ds/"
    out_dir = APP_DIR / "tpcds_tables"
    out_dir.mkdir(exist_ok=True)
    total_bytes = 0
    for tname in TPCDS_SCHEMA:
        rows = _load_csv_rows(f"{dir_tpcds}{tname}.csv.gz")
        p = out_dir / f"{tname}.pkl"
        with open(p, "wb") as f:
            pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
        total_bytes += p.stat().st_size
    print(f"Built {out_dir} ({total_bytes / 1024:.1f} KB, {len(TPCDS_SCHEMA)} tables)", file=sys.stderr)


if __name__ == "__main__":
    main()
