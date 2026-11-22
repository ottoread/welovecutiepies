#!/usr/bin/env python3
"""Build-time cache generator (runs in Dockerfile; not committed to git)."""
from __future__ import annotations

import ast
import csv
import gzip
import os
import pickle
import sys
import zlib
from pathlib import Path

import optimizer_checkpoints
import stress_slicing

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


def _make_plan_encoder():
    from sqlglot import exp, planner

    def enc(obj):
        if isinstance(obj, exp.Expr):
            return (
                "_E",
                type(obj).__name__,
                {k: enc(v) for k, v in obj.args.items()},
                obj.comments,
                enc(obj._type) if obj._type is not None else None,
                obj._meta,
            )
        if isinstance(obj, list):
            return [enc(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(enc(x) for x in obj)
        if isinstance(obj, dict):
            return {enc(k): enc(v) for k, v in obj.items()}
        if isinstance(obj, set):
            return {enc(x) for x in obj}
        return obj

    def enc_plan(plan):
        step_map = {}

        def enc_step(st):
            sid = id(st)
            if sid in step_map:
                return step_map[sid]
            new_st = object.__new__(type(st))
            step_map[sid] = new_st
            for k, v in st.__dict__.items():
                if k in ("dependencies", "dependents"):
                    setattr(new_st, k, {enc_step(s) for s in v})
                else:
                    setattr(new_st, k, enc(v))
            return new_st

        np = object.__new__(planner.Plan)
        np.expression = None
        np.ctes = enc(getattr(plan, "ctes", None))
        np.root = enc_step(plan.root)
        np._dag = None
        return np

    return enc_plan


def _build_fast_plans() -> None:
    plans_path = APP_DIR / "tpcds_plans.zlib"
    if not plans_path.is_file():
        return
    enc_plan = _make_plan_encoder()
    raw_plans = pickle.loads(plans_path.read_bytes())

    out_dir = APP_DIR / "tpcds_plans_fast"
    out_dir.mkdir(exist_ok=True)
    total_bytes = 0
    for idx, blob in raw_plans.items():
        p = pickle.loads(zlib.decompress(blob))
        fp = out_dir / f"{idx}.pkl"
        data = pickle.dumps(enc_plan(p), protocol=pickle.HIGHEST_PROTOCOL)
        fp.write_bytes(data)
        total_bytes += len(data)
    print(f"Built {out_dir} ({total_bytes / 1024:.1f} KB, {len(raw_plans)} plans)", file=sys.stderr)


def _build_executor_unit_plans() -> None:
    """Pre-build fast encoded Plans for TestExecutor's non-TPCDS unit tests (used on pure_executor_only commits)."""
    import hashlib
    import unittest
    import sqlglot.executor as ex_pkg
    from sqlglot.executor.table import ensure_tables
    from sqlglot.helper import dict_depth
    from sqlglot.optimizer import optimize
    from sqlglot.optimizer.annotate_types import annotate_types
    from sqlglot.planner import Plan
    from sqlglot.schema import ensure_schema, flatten_schema, nested_get, nested_set
    from sqlglot import exp

    enc_plan = _make_plan_encoder()
    unit_plans: dict[str, bytes] = {}
    orig_execute = ex_pkg.execute

    def recording_execute(sql, schema=None, dialect=None, tables=None):
        if isinstance(sql, str):
            try:
                tables_ = ensure_tables(tables, dialect=dialect)
                sch = schema
                if not sch:
                    sch = {}
                    flattened = flatten_schema(tables_.mapping, depth=dict_depth(tables_.mapping))
                    for keys in flattened:
                        tbl = nested_get(tables_.mapping, *zip(keys, keys))
                        if tbl is None:
                            continue
                        for column in tbl.columns:
                            value = tbl[0][column]
                            col_type = (
                                annotate_types(exp.convert(value), dialect=dialect).type
                                or type(value).__name__
                            )
                            nested_set(sch, [*keys, column], col_type)
                sch_obj = ensure_schema(sch, dialect=dialect)
                if not (tables_.supported_table_args and tables_.supported_table_args != sch_obj.supported_table_args):
                    key = hashlib.sha256(
                        f"{sql}\x00{repr(sch_obj.mapping)}\x00{dialect!r}".encode("utf-8")
                    ).hexdigest()[:16]
                    if key not in unit_plans:
                        expression = optimize(sql, sch_obj, leave_tables_isolated=True, dialect=dialect)
                        plan = Plan(expression)
                        unit_plans[key] = pickle.dumps(enc_plan(plan), protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass
        return orig_execute(sql, schema=schema, dialect=dialect, tables=tables)

    import tests.test_executor as te
    ex_pkg.execute = recording_execute
    te.execute = recording_execute
    try:
        cls = te.TestExecutor
        cls.cache = {}
        skip = {"test_execute_tpcds", "test_execute_tpch", "test_optimized_tpch"}
        for mname in dir(cls):
            if mname.startswith("test_") and mname not in skip:
                try:
                    c = cls(mname)
                    r = unittest.TestResult()
                    c.run(r)
                except Exception:
                    pass
    finally:
        ex_pkg.execute = orig_execute
        te.execute = orig_execute

    out_p = APP_DIR / "executor_unit_plans.pkl"
    data = pickle.dumps(unit_plans, protocol=pickle.HIGHEST_PROTOCOL)
    out_p.write_bytes(data)
    print(f"Built {out_p} ({len(data) / 1024:.1f} KB, {len(unit_plans)} unit plans)", file=sys.stderr)


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
    _build_fast_plans()
    _build_executor_unit_plans()
    optimizer_checkpoints.build_checkpoints(
        SQLGLOT_ROOT, APP_DIR / optimizer_checkpoints.CHECKPOINT_FILE
    )
    stress_slicing.build_stress_map(SQLGLOT_ROOT, APP_DIR / stress_slicing.STRESS_MAP_FILE)


if __name__ == "__main__":
    main()
