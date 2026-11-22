"""Build and load optimizer inputs from the pinned SQLGlot fixtures."""
from __future__ import annotations

import inspect
import io
import json
import pickle
import sys
import zlib
from collections import Counter
from functools import lru_cache
from pathlib import Path

CHECKPOINT_FILE = "optimizer_checkpoints.zlib"
# Estimated fraction of full verification CPU remaining at each rule boundary.
REMAINING_RATIO = (
    1.0, 0.76, 0.74, 0.70, 0.67, 0.62, 0.57, 0.50,
    0.46, 0.44, 0.42, 0.40, 0.32, 0.28, 0.22,
)
DIRECT_OPTIMIZER_TESTS = (
    "test_optimize_in_nulls",
    "test_optimize_uncorrelated_exists",
    "test_optimize_correlated_all",
    "test_optimize_subqueries_in_aggregates",
    "test_optimize_special_joins",
    "test_qualify_semi_anti_join_alias_shadowing_column",
)


def query_key(sql, schema, dialect, leave_isolated):
    return (
        sql, dialect, leave_isolated, repr(schema.mapping),
        repr(schema.visible), repr(schema.udf_mapping), schema.normalize,
        type(schema.dialect).__module__, type(schema.dialect).__qualname__,
    )


def _restore_expression(cls, args, comments, node_type, metadata, arg_key, index, self_type, hashed):
    node = object.__new__(cls)
    node.args = args
    node.comments = comments
    node._type = node if self_type else node_type
    node._meta = metadata
    node.parent = None
    node.arg_key = arg_key
    node.index = index
    node._hash = None
    if not node.is_primitive:
        for key, value in args.items():
            node._set_parent(key, value)
    if hashed:
        hash(node)
    return node


class _CheckpointPickler(pickle.Pickler):
    def __init__(self, target, expression_type):
        super().__init__(target, protocol=5)
        self.expression_type = expression_type

    def reducer_override(self, node):
        if not isinstance(node, self.expression_type):
            return NotImplemented
        # SQLGlot's standard reducer is intended for interchange and drops
        # None-valued arguments. A rule can distinguish None from a missing key.
        self_type = node._type is node
        return (
            _restore_expression,
            (type(node), node.args, node.comments, None if self_type else node._type,
             node._meta, node.arg_key, node.index, self_type, node._hash is not None),
        )


def _serialize(tree, expression_type) -> bytes:
    buffer = io.BytesIO()
    _CheckpointPickler(buffer, expression_type).dump(tree)
    return buffer.getvalue()


def _fixture_queries():
    from tests.helpers import TPCDS_SCHEMA, TPCH_SCHEMA, load_sql_fixture_pairs

    for name, schema in (
        ("optimizer/tpc-ds/tpc-ds.sql", TPCDS_SCHEMA),
        ("optimizer/tpc-h/tpc-h.sql", TPCH_SCHEMA),
        ("optimizer/optimizer.sql", {
            "x": {"a": "INT", "b": "INT"},
            "y": {"b": "INT", "c": "INT"},
            "z": {"b": "INT", "c": "INT"},
            "u": {"a": "INT", "b": "INT", "c": "INT"},
            "w": {"d": "INT", "e": "INT", "f": "TEXT"},
        }),
    ):
        for meta, sql, _ in load_sql_fixture_pairs(name):
            dialect = meta.get("dialect")
            if not (meta.keys() - {"title", "dialect", "execute", "execute_min_python", "note", "schema"}):
                mapping = json.loads(meta["schema"]) if meta.get("schema") else schema
                yield sql, mapping, dialect, False
            if name == "optimizer/tpc-ds/tpc-ds.sql" and meta.get("execute"):
                yield sql, schema, None, True


def _direct_optimizer_queries():
    import unittest

    import sqlglot.optimizer as optimizer
    from tests.test_optimizer import TestOptimizer

    optimize = optimizer.optimize
    queries = []

    def record(expression, *args, **kwargs):
        if (
            isinstance(expression, str) and not args
            and kwargs.keys() <= {"schema", "dialect"}
            and isinstance(kwargs.get("schema"), dict)
            and isinstance(kwargs.get("dialect"), (str, type(None)))
        ):
            queries.append((expression, kwargs["schema"], kwargs.get("dialect"), False))
        return optimize(expression, *args, **kwargs)

    optimizer.optimize = record
    try:
        suite = unittest.TestSuite(TestOptimizer(name) for name in DIRECT_OPTIMIZER_TESTS)
        suite.run(unittest.TestResult())
    finally:
        optimizer.optimize = optimize
        if connection := getattr(TestOptimizer, "conn", None):
            connection.close()
    return queries


def build_checkpoints(repo: Path, destination: Path, queries=None) -> None:
    from sqlglot import exp, parse_one
    from sqlglot.optimizer import RULES
    from sqlglot.schema import ensure_schema

    if queries is None:
        from itertools import chain

        queries = chain(_fixture_queries(), _direct_optimizer_queries())
    rule_params = [inspect.getfullargspec(rule).args for rule in RULES]
    root = str(repo) + "/"
    prefix = root + "sqlglot/"
    paths: dict[object, str | None] = {}
    seen: set[str] = set()
    events = sys.monitoring.events
    tool = sys.monitoring.OPTIMIZER_ID

    def record(code, offset):
        if code not in paths:
            filename = code.co_filename
            paths[code] = filename[len(root):] if filename.startswith(prefix) else None
        path = paths[code]
        if path is not None:
            seen.add(path)
        return sys.monitoring.DISABLE

    records = {}
    sys.monitoring.use_tool_id(tool, "blast-optimizer-checkpoints")
    sys.monitoring.register_callback(tool, events.LINE, record)
    try:
        for sql, mapping, dialect, leave_isolated in queries:
            schema = ensure_schema(mapping, dialect=dialect)
            key = query_key(sql, schema, dialect, leave_isolated)
            if key in records:
                continue
            inputs = {
                "schema": schema, "dialect": dialect, "db": None,
                "catalog": None, "sql": None, "isolate_tables": True,
                "quote_identifiers": False,
            }
            if leave_isolated:
                inputs["leave_tables_isolated"] = True
            seen.clear()
            sys.monitoring.restart_events()
            sys.monitoring.set_events(tool, events.LINE)
            try:
                tree = parse_one(sql, read=dialect)
                snapshots = []
                for stage, (rule, params) in enumerate(zip(RULES, rule_params), 1):
                    tree = rule(tree, **{k: inputs[k] for k in params if k in inputs})
                    sys.monitoring.set_events(tool, 0)
                    try:
                        snapshots.append((stage, frozenset(seen), _serialize(tree, exp.Expr)))
                    except (AttributeError, TypeError, ValueError, RecursionError, pickle.PicklingError):
                        pass
                    sys.monitoring.set_events(tool, events.LINE)
                records[key] = snapshots
            except Exception:
                # Some fixtures intentionally exercise optimizer exceptions.
                # A failed baseline never enters the resume cache.
                records[key] = []
            finally:
                sys.monitoring.set_events(tool, 0)
    finally:
        sys.monitoring.register_callback(tool, events.LINE, None)
        sys.monitoring.free_tool_id(tool)
    destination.write_bytes(zlib.compress(pickle.dumps(records, protocol=5)))
    profiles = Counter(
        tuple((stage, dependencies) for stage, dependencies, _ in snapshots)
        for snapshots in records.values()
    )
    destination.with_suffix(".index").write_bytes(zlib.compress(pickle.dumps(profiles, protocol=5)))


@lru_cache(maxsize=2)
def _load_index(path: Path) -> Counter:
    index = path.with_suffix(".index")
    return pickle.loads(zlib.decompress(index.read_bytes())) if index.is_file() else Counter()


def estimate_runtime_ratio(path: Path, changed_files: set[str]) -> float:
    profiles = _load_index(path)
    if not profiles:
        return 1.0
    total = 0.0
    for stages, count in profiles.items():
        ratio = next(
            (REMAINING_RATIO[stage] for stage, paths in reversed(stages)
             if paths.isdisjoint(changed_files)),
            1.0,
        )
        total += ratio * count
    return total / profiles.total()


def load_for_patch(path: Path, changed_files: set[str]) -> dict:
    if not path.is_file():
        return {}
    records = pickle.loads(zlib.decompress(path.read_bytes()))
    result = {}
    for key, snapshots in records.items():
        for stage, dependencies, tree in reversed(snapshots):
            if dependencies.isdisjoint(changed_files):
                result[key] = (stage, tree)
                break
    return result
