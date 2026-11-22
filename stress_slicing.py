"""Coverage for independent blocks of long, pinned SQLGlot tests."""
from __future__ import annotations

import ast
import importlib
import inspect
import pickle
import sys
import textwrap
import unittest
import zlib
from pathlib import Path

STRESS_MAP_FILE = "stress_blocks.zlib"
TPCDS_FIXTURE = "optimizer/tpc-ds/tpc-ds.sql"
OPTIMIZER = "tests/test_optimizer.py::TestOptimizer::"
BLOCKS = {
    OPTIMIZER + "test_scope": (("many_unions", "chain"), ("chain", "root")),
    OPTIMIZER + "test_deep_ast_type_annotation": (("union_sql", "binary_sql"), ("binary_sql", None)),
    OPTIMIZER + "test_qualify_struct_star_long_cte_chain": (("*", None),),
    "tests/test_serde.py::TestSerde::test_recursion": (("*", None),),
}


def _find_block(body: list[ast.stmt], start: str, stop: str | None) -> range:
    def index(name: str) -> int:
        return next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)

    return range(0 if start == "*" else index(start), len(body) if stop is None else index(stop))


def slice_method(method, skip: set[str], *, keep: set[str] | None = None):
    test_id = f"{method.__module__.replace('.', '/')}.py::{method.__qualname__.replace('.', '::')}"
    lines, start_line = inspect.getsourcelines(method)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    function = tree.body[0]
    body = function.body
    selected = set() if keep is not None else set(range(len(body)))
    for name, stop in BLOCKS[test_id]:
        block = _find_block(body, name, stop)
        if keep is not None:
            if name in keep:
                selected.update(block)
        elif name in skip:
            selected.difference_update(block)
    function.body = [stmt for i, stmt in enumerate(body) if i in selected] or [ast.Pass()]
    ast.increment_lineno(ast.fix_missing_locations(tree), start_line - 1)
    namespace = {}
    exec(compile(tree, method.__code__.co_filename, "exec"), method.__globals__, namespace)
    return namespace[function.name]


def build_stress_map(
    repo: Path,
    destination: Path,
    test_ids: set[str] | None = None,
    *,
    fixture_queries: list[str] | None = None,
) -> None:
    from sqlglot import exp, parse_one

    prefix = str(repo / "sqlglot") + "/"
    root = str(repo) + "/"
    events = sys.monitoring.events
    tool = sys.monitoring.COVERAGE_ID
    paths: dict[object, str | None] = {}
    source: dict[str, set[int]] = {}
    unsafe_flags: dict[type, set[str] | None] = {}
    original_errors = exp.Expression.error_messages

    def record(code, line):
        if code not in paths:
            filename = code.co_filename
            paths[code] = filename[len(root):] if filename.startswith(prefix) else None
        path = paths[code]
        if path is not None:
            source.setdefault(path, set()).add(line)
        return sys.monitoring.DISABLE

    def record_errors(self, *args, **kwargs):
        cls = type(self)
        try:
            result = original_errors(self, *args, **kwargs)
        except BaseException:
            unsafe_flags[cls] = None
            raise
        if result:
            # Even a flag that is satisfied can change the order of errors.
            unsafe_flags[cls] = None
        else:
            missing = unsafe_flags.setdefault(cls, set())
            if missing is not None:
                for key in self.arg_types:
                    value = self.args.get(key)
                    if value is None or (isinstance(value, list) and not value):
                        missing.add(key)
        return result

    def flag_keys() -> dict[str, set[str] | None]:
        result: dict[str, set[str] | None] = {}
        for expr, missing in unsafe_flags.items():
            for base in expr.__mro__:
                if not base.__module__.startswith("sqlglot."):
                    continue
                key = f"{base.__module__}:{base.__qualname__}"
                if missing is None:
                    result[key] = None
                else:
                    previous = result.setdefault(key, set())
                    if previous is not None:
                        previous.update(missing)
        return result

    result = {}
    sys.monitoring.use_tool_id(tool, "blast-stress-slicing")
    sys.monitoring.register_callback(tool, events.LINE, record)
    try:
        for test_id, blocks in BLOCKS.items():
            if test_ids is not None and test_id not in test_ids:
                continue
            filename, class_name, method_name = test_id.split("::")
            module = importlib.import_module(filename.removesuffix(".py").replace("/", "."))
            cls = getattr(module, class_name)
            method = getattr(cls, method_name)
            result[test_id] = {}
            for name, _ in blocks:
                case = cls(method_name)
                partial = slice_method(method, set(), keep={name})
                setattr(case, method_name, partial.__get__(case, cls))
                outcome = unittest.TestResult()
                source = {}
                unsafe_flags.clear()
                exp.Expression.error_messages = record_errors
                sys.monitoring.restart_events()
                sys.monitoring.set_events(tool, events.LINE)
                try:
                    case.run(outcome)
                finally:
                    sys.monitoring.set_events(tool, 0)
                    exp.Expression.error_messages = original_errors
                if outcome.wasSuccessful() and outcome.testsRun == 1 and not outcome.skipped:
                    result[test_id][name] = source, flag_keys()
    finally:
        sys.monitoring.register_callback(tool, events.LINE, None)
        sys.monitoring.free_tool_id(tool)

    if test_ids is None or fixture_queries is not None:
        from sqlglot.optimizer import optimize
        from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

        if fixture_queries is None:
            queries = [(meta, sql, TPCDS_SCHEMA) for meta, sql, _ in load_sql_fixture_pairs(TPCDS_FIXTURE)]
        else:
            queries = [({}, sql, {}) for sql in fixture_queries]
        requirements = []
        exp.Expression.error_messages = record_errors
        try:
            for meta, sql, schema in queries:
                if meta.keys() - {"title", "dialect", "execute", "execute_min_python"}:
                    requirements.append(None)
                    continue
                unsafe_flags.clear()
                dialect = meta.get("dialect")
                try:
                    tree = optimize(parse_one(sql, read=dialect), schema=schema, dialect=dialect)
                    tree.sql(dialect, pretty=True)
                except Exception:
                    requirements.append(None)
                    continue
                requirements.append(flag_keys())
        finally:
            exp.Expression.error_messages = original_errors
        result[TPCDS_FIXTURE] = requirements
    destination.write_bytes(zlib.compress(pickle.dumps(result, protocol=5)))


def load_map(path: Path) -> dict:
    return pickle.loads(zlib.decompress(path.read_bytes())) if path.is_file() else {}


def _flags_may_change_validation(unsafe_flags: dict, changed_flags: dict[str, set[str]]) -> bool:
    return any(
        (missing := unsafe_flags.get(cls, ())) is None or not flags.isdisjoint(missing)
        for cls, flags in changed_flags.items()
    )


def required_fixture_mask(coverage: dict, changed_flags: dict[str, set[str]]) -> int | None:
    requirements = coverage.get(TPCDS_FIXTURE)
    if requirements is None:
        return None
    return sum(
        1 << idx for idx, entry in enumerate(requirements)
        if entry is None or _flags_may_change_validation(entry, changed_flags)
    )


def skipped_blocks(
    coverage: dict,
    test_id: str,
    changed_files: set[str],
    changed_lines: dict[str, set[int]],
    changed_flags: dict[str, set[str]] | None = None,
) -> set[str]:
    skip = set()
    for block, (lines, unsafe_flags) in coverage.get(test_id, {}).items():
        if changed_flags and _flags_may_change_validation(unsafe_flags, changed_flags):
            continue
        if any(
            rel in lines and (rel not in changed_lines or changed_lines[rel] & lines[rel])
            for rel in changed_files
        ):
            continue
        skip.add(block)
    return skip
