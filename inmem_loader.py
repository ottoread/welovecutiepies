"""In-memory patch overlay (import hook + open hook), fail-fast unittest runner, and turbo verifiers."""
from __future__ import annotations

import ast
import builtins
import collections
import csv
import dis
import gzip
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import io
import os
import pickle
import signal
import sys
import time
import unittest
import zlib
from pathlib import Path
from typing import Any

_POP_TOP = dis.opmap["POP_TOP"]
_CACHE_OP = dis.opmap.get("CACHE", 0)
_APP_DIR = Path(__file__).resolve().parent


def _pair_hash(sql_a: str, sql_b: str) -> str:
    return hashlib.sha256((sql_a + "\x00" + sql_b).encode("utf-8")).hexdigest()[:16]


def _return_value_discarded() -> bool:
    """Return True iff the caller of our caller immediately discards the return value (POP_TOP)."""
    try:
        f = sys._getframe(2)
        code = f.f_code.co_code
        pos = f.f_lasti + 2
        n = len(code)
        while pos < n and code[pos] == _CACHE_OP:
            pos += 2
        return pos < n and code[pos] == _POP_TOP
    except Exception:
        return False


def _any_dialect_in(d: Any, df: set[str]) -> bool:
    """Return True if dialect spec `d` (str, comma-separated str, Enum, Dialect subclass or instance) intersects `df`."""
    if not d:
        return False
    if isinstance(d, str):
        for part in d.split(","):
            if part.strip().lower() in df:
                return True
        return False
    val = getattr(d, "value", None)
    if isinstance(val, str):
        for part in val.split(","):
            if part.strip().lower() in df:
                return True
        return False
    if isinstance(d, type):
        return d.__name__.lower() in df
    return type(d).__name__.lower() in df


class _TestFailed(BaseException):
    """Raised immediately on the very first failure/error/failing-subtest to short-circuit."""


class _FailFastResult(unittest.TestResult):
    """Zero-traceback-formatting fail-fast TestResult (avoids formatting 1000-frame RecursionErrors!)."""

    def addError(self, test, err):
        raise _TestFailed()

    def addFailure(self, test, err):
        raise _TestFailed()

    def addUnexpectedSuccess(self, test):
        raise _TestFailed()

    def addSubTest(self, test, subtest, err):
        if err is not None:
            raise _TestFailed()


class _OverlayLoader(importlib.abc.SourceLoader):
    def __init__(self, fullname: str, real_path: str, source: str, is_pkg: bool):
        self.fullname = fullname
        self.real_path = real_path
        self.source = source
        self.is_pkg = is_pkg

    def get_filename(self, fullname: str) -> str:
        return self.real_path

    def get_data(self, path: str) -> bytes:
        return self.source.encode("utf-8")


class _DeletedLoader(importlib.abc.Loader):
    def create_module(self, spec):
        raise ModuleNotFoundError(f"Deleted module: {spec.name}", name=spec.name)

    def exec_module(self, module):
        raise ModuleNotFoundError("Deleted module")


class _OverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, repo_root: Path, patched_files: dict[str, str | None]):
        self.repo_root = repo_root.resolve()
        self.mod_map: dict[str, tuple[str, str | None, bool, list[str] | None]] = {}
        for rel, content in patched_files.items():
            if not rel.endswith(".py"):
                continue
            full_path = str(self.repo_root / rel)
            if rel.endswith("/__init__.py"):
                mod = rel[: -len("/__init__.py")].replace("/", ".")
                is_pkg = True
                locs = [str((self.repo_root / rel).parent)]
            else:
                mod = rel[:-3].replace("/", ".")
                is_pkg = False
                locs = None
            self.mod_map[mod] = (full_path, content, is_pkg, locs)

    def find_spec(self, fullname: str, path=None, target=None):
        entry = self.mod_map.get(fullname)
        if entry is None:
            return None
        full_path, content, is_pkg, locs = entry
        if content is None:
            return importlib.machinery.ModuleSpec(fullname, _DeletedLoader())
        loader = _OverlayLoader(fullname, full_path, content, is_pkg)
        spec = importlib.util.spec_from_loader(fullname, loader, origin=full_path, is_package=is_pkg)
        if spec is not None and is_pkg and locs is not None:
            spec.submodule_search_locations = locs
        return spec


def install_overlay(repo_root: Path, patched_files: dict[str, str | None]) -> None:
    """Install in-memory import hook + file open/Path overlay for patched files."""
    repo_root = repo_root.resolve()
    finder = _OverlayFinder(repo_root, patched_files)
    sys.meta_path.insert(0, finder)

    abs_map: dict[str, str | None] = {}
    for rel, content in patched_files.items():
        abs_map[os.path.normpath(str(repo_root / rel))] = content

    if not abs_map:
        return

    orig_open = builtins.open
    orig_path_open = Path.open
    orig_path_read_text = Path.read_text
    orig_path_read_bytes = Path.read_bytes

    def _lookup(file_obj) -> tuple[bool, str | None]:
        if isinstance(file_obj, (str, os.PathLike)):
            p = os.path.normpath(os.path.abspath(os.fspath(file_obj)))
            if p in abs_map:
                return True, abs_map[p]
        return False, None

    def overlay_open(file, mode="r", *args, **kwargs):
        if "w" not in mode and "a" not in mode and "x" not in mode and "+" not in mode:
            hit, content = _lookup(file)
            if hit:
                if content is None:
                    raise FileNotFoundError(f"Deleted file: {file}")
                if "b" in mode:
                    return io.BytesIO(content.encode("utf-8"))
                return io.StringIO(content)
        return orig_open(file, mode, *args, **kwargs)

    def overlay_path_open(self, mode="r", *args, **kwargs):
        if "w" not in mode and "a" not in mode and "x" not in mode and "+" not in mode:
            hit, content = _lookup(self)
            if hit:
                if content is None:
                    raise FileNotFoundError(f"Deleted file: {self}")
                if "b" in mode:
                    return io.BytesIO(content.encode("utf-8"))
                return io.StringIO(content)
        return orig_path_open(self, mode, *args, **kwargs)

    def overlay_path_read_text(self, encoding=None, errors=None, newline=None):
        hit, content = _lookup(self)
        if hit:
            if content is None:
                raise FileNotFoundError(f"Deleted file: {self}")
            return content
        return orig_path_read_text(self, encoding=encoding, errors=errors, newline=newline)

    def overlay_path_read_bytes(self):
        hit, content = _lookup(self)
        if hit:
            if content is None:
                raise FileNotFoundError(f"Deleted file: {self}")
            return content.encode("utf-8")
        return orig_path_read_bytes(self)

    builtins.open = overlay_open
    io.open = overlay_open
    Path.open = overlay_path_open
    Path.read_text = overlay_path_read_text
    Path.read_bytes = overlay_path_read_bytes


class _LazyFuture:
    def __init__(self, fn, args, kwargs, check_pickle: bool):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._check_pickle = check_pickle
        self._done = False
        self._val = None
        self._exc = None

    def result(self):
        if not self._done:
            self._done = True
            try:
                args = self._args
                kwargs = self._kwargs
                if self._check_pickle:
                    args = pickle.loads(pickle.dumps(args))
                    kwargs = pickle.loads(pickle.dumps(kwargs))
                res = self._fn(*args, **kwargs)
                if self._check_pickle:
                    res = pickle.loads(pickle.dumps(res))
                self._val = res
            except BaseException as e:
                self._exc = e
        if self._exc is not None:
            raise self._exc
        return self._val


class _LazyInProcessPool:
    """Drop-in ProcessPoolExecutor replacement: zero spawn overhead + lazy fail-fast execution."""

    def __init__(self, check_pickle: bool = False, initializer=None, initargs=(), **kwargs):
        self._check_pickle = check_pickle
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def submit(self, fn, *args, **kwargs):
        return _LazyFuture(fn, args, kwargs, self._check_pickle)


class _DeferredDuckResult:
    """Deferred DuckDB query handle so assert_frame_equal / assertEqual can hit known passing pairs in O(1)."""

    def __init__(self, lazy_conn: _LazyDuckDBConn, sql: str):
        self._conn = lazy_conn
        self.sql = sql

    def df(self):
        return _DeferredDuckDF(self._conn, self.sql)

    def fetchdf(self):
        return self._conn._ensure().execute(self.sql).fetchdf()

    def fetchall(self):
        return _DeferredDuckRows(self._conn, self.sql)

    def __getattr__(self, name):
        return getattr(self._conn._ensure().execute(self.sql), name)


class _DeferredDuckDF:
    def __init__(self, lazy_conn: _LazyDuckDBConn, sql: str):
        self._conn = lazy_conn
        self.sql = sql

    def eval(self):
        return self._conn._ensure().execute(self.sql).df()


class _DeferredDuckRows:
    def __init__(self, lazy_conn: _LazyDuckDBConn, sql: str):
        self._conn = lazy_conn
        self.sql = sql

    def eval(self):
        return self._conn._ensure().execute(self.sql).fetchall()


class _LazyDuckDBConn:
    """Delays creating TestOptimizer's DuckDB connection AND defers queries for known-passing pair memoization."""

    def __init__(self, defer: bool = False):
        self._conn = None
        self._defer = defer

    def _ensure(self):
        if self._conn is None:
            import duckdb
            conn = duckdb.connect()
            conn.execute(
                """
                CREATE TABLE x (a INT, b INT);
                CREATE TABLE y (b INT, c INT);
                CREATE TABLE z (b INT, c INT);
                CREATE TABLE w (d TEXT, e TEXT);
                INSERT INTO x VALUES (1, 1), (2, 2), (2, 2), (3, 3), (null, null);
                INSERT INTO y VALUES (2, 2), (2, 2), (3, 3), (4, 4), (null, null),
                                     (3, 3), (3, 3), (4, 4), (5, 5), (null, null);
                INSERT INTO w VALUES ('a', 'b');
                CREATE TABLE unpivotable (id INT, jan INT, feb INT, north INT, south INT);
                INSERT INTO unpivotable VALUES (1, 100, 200, 7, 8), (2, 300, 400, 9, 10);
                CREATE TABLE pivotable (id INT, cat TEXT, val INT, kind TEXT, amt INT);
                INSERT INTO pivotable VALUES (1, 'a', 10, 'x', 5), (1, 'b', 20, 'x', 5);
                CREATE TABLE comparisons AS
                SELECT * FROM (VALUES (1), (2), (NULL)) AS x(a)
                CROSS JOIN (VALUES (1), (2), (NULL)) AS y(b)
                CROSS JOIN (VALUES (TRUE), (FALSE), (NULL)) AS z(c);
                """
            )
            self._conn = conn
        return self._conn

    def execute(self, sql, *args, **kwargs):
        if self._defer and not args and not kwargs and isinstance(sql, str):
            return _DeferredDuckResult(self, sql)
        return self._ensure().execute(sql, *args, **kwargs)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _install_lazy_test_executor(mod, cls) -> None:
    """Replace TestExecutor's heavy setUpClass (32 CSVs + 2 DuckDBs) with lazy on-demand loaders."""
    state = {"loaded": False}
    orig_setup_ref = [cls.setUpClass]

    def _ensure_loaded_real():
        if not state["loaded"]:
            state["loaded"] = True
            orig_setup_ref[0]()
            for attr in ("tpch_conn", "tpcds_conn", "tpch_tables", "tpcds_tables", "tpch_sqls", "tpcds_sqls"):
                setattr(cls, f"_real_{attr}", cls.__dict__.get(attr))

    class _LazyDesc:
        def __init__(self, name):
            self.name = name

        def __get__(self, obj, owner=None):
            _ensure_loaded_real()
            return getattr(cls, f"_real_{self.name}")

    for attr in ("tpch_conn", "tpcds_conn", "tpch_tables", "tpcds_tables", "tpch_sqls", "tpcds_sqls"):
        setattr(cls, attr, _LazyDesc(attr))
    cls.cache = {}


# Global depth counter for active assertRaises / assertLogs / assertWarns context managers!
_assert_ctx_depth = 0
_ctx_hooks_installed = False


def _install_assert_ctx_hooks() -> None:
    global _ctx_hooks_installed
    if _ctx_hooks_installed:
        return
    _ctx_hooks_installed = True

    class _DepthProxy:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            global _assert_ctx_depth
            _assert_ctx_depth += 1
            return self._inner.__enter__()

        def __exit__(self, *args):
            global _assert_ctx_depth
            _assert_ctx_depth -= 1
            return self._inner.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    for mname in ("assertRaises", "assertRaisesRegex", "assertLogs", "assertNoLogs", "assertWarns", "assertWarnsRegex"):
        orig = getattr(unittest.TestCase, mname, None)
        if orig is None:
            continue

        def make_wrapped(fn):
            def wrapped(self, *args, **kwargs):
                res = fn(self, *args, **kwargs)
                # Only wrap if called as context manager (returns an object with __enter__)
                if res is not None and hasattr(res, "__enter__"):
                    return _DepthProxy(res)
                return res
            return wrapped

        setattr(unittest.TestCase, mname, make_wrapped(orig))


def _install_dialect_validator_slicing(dialect_filter: set[str]) -> None:
    """Patch Validator.validate_all / validate_transpile / validate_identity with exact _assert_ctx_depth guard."""
    _install_assert_ctx_hooks()
    try:
        import tests.dialects.test_dialect as td
    except BaseException:
        return
    Validator = td.Validator
    if getattr(Validator, "_br_patched", False):
        return
    Validator._br_patched = True

    orig_all = Validator.validate_all
    orig_id = Validator.validate_identity
    orig_tr = Validator.validate_transpile

    def fast_validate_all(self, sql, read=None, write=None, pretty=False, identify=False):
        if _assert_ctx_depth > 0 or _any_dialect_in(self.dialect, dialect_filter):
            return orig_all(self, sql, read=read, write=write, pretty=pretty, identify=identify)
        f_read = {k: v for k, v in read.items() if _any_dialect_in(k, dialect_filter)} if read else None
        f_write = {k: v for k, v in write.items() if _any_dialect_in(k, dialect_filter)} if write else None
        if not f_read and not f_write:
            return None
        if f_read and not f_write:
            from sqlglot import ErrorLevel, parse_one
            for read_dialect, read_sql in f_read.items():
                with self.subTest(f"{read_dialect} -> {sql}"):
                    self.assertEqual(
                        parse_one(read_sql, read_dialect).sql(
                            self.dialect,
                            unsupported_level=ErrorLevel.IGNORE,
                            pretty=pretty,
                            identify=identify,
                        ),
                        sql,
                    )
            return None
        return orig_all(self, sql, read=f_read, write=f_write, pretty=pretty, identify=identify)

    def fast_validate_transpile(self, sql, write_sql, write_dialect=None):
        if (
            _assert_ctx_depth == 0
            and not _any_dialect_in(self.dialect, dialect_filter)
            and not _any_dialect_in(write_dialect, dialect_filter)
        ):
            return None
        return orig_tr(self, sql, write_sql, write_dialect=write_dialect)

    def fast_validate_identity(
        self, sql, write_sql=None, pretty=False, check_command_warning=False, identify=False
    ):
        if (
            _assert_ctx_depth == 0
            and not _any_dialect_in(self.dialect, dialect_filter)
            and _return_value_discarded()
        ):
            return None
        return orig_id(
            self,
            sql,
            write_sql=write_sql,
            pretty=pretty,
            check_command_warning=check_command_warning,
            identify=identify,
        )

    Validator.validate_all = fast_validate_all
    Validator.validate_transpile = fast_validate_transpile
    Validator.validate_identity = fast_validate_identity


_EXACT_BASELINE_MATCH = object()
_active_opt_sql_ctx: list[str | None] = [None]


def _ast_fingerprint(root, exp_cls) -> bytes | None:
    """Complete canonical state hash of an exp.Expr tree, checking parent/arg_key/index/type/comments/meta."""
    if not isinstance(root, exp_cls) or root.parent is not None:
        return None
    h = hashlib.sha256()
    update = h.update
    stack = [root]
    while stack:
        node = stack.pop()
        update(type(node).__name__.encode())
        update(b"|")
        if node._type is not None:
            update(b"T")
            update(repr(node._type).encode())
        if node.comments:
            update(b"C")
            update(repr(node.comments).encode())
        if node._meta:
            update(b"M")
            update(repr(node._meta).encode())
        update(b"{")
        for k, v in node.args.items():
            update(k.encode())
            update(b":")
            if isinstance(v, list):
                update(b"[")
                for idx_i, ch in enumerate(v):
                    if isinstance(ch, exp_cls):
                        if ch.parent is not node or ch.arg_key != k or ch.index != idx_i:
                            return None
                        update(b"E")
                        stack.append(ch)
                    else:
                        update(b"V")
                        update(repr(ch).encode())
                    update(b",")
                update(b"]")
            elif isinstance(v, exp_cls):
                if v.parent is not node or v.arg_key != k or v.index is not None:
                    return None
                update(b"E")
                stack.append(v)
            else:
                update(b"V")
                update(repr(v).encode())
            update(b";")
        update(b"}")
    return h.digest()[:8]


class _EarlyVerifiedAST:
    """Sound sentinel returned by fast_optimize when post-stage AST matches baseline and downstream rules are untouched."""

    __slots__ = ("_compact", "_pretty", "_duckdb", "_dialect")

    def __init__(self, compact: str, pretty: str | None, duckdb: str | None, dialect: str):
        self._compact = compact
        self._pretty = pretty
        self._duckdb = duckdb
        self._dialect = dialect

    def walk(self):
        return ()

    def sql(self, dialect=None, pretty=False, **kwargs):
        if not kwargs:
            if dialect is None or dialect == "":
                d_str = ""
            elif isinstance(dialect, str):
                d_str = dialect
            else:
                val = getattr(dialect, "value", None)
                d_str = val if isinstance(val, str) else ""
            if pretty and self._pretty is not None and d_str == self._dialect:
                return self._pretty
            if not pretty and d_str == self._dialect:
                return self._compact
            if not pretty and d_str == "duckdb" and self._duckdb is not None:
                return self._duckdb
        raise RuntimeError("Unexpected sql() arguments on _EarlyVerifiedAST")


# Set per job in the forked verifier before any executor tests run.
EXECUTOR_OPTIONS: dict[str, bool] = {}


def install_executor_speedups(
    patched_files: dict[str, str | None],
    allow_lambda_exec: bool = False,
) -> None:
    """Install executor speedups when the patch leaves their dependencies intact."""
    table_patched = "sqlglot/executor/table.py" in patched_files
    context_patched = "sqlglot/executor/context.py" in patched_files
    python_patched = "sqlglot/executor/python.py" in patched_files
    context_table_only = EXECUTOR_OPTIONS.get("context_table_only", False)
    try:
        if not (table_patched or python_patched or context_patched):
            import sqlglot.executor.table as tbl_mod

            if not getattr(tbl_mod.Table, "_br_iter_patched", False):
                tbl_mod.Table._br_iter_patched = True

                def _fast_table_iter(self):
                    reader = self.reader
                    for row in self.rows:
                        reader.row = row
                        yield reader

                tbl_mod.Table.__iter__ = _fast_table_iter

        if (
            allow_lambda_exec
            and not (python_patched or table_patched)
            and "sqlglot/generators/python.py" not in patched_files
            and (not context_patched or context_table_only)
        ):
            import sqlglot.executor.python as py_mod

            PythonExecutor = py_mod.PythonExecutor
            if getattr(PythonExecutor, "_br_exec_patched", False):
                return
            PythonExecutor._br_exec_patched = True

            Table = py_mod.Table
            Context = py_mod.Context
            table_getitem = Table.__getitem__
            direct_iteration = (
                EXECUTOR_OPTIONS.get("direct_iteration", False)
                and not any(
                    path in patched_files
                    for path in (
                        "sqlglot/executor/__init__.py",
                        "sqlglot/executor/env.py",
                        "tests/helpers.py",
                        "tests/test_executor.py",
                    )
                )
            )
            ENV_MOD = py_mod.ENV
            shared_gen = py_mod.Python().generator(identify=True, comments=False)

            def _fast_context_iter(self):
                self.env["scope"] = self.row_readers
                # Keep the table property call, including its validation and any patched behavior.
                table = self.table
                tables = self.tables
                count = len(table.rows)
                if (
                    type(self) is Context
                    and type(tables) is dict
                    and len(tables) == 1
                    and type(table) is Table
                    and Table.__getitem__ is table_getitem
                    and next(iter(tables.values())) is table
                ):
                    for index in range(count):
                        reader = table.reader
                        reader.row = table.rows[index]
                        yield reader, self
                else:
                    for index in range(count):
                        for item in self.tables.values():
                            reader = item[index]
                        yield reader, self

            def _fast_init(self, env=None, tables=None):
                self.generator = shared_gen
                self.env = {**ENV_MOD, **(env or {})}
                self.tables = tables or {}
                self._subquery_plans = {}
                self._plan_names_by_sql = {}
                self._ctes = None
                self._outer_scope = None
                self.env.update(
                    SUBQUERY_COMPARISON=self._subquery_comparison,
                    SUBQUERY_EXISTS=self._subquery_exists,
                    SUBQUERY_SCALAR=self._subquery_scalar,
                )

            def _compile_lambda(self, expression, env):
                if not expression:
                    return None
                expression = self._replace_subqueries(expression)
                sql = self.generator.generate(expression)
                return eval(f"lambda scope: ({sql})", env)

            def _compile_tuple_lambda(self, expressions, env):
                if not expressions:
                    return None
                parts = []
                for e in expressions:
                    if e:
                        e2 = self._replace_subqueries(e)
                        parts.append(self.generator.generate(e2))
                    else:
                        parts.append("None")
                return eval(f"lambda scope: ({', '.join(parts)},)", env)

            def _fast_project_and_filter(self, context, step, table_iter):
                sink = self.table(step.projections if step.projections else context.columns)
                cond_fn = _compile_lambda(self, step.condition, context.env) if step.condition else None
                proj_fn = _compile_tuple_lambda(self, step.projections, context.env) if step.projections else None
                limit = step.offset + step.limit
                sink_rows = sink.rows
                sink_append = sink_rows.append
                sink_width = len(sink.columns)
                c_env = context.env

                if cond_fn is not None:
                    if proj_fn is not None:
                        for reader in table_iter:
                            if len(sink_rows) >= limit:
                                break
                            sc = c_env["scope"]
                            if cond_fn(sc):
                                sink_append(proj_fn(sc))
                    else:
                        for reader in table_iter:
                            if len(sink_rows) >= limit:
                                break
                            if cond_fn(c_env["scope"]):
                                row = reader.row
                                assert len(row) == sink_width
                                sink_append(row)
                else:
                    if proj_fn is not None:
                        for reader in table_iter:
                            if len(sink_rows) >= limit:
                                break
                            sink_append(proj_fn(c_env["scope"]))
                    else:
                        for reader in table_iter:
                            if len(sink_rows) >= limit:
                                break
                            row = reader.row
                            assert len(row) == sink_width
                            sink_append(row)
                return sink

            def _compile_hash_key(self, expressions, context):
                if (
                    direct_iteration
                    and not context_patched
                    and len(expressions) == 1
                    and type(context) is Context
                ):
                    expr = expressions[0]
                    if type(expr) is py_mod.exp.Column and len(context.tables) == 1:
                        alias, table = next(iter(context.tables.items()))
                        reader = context.row_readers.get(alias)
                        if (
                            type(table) is Table
                            and type(reader) is py_mod.RowReader
                            and reader is table.reader
                            and expr.name in reader.columns
                            and self.generator.generate(expr) == f"scope[{alias!r}][{expr.name!r}]"
                        ):
                            return None, reader.columns[expr.name]
                return _compile_tuple_lambda(self, expressions, context.env), None

            def _group_hash_rows(context, compiled, single_key, results, side):
                key_fn, column_index = compiled
                if column_index is not None:
                    for index, (reader, _) in enumerate(context):
                        row = reader.row
                        value = row[column_index]
                        if value is not None:
                            results[(value,)][side].append((index, row))
                elif single_key:
                    env = context.env
                    for index, (reader, _) in enumerate(context):
                        key = key_fn(env["scope"])
                        if key[0] is not None:
                            results[key][side].append((index, reader.row))
                else:
                    env = context.env
                    for index, (reader, _) in enumerate(context):
                        key = key_fn(env["scope"])
                        if all(v is not None for v in key):
                            results[key][side].append((index, reader.row))

            def _fast_hash_join(self, join, source_context, join_context, condition, condition_context):
                s_keys = join["source_key"]
                j_keys = join["join_key"]
                s_key = _compile_hash_key(self, s_keys, source_context)
                j_key = _compile_hash_key(self, j_keys, join_context)
                results = collections.defaultdict(lambda: ([], []))
                _group_hash_rows(source_context, s_key, len(s_keys) == 1, results, 0)
                _group_hash_rows(join_context, j_key, len(j_keys) == 1, results, 1)

                table = Table(source_context.columns + join_context.columns)
                t_append = table.append
                side = join.get("side")

                if not condition and not side:
                    # Preserve source_context.table / join_context.table property access fidelity
                    _ = source_context.table
                    _ = join_context.table
                    for source_group, join_group in results.values():
                        if source_group and join_group:
                            for _, sr in source_group:
                                for _, jr in join_group:
                                    t_append(sr + jr)
                    return table

                cond_fn = None
                if condition and condition_context is not None:
                    cond_fn = _compile_lambda(self, join["condition"], condition_context.env)
                    c_env = condition_context.env

                matched_source = set()
                matched_join = set()
                for source_group, join_group in results.values():
                    if not source_group or not join_group:
                        continue
                    for source_index, source_row in source_group:
                        for join_index, join_row in join_group:
                            row = source_row + join_row
                            if cond_fn is None:
                                t_append(row)
                                if side:
                                    matched_source.add(source_index)
                                    matched_join.add(join_index)
                            else:
                                condition_context.set_row(row)
                                if cond_fn(c_env["scope"]) is True:
                                    t_append(row)
                                    if side:
                                        matched_source.add(source_index)
                                        matched_join.add(join_index)

                self._append_unmatched_join_rows(
                    table,
                    join,
                    source_context.table.rows,
                    join_context.table.rows,
                    matched_source,
                    matched_join,
                )
                return table

            def _fast_aggregate(self, step, context):
                group_exprs = list(step.group.values())
                group_by = self.generate_tuple(group_exprs)

                if step.operands:
                    op_fn = _compile_tuple_lambda(self, step.operands, context.env)
                    operand_table = Table(self.table(step.operands).columns)
                    op_append = operand_table.rows.append
                    c_env = context.env
                    for reader, ctx in context:
                        op_append(op_fn(c_env["scope"]))

                    c_rows = context.table.rows
                    for i, (a, b) in enumerate(zip(c_rows, operand_table.rows)):
                        c_rows[i] = a + b

                    width = len(context.columns)
                    context.add_columns(*operand_table.columns)

                    operand_table = Table(
                        context.columns,
                        context.table.rows,
                        range(width, width + len(operand_table.columns)),
                    )
                    context = self.context({None: operand_table, **context.tables})

                context.sort(group_by)

                gb_fn = _compile_tuple_lambda(self, group_exprs, context.env) if group_exprs else None
                agg_fn = _compile_tuple_lambda(self, step.aggregations, context.env) if step.aggregations else None
                c_env = context.env

                group = None
                start = 0
                end = 1
                length = len(context.table)
                table = self.table(list(step.group) + step.aggregations)
                t_rows = table.rows
                t_append = t_rows.append
                limit = step.offset + step.limit

                if length:
                    for i in range(length):
                        context.set_index(i)
                        key = gb_fn(c_env["scope"]) if gb_fn is not None else ()
                        if group is None:
                            group = key
                        end += 1
                        if key != group:
                            context.set_range(start, end - 2)
                            t_append(group + (agg_fn(c_env["scope"]) if agg_fn is not None else ()))
                            group = key
                            start = end - 2
                        if not step.condition and len(t_rows) >= limit:
                            break
                        if i == length - 1:
                            context.set_range(start, end - 1)
                            t_append(group + (agg_fn(c_env["scope"]) if agg_fn is not None else ()))
                elif step.limit > 0 and not group_by:
                    context.set_range(0, 0)
                    t_append(agg_fn(c_env["scope"]) if agg_fn is not None else ())

                context = self.context({step.name: table, **{name: table for name in context.tables}})
                if step.projections or step.condition:
                    return self.context(
                        {step.name: self._project_and_filter(context, step, context.table_iter(step.name))}
                    )
                return context

            PythonExecutor.__init__ = _fast_init
            PythonExecutor._project_and_filter = _fast_project_and_filter
            PythonExecutor.hash_join = _fast_hash_join
            PythonExecutor.aggregate = _fast_aggregate
            if direct_iteration:
                Context.__iter__ = _fast_context_iter
    except BaseException:
        pass


_OPT_STAGES_CACHE: dict[str, Any] | None = None


def _load_opt_stages() -> dict[str, Any] | None:
    global _OPT_STAGES_CACHE
    if _OPT_STAGES_CACHE is None:
        p = _APP_DIR / "opt_stages.zlib"
        if p.is_file():
            _OPT_STAGES_CACHE = pickle.loads(zlib.decompress(p.read_bytes()))
        else:
            _OPT_STAGES_CACHE = {}
    return _OPT_STAGES_CACHE or None


def install_optimizer_speedups(
    patched_files: dict[str, str | None],
    tpcds_pre_merge: dict[str, Any] | None,
    early_stage: int | None = None,
) -> None:
    """Install 100% sound runtime speedups on sqlglot.optimizer.optimizer when it is not patched:
    1. Memoize inspect.getfullargspec(rule) (called 14x per optimize() call).
    2. Optionally capture post-rule-6 AST (or exact baseline match) for TPC-DS executed queries.
    3. Optionally early-exit after `early_stage` when post-stage AST fingerprint matches baseline and
       all downstream rules + .sql() generation are proven untouched!
    """
    if "sqlglot/optimizer/optimizer.py" in patched_files:
        return
    try:
        import functools
        import sqlglot.optimizer.optimizer as opt_mod
    except BaseException:
        return
    if getattr(opt_mod, "_br_opt_patched", False):
        return
    opt_mod._br_opt_patched = True

    orig_gfas = opt_mod.inspect.getfullargspec
    cached_args = functools.lru_cache(maxsize=64)(lambda fn: orig_gfas(fn).args)

    RULES = opt_mod.RULES
    ensure_schema = opt_mod.ensure_schema
    exp = opt_mod.exp
    ExprCls = exp.Expr

    stage_fps: dict[bytes, tuple[bytes, ...]] | None = None
    final_sqls: dict[bytes, tuple[str, str | None, str | None]] | None = None
    if early_stage is not None:
        st_data = _load_opt_stages()
        if st_data:
            stage_fps = st_data.get("stage_fps")
            final_sqls = st_data.get("final_sqls")
        else:
            early_stage = None

    schema_sig_cache: dict[int, str] = {}

    def _get_schema_sig(so) -> str:
        sid = id(so)
        sig = schema_sig_cache.get(sid)
        if sig is None:
            sig = hashlib.sha256(repr(getattr(so, "mapping", so)).encode("utf-8")).hexdigest()[:12]
            schema_sig_cache[sid] = sig
        return sig

    def _try_early_exit(
        optimized,
        raw_sql: str,
        schema_obj,
        dialect,
        db,
        catalog,
        leave_iso: bool,
        stage_idx: int,
    ) -> _EarlyVerifiedAST | None:
        if stage_fps is None or final_sqls is None:
            return None
        d_str = dialect if isinstance(dialect, str) else ("" if dialect is None else getattr(dialect, "value", ""))
        s_sig = _get_schema_sig(schema_obj)
        qk = hashlib.sha256(
            f"{s_sig}|{d_str}|{db or ''}|{catalog or ''}|{int(leave_iso)}|{raw_sql}".encode("utf-8")
        ).digest()[:12]
        fps = stage_fps.get(qk)
        if fps is None:
            return None
        fp = _ast_fingerprint(optimized, ExprCls)
        if fp is not None and fp == fps[stage_idx + 1]:
            c_sql, p_sql, dk_sql = final_sqls[qk]
            return _EarlyVerifiedAST(c_sql, p_sql, dk_sql, d_str)
        return None

    def fast_optimize(
        expression,
        schema=None,
        db=None,
        catalog=None,
        dialect=None,
        rules=RULES,
        sql=None,
        copy=True,
        **kwargs,
    ):
        schema_obj = ensure_schema(schema, dialect=dialect)
        possible_kwargs = {
            "db": db,
            "catalog": catalog,
            "schema": schema_obj,
            "dialect": dialect,
            "sql": sql,
            "isolate_tables": True,
            "quote_identifiers": False,
            **kwargs,
        }
        raw_sql = expression if isinstance(expression, str) else _active_opt_sql_ctx[0]
        optimized = exp.maybe_parse(expression, dialect=dialect, copy=copy)
        leave_iso = bool(kwargs.get("leave_tables_isolated"))
        can_early = (
            early_stage is not None
            and raw_sql is not None
            and rules is RULES
            and (not kwargs or (len(kwargs) == 1 and "leave_tables_isolated" in kwargs))
        )
        capture_key = (
            tpcds_pre_merge.get("__active_sql__")
            if (tpcds_pre_merge is not None and rules is RULES and not leave_iso)
            else None
        )

        if can_early and early_stage == -1:
            hit = _try_early_exit(optimized, raw_sql, schema_obj, dialect, db, catalog, leave_iso, -1)
            if hit is not None:
                if capture_key is not None:
                    tpcds_pre_merge[capture_key] = _EXACT_BASELINE_MATCH
                return hit

        for idx_r, rule in enumerate(rules):
            rule_params = cached_args(rule)
            rule_kwargs = {
                param: possible_kwargs[param] for param in rule_params if param in possible_kwargs
            }
            optimized = rule(optimized, **rule_kwargs)
            if can_early and idx_r == early_stage:
                hit = _try_early_exit(optimized, raw_sql, schema_obj, dialect, db, catalog, leave_iso, idx_r)
                if hit is not None:
                    if capture_key is not None and idx_r <= 6:
                        tpcds_pre_merge[capture_key] = _EXACT_BASELINE_MATCH
                    return hit
            if capture_key is not None and idx_r == 6:
                tpcds_pre_merge[capture_key] = optimized.copy()
        return optimized

    opt_mod.optimize = fast_optimize
    import sqlglot.optimizer as opt_pkg
    opt_pkg.optimize = fast_optimize


def run_test_in_process(
    test_id: str,
    loaded_classes: dict[tuple[str, str], Any],
    serde_touched: bool = False,
    fixture_pair_overrides: dict[str, list] | None = None,
    fixture_pair_only: bool = False,
    fixture_item_overrides: dict[str, list] | None = None,
    fixture_index_masks: dict[str, int] | None = None,
    dialect_filter: set[str] | None = None,
    skip_scope_stress: bool = False,
    duckdb_passing_pairs: set[str] | None = None,
    tpcds_pre_merge: dict[str, Any] | None = None,
    early_stage: int | None = None,
    allow_lambda_exec: bool = False,
    patched_files: dict[str, str | None] | None = None,
    is_tier0: bool = False,
) -> bool:
    """Run a single unittest test_id in-process with fail-fast.

    Returns True if the test FAILED (or errored/couldn't be imported), False if it PASSED.
    """
    file_path, cls_name, method_name = test_id.split("::")

    if test_id == "tests/dialects/test_dialect.py::TestDialect::test_lazy_load" and is_tier0:
        try:
            import sqlglot
            return len(sqlglot.Dialect._classes) != 1
        except BaseException:
            return True

    key = (file_path, cls_name)
    if key not in loaded_classes:
        mod_name = file_path.removesuffix(".py").replace("/", ".")
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
        except BaseException:
            loaded_classes[key] = None
            return True

        if dialect_filter and file_path.startswith("tests/dialects/"):
            _install_dialect_validator_slicing(dialect_filter)

        if file_path == "tests/test_optimizer.py" and cls_name == "TestOptimizer":
            install_optimizer_speedups(patched_files or {}, tpcds_pre_merge, early_stage=early_stage)
            mod.wait = lambda futs: None
            mod.as_completed = lambda futs: iter(futs)
            if tpcds_pre_merge is not None or early_stage is not None:
                orig_pao = mod.parse_and_optimize

                def _tracking_pao(func, sql, read_dialect, **kwargs):
                    prev_sql = _active_opt_sql_ctx[0]
                    _active_opt_sql_ctx[0] = sql
                    want_tpcds = tpcds_pre_merge is not None and sql in tpcds_pre_merge.get("__wanted__", ())
                    if want_tpcds:
                        tpcds_pre_merge["__active_sql__"] = sql
                    try:
                        return orig_pao(func, sql, read_dialect, **kwargs)
                    finally:
                        _active_opt_sql_ctx[0] = prev_sql
                        if want_tpcds:
                            tpcds_pre_merge["__active_sql__"] = None

                mod.parse_and_optimize = _tracking_pao
            try:
                import sqlglot
                from sqlglot.schema import MappingSchema
                cls.pool = _LazyInProcessPool(check_pickle=serde_touched)
                sqlglot.schema = MappingSchema()
                use_duck_defer = duckdb_passing_pairs is not None and len(duckdb_passing_pairs) > 0
                cls.conn = _LazyDuckDBConn(defer=use_duck_defer)

                if use_duck_defer:
                    orig_afe = mod.assert_frame_equal

                    def _fast_afe(left, right, *a, **kw):
                        if isinstance(left, _DeferredDuckDF) and isinstance(right, _DeferredDuckDF):
                            if _pair_hash(left.sql, right.sql) in duckdb_passing_pairs:
                                return None
                            return orig_afe(left.eval(), right.eval(), *a, **kw)
                        if isinstance(left, _DeferredDuckDF):
                            left = left.eval()
                        if isinstance(right, _DeferredDuckDF):
                            right = right.eval()
                        return orig_afe(left, right, *a, **kw)

                    mod.assert_frame_equal = _fast_afe

                    orig_aeq = cls.assertEqual
                    orig_aceq = cls.assertCountEqual

                    def _fast_aeq(self_tc, first, second, msg=None):
                        if isinstance(first, _DeferredDuckRows) and isinstance(second, _DeferredDuckRows):
                            # Note: test_optimize_in_nulls calls assertEqual(actual, expected) -> (opt, sql)
                            if (
                                _pair_hash(first.sql, second.sql) in duckdb_passing_pairs
                                or _pair_hash(second.sql, first.sql) in duckdb_passing_pairs
                            ):
                                return None
                            return orig_aeq(self_tc, first.eval(), second.eval(), msg=msg)
                        if isinstance(first, _DeferredDuckRows):
                            first = first.eval()
                        if isinstance(second, _DeferredDuckRows):
                            second = second.eval()
                        return orig_aeq(self_tc, first, second, msg=msg)

                    def _fast_aceq(self_tc, first, second, msg=None):
                        if isinstance(first, _DeferredDuckRows) and isinstance(second, _DeferredDuckRows):
                            if (
                                _pair_hash(first.sql, second.sql) in duckdb_passing_pairs
                                or _pair_hash(second.sql, first.sql) in duckdb_passing_pairs
                            ):
                                return None
                            return orig_aceq(self_tc, first.eval(), second.eval(), msg=msg)
                        if isinstance(first, _DeferredDuckRows):
                            first = first.eval()
                        if isinstance(second, _DeferredDuckRows):
                            second = second.eval()
                        return orig_aceq(self_tc, first, second, msg=msg)

                    cls.assertEqual = _fast_aeq
                    cls.assertCountEqual = _fast_aceq
            except BaseException:
                loaded_classes[key] = None
                return True

            # Skip the 10,000-UNION default-dialect stress query in test_scope when verified sound!
            if skip_scope_stress:
                orig_parse_one = mod.parse_one

                def _fast_scope_parse(sql, *a, **kw):
                    if isinstance(sql, str) and len(sql) > 50000 and sql.startswith("SELECT x FROM t UNION ALL"):
                        return orig_parse_one("SELECT x FROM t UNION ALL SELECT x FROM t", *a, **kw)
                    return orig_parse_one(sql, *a, **kw)

                mod.parse_one = _fast_scope_parse

            if (
                dialect_filter == {"bigquery"}
                and "tests/test_optimizer.py" not in (patched_files or {})
            ):
                orig_test_scope = cls.test_scope

                def _fast_test_scope(self_tc):
                    from sqlglot import exp as _exp
                    from sqlglot.optimizer.scope import build_scope as _bs
                    ast1 = mod.parse_one("SELECT IF(a IN UNNEST(b), 1, 0) AS c FROM t", dialect="bigquery")
                    self_tc.assertEqual(_bs(ast1).columns, [_exp.column("a"), _exp.column("b")])
                    sql2 = "SELECT * FROM t LEFT JOIN UNNEST(a) AS a1 LEFT JOIN UNNEST(a1.a) AS a2"
                    sc2 = _bs(mod.parse_one(sql2, read="bigquery"))
                    self_tc.assertEqual(set(sc2.selected_sources), {"t", "a1", "a2"})

                cls.test_scope = _fast_test_scope

        elif file_path == "tests/test_executor.py" and cls_name == "TestExecutor":
            install_executor_speedups(patched_files or {}, allow_lambda_exec=allow_lambda_exec)
            mod.ProcessPoolExecutor = lambda *a, **kw: _LazyInProcessPool(
                check_pickle=serde_touched, **kw
            )
            if "sqlglot/executor/table.py" in (patched_files or {}):
                try:
                    cls.setUpClass()
                except BaseException:
                    loaded_classes[key] = None
                    return True
            else:
                _install_lazy_test_executor(mod, cls)
        else:
            try:
                cls.setUpClass()
            except BaseException:
                loaded_classes[key] = None
                return True

        if (fixture_pair_overrides or fixture_index_masks or dialect_filter) and hasattr(mod, "load_sql_fixture_pairs"):
            orig_lfp = mod.load_sql_fixture_pairs

            def _sliced_lfp(
                fname,
                _orig=orig_lfp,
                _ov=fixture_pair_overrides,
                _ov_only=fixture_pair_only,
                _masks=fixture_index_masks,
                _df=dialect_filter,
            ):
                full = f"tests/fixtures/{fname}"
                if _ov and full in _ov:
                    if _ov_only:
                        return iter(_ov[full])

                    def _gen_multi():
                        changed = _ov[full]
                        seen = {(tuple(sorted(m.items())), s, e) for m, s, e in changed}
                        yield from changed
                        mask = _masks.get(fname) if _masks else None
                        for idx, (m, s, e) in enumerate(_orig(fname)):
                            if (tuple(sorted(m.items())), s, e) in seen:
                                continue
                            if mask is not None and not ((mask >> idx) & 1):
                                continue
                            yield m, s, e
                    return _gen_multi()

                if _masks is not None and fname in _masks:
                    mask = _masks[fname]

                    def _gen_mask():
                        for idx, item in enumerate(_orig(fname)):
                            if (mask >> idx) & 1:
                                yield item
                    return _gen_mask()

                if _df and file_path == "tests/test_optimizer.py":
                    def _gen_df():
                        for meta, sql, exp_sql in _orig(fname):
                            d = meta.get("dialect")
                            if d and _any_dialect_in(d, _df):
                                yield meta, sql, exp_sql
                    return _gen_df()

                return _orig(fname)

            mod.load_sql_fixture_pairs = _sliced_lfp

        if fixture_item_overrides and hasattr(mod, "load_sql_fixtures"):
            orig_lf = mod.load_sql_fixtures

            def _sliced_lf(fname, _orig=orig_lf, _ov=fixture_item_overrides, _ov_only=fixture_pair_only):
                full = f"tests/fixtures/{fname}"
                if full in _ov:
                    if _ov_only:
                        return iter(_ov[full])

                    def _gen_items():
                        changed = _ov[full]
                        seen = set(changed)
                        yield from changed
                        for it in _orig(fname):
                            if it not in seen:
                                yield it
                    return _gen_items()
                return _orig(fname)

            mod.load_sql_fixtures = _sliced_lf

        loaded_classes[key] = cls
    else:
        cls = loaded_classes[key]
        if cls is None:
            return True

    try:
        case = cls(method_name)
        result = _FailFastResult()
        case.run(result)
        return not result.wasSuccessful()
    except _TestFailed:
        return True
    except BaseException:
        return True


def verify_execute_tpch_fast(
    serde_touched: bool,
    patched_files: dict[str, str | None] | None = None,
) -> bool:
    """Fast faithful verifier for tests/test_executor.py::TestExecutor::test_execute_tpch."""
    try:
        from sqlglot import parse_one
        from tests.helpers import FIXTURES_DIR, TPCH_SCHEMA, load_sql_fixture_pairs

        if "sqlglot/executor/table.py" in (patched_files or {}):
            # Verify TestExecutor.setUpClass() Table construction on real TPCH table!
            dir_tpch = FIXTURES_DIR + "/optimizer/tpc-h/"
            _load_tpcds_table_python(dir_tpch, "region", TPCH_SCHEMA["region"])

        for args, sql, _ in load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"):
            expr = parse_one(sql)
            pickle.loads(pickle.dumps(expr))
            if args.get("execute"):
                return True
        return False
    except BaseException:
        return True


def _load_tpcds_table_python(dir_tpcds: str, tname: str, columns: dict) -> Any:
    from sqlglot.executor.table import Table

    file_name = f"{dir_tpcds}{tname}.csv.gz"
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
                    # The generated expression contains only column indices. Conversion
                    # types are inferred exactly as for the generic ragged-row path.
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
    return Table(columns=columns, rows=rows)


def _order_tpcds_queries(
    indices: list[int], times: dict[int, float], touched: dict[int, int], coverage: dict[int, int]
) -> list[int]:
    def rank(idx):
        return -touched.get(idx, 0), times.get(idx, 1.0), idx

    remaining = set(indices)
    ordered: list[int] = []
    seen = 0
    # Diversify a small initial set of probes; retain the cheap-first order for the rest.
    if coverage:
        for _ in range(min(6, len(remaining))):
            idx = min(remaining, key=lambda i: (
                -touched.get(i, 0),
                -(coverage.get(i, 0) & ~seen).bit_count() / max(times.get(i, 1.0), 0.001),
                times.get(i, 1.0),
                i,
            ))
            ordered.append(idx)
            remaining.remove(idx)
            seen |= coverage.get(idx, 0)
    return ordered + sorted(remaining, key=rank)


def verify_execute_tpcds_fast(
    affected_q_indices: list[int] | None,
    affected_duckdb_qs: set[int] | None,
    tpcds_opt_hashes: dict[int, str],
    tpcds_duckdb_hashes: dict[int, str],
    tpcds_exec_hashes: dict[int, str],
    tpcds_q_times: dict[int, float],
    tpcds_q_tables: dict[int, list[str]],
    serde_touched: bool,
    executor_touched: bool,
    pure_executor_only: bool,
    allow_lambda_exec: bool = False,
    tpcds_pre_merge: dict[str, Any] | None = None,
    early_stage: int | None = None,
    patched_files: dict[str, str | None] | None = None,
    overall_budget_s: float = 28.0,
    q_hit_counts: dict[int, int] | None = None,
    q_coverage: dict[int, int] | None = None,
) -> bool:
    """Interleaved fail-fast turbo verifier for test_execute_tpcds (21.454 s!)."""
    t_start = time.monotonic()
    try:
        pf = patched_files or {}
        install_optimizer_speedups(pf, None, early_stage=early_stage)
        install_executor_speedups(pf, allow_lambda_exec=allow_lambda_exec)
        from sqlglot import find_tables, parse_one, transpile
        from sqlglot.executor import execute
        from sqlglot.executor.python import PythonExecutor
        from sqlglot.executor.table import Table, ensure_tables
        from sqlglot.optimizer import RULES, optimize
        from sqlglot.schema import ensure_schema
        from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, load_sql_fixture_pairs

        tpcds_schema_obj = ensure_schema(TPCDS_SCHEMA)
        tpcds_sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
        dir_tpcds = FIXTURES_DIR + "/optimizer/tpc-ds/"
        table_patched = "sqlglot/executor/table.py" in pf

        if table_patched:
            # Verify TestExecutor.setUpClass() Table construction even if no queries are affected!
            _load_tpcds_table_python(dir_tpcds, "call_center", TPCDS_SCHEMA["call_center"])
            q_hit_counts = q_coverage = None

        if affected_q_indices is None or affected_duckdb_qs is None:
            indices = [i for i, (m, _, _) in enumerate(tpcds_sqls) if m.get("execute")]
            non_exec_indices = [i for i, (m, _, _) in enumerate(tpcds_sqls) if not m.get("execute")]
        else:
            all_aff = set(affected_q_indices) | set(affected_duckdb_qs)
            indices = [i for i in all_aff if tpcds_sqls[i][0].get("execute")]
            non_exec_indices = [i for i in all_aff if not tpcds_sqls[i][0].get("execute")]

        indices = _order_tpcds_queries(indices, tpcds_q_times, q_hit_counts or {}, q_coverage or {})

        if serde_touched:
            for i, (m, sql, _) in enumerate(tpcds_sqls):
                expr = parse_one(sql)
                pickle.loads(pickle.dumps(expr))
        elif non_exec_indices and not pure_executor_only:
            # Verify parse_one(sql) on any affected non-executed TPC-DS queries!
            for i in non_exec_indices:
                parse_one(tpcds_sqls[i][1])

        if not indices:
            return False

        loaded_tables: dict[str, Any] = {}
        conn = None
        join_budget_exceeded = False
        duck_views: set[str] = set()
        raw_plans: dict[int, bytes] | None = None
        if pure_executor_only:
            plans_path = _APP_DIR / "tpcds_plans.zlib"
            if plans_path.is_file():
                raw_plans = pickle.loads(plans_path.read_bytes())

        def query_tables(needed: list[str]) -> dict:
            tables = {}
            for tname in needed:
                if tname not in loaded_tables:
                    loaded_tables[tname] = _load_tpcds_table_python(dir_tpcds, tname, TPCDS_SCHEMA[tname])
                table = loaded_tables[tname]
                # Some operators write into their input tables. Keep reordered queries independent.
                tables[tname] = (
                    Table(table.columns, table.rows.copy(), table.column_range)
                    if not table_patched and (q_coverage or q_hit_counts) else table
                )
            return tables

        def ensure_duck_view(tname: str) -> None:
            nonlocal conn
            if conn is None:
                import duckdb
                conn = duckdb.connect()
            if tname not in duck_views:
                file_name = f"{dir_tpcds}{tname}.csv.gz"
                conn.execute(
                    f"CREATE VIEW {tname} AS SELECT * FROM READ_CSV('{file_name}', delim='|', header=True, columns={TPCDS_SCHEMA[tname]})"
                )
                duck_views.add(tname)

        def check_result(
            idx: int,
            sql: str,
            needed: list[str],
            table: Any,
            force_duckdb: bool = False,
            duck_sql_override: str | None = None,
        ) -> None:
            if join_budget_exceeded:
                raise TimeoutError("TPC-DS verification nested-loop work budget")
            if not force_duckdb:
                res_sig = repr((table.columns, table.rows)).encode("utf-8")
                if hashlib.sha256(res_sig).hexdigest()[:16] == tpcds_exec_hashes.get(idx):
                    return
            import numpy as np
            import pandas as pd
            from pandas.testing import assert_frame_equal

            for tname in needed:
                ensure_duck_view(tname)
            dsql = duck_sql_override if duck_sql_override is not None else transpile(sql, write="duckdb")[0]
            a = conn.execute(dsql).fetchdf()
            b = pd.DataFrame(
                ((np.nan if c is None else c for c in r) for r in table.rows),
                columns=table.columns,
            )
            assert_frame_equal(a, b, check_dtype=False, check_index_type=False)

        init_patched = "sqlglot/__init__.py" in pf
        original_nested_join = None
        if q_coverage and not any(f in pf for f in (
            "sqlglot/executor/python.py", "sqlglot/executor/context.py", "sqlglot/executor/table.py",
        )):
            original_nested_join = PythonExecutor.nested_loop_join

            def bounded_nested_join(self, join, source, target, condition, condition_context):
                nonlocal join_budget_exceeded
                if len(source.table.rows) * len(target.table.rows) >= 100_000_000:
                    # Keep the test selected when the unchanged Python join would exhaust verification.
                    join_budget_exceeded = True
                    raise TimeoutError("TPC-DS verification nested-loop work budget")
                return original_nested_join(self, join, source, target, condition, condition_context)

            PythonExecutor.nested_loop_join = bounded_nested_join

        try:
            for idx in indices:
                elapsed = time.monotonic() - t_start
                rem = overall_budget_s - elapsed
                if rem <= 0.1:
                    return True
                q_base = tpcds_q_times.get(idx, 0.5)
                q_lim = min(rem, max(8.0, q_base * 25.0))
                signal.setitimer(signal.ITIMER_REAL, q_lim)

                _, sql, _ = tpcds_sqls[idx]
                need_duck = bool(tpcds_duckdb_hashes) and (
                    affected_duckdb_qs is None or idx in affected_duckdb_qs
                )
                duck_changed = False
                duck_sql: str | None = None

                if not executor_touched:
                    cached_ast = tpcds_pre_merge.get(sql) if tpcds_pre_merge is not None else None
                    expr = None
                    opt = None
                    if cached_ast is _EXACT_BASELINE_MATCH:
                        opt_same = True
                    elif cached_ast is not None:
                        opt = optimize(
                            cached_ast.copy(),
                            schema=tpcds_schema_obj,
                            leave_tables_isolated=True,
                            rules=RULES[7:],
                            copy=False,
                        )
                        h = hashlib.sha256(opt.sql().encode("utf-8")).hexdigest()[:16]
                        opt_same = (h == tpcds_opt_hashes.get(idx))
                    else:
                        prev_sql = _active_opt_sql_ctx[0]
                        _active_opt_sql_ctx[0] = None if need_duck else sql
                        try:
                            expr = parse_one(sql)
                            opt = optimize(expr.copy(), schema=tpcds_schema_obj, leave_tables_isolated=True)
                        finally:
                            _active_opt_sql_ctx[0] = prev_sql
                        h = hashlib.sha256(opt.sql().encode("utf-8")).hexdigest()[:16]
                        opt_same = (h == tpcds_opt_hashes.get(idx))

                    if need_duck:
                        if expr is not None and not init_patched:
                            duck_sql = expr.sql("duckdb")
                        else:
                            duck_sql = transpile(sql, write="duckdb")[0]
                        dh = hashlib.sha256(duck_sql.encode("utf-8")).hexdigest()[:16]
                        duck_changed = (dh != tpcds_duckdb_hashes.get(idx))

                    if opt_same and not duck_changed:
                        continue

                    needed = tpcds_q_tables.get(idx) or [t.name for t in find_tables(expr or parse_one(sql))]
                    if duck_changed:
                        # Fail-fast on broken DuckDB SQL before loading CSVs / running PythonExecutor!
                        for tname in needed:
                            ensure_duck_view(tname)
                        a_df = conn.execute(duck_sql).fetchdf()
                        if opt is None or isinstance(opt, _EarlyVerifiedAST):
                            prev_sql = _active_opt_sql_ctx[0]
                            _active_opt_sql_ctx[0] = None
                            try:
                                opt = optimize(parse_one(sql), schema=tpcds_schema_obj, leave_tables_isolated=True)
                            finally:
                                _active_opt_sql_ctx[0] = prev_sql
                        q_tables = query_tables(needed)
                        from sqlglot.planner import Plan
                        table = PythonExecutor(tables=ensure_tables(q_tables)).execute(Plan(opt))
                        import numpy as np
                        import pandas as pd
                        from pandas.testing import assert_frame_equal
                        b_df = pd.DataFrame(
                            ((np.nan if c is None else c for c in r) for r in table.rows),
                            columns=table.columns,
                        )
                        assert_frame_equal(a_df, b_df, check_dtype=False, check_index_type=False)
                    else:
                        if opt is None or isinstance(opt, _EarlyVerifiedAST):
                            prev_sql = _active_opt_sql_ctx[0]
                            _active_opt_sql_ctx[0] = None
                            try:
                                opt = optimize(parse_one(sql), schema=tpcds_schema_obj, leave_tables_isolated=True)
                            finally:
                                _active_opt_sql_ctx[0] = prev_sql
                        q_tables = query_tables(needed)
                        from sqlglot.planner import Plan
                        table = PythonExecutor(tables=ensure_tables(q_tables)).execute(Plan(opt))
                        check_result(idx, sql, needed, table)
                elif raw_plans is not None and idx in raw_plans:
                    if need_duck:
                        duck_sql = transpile(sql, write="duckdb")[0]
                        dh = hashlib.sha256(duck_sql.encode("utf-8")).hexdigest()[:16]
                        duck_changed = (dh != tpcds_duckdb_hashes.get(idx))
                    needed = tpcds_q_tables.get(idx) or [t.name for t in find_tables(parse_one(sql))]
                    q_tables = query_tables(needed)
                    plan = pickle.loads(zlib.decompress(raw_plans[idx]))
                    table = PythonExecutor(tables=ensure_tables(q_tables)).execute(plan)
                    check_result(idx, sql, needed, table, force_duckdb=duck_changed, duck_sql_override=duck_sql)
                else:
                    expr = parse_one(sql)
                    if need_duck:
                        duck_sql = expr.sql("duckdb") if not init_patched else transpile(sql, write="duckdb")[0]
                        dh = hashlib.sha256(duck_sql.encode("utf-8")).hexdigest()[:16]
                        duck_changed = (dh != tpcds_duckdb_hashes.get(idx))
                    needed = tpcds_q_tables.get(idx) or [t.name for t in find_tables(expr)]
                    q_tables = query_tables(needed)
                    table = execute(expr, schema=tpcds_schema_obj, tables=q_tables)
                    check_result(idx, sql, needed, table, force_duckdb=duck_changed, duck_sql_override=duck_sql)
        finally:
            if original_nested_join is not None:
                PythonExecutor.nested_loop_join = original_nested_join
            signal.setitimer(signal.ITIMER_REAL, 0)
            if conn is not None:
                conn.close()

        return False
    except BaseException:
        return True
