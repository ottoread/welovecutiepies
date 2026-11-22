#!/usr/bin/env python3
"""Blast Radius selector: sound static+dynamic dependency maps + 2-tier fork-server fail-fast verifier."""
from __future__ import annotations

import json
import os
import pickle
import re
import signal
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any

from inmem_loader import (
    EXECUTOR_OPTIONS,
    install_overlay,
    run_test_in_process,
    verify_execute_tpcds_fast,
    verify_execute_tpch_fast,
)
from patch_utils import (
    analyze_python_edit,
    apply_file_patch,
    diff_sql_fixture_pairs,
    diff_sql_fixtures,
    has_only_shadowed_metadata_inserts,
    is_normalization_enum_comparison_edit,
    is_numeric_escape_value_edit,
    parse_patch,
)
from worker_pack import WORKER_COST_MS, duration_ms, pack_cost_ms, pack_workers

DATA = Path(os.environ.get("DATA_DIR", "/data"))
SQLGLOT_ROOT = Path(os.environ.get("SQLGLOT_ROOT", "/app/sqlglot")).resolve()
APP_DIR = Path(__file__).resolve().parent
MAPS_PATH = APP_DIR / "maps.zlib"
FIXTURE_COV_PATH = APP_DIR / "fixture_cov.zlib"
OPT_STAGES_PATH = APP_DIR / "opt_stages.zlib"

DEFAULT_CORE_FILES = {
    "sqlglot/__init__.py",
    "sqlglot/_typing.py",
    "sqlglot/dialects/__init__.py",
    "sqlglot/dialects/dialect.py",
    "sqlglot/diff.py",
    "sqlglot/errors.py",
    "sqlglot/expressions/__init__.py",
    "sqlglot/expressions/aggregate.py",
    "sqlglot/expressions/array.py",
    "sqlglot/expressions/builders.py",
    "sqlglot/expressions/constraints.py",
    "sqlglot/expressions/core.py",
    "sqlglot/expressions/datatypes.py",
    "sqlglot/expressions/ddl.py",
    "sqlglot/expressions/dml.py",
    "sqlglot/expressions/functions.py",
    "sqlglot/expressions/json.py",
    "sqlglot/expressions/math.py",
    "sqlglot/expressions/properties.py",
    "sqlglot/expressions/query.py",
    "sqlglot/expressions/string.py",
    "sqlglot/expressions/temporal.py",
    "sqlglot/generator.py",
    "sqlglot/helper.py",
    "sqlglot/jsonpath.py",
    "sqlglot/optimizer/__init__.py",
    "sqlglot/optimizer/scope.py",
    "sqlglot/parser.py",
    "sqlglot/parsers/__init__.py",
    "sqlglot/parsers/base.py",
    "sqlglot/schema.py",
    "sqlglot/time.py",
    "sqlglot/tokenizer_core.py",
    "sqlglot/tokens.py",
    "sqlglot/trie.py",
    "sqlglot/typing/__init__.py",
    "tests/__init__.py",
    "tests/helpers.py",
}

TEST_EXECUTE_TPCDS = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
TEST_EXECUTE_TPCH = "tests/test_executor.py::TestExecutor::test_execute_tpch"
TEST_MAX_NODES = "tests/test_parser.py::TestParser::test_max_nodes"
TEST_GEN_MEM = "tests/test_generator.py::TestGenerator::test_generator_memory"
TEST_TYPING = "tests/test_typing.py::TestTyping::test_no_redundant_expression_metadata"
TEST_LAZY_LOAD = "tests/dialects/test_dialect.py::TestDialect::test_lazy_load"

SINGLE_LINE_SQL_FIXTURES = {
    "tests/fixtures/identity.sql",
    "tests/fixtures/partial.sql",
    "tests/fixtures/optimizer/qualify_columns__invalid.sql",
}


def file_to_module(rp: str) -> str:
    if rp.endswith("/__init__.py"):
        return rp[: -len("/__init__.py")].replace("/", ".")
    return rp.removesuffix(".py").replace("/", ".")


def _send_msg(fd: int, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    hdr = struct.pack("!I", len(data))
    mv = memoryview(hdr + data)
    off = 0
    while off < len(mv):
        n = os.write(fd, mv[off:])
        off += n


def _recv_msg(fd: int) -> Any:
    hdr = _read_exact(fd, 4)
    if not hdr:
        return None
    (length,) = struct.unpack("!I", hdr)
    body = _read_exact(fd, length)
    return pickle.loads(body)


def _read_exact(fd: int, n: int) -> bytes:
    chunks = []
    rem = n
    while rem > 0:
        b = os.read(fd, rem)
        if not b:
            return b""
        chunks.append(b)
        rem -= len(b)
    return b"".join(chunks)


def _silence_stdout_and_stderr_in_child() -> None:
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


class _TestTimeout(BaseException):
    """Raised by SIGALRM if a patched function enters an infinite loop inside a test."""


def _alarm_handler(signum, frame):
    raise _TestTimeout()


def _execute_job_in_child(
    job: dict, is_tier0: bool
) -> tuple[str, list[str], float] | tuple[str, list[int], float, list[tuple[str, float]]]:
    """Run inside a freshly forked leaf process with stdout/stderr silenced."""
    _silence_stdout_and_stderr_in_child()
    signal.signal(signal.SIGALRM, _alarm_handler)
    os.environ["SKIP_INTEGRATION"] = "0"
    os.chdir(SQLGLOT_ROOT)
    if str(SQLGLOT_ROOT) not in sys.path:
        sys.path.insert(0, str(SQLGLOT_ROOT))

    patched_files: dict[str, str | None] = job["patched_files"]
    install_overlay(SQLGLOT_ROOT, patched_files)
    EXECUTOR_OPTIONS.update(
        direct_iteration=job.get("allow_direct_iteration", False),
        context_table_only=job.get("context_table_only", False),
    )

    test_items: list[tuple[int, str]] = list(job["tests"])
    failing: list[int] = []

    # Run TEST_LAZY_LOAD FIRST in Tier 0 BEFORE check_import_rels or any other test imports any dialect!
    if is_tier0:
        lazy_items = [x for x in test_items if x[1] == TEST_LAZY_LOAD]
        if lazy_items:
            test_items = [x for x in test_items if x[1] != TEST_LAZY_LOAD]
            signal.setitimer(signal.ITIMER_REAL, 5.0)
            try:
                import sqlglot
                lazy_failed = len(sqlglot.Dialect._classes) != 1
            except BaseException:
                lazy_failed = True
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            if lazy_failed:
                for lnum, _ in lazy_items:
                    failing.append(lnum)

    # Check if any class/module-level modified file crashes (or infinite-loops!) at import time!
    check_import_rels: list[str] = job.get("check_import_rels", [])
    if check_import_rels:
        import importlib
        broken_rels: list[str] = []
        for rel in check_import_rels:
            mod_name = file_to_module(rel)
            signal.setitimer(signal.ITIMER_REAL, 5.0)
            try:
                importlib.import_module(mod_name)
            except BaseException:
                broken_rels.append(rel)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        if broken_rels:
            return ("import_err", broken_rels, time.process_time())

    serde_touched: bool = job["serde_touched"]
    fixture_pair_ov: dict[str, list] | None = job.get("fixture_pair_overrides")
    fixture_pair_only: bool = job.get("fixture_pair_only", False)
    fixture_item_ov: dict[str, list] | None = job.get("fixture_item_overrides")
    fixture_index_masks: dict[str, int] | None = job.get("fixture_index_masks")
    dialect_filter: set[str] | None = job.get("dialect_filter")
    skip_scope_stress: bool = job.get("skip_scope_stress", False)
    duckdb_passing_pairs: set[str] | None = job.get("duckdb_passing_pairs")
    tpcds_fast: dict | None = job.get("tpcds_fast")
    tpch_fast: bool = job.get("tpch_fast", False)
    gen_mem_dialects: list[str] | None = job.get("gen_mem_dialects")
    early_stage: int | None = job.get("early_stage")
    allow_lambda_exec: bool = job.get("allow_lambda_exec", False)
    dur_ms_map: dict[int, int] = job.get("dur_ms_map", {})
    verification_cpu_ms: dict[int, float] = job.get("verification_cpu_ms", {})
    auto_include: set[int] = job.get("auto_include", set())
    per_commit_cpu_cap: float = job.get("cpu_cap", 25.0)

    loaded_classes: dict[tuple[str, str], Any] = {}
    t_cpu0 = time.process_time()

    exec_items = [x for x in test_items if x[1] == TEST_EXECUTE_TPCDS]
    opt_items = [
        x for x in test_items if x[1] == "tests/test_optimizer.py::TestOptimizer::test_tpcds"
    ]
    tpcds_pre_merge: dict[str, Any] | None = None
    share_tpcds = (
        bool(opt_items and exec_items)
        and tpcds_fast is not None
        and not tpcds_fast["executor_touched"]
        and "sqlglot/optimizer/optimizer.py" not in patched_files
    )
    if share_tpcds:
        try:
            from tests.helpers import load_sql_fixture_pairs
            wanted = {
                sql
                for m, sql, _ in load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql")
                if m.get("execute")
            }
            tpcds_pre_merge = {"__wanted__": wanted}
        except BaseException:
            tpcds_pre_merge = None

    if exec_items:
        # Resolve the expensive test early when it has a fast verifier, preserving its cache dependency.
        if tpcds_fast is not None:
            front = (opt_items if tpcds_pre_merge is not None else []) + exec_items
            front_nums = {n for n, _ in front}
            test_items = front + [x for x in test_items if x[0] not in front_nums]
        else:
            test_items = [x for x in test_items if x[1] != TEST_EXECUTE_TPCDS] + exec_items

    remaining = {n for n, _ in test_items}
    remaining_cpu_ms = sum(verification_cpu_ms.get(n, 0.0) for n in remaining)
    have_anchor = any(dur_ms_map.get(n, 0) >= WORKER_COST_MS for n in auto_include | set(failing))
    next_pack_check = 0
    timings: list[tuple[str, float]] = []
    for idx, (tnum, tid) in enumerate(test_items):
        test_cpu_start = time.process_time()
        if idx > 0 and (test_cpu_start - t_cpu0) > per_commit_cpu_cap:
            failing.extend(remaining)
            break

        if have_anchor and verification_cpu_ms and idx >= next_pack_check:
            required = auto_include.union(failing)
            savings_ms = pack_cost_ms(required | remaining, dur_ms_map) - pack_cost_ms(required, dur_ms_map)
            if savings_ms < remaining_cpu_ms:
                # Keep every unchecked candidate, including any undiscovered failures.
                failing.extend(remaining)
                break
            next_pack_check = idx + 16

        remaining.remove(tnum)
        remaining_cpu_ms -= verification_cpu_ms.get(tnum, 0.0)
        prior_failures = len(failing)

        # Per-test infinite-loop protection via SIGALRM (timeout = test failure, matching pytest-timeout!)
        rec_s = dur_ms_map.get(tnum, 500) / 1000.0
        if tid == TEST_EXECUTE_TPCDS:
            t_limit = 30.0
        elif tid == TEST_GEN_MEM:
            t_limit = 15.0
        else:
            t_limit = max(3.5, min(15.0, rec_s * 4.0 + 2.5))

        signal.setitimer(signal.ITIMER_REAL, t_limit)
        try:
            if tid == TEST_EXECUTE_TPCDS and tpcds_fast is not None:
                failed = verify_execute_tpcds_fast(
                    tpcds_fast["affected_q_indices"],
                    tpcds_fast["affected_duckdb_qs"],
                    tpcds_fast["opt_hashes"],
                    tpcds_fast["duckdb_hashes"],
                    tpcds_fast["exec_hashes"],
                    tpcds_fast["q_times"],
                    tpcds_fast["q_tables"],
                    serde_touched,
                    tpcds_fast["executor_touched"],
                    tpcds_fast["pure_executor_only"],
                    allow_lambda_exec=allow_lambda_exec,
                    tpcds_pre_merge=tpcds_pre_merge,
                    early_stage=early_stage,
                    patched_files=patched_files,
                    q_hit_counts=tpcds_fast["q_hit_counts"],
                    q_coverage=tpcds_fast["q_coverage"],
                )
                if failed:
                    failing.append(tnum)
                continue

            if tid == TEST_EXECUTE_TPCH and tpch_fast:
                if verify_execute_tpch_fast(serde_touched, patched_files=patched_files):
                    failing.append(tnum)
                continue

            if tid == TEST_GEN_MEM and gen_mem_dialects is not None:
                if _verify_generator_memory_subset(gen_mem_dialects):
                    failing.append(tnum)
                continue

            if run_test_in_process(
                tid,
                loaded_classes,
                serde_touched=serde_touched,
                fixture_pair_overrides=fixture_pair_ov,
                fixture_pair_only=fixture_pair_only,
                fixture_item_overrides=fixture_item_ov,
                fixture_index_masks=fixture_index_masks,
                dialect_filter=dialect_filter,
                skip_scope_stress=skip_scope_stress,
                duckdb_passing_pairs=duckdb_passing_pairs,
                tpcds_pre_merge=tpcds_pre_merge,
                early_stage=early_stage,
                allow_lambda_exec=allow_lambda_exec,
                patched_files=patched_files,
                is_tier0=is_tier0,
            ):
                failing.append(tnum)
        except _TestTimeout:
            failing.append(tnum)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            timings.append((tid, round(time.process_time() - test_cpu_start, 3)))
            if len(failing) > prior_failures and dur_ms_map.get(tnum, 0) >= WORKER_COST_MS:
                have_anchor = True
                next_pack_check = idx + 1

    return ("ok", failing, time.process_time(), timings)


def _verify_generator_memory_subset(dialect_names: list[str]) -> bool:
    """Fast dialect-specific version of test_generator_memory."""
    try:
        import gc
        import tracemalloc
        from sqlglot.dialects.dialect import Dialect

        for name in dialect_names:
            dialect = Dialect.get_or_raise(name)
            for _ in range(100):
                dialect.generator()
            gc.collect()
            tracemalloc.start()
            try:
                before = tracemalloc.get_traced_memory()[0]
                for _ in range(1000):
                    dialect.generator()
                gc.collect()
                after = tracemalloc.get_traced_memory()[0]
            finally:
                tracemalloc.stop()
            if after - before >= 1024:
                return True
        return False
    except BaseException:
        return True


class ForkRunner:
    """Manages Tier 0 (clean pre-sqlglot) and Tier 1 (warm sqlglot-core + numpy/pandas/duckdb)."""

    def __init__(self) -> None:
        self.t1_pid: int | None = None
        self.t1_req_w: int | None = None
        self.t1_rep_r: int | None = None
        self.t1_loaded_files: set[str] = set(DEFAULT_CORE_FILES)
        self._heavy_preloaded = False
        self.child_cpu_total = 0.0
        self.last_timings = None

    def _ensure_heavy_third_party(self) -> None:
        """Pre-import numpy, pandas, duckdb ONCE in parent (verified OS num_threads == 1)."""
        if self._heavy_preloaded:
            return
        self._heavy_preloaded = True
        saved_stdout = sys.stdout
        try:
            sys.stdout = sys.stderr
            import duckdb  # noqa: F401
            import numpy  # noqa: F401
            import pandas  # noqa: F401
            from pandas.testing import assert_frame_equal  # noqa: F401
        except BaseException:
            pass
        finally:
            sys.stdout = saved_stdout

    def _ensure_tier1(self) -> None:
        if self.t1_pid is not None:
            return
        self._ensure_heavy_third_party()
        req_r, req_w = os.pipe()
        rep_r, rep_w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(req_w)
            os.close(rep_r)
            try:
                _silence_stdout_and_stderr_in_child()
                os.environ["SKIP_INTEGRATION"] = "0"
                os.chdir(SQLGLOT_ROOT)
                if str(SQLGLOT_ROOT) not in sys.path:
                    sys.path.insert(0, str(SQLGLOT_ROOT))
                import sqlglot  # noqa: F401
                import tests.helpers  # noqa: F401

                repo_prefix = str(SQLGLOT_ROOT) + os.sep
                loaded: set[str] = set()
                for m in list(sys.modules.values()):
                    f = getattr(m, "__file__", None)
                    if f and f.startswith(repo_prefix):
                        loaded.add(f[len(repo_prefix):])
                _send_msg(rep_w, loaded)

                while True:
                    job = _recv_msg(req_r)
                    if job is None:
                        break
                    c_r, c_w = os.pipe()
                    cpid = os.fork()
                    if cpid == 0:
                        os.close(c_r)
                        os.close(req_r)
                        os.close(rep_w)
                        try:
                            res = _execute_job_in_child(job, is_tier0=False)
                            _send_msg(c_w, res)
                        except BaseException as exc:
                            _send_msg(c_w, ("err", repr(exc)))
                        finally:
                            os.close(c_w)
                            os._exit(0)
                    else:
                        os.close(c_w)
                        msg = _recv_msg(c_r)
                        os.close(c_r)
                        os.waitpid(cpid, 0)
                        _send_msg(rep_w, msg)
            except BaseException:
                pass
            finally:
                os.close(req_r)
                os.close(rep_w)
                os._exit(0)
        else:
            os.close(req_r)
            os.close(rep_w)
            self.t1_pid = pid
            self.t1_req_w = req_w
            self.t1_rep_r = rep_r
            loaded = _recv_msg(rep_r)
            if isinstance(loaded, set):
                self.t1_loaded_files = loaded

    def run_job(self, job: dict, need_tier0: bool) -> tuple[str, Any]:
        if not need_tier0:
            self._ensure_tier1()
            if not any(p in self.t1_loaded_files for p in job["patched_files"]):
                assert self.t1_req_w is not None and self.t1_rep_r is not None
                _send_msg(self.t1_req_w, job)
                msg = _recv_msg(self.t1_rep_r)
                if msg and len(msg) > 2:
                    self.child_cpu_total += msg[2]
                    self.last_timings = msg[3] if len(msg) > 3 else None
                if msg and msg[0] in ("ok", "import_err"):
                    return msg[0], msg[1]
                return ("ok", [n for n, _ in job["tests"]])

        self._ensure_heavy_third_party()
        c_r, c_w = os.pipe()
        cpid = os.fork()
        if cpid == 0:
            os.close(c_r)
            if self.t1_req_w is not None:
                os.close(self.t1_req_w)
            if self.t1_rep_r is not None:
                os.close(self.t1_rep_r)
            try:
                res = _execute_job_in_child(job, is_tier0=True)
                _send_msg(c_w, res)
            except BaseException as exc:
                _send_msg(c_w, ("err", repr(exc)))
            finally:
                os.close(c_w)
                os._exit(0)
        else:
            os.close(c_w)
            msg = _recv_msg(c_r)
            os.close(c_r)
            os.waitpid(cpid, 0)
            if msg and len(msg) > 2:
                self.child_cpu_total += msg[2]
                self.last_timings = msg[3] if len(msg) > 3 else None
            if msg and msg[0] in ("ok", "import_err"):
                return msg[0], msg[1]
            return ("ok", [n for n, _ in job["tests"]])

    def close(self) -> None:
        if self.t1_pid is not None:
            try:
                assert self.t1_req_w is not None and self.t1_rep_r is not None
                os.close(self.t1_req_w)
                os.close(self.t1_rep_r)
                os.waitpid(self.t1_pid, 0)
            except BaseException:
                pass
            self.t1_pid = None


class BlastRadiusSelector:
    def __init__(self, tests_list: list[dict], maps: dict) -> None:
        self.tests = tests_list
        self.by_num = {t["number"]: t for t in tests_list}
        self.by_id = {t["id"]: t["number"] for t in tests_list}
        self.dur_ms = {t["number"]: duration_ms(t["seconds"]) for t in tests_list}
        self.tests_by_file: dict[str, list[int]] = {}
        self.tests_by_class: dict[tuple[str, str], list[int]] = {}
        for t in tests_list:
            f, cls, _ = t["id"].split("::")
            self.tests_by_file.setdefault(f, []).append(t["number"])
            self.tests_by_class.setdefault((f, cls), []).append(t["number"])

        self.line_to_tests: dict[str, dict[int, list[int]]] = maps["line_to_tests"]
        self.file_exec_to_tests: dict[str, list[int]] = maps["file_exec_to_tests"]
        self.open_file_to_tests: dict[str, list[int]] = maps["open_file_to_tests"]
        self.dialect_to_tests: dict[str, list[int]] = maps["dialect_to_tests"]
        self.expr_to_tests: dict[str, list[int]] = maps["expr_to_tests"]
        self.rev_import_test_files: dict[str, list[str]] = maps["rev_import_test_files"]
        self.rev_import_dialect_mods: dict[str, list[str]] = maps.get("rev_import_dialect_mods", {})
        self.lazy_import_tests: dict[str, list[int]] = maps.get("lazy_import_tests", {})
        self.symbol_refs: dict[str, list[tuple[str, int]]] = maps["symbol_refs"]
        self.attr_read_tests: dict[str, set[int]] = {
            k: set(v) for k, v in maps.get("attr_read_tests", {}).items()
        }
        self.metaclass_read_attrs: set[str] = set(maps.get("metaclass_read_attrs", ()))
        self.dialect_descendants: dict[str, set[str]] = maps.get("dialect_descendants", {})
        self.tpcds_opt_hashes: dict[int, str] = maps["tpcds_opt_hashes"]
        self.tpcds_duckdb_hashes: dict[int, str] = maps.get("tpcds_duckdb_hashes", {})
        self.tpcds_exec_hashes: dict[int, str] = maps.get("tpcds_exec_hashes", {})
        self.tpcds_q_times: dict[int, float] = maps.get("tpcds_q_times", {})
        self.tpcds_q_tables: dict[int, list[str]] = maps.get("tpcds_q_tables", {})
        self.tpcds_line_to_q: dict[str, dict[int, list[int]]] = maps["tpcds_line_to_q"]
        self.tpcds_duckdb_line_to_q: dict[str, dict[int, list[int]]] = maps.get("tpcds_duckdb_line_to_q", {})
        self.tpcds_exec_line_to_q: dict[str, dict[int, list[int]]] = maps.get("tpcds_exec_line_to_q", {})
        self._tpcds_optimizer_cover: dict[int, int] | None = None
        self.scope_stress_lines: dict[str, set[int]] = maps.get("scope_stress_lines", {})
        self.duckdb_passing_pairs: set[str] = maps.get("duckdb_passing_pairs", set())

        self._fixture_cov_loaded = False
        self._src_to_fixture_masks: dict[str, dict[str, dict[int, int]]] = {}
        self._fixture_pair_counts: dict[str, int] = {}
        self._opt_stages_loaded = False
        self._rule_line_masks: dict[str, dict[int, int]] = {}

        self.mod_to_dialect_tests: dict[str, set[int]] = {}
        for dkey, tnums in self.dialect_to_tests.items():
            mod = dkey.split(":", 1)[0]
            self.mod_to_dialect_tests.setdefault(mod, set()).update(tnums)

        self.all_test_nums = [t["number"] for t in tests_list]
        self.tpcds_num = self.by_id.get(TEST_EXECUTE_TPCDS)
        self.tpch_num = self.by_id.get(TEST_EXECUTE_TPCH)
        self.max_nodes_num = self.by_id.get(TEST_MAX_NODES)
        self.gen_mem_num = self.by_id.get(TEST_GEN_MEM)
        self.typing_num = self.by_id.get(TEST_TYPING)
        self.lazy_load_num = self.by_id.get(TEST_LAZY_LOAD)

        self.runner = ForkRunner()
        self._orig_cache: dict[str, str | None] = {}
        self._external_read_cache: dict[tuple[str, str], bool] = {}
        self._batch_t0 = time.monotonic()
        self._batch_wall_budget = float(os.environ.get("BATCH_WALL_BUDGET", "145.0"))

    def _ensure_fixture_cov(self) -> None:
        if self._fixture_cov_loaded:
            return
        self._fixture_cov_loaded = True
        if FIXTURE_COV_PATH.is_file():
            fc = pickle.loads(zlib.decompress(FIXTURE_COV_PATH.read_bytes()))
            self._src_to_fixture_masks = fc["src_to_fixture_masks"]
            self._fixture_pair_counts = fc["fixture_pair_counts"]

    def _ensure_opt_stages(self) -> None:
        if self._opt_stages_loaded:
            return
        self._opt_stages_loaded = True
        if OPT_STAGES_PATH.is_file():
            st = pickle.loads(zlib.decompress(OPT_STAGES_PATH.read_bytes()))
            self._rule_line_masks = st.get("rule_line_masks", {})

    def _tpcds_coverage(self) -> dict[int, int]:
        if self._tpcds_optimizer_cover is None:
            cover: dict[int, int] = {}
            bit = 1
            for path, lines in self.tpcds_exec_line_to_q.items():
                if path.startswith("sqlglot/optimizer/"):
                    for queries in lines.values():
                        for idx in queries:
                            cover[idx] = cover.get(idx, 0) | bit
                        bit <<= 1
            self._tpcds_optimizer_cover = cover
        return self._tpcds_optimizer_cover

    def _read_orig(self, rel_path: str) -> str | None:
        if rel_path not in self._orig_cache:
            p = SQLGLOT_ROOT / rel_path
            if p.is_file():
                try:
                    self._orig_cache[rel_path] = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    self._orig_cache[rel_path] = None
            else:
                self._orig_cache[rel_path] = None
        return self._orig_cache[rel_path]

    def _has_external_reader(self, rel_path: str, symbol: str) -> bool:
        key = (rel_path, symbol)
        if key not in self._external_read_cache:
            module = file_to_module(rel_path)
            token = re.compile(rf"\b{re.escape(symbol)}\b")
            self._external_read_cache[key] = bool(self.symbol_refs.get(f"{module}:*")) or any(
                token.search(source.read_text(encoding="utf-8", errors="replace"))
                for root in ("sqlglot", "tests")
                for source in (SQLGLOT_ROOT / root).rglob("*.py")
                if source != SQLGLOT_ROOT / rel_path
            )
        return self._external_read_cache[key]

    def _tests_for_rev_imports(self, rel_path: str) -> set[int]:
        out: set[int] = set()
        for tf in self.rev_import_test_files.get(rel_path, ()):
            out.update(self.tests_by_file.get(tf, ()))
        if rel_path in self.tests_by_file:
            out.update(self.tests_by_file[rel_path])
        return out

    def _import_blast_radius(self, rel: str) -> set[int]:
        """Full sound blast radius when importing `rel` fails (SyntaxError, ImportError, NameError, etc.)."""
        out: set[int] = set()
        out.update(self.file_exec_to_tests.get(rel, ()))
        out.update(self.open_file_to_tests.get(rel, ()))
        out.update(self._tests_for_rev_imports(rel))
        out.update(self.lazy_import_tests.get(rel, ()))
        mod = file_to_module(rel)
        out.update(self.mod_to_dialect_tests.get(mod, ()))
        for dmod in self.rev_import_dialect_mods.get(rel, ()):
            out.update(self.mod_to_dialect_tests.get(dmod, ()))
        if rel.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
            stem = Path(rel).stem
            if stem not in ("__init__", "dialect", "base", "python"):
                for d in {stem} | set(self.dialect_descendants.get(stem, ())):
                    out.update(self.mod_to_dialect_tests.get(f"sqlglot.dialects.{d}", ()))
                    out.update(self.mod_to_dialect_tests.get(f"sqlglot.parsers.{d}", ()))
                    out.update(self.mod_to_dialect_tests.get(f"sqlglot.generators.{d}", ()))
                # Any dialect import failure breaks Dialect.classes / test_dialect / test_dialect_imports / test_generator_memory!
                out.update(self.tests_by_file.get("tests/dialects/test_dialect.py", ()))
                out.update(self.tests_by_file.get("tests/test_dialect_imports.py", ()))
                if self.gen_mem_num is not None:
                    out.add(self.gen_mem_num)
                if stem == "duckdb" or "duckdb" in self.dialect_descendants.get(stem, ()):
                    out.update(self.tests_by_file.get("tests/test_optimizer.py", ()))
                    out.update(self.tests_by_file.get("tests/test_executor.py", ()))
        if rel.startswith("sqlglot/typing/") and self.typing_num is not None:
            out.add(self.typing_num)
        if rel in DEFAULT_CORE_FILES or rel.startswith("sqlglot/expressions/"):
            out.update(self.all_test_nums)
        return out

    def select_for_commit(self, patch_text: str) -> list[list[int]]:
        file_patches = parse_patch(patch_text)
        if not file_patches:
            return []

        patched_files: dict[str, str | None] = {}
        candidates: set[int] = set()
        fixture_pair_overrides: dict[str, list] = {}
        fixture_item_overrides: dict[str, list] = {}
        real_code_files: list[str] = []
        has_fixture_change = False
        has_syntax_or_import_break = False
        serde_touched = False
        executor_touched = False
        tpcds_inputs_touched = False
        tpch_inputs_touched = False
        tpcds_affected_qs: set[int] | None = set()
        tpcds_duckdb_affected_qs: set[int] | None = set()
        tpcds_exec_affected_qs: set[int] | None = set()
        tpcds_hit_counts: dict[int, int] = {}
        touched_dialect_names: set[str] | None = set()
        allow_lambda_exec = True
        context_table_only = False

        # Track whether per-fixture-index bitmask slicing is 100% sound for this commit!
        can_slice_fixtures = True
        fixture_enum_equivalent = False
        py_hit_lines_for_fixtures: list[tuple[str, set[int]]] = []
        check_import_rels: list[str] = []
        has_binary_change = False

        for fp in file_patches:
            rel = fp.path
            if not rel:
                continue

            if fp.is_binary:
                has_binary_change = True
                can_slice_fixtures = False
                if rel in self.open_file_to_tests:
                    candidates.update(self.open_file_to_tests[rel])
                elif "tpc-ds" in rel or "tpc-h" in rel:
                    candidates.update(self.tests_by_file.get("tests/test_executor.py", ()))
                continue

            orig_text = self._read_orig(fp.old_path) if fp.old_path else None
            new_text, orig_changed, new_changed = apply_file_patch(orig_text, fp)

            if not rel.endswith(".py"):
                if rel.endswith(".sql") and rel.startswith("tests/fixtures/"):
                    if orig_text is not None and new_text is not None:
                        if rel in SINGLE_LINE_SQL_FIXTURES:
                            diff_items = diff_sql_fixtures(orig_text, new_text)
                            if diff_items == []:
                                continue
                            if diff_items is not None:
                                fixture_item_overrides[rel] = diff_items
                        else:
                            diff_pairs = diff_sql_fixture_pairs(orig_text, new_text)
                            if diff_pairs == []:
                                continue
                            if diff_pairs is not None:
                                fixture_pair_overrides[rel] = diff_pairs

                    has_fixture_change = True
                    patched_files[rel] = new_text
                    if "tpc-ds" in rel:
                        tpcds_inputs_touched = True
                    if "tpc-h" in rel:
                        tpch_inputs_touched = True
                    opened_by = self.open_file_to_tests.get(rel)
                    if opened_by:
                        candidates.update(opened_by)
                    else:
                        candidates.update(self.tests_by_file.get("tests/test_optimizer.py", ()))
                        candidates.update(self.tests_by_file.get("tests/test_transpile.py", ()))
                        candidates.update(self.tests_by_file.get("tests/test_executor.py", ()))
                else:
                    patched_files[rel] = new_text
                    opened_by = self.open_file_to_tests.get(rel)
                    if opened_by:
                        has_fixture_change = True
                        candidates.update(opened_by)
                    elif rel.startswith("tests/fixtures/"):
                        has_fixture_change = True
                        candidates.update(self.tests_by_file.get("tests/test_optimizer.py", ()))
                        candidates.update(self.tests_by_file.get("tests/test_executor.py", ()))
                continue

            # Python file!
            edit = analyze_python_edit(rel, orig_text, new_text, orig_changed, new_changed)
            if edit.ast_identical:
                continue

            real_code_files.append(rel)
            patched_files[rel] = new_text
            if fp.old_path and fp.old_path != rel:
                patched_files[fp.old_path] = None

            if (
                len(file_patches) == 1
                and rel == "sqlglot/typing/duckdb.py"
                and fp.old_path == rel
                and self.typing_num is not None
                and orig_text is not None
                and new_text is not None
                and not edit.syntax_error
                and (parent_metadata := self._read_orig("sqlglot/typing/__init__.py"))
                and has_only_shadowed_metadata_inserts(orig_text, new_text, parent_metadata)
            ):
                # Optimizer readers use keyed lookups in this mapping. The typing suite
                # enumerates it, so keep that check and still verify import-time evaluation.
                candidates.update(self.tests_by_file.get("tests/test_typing.py", ()))
                candidates.update(self.open_file_to_tests.get(rel, ()))
                check_import_rels.append(rel)
                can_slice_fixtures = False
                continue

            if (
                rel == "sqlglot/serde.py"
                or rel == "sqlglot/expressions/core.py"
                or rel == "sqlglot/expressions/__init__.py"
            ):
                serde_touched = True
                can_slice_fixtures = False
            if rel.startswith("sqlglot/executor/") or rel == "sqlglot/planner.py" or rel == "sqlglot/generators/python.py":
                executor_touched = True
            if rel == "tests/test_executor.py" or rel == "tests/helpers.py" or rel == "tests/test_optimizer.py":
                can_slice_fixtures = False
                if rel != "tests/test_optimizer.py":
                    executor_touched = True
                    tpcds_inputs_touched = True
                    tpch_inputs_touched = True

            if rel.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
                stem = Path(rel).stem
                if stem in ("dialect", "base", "__init__", "python"):
                    touched_dialect_names = None
                elif touched_dialect_names is not None:
                    touched_dialect_names.add(stem)
            else:
                touched_dialect_names = None

            if rel in self.open_file_to_tests:
                candidates.update(self.open_file_to_tests[rel])

            if (
                len(file_patches) == 1
                and fp.old_path == rel
                and orig_text is not None
                and new_text is not None
                and not edit.syntax_error
            ):
                if rel == "sqlglot/dialects/dialect.py" and edit.enclosing_funcs == {"Dialect.case_sensitive"}:
                    fixture_enum_equivalent = is_normalization_enum_comparison_edit(orig_text, new_text)
                if (
                    rel.startswith("sqlglot/dialects/")
                    and any("NUMERIC_ESCAPES" in names for names in edit.class_attr_edits.values())
                    and is_numeric_escape_value_edit(orig_text, new_text)
                ):
                    # Metaclass and generator readers only inspect membership or truthiness,
                    # which are unchanged. Only the tokenizer reads the literal tuple values.
                    core = "sqlglot/tokenizer_core.py"
                    reader_lines = {
                        ln for ln, line in enumerate((self._read_orig(core) or "").splitlines(), 1)
                        if "spec = self.numeric_escapes.get(" in line
                    }
                    core_coverage = self.line_to_tests.get(core, {})
                    if len(reader_lines) == 2 and reader_lines.intersection(core_coverage):
                        for ln in reader_lines:
                            candidates.update(core_coverage.get(ln, ()))
                        py_hit_lines_for_fixtures.append((core, reader_lines))
                        check_import_rels.append(rel)
                        tpcds_affected_qs = tpcds_duckdb_affected_qs = tpcds_exec_affected_qs = None
                        continue

            if rel in ("sqlglot/executor/python.py", "sqlglot/generators/python.py", "sqlglot/executor/table.py"):
                allow_lambda_exec = False
            elif rel == "sqlglot/executor/context.py":
                context_table_only = (
                    edit.enclosing_funcs == {"Context.table"}
                    and not edit.class_level_edits
                    and not edit.module_level_edit
                    and not edit.scope_changed_funcs
                )
                if (
                    edit.class_level_edits
                    or edit.module_level_edit
                    or (edit.enclosing_funcs & {"Context.eval", "Context.eval_tuple", "Context.__init__"})
                ):
                    allow_lambda_exec = False

            if edit.syntax_error or edit.is_added_or_deleted:
                has_syntax_or_import_break = True
                can_slice_fixtures = False
                allow_lambda_exec = False
                tpcds_affected_qs = None
                tpcds_duckdb_affected_qs = None
                tpcds_exec_affected_qs = None
                touched_dialect_names = None
                candidates.update(self._import_blast_radius(rel))
                continue

            if (edit.class_level_edits or edit.module_level_edit) and rel.startswith(("sqlglot/", "tests/")):
                check_import_rels.append(rel)

            f_lines = self.line_to_tests.get(rel, {})
            reader_lines = None
            if (
                len(file_patches) == 1
                and rel.startswith("sqlglot/")
                and fp.old_path == rel
                and edit.const_reader_symbol is not None
                and not self._has_external_reader(rel, edit.const_reader_symbol)
            ):
                reader_lines = edit.const_reader_lines
            broad_scope_edit = (edit.class_level_edits or edit.module_level_edit) and reader_lines is None
            hit_lines = reader_lines if reader_lines is not None else edit.stmt_lines | edit.orig_changed_lines

            if broad_scope_edit or not rel.startswith("sqlglot/"):
                can_slice_fixtures = False
            else:
                py_hit_lines_for_fixtures.append((rel, hit_lines | edit.func_entry_lines))

            # 1. Line-level coverage
            for ln in hit_lines:
                if ln in f_lines:
                    candidates.update(f_lines[ln])

            # Track TPC-DS opt, DuckDB-transpile, & exec affected queries
            if tpcds_affected_qs is not None:
                q_map = self.tpcds_line_to_q.get(rel, {})
                if broad_scope_edit:
                    tpcds_affected_qs = None
                else:
                    for ln in hit_lines | edit.func_entry_lines:
                        if ln in q_map:
                            tpcds_affected_qs.update(q_map[ln])

            if tpcds_duckdb_affected_qs is not None:
                dq_map = self.tpcds_duckdb_line_to_q.get(rel, {})
                if broad_scope_edit:
                    tpcds_duckdb_affected_qs = None
                else:
                    for ln in hit_lines | edit.func_entry_lines:
                        if ln in dq_map:
                            tpcds_duckdb_affected_qs.update(dq_map[ln])

            if tpcds_exec_affected_qs is not None:
                eq_map = self.tpcds_exec_line_to_q.get(rel, {})
                if broad_scope_edit:
                    tpcds_exec_affected_qs = None
                else:
                    for ln in hit_lines | edit.func_entry_lines:
                        if ln in eq_map:
                            tpcds_exec_affected_qs.update(eq_map[ln])

            for ln in hit_lines:
                for idx in set(self.tpcds_line_to_q.get(rel, {}).get(ln, ())).union(
                    self.tpcds_duckdb_line_to_q.get(rel, {}).get(ln, ()),
                    self.tpcds_exec_line_to_q.get(rel, {}).get(ln, ()),
                ):
                    tpcds_hit_counts[idx] = tpcds_hit_counts.get(idx, 0) + 1

            # 2. Enclosing function entry lines (first executable line inside body, NOT def header!)
            max_nodes_already = self.max_nodes_num in candidates
            for entry_ln in edit.func_entry_lines:
                fn_tests = f_lines.get(entry_ln)
                if not fn_tests:
                    continue
                if rel.startswith("tests/") or edit.scope_changed_funcs or len(fn_tests) <= 150:
                    for tn in fn_tests:
                        if tn == self.max_nodes_num and not max_nodes_already:
                            continue
                        candidates.add(tn)

            # 3. Class-level edits (with sound attr_read_tests narrowing!)
            mod = file_to_module(rel)
            if edit.class_level_edits and reader_lines is None:
                for cname in edit.class_level_edits:
                    full_cls = f"{mod}:{cname}"
                    matched = False
                    cls_tests: set[int] = set()
                    if full_cls in self.dialect_to_tests:
                        cls_tests.update(self.dialect_to_tests[full_cls])
                        matched = True
                    if full_cls in self.expr_to_tests:
                        cls_tests.update(self.expr_to_tests[full_cls])
                        matched = True

                    if matched:
                        edited_attrs = edit.class_attr_edits.get(cname)
                        if (
                            edited_attrs
                            and cname not in edit.class_non_attr_edits
                            and "arg_types" not in edited_attrs
                            and all(
                                (a not in self.metaclass_read_attrs and a in self.attr_read_tests)
                                for a in edited_attrs
                            )
                        ):
                            attr_union: set[int] = set()
                            for a in edited_attrs:
                                attr_union.update(self.attr_read_tests[a])
                            candidates.update(cls_tests & attr_union)
                        else:
                            candidates.update(cls_tests)
                    else:
                        if rel.startswith("tests/"):
                            top_cls = cname.split(".")[0]
                            candidates.update(self.tests_by_class.get((rel, top_cls), ()))
                        else:
                            candidates.update(self.file_exec_to_tests.get(rel, ()))
                            candidates.update(self._tests_for_rev_imports(rel))

            # 4. Module-level edits
            if edit.module_level_edit and reader_lines is None:
                if rel.startswith("sqlglot/typing/"):
                    if self.typing_num is not None:
                        candidates.add(self.typing_num)
                    stem = Path(rel).stem
                    em_tests = self.attr_read_tests.get("EXPRESSION_METADATA")
                    if stem == "__init__":
                        candidates.update(self.file_exec_to_tests.get(rel, ()))
                        if em_tests:
                            candidates.update(em_tests)
                        else:
                            candidates.update(self.tests_by_file.get("tests/test_optimizer.py", ()))
                    else:
                        desc = {stem} | set(self.dialect_descendants.get(stem, ()))
                        d_tests: set[int] = set()
                        for d in desc:
                            d_tests.update(self.mod_to_dialect_tests.get(f"sqlglot.dialects.{d}", ()))
                        if em_tests:
                            candidates.update(d_tests & em_tests)
                        else:
                            candidates.update(d_tests)
                        candidates.update(self.file_exec_to_tests.get(rel, ()))
                elif rel.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/")):
                    candidates.update(self.mod_to_dialect_tests.get(mod, ()))
                    candidates.update(self.file_exec_to_tests.get(rel, ()))
                    if rel == "sqlglot/dialects/__init__.py":
                        candidates.update(self.tests_by_file.get("tests/dialects/test_dialect.py", ()))
                        candidates.update(self.tests_by_file.get("tests/test_dialect_imports.py", ()))
                else:
                    for sym in edit.module_level_names:
                        refs = self.symbol_refs.get(f"{mod}:{sym}")
                        if refs:
                            for rf, rln in refs:
                                candidates.update(self.file_exec_to_tests.get(rf, ()))
                                candidates.update(self._tests_for_rev_imports(rf))
                        if sym in self.attr_read_tests:
                            candidates.update(self.attr_read_tests[sym])
                    candidates.update(self.file_exec_to_tests.get(rel, ()))
                    if not edit.module_level_names or not all(
                        (s.isupper() or s.startswith("_")) for s in edit.module_level_names
                    ):
                        candidates.update(self._tests_for_rev_imports(rel))
                    if rel in DEFAULT_CORE_FILES and self.lazy_load_num is not None:
                        candidates.add(self.lazy_load_num)

            # 5. Direct test file edits (`tests/**/*.py`)
            if rel.startswith("tests/"):
                test_line_hits = any(ln in f_lines for ln in (hit_lines | edit.func_entry_lines))
                for qn in edit.enclosing_funcs:
                    parts = qn.split(".")
                    if len(parts) >= 2:
                        tid = f"{rel}::{parts[0]}::{parts[1]}"
                        if tid in self.by_id:
                            candidates.add(self.by_id[tid])
                        else:
                            cls_t = self.tests_by_class.get((rel, parts[0]), ())
                            if cls_t:
                                candidates.update(cls_t)
                            elif not test_line_hits:
                                # Previously unexecuted base helper method -> conservative rev_imports
                                candidates.update(self.tests_by_file.get(rel, ()))
                                candidates.update(self._tests_for_rev_imports(rel))
                    elif not test_line_hits:
                        candidates.update(self.tests_by_file.get(rel, ()))
                        candidates.update(self._tests_for_rev_imports(rel))

        if not candidates:
            return []

        if has_binary_change:
            return pack_workers(candidates, self.dur_ms)

        has_real_code_change = bool(real_code_files)
        fixture_pair_only = not has_real_code_change

        if has_syntax_or_import_break and len(candidates) > 600:
            return pack_workers(candidates, self.dur_ms)

        # Compute sound dialect_filter (preserved even on multi-file commits with unrelated fixtures!)
        dialect_filter: set[str] | None = None
        if touched_dialect_names and not has_syntax_or_import_break:
            expanded: set[str] = set()
            for d in touched_dialect_names:
                expanded.add(d)
                expanded.update(self.dialect_descendants.get(d, ()))
            if "duckdb" not in expanded:
                dialect_filter = expanded

        # Compute sound per-fixture-index bitmasks for TestOptimizer tests when eligible!
        fixture_index_masks: dict[str, int] | None = None
        has_opt_cand = any(self.by_num[n]["file"] == "tests/test_optimizer.py" for n in candidates)
        if can_slice_fixtures and py_hit_lines_for_fixtures and has_opt_cand:
            self._ensure_fixture_cov()
            if self._fixture_pair_counts:
                masks: dict[str, int] = {fname: 0 for fname in self._fixture_pair_counts}
                for src_f, lines_set in (() if fixture_enum_equivalent else py_hit_lines_for_fixtures):
                    f_by_fix = self._src_to_fixture_masks.get(src_f)
                    if not f_by_fix:
                        continue
                    for fname, ln_map in f_by_fix.items():
                        m = masks[fname]
                        for ln in lines_set:
                            lm = ln_map.get(ln)
                            if lm:
                                m |= lm
                        masks[fname] = m
                # Don't apply mask to any fixture file that was itself modified in this commit
                for pfile in patched_files:
                    if pfile.startswith("tests/fixtures/"):
                        short = pfile[len("tests/fixtures/"):]
                        masks.pop(short, None)
                fixture_index_masks = masks

        if fixture_enum_equivalent:
            # TPC-DS uses the same built-in dialect initialization as these optimizer fixtures.
            tpcds_affected_qs = tpcds_duckdb_affected_qs = set()

        pure_executor_only = (
            bool(real_code_files)
            and all(
                f.startswith("sqlglot/executor/") and f != "sqlglot/executor/__init__.py"
                for f in real_code_files
            )
            and not has_fixture_change
            and not serde_touched
            and not has_syntax_or_import_break
        )

        # Compute sound optimizer stage early-exit (Motherlode #3):
        # If ONLY function-body edits in sqlglot/ (no fixture/schema/optimizer/serde edits),
        # find the last rule in RULES[0..13] that can execute any modified line. If .sql() (bit 14)
        # is untouched and last_r <= 12, downstream rules are guaranteed identical when post-stage AST matches!
        early_stage: int | None = None
        if (
            can_slice_fixtures
            and py_hit_lines_for_fixtures
            and (has_opt_cand or self.tpcds_num in candidates)
            and not has_fixture_change
            and not serde_touched
            and not has_syntax_or_import_break
            and "sqlglot/optimizer/optimizer.py" not in patched_files
            and "sqlglot/schema.py" not in patched_files
        ):
            self._ensure_opt_stages()
            if self._rule_line_masks:
                touched_mask = 0
                for src_f, lines_set in py_hit_lines_for_fixtures:
                    rmap = self._rule_line_masks.get(src_f)
                    if not rmap:
                        continue
                    for ln in lines_set:
                        touched_mask |= rmap.get(ln, 0)
                if not (touched_mask & (1 << 14)):
                    if touched_mask == 0:
                        early_stage = -1
                    else:
                        last_r = touched_mask.bit_length() - 1
                        if last_r <= 12:
                            early_stage = last_r

        verify_list: list[tuple[int, str]] = []
        auto_include: set[int] = set()

        tpcds_fast_arg: dict | None = None
        tpch_fast_arg = False
        gen_mem_dialects_arg: list[str] | None = None

        normal_cands: list[int] = []
        for n in sorted(candidates):
            tid = self.by_num[n]["id"]
            if n == self.tpcds_num:
                if not tpcds_inputs_touched and self.tpcds_exec_hashes:
                    if executor_touched:
                        if pure_executor_only:
                            aff = sorted(tpcds_exec_affected_qs) if tpcds_exec_affected_qs is not None else None
                        elif tpcds_exec_affected_qs is not None and tpcds_affected_qs is not None:
                            aff = sorted(tpcds_exec_affected_qs | tpcds_affected_qs)
                        else:
                            aff = None
                    else:
                        aff = sorted(tpcds_affected_qs) if tpcds_affected_qs is not None else None
                    eff_duck_qs = set() if pure_executor_only else tpcds_duckdb_affected_qs
                    tpcds_fast_arg = {
                        "affected_q_indices": aff,
                        "affected_duckdb_qs": eff_duck_qs,
                        "opt_hashes": self.tpcds_opt_hashes,
                        "duckdb_hashes": self.tpcds_duckdb_hashes,
                        "exec_hashes": self.tpcds_exec_hashes,
                        "q_times": self.tpcds_q_times,
                        "q_tables": self.tpcds_q_tables,
                        "q_hit_counts": tpcds_hit_counts,
                        "q_coverage": self._tpcds_coverage() if (
                            "sqlglot/executor/__init__.py" in patched_files or "sqlglot/planner.py" in patched_files
                        ) else {},
                        "executor_touched": executor_touched,
                        "pure_executor_only": pure_executor_only,
                    }
                    verify_list.append((n, tid))
                else:
                    normal_cands.append(n)
            elif n == self.tpch_num:
                if not tpch_inputs_touched:
                    tpch_fast_arg = True
                    verify_list.append((n, tid))
                else:
                    normal_cands.append(n)
            elif n == self.gen_mem_num:
                if touched_dialect_names is not None and len(touched_dialect_names) <= 4:
                    expanded_gm: set[str] = set()
                    for d in touched_dialect_names:
                        expanded_gm.add(d)
                        expanded_gm.update(self.dialect_descendants.get(d, ()))
                    gen_mem_dialects_arg = sorted(expanded_gm)
                    verify_list.append((n, tid))
                else:
                    normal_cands.append(n)
            elif n == self.max_nodes_num:
                auto_include.add(n)
            else:
                normal_cands.append(n)

        verification_cpu_ms: dict[int, float] = {}
        for n in candidates - auto_include:
            file = self.by_num[n]["file"]
            if n == self.tpcds_num:
                ratio = 0.35 if tpcds_fast_arg is not None else 0.85
            elif n == self.tpch_num and tpch_fast_arg:
                ratio = 0.35
            elif n == self.gen_mem_num and gen_mem_dialects_arg is not None:
                ratio = 0.20
            elif file == "tests/test_optimizer.py":
                ratio = 0.10 if (dialect_filter or fixture_index_masks) else 0.35
            elif file.startswith("tests/dialects/") and dialect_filter:
                ratio = 0.20
            else:
                ratio = 0.90
            verification_cpu_ms[n] = max(1.0, self.dur_ms[n] * ratio)

        if normal_cands:
            est_cpu_ms = sum(verification_cpu_ms[n] for n in normal_cands) + 40.0
            sim_cost_ms = pack_cost_ms(normal_cands, self.dur_ms)
            if len(normal_cands) <= 1200 and (est_cpu_ms < sim_cost_ms * 0.92 or est_cpu_ms <= 5500):
                for n in sorted(normal_cands, key=lambda x: self.dur_ms[x]):
                    verify_list.append((n, self.by_num[n]["id"]))
            else:
                for n in normal_cands:
                    if self.by_num[n]["file"] == "tests/test_optimizer.py" and self.dur_ms[n] >= 500:
                        verify_list.append((n, self.by_num[n]["id"]))
                    else:
                        auto_include.add(n)

        final_failing: set[int] = set(auto_include)

        if verify_list:
            # Global wall-time safety guard: never risk hitting the 3-min/4.5-min container timeout!
            if (time.monotonic() - self._batch_t0) > self._batch_wall_budget:
                final_failing.update(n for n, _ in verify_list)
                return pack_workers(final_failing, self.dur_ms)

            # Soundness check for skip_scope_stress: True iff all edits are function-body edits in sqlglot/
            # whose hit lines do NOT intersect scope_stress_lines (the 10,000-UNION default-dialect stress lines)!
            skip_scope_stress = False
            if can_slice_fixtures and py_hit_lines_for_fixtures and self.scope_stress_lines:
                hits_stress = False
                for src_f, lines_set in py_hit_lines_for_fixtures:
                    sl = self.scope_stress_lines.get(src_f)
                    if sl and (lines_set & sl):
                        hits_stress = True
                        break
                if not hits_stress:
                    skip_scope_stress = True

            # Soundness check for duckdb_passing_pairs: safe whenever tests/test_optimizer.py wasn't edited!
            use_duck_pairs = (
                self.duckdb_passing_pairs
                if ("tests/test_optimizer.py" not in patched_files and has_opt_cand)
                else None
            )

            need_tier0 = (
                any(p in DEFAULT_CORE_FILES for p in patched_files)
                or any(p.startswith("sqlglot/expressions/") for p in patched_files)
                or any(n == self.lazy_load_num for n, _ in verify_list)
            )
            job = {
                "patched_files": patched_files,
                "check_import_rels": check_import_rels,
                "serde_touched": serde_touched,
                "fixture_pair_overrides": fixture_pair_overrides or None,
                "fixture_pair_only": fixture_pair_only,
                "fixture_item_overrides": fixture_item_overrides or None,
                "fixture_index_masks": fixture_index_masks,
                "dialect_filter": dialect_filter,
                "skip_scope_stress": skip_scope_stress,
                "duckdb_passing_pairs": use_duck_pairs,
                "tpcds_fast": tpcds_fast_arg,
                "tpch_fast": tpch_fast_arg,
                "gen_mem_dialects": gen_mem_dialects_arg,
                "early_stage": early_stage,
                "allow_lambda_exec": allow_lambda_exec,
                "allow_direct_iteration": allow_lambda_exec,
                "context_table_only": context_table_only,
                "tests": verify_list,
                "dur_ms_map": {n: self.dur_ms[n] for n in candidates},
                "verification_cpu_ms": verification_cpu_ms,
                "auto_include": auto_include,
                "cpu_cap": 25.0,
            }
            status, payload = self.runner.run_job(job, need_tier0=need_tier0)
            if status == "import_err":
                # Runtime import-time crash (e.g. NameError/ZeroDivisionError in class body)!
                expanded_err: set[int] = set(final_failing)
                for br in payload:
                    expanded_err.update(self._import_blast_radius(br))
                if len(expanded_err) <= 450:
                    job2 = {
                        "patched_files": patched_files,
                        "check_import_rels": [],
                        "serde_touched": serde_touched,
                        "tests": [(n, self.by_num[n]["id"]) for n in sorted(expanded_err)],
                        "cpu_cap": 20.0,
                    }
                    _, verified_fail2 = self.runner.run_job(job2, need_tier0=True)
                    return pack_workers(set(verified_fail2), self.dur_ms)
                return pack_workers(expanded_err, self.dur_ms)
            final_failing.update(payload)

        return pack_workers(final_failing, self.dur_ms)

    def close(self) -> None:
        self.runner.close()


def _remap_maps_if_needed(maps: dict, runtime_tests: list[dict]) -> dict:
    """If runtime tests.json has a different ordering or numbering than our build-time tests.json,
    seamlessly remap every test number in maps by exact pytest node id."""
    stored_ids = maps.get("num_to_id")
    if not stored_ids:
        return maps
    runtime_by_num = sorted(runtime_tests, key=lambda x: x["number"])
    runtime_ids = [t["id"] for t in runtime_by_num]
    if runtime_ids == stored_ids and all(t["number"] == i for i, t in enumerate(runtime_by_num)):
        return maps
    id_to_rt = {t["id"]: t["number"] for t in runtime_tests}
    remap = {old_i: id_to_rt[tid] for old_i, tid in enumerate(stored_ids) if tid in id_to_rt}

    def rlist(xs):
        return [remap[x] for x in xs if x in remap]

    maps["line_to_tests"] = {
        f: {ln: rlist(ts) for ln, ts in d.items()} for f, d in maps["line_to_tests"].items()
    }
    for key in (
        "file_exec_to_tests",
        "open_file_to_tests",
        "dialect_to_tests",
        "expr_to_tests",
        "attr_read_tests",
        "lazy_import_tests",
    ):
        if key in maps:
            maps[key] = {k: rlist(v) for k, v in maps[key].items()}
    return maps


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text(encoding="utf-8"))["tests"]
    maps = pickle.loads(zlib.decompress(MAPS_PATH.read_bytes()))
    maps = _remap_maps_if_needed(maps, tests)
    patch_paths = sorted((DATA / "commits").glob("*.patch"))
    selector = BlastRadiusSelector(tests, maps)
    if "BATCH_WALL_BUDGET" not in os.environ:
        selector._batch_wall_budget = 235.0 if len(patch_paths) > 48 else 150.0
    try:
        for patch_path in patch_paths:
            cid = patch_path.name.removesuffix(".patch")
            try:
                patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
                workers = selector.select_for_commit(patch_text)
            except BaseException:
                workers = pack_workers(selector.all_test_nums, selector.dur_ms)
            sys.stdout.write(json.dumps({"commit": cid, "workers": workers}) + "\n")
        sys.stdout.flush()
    finally:
        selector.close()


if __name__ == "__main__":
    main()
