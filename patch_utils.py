"""Unified diff parser, in-memory patch applier, and Python AST / SQL fixture diff analyzer."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_CODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


@dataclass
class FilePatch:
    old_path: str | None  # None for /dev/null (added file)
    new_path: str | None  # None for /dev/null (deleted file)
    hunks: list[Hunk] = field(default_factory=list)
    is_binary: bool = False

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""


def _clean_diff_path(raw: str) -> str | None:
    raw = raw.strip()
    if raw == "/dev/null":
        return None
    # Strip trailing timestamp if any (tab-separated)
    if "\t" in raw:
        raw = raw.split("\t", 1)[0]
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    if raw.startswith("a/") or raw.startswith("b/"):
        raw = raw[2:]
    return raw


def parse_patch(patch_text: str) -> list[FilePatch]:
    """Parse a git unified diff into FilePatch objects."""
    lines = patch_text.splitlines(keepends=True)
    patches: list[FilePatch] = []
    cur: FilePatch | None = None
    cur_hunk: Hunk | None = None
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if line.startswith("diff --git "):
            cur = None
            cur_hunk = None
            i += 1
            continue
        if line.startswith("rename from "):
            old_p = line[len("rename from "):].strip()
            new_p = None
            if i + 1 < n and lines[i + 1].startswith("rename to "):
                new_p = lines[i + 1][len("rename to "):].strip()
                i += 2
            else:
                i += 1
            cur = FilePatch(old_path=old_p, new_path=new_p)
            patches.append(cur)
            cur_hunk = None
            continue
        if line.startswith("Binary files "):
            if cur is not None:
                cur.is_binary = True
            i += 1
            continue
        if line.startswith("--- "):
            old_p = _clean_diff_path(line[4:])
            new_p = None
            if i + 1 < n and lines[i + 1].startswith("+++ "):
                new_p = _clean_diff_path(lines[i + 1][4:])
                i += 2
            else:
                i += 1
            if cur is not None and (cur.old_path == old_p or cur.new_path == new_p):
                cur.old_path = old_p
                cur.new_path = new_p
            else:
                cur = FilePatch(old_path=old_p, new_path=new_p)
                patches.append(cur)
            cur_hunk = None
            continue
        if line.startswith("+++ ") and cur is not None and cur.new_path is None:
            cur.new_path = _clean_diff_path(line[4:])
            i += 1
            continue
        m = _HUNK_RE.match(line)
        if m and cur is not None:
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) is not None else 1
            cur_hunk = Hunk(old_start, old_count, new_start, new_count, [])
            cur.hunks.append(cur_hunk)
            i += 1
            continue
        if cur_hunk is not None and (
            line.startswith(" ")
            or line.startswith("+")
            or line.startswith("-")
            or line.startswith("\\ No newline at end of file")
        ):
            cur_hunk.lines.append(line)
            i += 1
            continue
        i += 1

    return patches


def apply_file_patch(orig_text: str | None, fp: FilePatch) -> tuple[str | None, set[int], set[int]]:
    """Apply a FilePatch in memory.

    Returns (new_text_or_None_if_deleted, orig_changed_lines_1based, new_changed_lines_1based).
    """
    if fp.new_path is None:
        # File deleted
        orig_lines = orig_text.splitlines(keepends=True) if orig_text else []
        return None, set(range(1, len(orig_lines) + 1)), set()

    orig_lines = orig_text.splitlines(keepends=True) if orig_text is not None else []
    out_lines: list[str] = []
    orig_changed: set[int] = set()
    new_changed: set[int] = set()

    orig_idx = 0  # 0-based index into orig_lines

    for hunk in fp.hunks:
        target_orig_idx = max(0, hunk.old_start - 1) if hunk.old_count > 0 else hunk.old_start
        # Copy untouched lines before this hunk
        while orig_idx < target_orig_idx and orig_idx < len(orig_lines):
            out_lines.append(orig_lines[orig_idx])
            orig_idx += 1

        # Process hunk lines
        idx_h = 0
        hlines = hunk.lines
        while idx_h < len(hlines):
            hl = hlines[idx_h]
            if hl.startswith("\\"):
                idx_h += 1
                continue
            tag = hl[0]
            content = hl[1:]
            # Check if next line is "\ No newline at end of file"
            no_nl = idx_h + 1 < len(hlines) and hlines[idx_h + 1].startswith("\\")
            if no_nl and content.endswith("\n"):
                content = content[:-1]

            if tag == " ":
                if orig_idx < len(orig_lines):
                    out_lines.append(orig_lines[orig_idx])
                    orig_idx += 1
                else:
                    out_lines.append(content)
            elif tag == "-":
                orig_changed.add(orig_idx + 1)
                orig_idx += 1
            elif tag == "+":
                out_lines.append(content)
                new_changed.add(len(out_lines))
                # For pure insertions, also mark adjacent original lines
                if orig_idx > 0:
                    orig_changed.add(orig_idx)
                if orig_idx < len(orig_lines):
                    orig_changed.add(orig_idx + 1)
            idx_h += 1

    # Copy remaining lines after last hunk
    while orig_idx < len(orig_lines):
        out_lines.append(orig_lines[orig_idx])
        orig_idx += 1

    return "".join(out_lines), orig_changed, new_changed


def has_coding_cookie(text: str) -> str | None:
    lines = text.splitlines()[:2]
    for ln in lines:
        m = _CODING_RE.match(ln)
        if m:
            return m.group(1).lower()
    return None


@dataclass
class PyEditInfo:
    path: str
    syntax_error: bool = False
    ast_identical: bool = False
    is_added_or_deleted: bool = False
    # Enclosing statement lines in orig_ast for every modified line
    stmt_lines: set[int] = field(default_factory=set)
    # Enclosing function/method entry lines in orig_ast
    func_entry_lines: set[int] = field(default_factory=set)
    # Enclosing function names (qualnames inside file, e.g. "TestParser.test_foo" or "func")
    enclosing_funcs: set[str] = field(default_factory=set)
    # Class-level (outside method bodies) edited classes (qualnames inside file)
    class_level_edits: set[str] = field(default_factory=set)
    # Class-level attribute assignments edited: class_qualname -> set of attr names
    class_attr_edits: dict[str, set[str]] = field(default_factory=dict)
    # Class-level non-simple-attr edits (bases, decorators, method sigs, non-assign)
    class_non_attr_edits: set[str] = field(default_factory=set)
    # All enclosing classes (even if edit is inside method body)
    enclosing_classes: set[str] = field(default_factory=set)
    # Module-level edits (outside any class or function)
    module_level_edit: bool = False
    # Module-level edited symbol names (constants / functions / imports)
    module_level_names: set[str] = field(default_factory=set)
    # Whether lexical scoping inside an enclosing function changed
    scope_changed_funcs: set[str] = field(default_factory=set)
    # Raw orig changed lines
    orig_changed_lines: set[int] = field(default_factory=set)
    # Dict-key diffs for class/module dict assignments: "Scope:ATTR" -> (changed_key_tags_or_None, keys_added_or_removed)
    dict_key_edits: dict[str, tuple[set[str] | None, bool]] = field(default_factory=dict)
    # Exact runtime reads and the name to check for consumers outside this module.
    const_reader_lines: set[int] | None = None
    const_reader_symbol: str | None = None
    # Single pure operator swap (`is`<->`==`, `<`<->`<=`, `if A and/or B:`) inside one function body.
    single_inv_op: tuple[str, int, int] | None = None


def _node_span(node: ast.AST) -> tuple[int, int]:
    start = getattr(node, "lineno", 0)
    end = getattr(node, "end_lineno", start) or start
    # Include decorators!
    decs = getattr(node, "decorator_list", None)
    if decs:
        start = min(start, min(getattr(d, "lineno", start) for d in decs))
    return start, end


def _collect_local_bindings(fn_node: ast.AST) -> set[str]:
    """Collect variables bound locally in a function (excluding nested defs/classes)."""
    bound: set[str] = set()
    stack = list(ast.iter_child_nodes(fn_node))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
            continue
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound.update(n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for alias in n.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        stack.extend(ast.iter_child_nodes(n))
    return bound


def analyze_python_edit(
    path: str,
    orig_text: str | None,
    new_text: str | None,
    orig_changed: set[int],
    new_changed: set[int],
) -> PyEditInfo:
    info = PyEditInfo(path=path, orig_changed_lines=set(orig_changed))

    if orig_text is None or new_text is None:
        info.is_added_or_deleted = True
        info.module_level_edit = True
        if new_text is not None:
            try:
                ast.parse(new_text.encode("utf-8"), filename=path)
            except (SyntaxError, ValueError):
                info.syntax_error = True
        return info

    # Parse UTF-8 bytes so `# coding: ...` encoding cookies are enforced identically to Python's importer!
    try:
        new_ast = ast.parse(new_text.encode("utf-8"), filename=path)
    except (SyntaxError, ValueError):
        info.syntax_error = True
        return info

    try:
        orig_ast = ast.parse(orig_text.encode("utf-8"), filename=path)
    except (SyntaxError, ValueError):
        info.module_level_edit = True
        return info

    # Check exact AST equivalence (ignoring line/col attributes) AND coding cookie
    if (
        has_coding_cookie(orig_text) == has_coding_cookie(new_text)
        and ast.dump(orig_ast, include_attributes=False) == ast.dump(new_ast, include_attributes=False)
    ):
        info.ast_identical = True
        return info

    # Index functions & classes in both orig_ast and new_ast
    orig_funcs: dict[str, ast.AST] = {}
    new_funcs: dict[str, ast.AST] = {}
    orig_classes: dict[str, ast.ClassDef] = {}
    new_classes: dict[str, ast.ClassDef] = {}

    def index_ast(
        node: ast.AST,
        funcs_out: dict[str, ast.AST],
        classes_out: dict[str, ast.ClassDef] | None,
        prefix: str = "",
    ) -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qn = f"{prefix}.{child.name}" if prefix else child.name
                if classes_out is not None:
                    classes_out[qn] = child
                index_ast(child, funcs_out, classes_out, qn)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qn = f"{prefix}.{child.name}" if prefix else child.name
                funcs_out[qn] = child
                index_ast(child, funcs_out, classes_out, qn)

    def _func_header_sig(fn: ast.AST) -> tuple:
        return (
            type(fn).__name__,
            ast.dump(fn.args, include_attributes=False),
            tuple(ast.dump(d, include_attributes=False) for d in fn.decorator_list),
            ast.dump(fn.returns, include_attributes=False) if getattr(fn, "returns", None) is not None else None,
        )

    def _class_header_sig(cls: ast.ClassDef) -> tuple:
        return (
            tuple(ast.dump(b, include_attributes=False) for b in cls.bases),
            tuple(ast.dump(kw, include_attributes=False) for kw in cls.keywords),
            tuple(ast.dump(d, include_attributes=False) for d in cls.decorator_list),
        )

    index_ast(orig_ast, orig_funcs, orig_classes)
    index_ast(new_ast, new_funcs, new_classes)

    unchanged_funcs: set[str] = set()
    for qn, ofn in orig_funcs.items():
        nfn = new_funcs.get(qn)
        if nfn is None:
            continue
        os_, oe_ = _node_span(ofn)
        ns_, ne_ = _node_span(nfn)
        if not (
            any(os_ <= ln <= oe_ for ln in orig_changed)
            or any(ns_ <= ln <= ne_ for ln in new_changed)
        ):
            continue
        if ast.dump(ofn, include_attributes=False) == ast.dump(nfn, include_attributes=False):
            unchanged_funcs.add(qn)

    def _first_exec_line(fn_node: ast.AST) -> int | None:
        for bstmt in getattr(fn_node, "body", []):
            if (
                isinstance(bstmt, ast.Expr)
                and isinstance(bstmt.value, ast.Constant)
                and isinstance(bstmt.value.value, str)
            ):
                continue
            return bstmt.lineno
        return None

    orig_lambda_classes: set[str] = set()

    # Walk orig_ast to find enclosing scopes for each changed line
    def walk_scope(
        body: list[ast.stmt],
        class_stack: list[str],
        func_stack: list[tuple[str, ast.AST]],
    ) -> None:
        for stmt in body:
            s, e = _node_span(stmt)
            # Check if any orig_changed line intersects [s, e]
            hit_lines = [ln for ln in orig_changed if s <= ln <= e]
            if not hit_lines:
                continue

            if isinstance(stmt, ast.ClassDef):
                cname = ".".join(class_stack + [stmt.name])
                info.enclosing_classes.add(cname)
                # Exact AST header/decorator/base check (immune to pure-insertion adjacent-line spill!)
                ncls = new_classes.get(cname)
                if ncls is None or _class_header_sig(stmt) != _class_header_sig(ncls):
                    info.class_level_edits.add(cname)
                    info.class_non_attr_edits.add(cname)
                # Recurse into class body
                walk_scope(stmt.body, class_stack + [stmt.name], func_stack)
                continue

            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qn = ".".join(class_stack + [f[0] for f in func_stack] + [stmt.name])
                if qn in unchanged_funcs:
                    # Pure adjacent-line spill or comment inside unchanged function -> ignore!
                    for ln in hit_lines:
                        info.orig_changed_lines.discard(ln)
                    continue
                info.enclosing_funcs.add(qn)
                # First EXECUTABLE statement inside function body (skip def header & docstring!)
                first_ln = _first_exec_line(stmt)
                if first_ln is not None:
                    info.func_entry_lines.add(first_ln)
                # Exact AST decorator/signature/defaults check (immune to pure-insertion adjacent-line spill!)
                nfn = new_funcs.get(qn)
                if nfn is None or _func_header_sig(stmt) != _func_header_sig(nfn):
                    if class_stack and not func_stack:
                        cname = ".".join(class_stack)
                        info.class_level_edits.add(cname)
                        info.class_non_attr_edits.add(cname)
                    elif not class_stack and not func_stack:
                        info.module_level_edit = True
                        info.module_level_names.add(stmt.name)
                # Check if lexical bindings changed
                if nfn is not None:
                    if _collect_local_bindings(stmt) != _collect_local_bindings(nfn):
                        info.scope_changed_funcs.add(qn)
                walk_scope(stmt.body, class_stack, func_stack + [(stmt.name, stmt)])
                continue

            # Regular or compound statement
            if not func_stack:
                # We are at class level or module level!
                if class_stack:
                    cname = ".".join(class_stack)
                    # Check if the edit is ONLY inside a lambda inside a class dict (like FUNCTIONS)
                    inside_lambda = _find_enclosing_lambda(stmt, hit_lines)
                    if inside_lambda is not None:
                        ls, le = _node_span(inside_lambda)
                        info.stmt_lines.update(range(ls, le + 1))
                        orig_lambda_classes.add(cname)
                        if any(ln < ls or ln > le for ln in hit_lines):
                            info.class_level_edits.add(cname)
                            info.class_non_attr_edits.add(cname)
                    else:
                        info.class_level_edits.add(cname)
                        info.stmt_lines.update(range(s, e + 1))
                        # Track simple attribute assignment name (e.g. WEEK_OFFSET = -1, SHOW_PARSERS = {...})
                        attr_names: set[str] = set()
                        if isinstance(stmt, ast.Assign) and all(isinstance(t, ast.Name) for t in stmt.targets):
                            for t in stmt.targets:
                                attr_names.add(t.id)
                        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                            attr_names.add(stmt.target.id)
                        if attr_names:
                            info.class_attr_edits.setdefault(cname, set()).update(attr_names)
                        else:
                            info.class_non_attr_edits.add(cname)
                else:
                    info.module_level_edit = True
                    info.stmt_lines.update(range(s, e + 1))
                    _collect_assigned_names(stmt, info.module_level_names)
                continue

            # Inside a function body!
            # Record enclosing function entry line & enclosing statement lines
            if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match)):
                # Check if edit is in the header/test vs inside a child body
                child_bodies: list[list[ast.stmt]] = []
                if hasattr(stmt, "body") and isinstance(stmt.body, list):
                    child_bodies.append(stmt.body)
                if hasattr(stmt, "orelse") and isinstance(stmt.orelse, list):
                    child_bodies.append(stmt.orelse)
                if hasattr(stmt, "finalbody") and isinstance(stmt.finalbody, list):
                    child_bodies.append(stmt.finalbody)
                if hasattr(stmt, "handlers"):
                    for h in stmt.handlers:
                        child_bodies.append(h.body)
                if hasattr(stmt, "cases"):
                    for c in stmt.cases:
                        child_bodies.append(c.body)

                # Header lines (before first body stmt, or condition lines)
                first_body_ln = min((b[0].lineno for b in child_bodies if b), default=e + 1)
                header_hits = [ln for ln in hit_lines if ln < first_body_ln]
                if header_hits:
                    # Condition/header edited -> include ALL header lines
                    info.stmt_lines.update(range(s, min(first_body_ln, e + 1)))
                for cb in child_bodies:
                    if cb:
                        cs = _node_span(cb[0])[0]
                        ce = _node_span(cb[-1])[1]
                        if any(cs <= ln <= ce for ln in hit_lines):
                            walk_scope(cb, class_stack, func_stack)
                # Also if hit line fell on `elif` / `else:` / `except:` header between bodies
                for ln in hit_lines:
                    info.stmt_lines.add(ln)
                # Include the compound stmt's own start line (for short-circuit / branch flips!)
                info.stmt_lines.add(s)
            else:
                # Leaf statement (Assign, Return, Expr, AugAssign, etc.) -> include full statement span!
                info.stmt_lines.update(range(s, e + 1))

    walk_scope(orig_ast.body, [], [])

    # Also walk new_ast with new_changed to catch pure insertions at top-level, class-level, or brand-new methods!
    def walk_new_scope(
        body: list[ast.stmt],
        class_stack: list[str],
        func_stack: list[str],
    ) -> None:
        for stmt in body:
            s, e = _node_span(stmt)
            hit_lines = [ln for ln in new_changed if s <= ln <= e]
            if not hit_lines:
                continue

            if isinstance(stmt, ast.ClassDef):
                cname = ".".join(class_stack + [stmt.name])
                info.enclosing_classes.add(cname)
                ocls = orig_classes.get(cname)
                if ocls is None or _class_header_sig(ocls) != _class_header_sig(stmt):
                    info.class_level_edits.add(cname)
                    info.class_non_attr_edits.add(cname)
                    if not class_stack and not func_stack:
                        info.module_level_edit = True
                        info.module_level_names.add(stmt.name)
                walk_new_scope(stmt.body, class_stack + [stmt.name], func_stack)
                continue

            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qn = ".".join(class_stack + func_stack + [stmt.name])
                if qn in unchanged_funcs:
                    continue
                info.enclosing_funcs.add(qn)
                if qn not in orig_funcs:
                    # Brand-new function or method override!
                    if class_stack and not func_stack:
                        cname = ".".join(class_stack)
                        info.class_level_edits.add(cname)
                        info.class_non_attr_edits.add(cname)
                    elif not class_stack and not func_stack:
                        info.module_level_edit = True
                        info.module_level_names.add(stmt.name)
                else:
                    first_ln = _first_exec_line(orig_funcs[qn])
                    if first_ln is not None:
                        info.func_entry_lines.add(first_ln)
                    if _func_header_sig(orig_funcs[qn]) != _func_header_sig(stmt):
                        if class_stack and not func_stack:
                            cname = ".".join(class_stack)
                            info.class_level_edits.add(cname)
                            info.class_non_attr_edits.add(cname)
                        elif not class_stack and not func_stack:
                            info.module_level_edit = True
                            info.module_level_names.add(stmt.name)
                walk_new_scope(stmt.body, class_stack, func_stack + [stmt.name])
                continue

            if not func_stack:
                if class_stack:
                    cname = ".".join(class_stack)
                    inside_lambda = _find_enclosing_lambda(stmt, hit_lines)
                    if inside_lambda is not None and cname in orig_lambda_classes:
                        continue
                    info.class_level_edits.add(cname)
                    attr_names: set[str] = set()
                    if isinstance(stmt, ast.Assign) and all(isinstance(t, ast.Name) for t in stmt.targets):
                        for t in stmt.targets:
                            attr_names.add(t.id)
                    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        attr_names.add(stmt.target.id)
                    if attr_names:
                        info.class_attr_edits.setdefault(cname, set()).update(attr_names)
                    else:
                        info.class_non_attr_edits.add(cname)
                else:
                    info.module_level_edit = True
                    _collect_assigned_names(stmt, info.module_level_names)

    walk_new_scope(new_ast.body, [], [])

    # Safety: if a changed line was NOT inside any statement in orig_ast or new_ast, treat as module-level edit!
    if not info.stmt_lines and not info.enclosing_funcs and not info.class_level_edits:
        info.module_level_edit = True

    # Dict-key diffing for edited class attributes and module-level dict assignments!
    _analyze_dict_key_edits(orig_ast, new_ast, info)

    if (info.module_level_edit or info.class_attr_edits) and not info.enclosing_funcs:
        _analyze_const_readers(orig_ast, new_ast, info)

    # Fast single-operator invariant edit detection reusing already-parsed single edited function!
    if (
        not info.module_level_edit
        and not info.class_level_edits
        and len(info.enclosing_funcs) == 1
        and len(orig_changed) <= 3
        and len(new_changed) <= 3
    ):
        qn = next(iter(info.enclosing_funcs))
        ofn = orig_funcs.get(qn)
        nfn = new_funcs.get(qn)
        if ofn is not None and nfn is not None and _raw_lines_may_be_inv_op(orig_text, new_text, orig_changed, new_changed):
            info.single_inv_op = _find_single_invariant_op_in_nodes(ofn, nfn)

    return info


def _extract_dict_key_tag(k: ast.AST) -> str | None:
    if isinstance(k, ast.Constant) and isinstance(k.value, str):
        return f"str:{k.value}"
    if isinstance(k, ast.Attribute):
        # exp.Abs -> "exp:Abs", exp.DType.BOOLEAN -> "dtype:BOOLEAN", exp.DataType.Type.BOOLEAN -> "dtype:BOOLEAN"
        if isinstance(k.value, ast.Name) and k.value.id == "exp":
            return f"exp:{k.attr}"
        if (
            isinstance(k.value, ast.Attribute)
            and k.value.attr == "DType"
            and isinstance(k.value.value, ast.Name)
            and k.value.value.id == "exp"
        ):
            return f"dtype:{k.attr}"
        if (
            isinstance(k.value, ast.Attribute)
            and k.value.attr == "Type"
            and isinstance(k.value.value, ast.Attribute)
            and k.value.value.attr == "DataType"
        ):
            return f"dtype:{k.attr}"
    return None


def _diff_dict_nodes(orig_d: ast.Dict, new_d: ast.Dict) -> tuple[set[str] | None, bool]:
    orig_spreads = [ast.dump(v, include_attributes=False) for k, v in zip(orig_d.keys, orig_d.values) if k is None]
    new_spreads = [ast.dump(v, include_attributes=False) for k, v in zip(new_d.keys, new_d.values) if k is None]
    if orig_spreads != new_spreads:
        return None, True

    orig_map: dict[str, tuple[ast.AST, str]] = {}
    for k, v in zip(orig_d.keys, orig_d.values):
        if k is None:
            continue
        kd = ast.dump(k, include_attributes=False)
        if kd in orig_map:
            return None, True
        orig_map[kd] = (k, ast.dump(v, include_attributes=False))

    new_map: dict[str, tuple[ast.AST, str]] = {}
    for k, v in zip(new_d.keys, new_d.values):
        if k is None:
            continue
        kd = ast.dump(k, include_attributes=False)
        if kd in new_map:
            return None, True
        new_map[kd] = (k, ast.dump(v, include_attributes=False))

    keys_added_or_removed = set(orig_map.keys()) != set(new_map.keys())
    tags: set[str] = set()
    for kd in set(orig_map.keys()) | set(new_map.keys()):
        ov = orig_map.get(kd)
        nv = new_map.get(kd)
        if ov is None or nv is None or ov[1] != nv[1]:
            k_node = nv[0] if nv is not None else ov[0]
            tag = _extract_dict_key_tag(k_node)
            if tag is None:
                return None, keys_added_or_removed
            tags.add(tag)
    return tags, keys_added_or_removed


def _index_dict_assigns(tree: ast.AST) -> dict[str, ast.Dict]:
    out: dict[str, ast.Dict] = {}

    def walk(body: list[ast.stmt], cls_prefix: str) -> None:
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                qn = f"{cls_prefix}.{stmt.name}" if cls_prefix else stmt.name
                walk(stmt.body, qn)
            elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Dict) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                key = f"{cls_prefix or '__module__'}:{stmt.targets[0].id}"
                out[key] = stmt.value
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Dict) and isinstance(stmt.target, ast.Name):
                key = f"{cls_prefix or '__module__'}:{stmt.target.id}"
                out[key] = stmt.value

    walk(getattr(tree, "body", []), "")
    return out


def _analyze_dict_key_edits(orig_ast: ast.AST, new_ast: ast.AST, info: PyEditInfo) -> None:
    targets: list[str] = []
    for cname, attrs in info.class_attr_edits.items():
        if cname not in info.class_non_attr_edits:
            for a in attrs:
                targets.append(f"{cname}:{a}")
    if info.module_level_edit and info.module_level_names:
        for nm in info.module_level_names:
            targets.append(f"__module__:{nm}")
    if not targets:
        return
    orig_dicts = _index_dict_assigns(orig_ast)
    new_dicts = _index_dict_assigns(new_ast)
    for tkey in targets:
        od = orig_dicts.get(tkey)
        nd = new_dicts.get(tkey)
        if od is not None and nd is not None:
            info.dict_key_edits[tkey] = _diff_dict_nodes(od, nd)


def _single_changed_statement(old_body, new_body):
    if len(old_body) == len(new_body):
        changes = [
            (old, new) for old, new in zip(old_body, new_body) if ast.dump(old) != ast.dump(new)
        ]
        if len(changes) == 1:
            return changes[0]
    return None


def _changed_class_member(original: str, patched: str) -> tuple[tuple[str, ...], tuple] | None:
    change = _single_changed_statement(ast.parse(original).body, ast.parse(patched).body)
    owners = []
    while change is not None:
        old, new = change
        if not isinstance(old, ast.ClassDef) or not isinstance(new, ast.ClassDef):
            return (tuple(owners), change) if owners else None
        old_body, new_body = old.body, new.body
        old.body, new.body = [], []
        if ast.dump(old) != ast.dump(new):
            return None
        owners.append(old.name)
        change = _single_changed_statement(old_body, new_body)
    return None


def is_parser_max_nodes_edit(edit: PyEditInfo) -> bool:
    """Return True if a sqlglot/parser.py edit could alter max_nodes / _node_count / raise_error logic."""
    if edit.path != "sqlglot/parser.py":
        return False
    if edit.syntax_error or edit.is_added_or_deleted or edit.module_level_edit or edit.class_level_edits:
        return True
    sensitive = {
        "Parser.__init__",
        "Parser.reset",
        "Parser.validate_expression",
        "Parser.raise_error",
        "Parser.parse",
        "Parser._parse",
    }
    return bool(edit.enclosing_funcs & sensitive)


def is_numeric_escape_value_edit(original: str, patched: str) -> bool:
    change = _changed_class_member(original, patched)
    if change is None or change[0][-1] != "Tokenizer":
        return False
    pair = change[1]
    if not all(
        isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "NUMERIC_ESCAPES"
        and isinstance(stmt.value, ast.Dict) and stmt.value.keys
        and all(isinstance(k, ast.Constant) and type(k.value) is str for k in stmt.value.keys)
        and all(isinstance(v, ast.Tuple) and len(v.elts) == 4 and all(
            isinstance(n, ast.Constant) and type(n.value) is int for n in v.elts
        ) for v in stmt.value.values)
        for stmt in pair
    ):
        return False
    old, new = pair
    return [k.value for k in old.value.keys] == [k.value for k in new.value.keys]


def is_normalization_enum_comparison_edit(original: str, patched: str) -> bool:
    change = _changed_class_member(original, patched)
    if change is None or change[0] != ("Dialect",):
        return False
    pair = change[1]
    if not all(isinstance(node, ast.FunctionDef) and node.name == "case_sensitive" for node in pair):
        return False

    # Built-in fixture dialects initialize this slot with NormalizationStrategy members.
    # Equality and identity agree for two members of that enum.
    for method in pair:
        for node in ast.walk(method):
            if (
                isinstance(node, ast.Compare) and len(node.ops) == 1
                and ast.unparse(node.left) == "self.normalization_strategy"
                and isinstance(right := node.comparators[0], ast.Attribute)
                and isinstance(right.value, ast.Name) and right.value.id == "NormalizationStrategy"
            ):
                if isinstance(node.ops[0], ast.Is):
                    node.ops[0] = ast.Eq()
                elif isinstance(node.ops[0], ast.IsNot):
                    node.ops[0] = ast.NotEq()
    return ast.dump(pair[0]) == ast.dump(pair[1])


def has_only_shadowed_metadata_inserts(original: str, patched: str, parent: str) -> bool:
    """Prove that new leading entries are overwritten by the original parent spread."""
    name = "EXPRESSION_METADATA"
    change = _single_changed_statement(ast.parse(original).body, ast.parse(patched).body)
    if change is None or not all(
        isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == name
        and isinstance(stmt.value, ast.Dict) for stmt in change
    ):
        return False
    old, new = (stmt.value for stmt in change)
    if not old.keys or old.keys[0] is not None or not (
        isinstance(old.values[0], ast.Name) and old.values[0].id == name
    ):
        return False
    added = len(new.keys) - len(old.keys)
    if added <= 0 or any(
        ast.dump(before) != ast.dump(after) if before is not None and after is not None
        else before is not after
        for before, after in zip(old.keys, new.keys[added:])
    ) or any(ast.dump(a) != ast.dump(b) for a, b in zip(old.values, new.values[added:])):
        return False

    inherited: set[str] = set()
    for stmt in ast.parse(parent).body:
        if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
                and stmt.target.id == name and isinstance(stmt.value, ast.Dict)):
            continue
        for key, value in zip(stmt.value.keys, stmt.value.values):
            if key is not None:
                inherited.add(ast.dump(key))
            elif isinstance(value, ast.DictComp) and len(value.generators) == 1:
                source = value.generators[0]
                if (not source.ifs and isinstance(source.iter, ast.Set)
                    and isinstance(value.key, ast.Name) and isinstance(source.target, ast.Name)
                    and value.key.id == source.target.id):
                    inherited.update(ast.dump(item) for item in source.iter.elts)
    return all(
        isinstance(key, ast.Attribute) and isinstance(key.value, ast.Name) and key.value.id == "exp"
        and ast.dump(key) in inherited
        and isinstance(value, ast.Dict) and len(value.keys) == 1
        and isinstance(value.keys[0], ast.Constant) and value.keys[0].value == "returns"
        and isinstance(value.values[0], ast.Attribute) and ast.unparse(value.values[0].value) == "exp.DType"
        for key, value in zip(new.keys[:added], new.values[:added])
    )


def _analyze_const_readers(orig_ast: ast.Module, new_ast: ast.Module, info: PyEditInfo) -> None:
    """Trace a literal container used only by direct runtime membership and lookup operations."""
    change = _single_changed_statement(orig_ast.body, new_ast.body)
    if change is None:
        return
    old, new = change
    owner = None
    if isinstance(old, ast.ClassDef) and isinstance(new, ast.ClassDef):
        if old.name != new.name or any(
            cls.bases or cls.keywords or cls.decorator_list or cls.type_params for cls in (old, new)
        ):
            return
        owner = old.name
        change = _single_changed_statement(old.body, new.body)
        if change is None:
            return
        old, new = change
    if isinstance(old, ast.AnnAssign) and isinstance(new, ast.AnnAssign):
        if (
            not isinstance(old.target, ast.Name)
            or not isinstance(new.target, ast.Name)
            or ast.dump(old.annotation) != ast.dump(new.annotation)
            or old.value is None
            or new.value is None
        ):
            return
        old_target, new_target = old.target, new.target
    elif all(
        isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name) for stmt in (old, new)
    ):
        old_target, new_target = old.targets[0], new.targets[0]
    else:
        return
    name = old_target.id
    if name != new_target.id or not name.isupper():
        return
    allowed_types = (ast.Dict,) if owner else (ast.Dict, ast.Set, ast.Tuple)
    try:
        old_literal = ast.literal_eval(old.value)
        new_literal = ast.literal_eval(new.value)
    except (SyntaxError, TypeError, ValueError):
        return
    scalar = owner is None and type(old_literal) is type(new_literal) and type(old_literal) in (
        str, bytes, int, float, complex, bool, type(None),
    )
    if not scalar and (type(old.value) is not type(new.value) or not isinstance(old.value, allowed_types)):
        return

    parents = {child: node for node in ast.walk(orig_ast) for child in ast.iter_child_nodes(node)}
    runtime = {
        node: fn.name
        for fn in ast.walk(orig_ast)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        for stmt in fn.body
        for node in ast.walk(stmt)
    }
    symbol = owner or name
    lines: set[int] = set()
    readers: set[str] = set()
    for node in parents:
        if isinstance(node, ast.Constant) and node.value in (symbol, name):
            return
        if not isinstance(node, ast.Name):
            continue
        if node.id in {"globals", "locals", "vars", "eval", "exec"}:
            return
        if node.id != symbol or node is old_target:
            continue
        if not isinstance(node.ctx, ast.Load) or node not in runtime:
            return
        readers.add(runtime[node])
        if scalar:
            lines.add(node.lineno)
            continue
        access = node
        if owner:
            access = parents[node]
            if (not isinstance(access, ast.Attribute) or access.attr != name
                or not isinstance(access.ctx, ast.Load)):
                return
        reader = parents[access]
        if isinstance(reader, ast.Compare) and any(
            right is access and isinstance(op, (ast.In, ast.NotIn))
            for op, right in zip(reader.ops, reader.comparators)
        ):
            pass
        elif isinstance(old.value, ast.Dict) and (
            (isinstance(reader, ast.Subscript) and reader.value is access and isinstance(reader.ctx, ast.Load))
            or (isinstance(reader, ast.Attribute) and reader.attr == "get"
                and isinstance(parents[reader], ast.Call) and parents[reader].func is reader)
        ):
            pass
        else:
            return
        lines.update((reader.lineno, access.lineno))
    if not lines:
        return

    # A direct or transitive call from module/class initialization can cache the
    # changed value before fixture coverage starts. Keep the import-wide fallback.
    referrers: dict[str, set[str | None]] = {}
    for node in parents:
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            ref = node.id if isinstance(node, ast.Name) else node.attr
            referrers.setdefault(ref, set()).add(runtime.get(node))
    pending = list(readers)
    while pending:
        for caller in referrers.get(pending.pop(), ()):
            if caller is None:
                return
            if caller not in readers:
                readers.add(caller)
                pending.append(caller)

    info.const_reader_lines = lines
    info.const_reader_symbol = symbol


def _find_enclosing_lambda(node: ast.AST, hit_lines: list[int]) -> ast.Lambda | None:
    """If ALL hit_lines lie strictly inside a single ast.Lambda inside node, return that Lambda."""
    best: ast.Lambda | None = None
    for sub in ast.walk(node):
        if isinstance(sub, ast.Lambda):
            ls, le = _node_span(sub)
            if all(ls <= ln <= le for ln in hit_lines):
                if best is None or (le - ls) < (_node_span(best)[1] - _node_span(best)[0]):
                    best = sub
    return best


def _collect_assigned_names(stmt: ast.stmt, out: set[str]) -> None:
    if isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    out.add(n.id)
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        out.add(stmt.target.id)
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        for a in stmt.names:
            out.add(a.asname or a.name.split(".")[0])


# --- SQL fixture parsing (exact mirror of tests/helpers.py) ---

def _filter_sql_comments(s: str) -> str:
    return "\n".join([line for line in s.splitlines() if line and not line.startswith("--")])


def _extract_meta(sql: str) -> tuple[str, dict[str, str]]:
    meta: dict[str, str] = {}
    sql_lines = sql.split("\n")
    i = 0
    while i < len(sql_lines) and sql_lines[i].startswith("#"):
        key, val = sql_lines[i].split(":", maxsplit=1)
        meta[key.lstrip("#").strip()] = val.strip()
        i += 1
    return "\n".join(sql_lines[i:]), meta


def parse_sql_fixture_pairs(text: str) -> list[tuple[dict[str, str], str, str]]:
    statements = _filter_sql_comments(text).split(";")
    size = len(statements)
    pairs = []
    for i in range(0, size, 2):
        if i + 1 < size:
            sql = statements[i].strip()
            sql, meta = _extract_meta(sql)
            expected = statements[i + 1].strip()
            pairs.append((meta, sql, expected))
    return pairs


def parse_sql_fixtures(text: str) -> list[str]:
    items = []
    lines = []
    for line in _filter_sql_comments(text).splitlines():
        lines.append(line)
        if not line.startswith("#"):
            items.append("\n".join(lines))
            lines.clear()
    return items


def diff_sql_fixture_pairs(orig_text: str, new_text: str) -> list[tuple[dict[str, str], str, str]] | None:
    """Return ONLY the changed/added pairs (in new_text) if pair count matches and diff is clean,
    or None if the fixture is unchanged (empty list) vs malformed.
    Returns [] if parsed pairs are 100% identical!
    """
    try:
        orig_pairs = parse_sql_fixture_pairs(orig_text)
        new_pairs = parse_sql_fixture_pairs(new_text)
    except Exception:
        return None  # Malformed metadata header -> must run full test (it will raise!)

    if orig_pairs == new_pairs:
        return []

    orig_set = {(tuple(sorted(m.items())), s, e) for m, s, e in orig_pairs}
    changed = [
        (m, s, e)
        for m, s, e in new_pairs
        if (tuple(sorted(m.items())), s, e) not in orig_set
    ]
    if not changed:
        # Only deletions? If pairs were deleted, tests might still pass (remaining pairs unchanged)
        # Returning [] is safe ONLY if remaining pairs are identical and test doesn't assert count!
        # To be safe, return None if deletions only.
        return None
    return changed


def diff_sql_fixtures(orig_text: str, new_text: str) -> list[str] | None:
    try:
        orig_items = parse_sql_fixtures(orig_text)
        new_items = parse_sql_fixtures(new_text)
    except Exception:
        return None
    if orig_items == new_items:
        return []
    orig_set = set(orig_items)
    changed = [it for it in new_items if it not in orig_set]
    return changed if changed else None


_IS_OP_PAIRS = {
    (ast.Is, ast.Eq),
    (ast.Eq, ast.Is),
    (ast.IsNot, ast.NotEq),
    (ast.NotEq, ast.IsNot),
}
_ORD_OP_PAIRS = {
    (ast.Lt, ast.LtE),
    (ast.LtE, ast.Lt),
    (ast.Gt, ast.GtE),
    (ast.GtE, ast.Gt),
}
_BOOL_OP_PAIRS = {
    (ast.And, ast.Or),
    (ast.Or, ast.And),
}


_INV_OP_RE = re.compile(r"\b(?:is|and|or)\b|==|!=|<=|>=|<|>")


def _raw_lines_may_be_inv_op(
    orig_text: str,
    new_text: str,
    orig_changed: set[int],
    new_changed: set[int],
) -> bool:
    orig_lines = orig_text.splitlines()
    new_lines = new_text.splitlines()
    o_hit = any(
        1 <= ln <= len(orig_lines) and _INV_OP_RE.search(orig_lines[ln - 1])
        for ln in orig_changed
    )
    if not o_hit:
        return False
    return any(
        1 <= ln <= len(new_lines) and _INV_OP_RE.search(new_lines[ln - 1])
        for ln in new_changed
    )


def _ast_equal(a: object, b: object) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, ast.AST):
        for f in a._fields:
            if not _ast_equal(getattr(a, f), getattr(b, f)):
                return False
        return True
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_ast_equal(x, y) for x, y in zip(a, b))
    return a == b


def _find_single_invariant_op_in_nodes(orig_node: ast.AST, new_node: ast.AST) -> tuple[str, int, int] | None:
    found: list[tuple[str, int, int]] = []
    aborted = False

    def _walk(o: object, n: object) -> None:
        nonlocal aborted
        if aborted:
            return
        if type(o) is not type(n):
            aborted = True
            return
        if isinstance(o, ast.AST):
            if isinstance(o, ast.Compare) and len(o.ops) == 1 and len(n.ops) == 1:
                pair = (type(o.ops[0]), type(n.ops[0]))
                if (
                    (pair in _IS_OP_PAIRS or pair in _ORD_OP_PAIRS)
                    and _ast_equal(o.left, n.left)
                    and _ast_equal(o.comparators[0], n.comparators[0])
                ):
                    kind = "is" if pair in _IS_OP_PAIRS else "ord"
                    found.append((kind, o.lineno, o.col_offset))
                    if len(found) > 1:
                        aborted = True
                    return
            if (
                isinstance(o, ast.If)
                and isinstance(o.test, ast.BoolOp)
                and isinstance(n.test, ast.BoolOp)
                and len(o.test.values) == 2
                and len(n.test.values) == 2
                and (type(o.test.op), type(n.test.op)) in _BOOL_OP_PAIRS
                and _ast_equal(o.test.values[0], n.test.values[0])
                and _ast_equal(o.test.values[1], n.test.values[1])
                and _ast_equal(o.body, n.body)
                and _ast_equal(o.orelse, n.orelse)
            ):
                found.append(("bool", o.test.lineno, o.test.col_offset))
                if len(found) > 1:
                    aborted = True
                return
            for field in o._fields:
                _walk(getattr(o, field), getattr(n, field))
        elif isinstance(o, list):
            if len(o) != len(n):
                aborted = True
                return
            for a, b in zip(o, n):
                _walk(a, b)
        else:
            if o != n:
                aborted = True

    _walk(orig_node, new_node)
    if aborted or len(found) != 1:
        return None
    return found[0]


def detect_single_invariant_op_edit(orig_text: str, new_text: str) -> tuple[str, int, int] | None:
    """If the ENTIRE file diff is swapping a single pure operator (`is`<->`==`, `<`<->`<=`, or `if A and/or B:`),
    return `(kind, orig_lineno, orig_col_offset)` where kind in {'is', 'ord', 'bool'}. Otherwise None."""
    try:
        orig_ast = ast.parse(orig_text)
        new_ast = ast.parse(new_text)
    except SyntaxError:
        return None
    return _find_single_invariant_op_in_nodes(orig_ast, new_ast)
