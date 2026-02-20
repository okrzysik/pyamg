"""Contract tests for the refactored LS–AMG–DD implementation.

These tests are intentionally lightweight and fast. They enforce refactor
invariants that prevent regressions during future algorithm work.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


def _pyamg_dir() -> Path:
    # .../pyamg/tests/schwarz/test_*.py -> parents[2] is .../pyamg
    return Path(__file__).resolve().parents[2]


def _lsdd_files() -> list[Path]:
    pyamg_dir = _pyamg_dir()
    lsdd_dir = pyamg_dir / "schwarz" / "lsdd"
    exp = pyamg_dir / "schwarz" / "least_squares_dd_exp.py"

    files = sorted(lsdd_dir.glob("*.py"))
    files.append(exp)
    return files


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _has_module_docstring_as_first_stmt(tree: ast.Module) -> bool:
    if not tree.body:
        return False
    first = tree.body[0]
    if isinstance(first, ast.Expr):
        val = first.value
        return isinstance(val, ast.Constant) and isinstance(val.value, str)
    return False


def _top_level_defs(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node


def _find_print_calls(tree: ast.Module) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            calls.append(node)
    return calls


@pytest.mark.parametrize("path", _lsdd_files())
def test_module_docstring_first_statement(path: Path) -> None:
    tree = _parse(path)
    assert _has_module_docstring_as_first_stmt(tree), (
        f"{path} must start with a module docstring as the first statement"
    )


@pytest.mark.parametrize("path", _lsdd_files())
def test_top_level_definitions_have_docstrings(path: Path) -> None:
    tree = _parse(path)
    missing: list[str] = []
    for node in _top_level_defs(tree):
        if ast.get_docstring(node) is None:
            missing.append(f"{node.name} (line {node.lineno})")
    assert not missing, f"{path} missing docstrings for: {', '.join(missing)}"


@pytest.mark.parametrize("path", _lsdd_files())
def test_print_policy(path: Path) -> None:
    tree = _parse(path)
    prints = _find_print_calls(tree)
    if path.name == "stats.py":
        return
    assert not prints, f"{path} has print() calls; printing must be confined to lsdd/stats.py"


@pytest.mark.parametrize("path", _lsdd_files())
def test_no_legacy_level_attributes(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    banned_patterns = [
        r"\blevel\.N\b",
        r"\blevel\.nIi\b",
        r"\blevel\.ni\b",
        r"\blevel\.nonoverlapping_subdomain\b",
        r"\blevel\.overlapping_subdomain\b",
        r"\blevel\.overlapping_rows\b",
        r"\blevel\.subdomain_ptr\b",
        r"\blevel\.subdomain\b",
        r"\blevel\.threshold\b",
        r"\blevel\.nev\b",
        r"\blevel\.min_ev\b",
    ]

    hits: list[str] = []
    for pat in banned_patterns:
        if re.search(pat, text):
            hits.append(pat)

    assert not hits, f"{path} contains legacy level attributes: {hits}"
