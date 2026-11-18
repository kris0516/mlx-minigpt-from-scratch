"""Read-only, PARTIAL static release checker. Does NOT import project code.

When explicitly run later, checks .py syntax with ast.parse and Notebook
JSON structure, nbformat major/minor, present/unique cell IDs, unexecuted
code cells, and parseable plain-Python cell syntax. Notebook IPython magic
cells are counted as skipped (not parsed); old masters with missing cell IDs
are counted as legacy warnings. This is NOT full nbformat schema validation,
not a runtime smoke test, and not an execution of Notebook code.
"""
import ast
import json
import re
from pathlib import Path


REQUIRED = (
    "README.md", "pyproject.toml", "uv.lock", "LICENSE",
    "docs/course-roadmap.md", "docs/INDEX.md",
    "src/minigpt/data.py", "src/minigpt/model.py",
    "src/minigpt/train.py", "src/minigpt/generate.py",
    "src/minigpt/config.py", "src/minigpt/cli.py",
)
CELL_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _cell_source(cell, label, errors):
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(s, str) for s in source):
        return "".join(source)
    errors.append(f"invalid cell source: {label}")
    return None


def audit(root: Path):
    root = root.resolve()
    errors = []
    for relative in REQUIRED:
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")

    source_files = sorted((root / "src" / "minigpt").glob("*.py"))
    source_files += sorted((root / "scripts").glob("*.py"))
    source_files += sorted((root / "tests").glob("*.py"))
    for path in source_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"Python source invalid: {path.relative_to(root)}: {exc}")

    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    python_cells_parsed = 0
    magic_cells_skipped = 0
    legacy_missing_ids = 0
    for path in notebooks:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                errors.append(f"notebook JSON root must be object: {path.name}")
                continue
            if data.get("nbformat") != 4:
                errors.append(f"not nbformat4: {path.name}")
            minor = data.get("nbformat_minor")
            if type(minor) is not int or minor < 0:
                errors.append(f"invalid nbformat_minor: {path.name}")
            cells = data.get("cells")
            if not isinstance(cells, list):
                errors.append(f"cells is not a list: {path.name}")
                continue
            seen_ids = set()
            numeric_prefix = path.name.split("_", 1)[0]
            new_master = numeric_prefix.isdecimal() and 102 <= int(numeric_prefix) <= 116
            for i, cell in enumerate(cells):
                label = f"{path.name} cell {i}"
                if not isinstance(cell, dict):
                    errors.append(f"cell is not an object: {label}")
                    continue
                cid = cell.get("id")
                if cid is None:
                    if new_master:
                        errors.append(f"new master missing cell ID: {label}")
                    else:
                        legacy_missing_ids += 1  # Preserve old, previously published masters.
                elif not isinstance(cid, str) or CELL_ID.fullmatch(cid) is None:
                    errors.append(f"invalid cell ID: {label}")
                elif cid in seen_ids:
                    errors.append(f"duplicate cell ID: {label}")
                else:
                    seen_ids.add(cid)
                kind = cell.get("cell_type")
                if kind not in ("code", "markdown", "raw"):
                    errors.append(f"invalid cell_type: {label}")
                    continue
                source = _cell_source(cell, label, errors)
                if kind == "code":
                    if cell.get("execution_count") is not None or cell.get("outputs") != []:
                        errors.append(f"executed/invalid output cell: {label}")
                    if source is None:
                        continue
                    # IPython-specific syntax cannot be parsed with Python ast.parse.
                    # Explicitly count such cells as uncovered, not as PASS.
                    if any(line.lstrip().startswith(("%", "!", "?"))
                           for line in source.splitlines()):
                        magic_cells_skipped += 1
                        continue
                    try:
                        ast.parse(source, filename=label)
                        python_cells_parsed += 1
                    except SyntaxError as exc:
                        errors.append(f"Notebook plain-Python syntax invalid: {label}: {exc}")
        except (ValueError, UnicodeError, OSError) as exc:
            errors.append(f"cannot inspect notebook: {path.name}: {exc}")

    index_file = root / "docs" / "INDEX.md"
    if index_file.exists():
        index = index_file.read_text(encoding="utf-8")
        for rel in re.findall(r"\]\(\.\./(notebooks/[^)]+\.ipynb)\)", index):
            if not (root / rel).is_file():
                errors.append(f"broken notebook index link: {rel}")

    return {
        "source_files_python_ast_parsed": len(source_files),
        "master_notebooks_inspected": len(notebooks),
        "notebook_plain_python_cells_ast_parsed": python_cells_parsed,
        "notebook_magic_cells_ast_skipped": magic_cells_skipped,
        "legacy_cells_missing_ids_warning": legacy_missing_ids,
        "issues": errors,
        "scope": "partial static structural and Python syntax checks; NOT full schema",
        "static_only": True,
        "executed_tests": False,
        "runtime_certified": False,
    }


def main():
    report = audit(Path(__file__).resolve().parents[1])
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
