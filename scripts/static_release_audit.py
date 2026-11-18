"""Read-only static structural audit, never runs imported course code.

Use only when explicitly requested:
python scripts/static_release_audit.py
Checks JSON/AST/path structure; not a unit test or runtime smoke test.
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
    for path in notebooks:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("nbformat") != 4:
                errors.append(f"not nbformat4: {path.name}")
            cells = data.get("cells", [])
            cell_ids = [cell.get("id") for cell in cells if "id" in cell]
            if len(cell_ids) != len(set(cell_ids)):
                errors.append(f"duplicate notebook cell IDs: {path.name}")
            for i, cell in enumerate(cells):
                if cell.get("cell_type") == "code":
                    if cell.get("execution_count") is not None or cell.get("outputs", []) != []:
                        errors.append(f"executed/output cell: {path.name} #{i}")
        except (ValueError, UnicodeError, KeyError, TypeError) as exc:
            errors.append(f"bad notebook: {path.name}: {exc}")

    index_file = root / "docs" / "INDEX.md"
    if index_file.exists():
        index = index_file.read_text(encoding="utf-8")
        for rel in re.findall(r"\]\(\.\./(notebooks/[^)]+\.ipynb)\)", index):
            if not (root / rel).is_file():
                errors.append(f"broken notebook index link: {rel}")

    return {"source_files_parsed": len(source_files),
            "master_notebooks_inspected": len(notebooks),
            "issues": errors, "static_only": True,
            "executed_tests": False, "runtime_certified": False}


def main():
    root = Path(__file__).resolve().parents[1]
    report = audit(root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
