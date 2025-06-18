import re
from pathlib import Path

from src.tools.base import ToolRegistry

DOC_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "docs"
    / "user-guide"
    / "tool-catalogue.md"
)


def extract_tool_names(md_text: str):
    # assumes table rows start with `| ` and tool names are in backticks
    pattern = re.compile(r"\|\s+`([^`]+)`")
    return set(pattern.findall(md_text))


def test_tool_catalogue_up_to_date():
    if not DOC_PATH.exists():
        # Skip if doc has not been generated yet
        import pytest

        pytest.skip("tool-catalogue.md not generated")

    doc_names = extract_tool_names(DOC_PATH.read_text(encoding="utf-8"))

    # Ensure all modules imported so registry is populated
    import importlib
    import pkgutil

    base_pkg = importlib.import_module("src.tools")
    for _, modname, _ in pkgutil.walk_packages(
        base_pkg.__path__, prefix=f"{base_pkg.__name__}."
    ):
        try:
            importlib.import_module(modname)
        except Exception:
            pass

    code_names = set(ToolRegistry.list_tools())
    missing_in_docs = code_names - doc_names
    extra_in_docs = doc_names - code_names

    assert (
        not missing_in_docs and not extra_in_docs
    ), f"Docs out of sync: missing={sorted(missing_in_docs)}, extra={sorted(extra_in_docs)}"
