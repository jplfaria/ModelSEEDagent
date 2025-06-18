from pathlib import Path

from src.tools.base import ToolRegistry

DOC_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "tool-reference"


def test_tool_pages_exist():
    missing = []
    for name in ToolRegistry.list_tools():
        if not (DOC_DIR / f"{name}.md").exists():
            missing.append(name)
    assert not missing, f"Missing tool reference pages: {missing}"
