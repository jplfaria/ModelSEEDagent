import importlib
import inspect
import sys
from pathlib import Path
from textwrap import dedent
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.tools.base import ToolRegistry
except ImportError as e:
    print(f"Cannot import ToolRegistry: {e}")
    print("Tool pages generation requires full package installation - skipping")
    sys.exit(0)

DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs" / "tool-reference"

SAFE_IMPORT_BASE = "src.tools"


def import_all_tools():
    """Import every module under src.tools so registry is populated."""
    try:
        base = importlib.import_module(SAFE_IMPORT_BASE)
        for modinfo in Path(base.__file__).parent.rglob("*.py"):
            module_name = f"{SAFE_IMPORT_BASE}." + ".".join(
                modinfo.relative_to(Path(base.__file__).parent).with_suffix("").parts
            )
            try:
                importlib.import_module(module_name)
            except Exception:
                # Ignore modules that fail to import (e.g., service-dependent)
                continue
    except Exception as e:
        print(f"Warning: Could not import tool modules: {e}")
        # Continue anyway - may have some tools registered


def docstring_summary(obj) -> str:
    doc = inspect.getdoc(obj) or "(no description)"
    return dedent(doc).strip()


def generate_page(tool_name: str, tool_cls: type) -> str:
    md: List[str] = []
    md.append(f"# `{tool_name}`\n")
    md.append(docstring_summary(tool_cls))
    md.append("\n## Import\n")
    md.append(
        f"```python\nfrom {tool_cls.__module__} import {tool_cls.__name__}\n````\n"
    )

    # Input schema (if _run_tool uses pydantic model)
    sig = inspect.signature(tool_cls._run_tool)  # type: ignore[attr-defined]
    params = [p for p in sig.parameters.values() if p.name not in ("self",)]
    if params:
        md.append("## Parameters\n")
        md.append("| Name | Type | Description |\n|-----|------|-------------|")
        for p in params:
            md.append(f"| {p.name} | {p.annotation} | |")
    return "\n".join(md)


def main():
    import_all_tools()
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    index_lines: List[str] = [
        "# Tool Reference\n",
        "Auto-generated pages for every registered tool.\n",
    ]
    for name in sorted(ToolRegistry.list_tools()):
        cls = ToolRegistry.get_tool(name)
        if cls is None:
            continue
        page_md = generate_page(name, cls)
        page_path = DOCS_ROOT / f"{name}.md"
        page_path.write_text(page_md, encoding="utf-8")
        index_lines.append(f"- [{name}]({name}.md)")
    (DOCS_ROOT / "index.md").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"Generated {len(ToolRegistry.list_tools())} tool pages in {DOCS_ROOT}")


if __name__ == "__main__":
    main()
