import inspect
import sys
import textwrap
from pathlib import Path
from typing import Dict, List

import typer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Path where markdown reference will be written
OUTPUT_MD = (
    Path(__file__).resolve().parent.parent / "docs" / "user-guide" / "cli-reference.md"
)

# Import Typer app lazily to avoid heavy deps during import of this script.
from importlib import import_module

APP = import_module("src.cli.main").app  # type: ignore[attr-defined]


def _format_default(value):
    if value is None or value == inspect._empty:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    # Handle Typer option/argument objects
    if hasattr(value, "default"):
        return _format_default(value.default)
    # Handle various object types
    if hasattr(value, "__dict__") and not isinstance(value, (str, int, float)):
        return "—"
    return str(value)


def describe_command(command: typer.models.CommandInfo) -> str:
    """Return markdown table for a Typer command (excluding subcommands)."""
    md_lines: List[str] = []
    cmd_name = getattr(command, "name", "Unknown")
    md_lines.append(f"### `{cmd_name}`\n")
    help_text = (
        getattr(command, "help", None)
        or getattr(command.callback, "__doc__", "").strip()
        if hasattr(command, "callback")
        else ""
    )
    if help_text:
        md_lines.append(f"{help_text}\n")

    sig = inspect.signature(command.callback)
    params = [p for p in sig.parameters.values() if p.name not in ("ctx",)]
    if not params:
        md_lines.append("*No options*\n")
        return "\n".join(md_lines)

    headers = ["Parameter", "Type", "Default"]
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for p in params:
        type_name = (
            p.annotation.__name__
            if hasattr(p.annotation, "__name__")
            else str(p.annotation)
        )
        md_lines.append(f"| `{p.name}` | {type_name} | {_format_default(p.default)} |")
    md_lines.append("")
    return "\n".join(md_lines)


def walk_app(app: typer.Typer, path: str = "") -> Dict[str, typer.models.CommandInfo]:
    """Return mapping of command path to CommandInfo (Typer 0.9)."""
    commands: Dict[str, typer.models.CommandInfo] = {}
    for cmd in app.registered_commands:  # list of CommandInfo
        full_path = f"{path} {cmd.name}".strip()
        commands[full_path] = cmd
    for sub in getattr(app, "registered_groups", []):  # list of TyperInfo
        sub_app = sub.typer_instance  # type: ignore[attr-defined]
        sub_name = sub.name
        commands.update(walk_app(sub_app, path=f"{path} {sub_name}".strip()))
    return commands


def build_markdown() -> str:
    md: List[str] = []
    md.append("# CLI Reference\n")
    md.append("This page is auto-generated from the Typer app; do not edit manually.\n")

    commands = walk_app(APP)
    for path, cmd in sorted(commands.items()):
        if path.strip():  # Only process non-empty paths
            md.append(f"## `{path}`\n")
            md.append(describe_command(cmd))
    return "\n".join(md)


def main():
    # We have a manually-maintained comprehensive CLI reference
    # Skip auto-generation to avoid overwriting good manual documentation
    print(
        f"Skipping auto-generation - using manually maintained {OUTPUT_MD.relative_to(Path.cwd())}"
    )
    return

    # Original auto-generation code (disabled)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(build_markdown(), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
