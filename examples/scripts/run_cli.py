#!/usr/bin/env python3
"""
Entry point script for ModelSEEDagent CLI

This script fixes import path issues by setting up the Python path correctly
before importing and running the CLI modules.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_main_cli():
    """Run the main CLI"""
    try:
        from src.cli.main import app

        app()
    except ImportError as e:
        print(f"❌ Import error in main CLI: {e}")
        print("🔧 Try using the standalone CLI instead: python src/cli/standalone.py")
        sys.exit(1)


def run_standalone_cli():
    """Run the standalone CLI"""
    from src.cli.standalone import app

    app()


def run_interactive():
    """Run the interactive interface"""
    try:
        from src.interactive.interactive_cli import main

        main()
    except ImportError as e:
        print(f"❌ Import error in interactive CLI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ModelSEEDagent CLI Launcher")
    parser.add_argument(
        "mode", choices=["main", "standalone", "interactive"], help="CLI mode to run"
    )
    parser.add_argument("args", nargs="*", help="Arguments to pass to the CLI")

    args = parser.parse_args()

    # Modify sys.argv to pass arguments correctly
    sys.argv = [sys.argv[0]] + args.args

    if args.mode == "main":
        run_main_cli()
    elif args.mode == "standalone":
        run_standalone_cli()
    elif args.mode == "interactive":
        run_interactive()
