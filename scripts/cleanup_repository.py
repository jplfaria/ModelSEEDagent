#!/usr/bin/env python3
"""
Repository Cleanup Script
Organizes test files, removes obsolete content, and archives old results.
"""

import glob
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def move_file(src, dst):
    """Move file with directory creation."""
    ensure_dir(os.path.dirname(dst))
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved: {src} → {dst}")


def remove_if_exists(path):
    """Remove file or directory if it exists."""
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed directory: {path}")
        else:
            os.remove(path)
            print(f"Removed file: {path}")


def main():
    print("🧹 Starting repository cleanup...")

    # 1. Move test files from root to tests/manual/
    print("\n📁 Moving test files to proper directories...")
    test_files = [
        "test_comprehensive_toolset.py",
        "test_direct_tools.py",
        "test_infrastructure_final.py",
        "test_media_infrastructure.py",
        "test_simple_cli.py",
        "test_timeout_debug.py",
        "test_interactive_debug.py",
        "test_user_scenario.py",
        "test_multi_tool_analysis.py",
    ]

    ensure_dir("tests/manual")
    for test_file in test_files:
        if os.path.exists(test_file):
            move_file(test_file, f"tests/manual/{test_file}")

    # 2. Archive recent testbed results
    print("\n📦 Archiving testbed results...")
    ensure_dir("testbed_results/archive")

    testbed_dirs = glob.glob("comprehensive_testbed_results_*") + glob.glob(
        "direct_testbed_results_*"
    )
    for dir_name in testbed_dirs:
        if os.path.isdir(dir_name):
            move_file(dir_name, f"testbed_results/archive/{dir_name}")

    # 3. Clean up old log files (keep last 10 of each type)
    print("\n🗂️ Cleaning up old log files...")

    def clean_log_dirs(pattern, keep_count=10):
        dirs = sorted(glob.glob(pattern), reverse=True)  # Most recent first
        for dir_path in dirs[keep_count:]:  # Remove all but the most recent
            remove_if_exists(dir_path)

    clean_log_dirs("logs/langgraph_run_*", 10)
    clean_log_dirs("logs/realtime_run_*", 10)

    # 4. Remove duplicate model files
    print("\n🗃️ Removing duplicate model files...")

    # Keep models in data/models/, remove from data/examples/ if duplicated
    models_to_check = ["e_coli_core.xml", "iML1515.xml"]
    for model in models_to_check:
        examples_path = f"data/examples/{model}"
        models_path = f"data/models/{model}"

        if os.path.exists(examples_path) and os.path.exists(models_path):
            remove_if_exists(examples_path)
            print(f"Removed duplicate: {examples_path} (kept in data/models/)")

    # 5. Remove intermediate Mycoplasma model versions
    print("\n🧬 Cleaning up Mycoplasma model variants...")
    mycoplasma_variants = [
        "data/examples/Mycoplasma_G37_fixed.xml",
        "data/examples/Mycoplasma_G37_fixed2.xml",
        "data/examples/Mycoplasma_G37.GMM.mdl_fixed.xml",
        "data/examples/Mycoplasma_G37.GMM.mdl_comprehensive_fix.xml",
    ]

    for variant in mycoplasma_variants:
        remove_if_exists(variant)

    # 6. Clean up large temporary files
    print("\n📊 Removing large temporary files...")
    large_files = [
        "data/examples/EcoliMG1655.xml",  # 7MB - likely temporary
        "data/examples/pputida.faa",  # 2MB - temporary protein file
    ]

    for large_file in large_files:
        remove_if_exists(large_file)

    # 7. Clean up session files (keep last 15)
    print("\n💾 Cleaning up session files...")
    session_files = sorted(
        glob.glob("sessions/*.json"), key=lambda x: os.path.getmtime(x), reverse=True
    )

    for session_file in session_files[15:]:  # Keep last 15 sessions
        remove_if_exists(session_file)

    # 8. Remove redundant documentation
    print("\n📚 Cleaning up redundant documentation...")
    redundant_docs = [
        "docs/archive/redundant",
        "docs/archive/session_logs",  # Large and outdated
    ]

    for doc_path in redundant_docs:
        remove_if_exists(doc_path)

    # 9. Clean up Python cache files
    print("\n🐍 Removing Python cache files...")
    os.system("find . -name '*.pyc' -delete")
    os.system("find . -name '__pycache__' -type d -delete")

    # 10. Create .gitignore entries for future cleanup
    print("\n📝 Updating .gitignore...")
    gitignore_additions = [
        "# Test results and logs",
        "comprehensive_testbed_results_*/",
        "direct_testbed_results_*/",
        "test_*.py",  # Temporary test files in root
        "logs/langgraph_run_*/",
        "logs/realtime_run_*/",
        "sessions/*.json",
        "",
        "# Large files",
        "*.xml",
        "*.faa",
        "!data/examples/e_coli_core.xml",
        "!data/examples/glucose_minimal.*",
        "!data/examples/mycoplasma_minimal.*",
        "",
    ]

    with open(".gitignore", "a") as f:
        f.write("\n".join(gitignore_additions))

    print("\n✅ Repository cleanup completed!")
    print("\n📊 Summary:")
    print("• Moved test files to tests/manual/")
    print("• Archived testbed results")
    print("• Cleaned up old log files")
    print("• Removed duplicate and large model files")
    print("• Cleaned up session files")
    print("• Removed redundant documentation")
    print("• Updated .gitignore for future cleanup")


if __name__ == "__main__":
    main()
