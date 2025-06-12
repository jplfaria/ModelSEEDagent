#!/usr/bin/env python3
"""
Automated log cleanup script for ModelSEEDagent

This script implements a retention policy to keep the logs directory manageable:
- Keep last 10 LangGraph runs
- Keep last 5 RealTime runs
- Archive older runs to compressed format
- Remove runs older than 30 days from archive
"""

import os
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path


def cleanup_logs(logs_dir: Path, dry_run: bool = False):
    """Clean up log directories according to retention policy"""
    print(f"🧹 Cleaning up logs in: {logs_dir}")

    # Create archive directory
    archive_dir = logs_dir / "archive"
    archive_dir.mkdir(exist_ok=True)

    # Find all run directories
    langgraph_runs = sorted(logs_dir.glob("langgraph_run_*"))
    realtime_runs = sorted(logs_dir.glob("realtime_run_*"))

    print(
        f"📊 Found {len(langgraph_runs)} LangGraph runs, {len(realtime_runs)} RealTime runs"
    )

    # Archive older runs
    to_archive = []

    # Keep last 10 LangGraph runs
    if len(langgraph_runs) > 10:
        to_archive.extend(langgraph_runs[:-10])

    # Keep last 5 RealTime runs
    if len(realtime_runs) > 5:
        to_archive.extend(realtime_runs[:-5])

    if to_archive:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"archived_runs_{timestamp}"
        archive_path = archive_dir / archive_name

        if not dry_run:
            archive_path.mkdir(exist_ok=True)

            for run_dir in to_archive:
                print(f"📦 Archiving: {run_dir.name}")
                shutil.move(str(run_dir), str(archive_path))

            # Compress archive
            with tarfile.open(f"{archive_path}.tar.gz", "w:gz") as tar:
                tar.add(archive_path, arcname=archive_name)

            # Remove uncompressed archive
            shutil.rmtree(archive_path)

            print(f"✅ Archived {len(to_archive)} runs to {archive_name}.tar.gz")
        else:
            print(f"🔍 Would archive {len(to_archive)} runs")

    # Clean old archives (older than 30 days)
    cutoff_date = datetime.now() - timedelta(days=30)
    old_archives = []

    for archive_file in archive_dir.glob("*.tar.gz"):
        if archive_file.stat().st_mtime < cutoff_date.timestamp():
            old_archives.append(archive_file)

    if old_archives:
        if not dry_run:
            for archive_file in old_archives:
                print(f"🗑️  Removing old archive: {archive_file.name}")
                archive_file.unlink()
        else:
            print(f"🔍 Would remove {len(old_archives)} old archives")

    # Report final stats
    remaining_runs = len(list(logs_dir.glob("*run_*")))
    total_size = sum(f.stat().st_size for f in logs_dir.rglob("*") if f.is_file())

    print(f"📈 Final state: {remaining_runs} runs, {total_size / (1024*1024):.1f}MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean up ModelSEEDagent logs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--logs-dir", type=Path, default="logs", help="Path to logs directory"
    )

    args = parser.parse_args()

    cleanup_logs(args.logs_dir, args.dry_run)
