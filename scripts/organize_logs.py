#!/usr/bin/env python3
"""
Log Organization Script

Manages log retention and organization:
- Keeps recent logs accessible
- Archives older logs with compression
- Implements automatic cleanup
"""

import gzip
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path


def setup_logging():
    """Setup logging for this script"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def organize_logs(logs_dir="logs", max_recent=10, archive_days=30):
    """
    Organize logs with retention policy

    Args:
        logs_dir: Directory containing logs
        max_recent: Number of recent runs to keep unarchived
        archive_days: Days to keep in archive before deletion
    """
    logger = setup_logging()
    logs_path = Path(logs_dir)

    if not logs_path.exists():
        logger.info(f"Logs directory {logs_path} does not exist")
        return

    # Create organization structure
    recent_dir = logs_path / "recent"
    archive_dir = logs_path / "archive"

    recent_dir.mkdir(exist_ok=True)
    archive_dir.mkdir(exist_ok=True)

    # Find all run directories
    run_dirs = [
        d for d in logs_path.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]

    # Sort by creation time (newest first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    logger.info(f"Found {len(run_dirs)} run directories")

    # Keep recent runs
    recent_count = 0
    archived_count = 0

    for run_dir in run_dirs:
        if recent_count < max_recent:
            # Keep in recent
            recent_path = recent_dir / run_dir.name
            if not recent_path.exists():
                shutil.move(str(run_dir), str(recent_path))
                logger.info(f"Moved {run_dir.name} to recent/")
            recent_count += 1
        else:
            # Archive older runs
            archive_run(run_dir, archive_dir, logger)
            archived_count += 1

    # Clean up old archives
    cleanup_old_archives(archive_dir, archive_days, logger)

    logger.info(
        f"Organization complete: {recent_count} recent, {archived_count} archived"
    )


def archive_run(run_dir, archive_dir, logger):
    """Archive a run directory with compression"""
    try:
        # Create compressed archive
        archive_name = f"{run_dir.name}.tar.gz"
        archive_path = archive_dir / archive_name

        if not archive_path.exists():
            shutil.make_archive(
                str(archive_dir / run_dir.name),
                "gztar",
                str(run_dir.parent),
                str(run_dir.name),
            )
            logger.info(f"Archived {run_dir.name} to {archive_name}")

        # Remove original
        shutil.rmtree(run_dir)

    except Exception as e:
        logger.error(f"Failed to archive {run_dir.name}: {e}")


def cleanup_old_archives(archive_dir, max_age_days, logger):
    """Remove archives older than max_age_days"""
    cutoff_date = datetime.now() - timedelta(days=max_age_days)

    for archive_file in archive_dir.glob("*.tar.gz"):
        file_time = datetime.fromtimestamp(archive_file.stat().st_mtime)

        if file_time < cutoff_date:
            archive_file.unlink()
            logger.info(f"Deleted old archive: {archive_file.name}")


def get_logs_summary(logs_dir="logs"):
    """Get summary of current logs organization"""
    logger = setup_logging()
    logs_path = Path(logs_dir)

    summary = {"recent_runs": 0, "archived_runs": 0, "total_size_mb": 0}

    if not logs_path.exists():
        return summary

    # Count recent runs
    recent_dir = logs_path / "recent"
    if recent_dir.exists():
        summary["recent_runs"] = len([d for d in recent_dir.iterdir() if d.is_dir()])

    # Count archived runs
    archive_dir = logs_path / "archive"
    if archive_dir.exists():
        summary["archived_runs"] = len(list(archive_dir.glob("*.tar.gz")))

    # Calculate total size
    for item in logs_path.rglob("*"):
        if item.is_file():
            summary["total_size_mb"] += item.stat().st_size / (1024 * 1024)

    logger.info(
        f"Logs summary: {summary['recent_runs']} recent, "
        f"{summary['archived_runs']} archived, "
        f"{summary['total_size_mb']:.1f} MB total"
    )

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Organize ModelSEEDagent logs")
    parser.add_argument("--logs-dir", default="logs", help="Logs directory path")
    parser.add_argument(
        "--max-recent", type=int, default=10, help="Number of recent runs to keep"
    )
    parser.add_argument(
        "--archive-days", type=int, default=30, help="Days to keep archived logs"
    )
    parser.add_argument("--summary", action="store_true", help="Show logs summary only")

    args = parser.parse_args()

    if args.summary:
        get_logs_summary(args.logs_dir)
    else:
        organize_logs(args.logs_dir, args.max_recent, args.archive_days)
