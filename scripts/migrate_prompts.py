#!/usr/bin/env python3
"""
Run prompt migration to centralize all scattered prompts

This script migrates all 28+ identified prompts from various files
into the centralized prompt registry system.
"""

import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.prompts.migration_script import PromptMigrationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "logs" / "prompt_migration.log"),
    ],
)

logger = logging.getLogger(__name__)


def main():
    """Run the complete prompt migration process"""
    logger.info("🚀 Starting ModelSEEDagent Prompt Migration")
    logger.info("=" * 60)

    try:
        # Create logs directory if it doesn't exist
        (project_root / "logs").mkdir(exist_ok=True)

        # Initialize migration manager
        migration_manager = PromptMigrationManager()

        # Run migration
        logger.info("📋 Beginning migration of 28+ prompts...")
        success = migration_manager.migrate_all_prompts()

        # Generate report
        report = migration_manager.get_migration_report()

        # Display results
        logger.info("=" * 60)
        logger.info("📊 MIGRATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total prompts processed: {report['total_prompts']}")
        logger.info(f"Successful migrations: {report['successful_migrations']}")
        logger.info(f"Failed migrations: {report['failed_migrations']}")
        logger.info(f"Success rate: {report['success_rate']:.1%}")

        if success:
            logger.info("✅ Migration completed successfully!")
            logger.info("🎯 All prompts are now centralized and ready for Phase 2")
        else:
            logger.warning("⚠️ Migration completed with some issues")
            logger.info("📝 Check migration log for details")

        # Save detailed report
        report_file = project_root / "logs" / "migration_report.json"
        import json

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"📄 Detailed report saved to: {report_file}")

        # Verify registry status
        registry_stats = report["registry_stats"]
        logger.info(
            f"📚 Registry now contains {registry_stats['prompt_count']} prompts"
        )

        return 0 if success else 1

    except Exception as e:
        logger.error(f"💥 Migration failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
