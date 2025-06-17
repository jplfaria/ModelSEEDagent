#!/usr/bin/env python3
"""
ModelSEED Database Setup Script

This script automatically downloads and sets up the ModelSEED biochemistry database
for enhanced metabolic modeling capabilities. The database provides access to:
- 45,706+ compounds with chemical properties
- 56,009+ reactions with thermodynamic data
- Cross-references across 55+ databases (KEGG, BiGG, MetaCyc, ChEBI, etc.)
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


class ModelSEEDDatabaseSetup:
    """Handle ModelSEED database setup and verification"""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            # Default to repo root/data
            script_dir = Path(__file__).parent
            base_dir = script_dir.parent / "data"

        self.base_dir = Path(base_dir)
        self.database_dir = self.base_dir / "ModelSEEDDatabase"
        self.github_url = "https://github.com/ModelSEED/ModelSEEDDatabase.git"
        self.branch = "dev"

    def is_database_available(self) -> bool:
        """Check if ModelSEED database is already available"""
        return (
            self.database_dir.exists()
            and (self.database_dir / ".git").exists()
            and (self.database_dir / "Biochemistry").exists()
        )

    def check_git_available(self) -> bool:
        """Check if git is available on the system"""
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def check_modelseedpy_available(self) -> bool:
        """Check if modelseedpy is available"""
        try:
            import modelseedpy

            return True
        except ImportError:
            return False

    def clone_database(self) -> bool:
        """Clone the ModelSEED database repository"""
        try:
            print(f"📥 Cloning ModelSEED database from {self.github_url}")
            print(f"📂 Target directory: {self.database_dir}")

            # Ensure parent directory exists
            self.database_dir.parent.mkdir(parents=True, exist_ok=True)

            # Clone the repository
            cmd = [
                "git",
                "clone",
                "-b",
                self.branch,
                "--depth",
                "1",  # Shallow clone for faster download
                self.github_url,
                str(self.database_dir),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Successfully cloned ModelSEED database")
                return True
            else:
                print(f"❌ Failed to clone database: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error cloning database: {e}")
            return False

    def test_database_access(self) -> bool:
        """Test that the database can be loaded with modelseedpy"""
        try:
            import modelseedpy.biochem

            print("🧪 Testing database access...")
            database = modelseedpy.biochem.from_local(str(self.database_dir))

            # Basic verification
            compound_count = len(database.compounds)
            reaction_count = len(database.reactions)

            print(f"✅ Database loaded successfully!")
            print(f"   📊 Compounds: {compound_count:,}")
            print(f"   📊 Reactions: {reaction_count:,}")

            if compound_count < 40000 or reaction_count < 50000:
                print("⚠️  Warning: Database seems incomplete")
                return False

            return True

        except Exception as e:
            print(f"❌ Failed to load database: {e}")
            return False

    def setup_database(self) -> bool:
        """Complete database setup process"""
        print("🧬 ModelSEED Biochemistry Database Setup")
        print("=" * 50)

        # Check if already available
        if self.is_database_available():
            print("✅ ModelSEED database already available")
            if self.test_database_access():
                print("🎉 Database setup complete!")
                return True
            else:
                print("🔄 Database exists but seems corrupted, re-downloading...")
                # Remove corrupted database
                import shutil

                shutil.rmtree(self.database_dir, ignore_errors=True)

        # Check prerequisites
        if not self.check_modelseedpy_available():
            print("❌ modelseedpy not available. Please install with:")
            print("   pip install modelseedpy")
            return False

        if not self.check_git_available():
            print("❌ git not available. Please install git first.")
            return False

        # Clone database
        if not self.clone_database():
            print("❌ Failed to download ModelSEED database")
            return False

        # Test access
        if not self.test_database_access():
            print("❌ Database downloaded but cannot be accessed")
            return False

        print("🎉 ModelSEED database setup complete!")
        print("✨ Enhanced biochemical capabilities now available:")
        print("   - 45,706+ compounds with chemical properties")
        print("   - 56,009+ reactions with thermodynamic data")
        print("   - Cross-references across 55+ databases")
        print("   - Universal ID translation capabilities")

        return True

    def update_database(self) -> bool:
        """Update existing database to latest version"""
        if not self.is_database_available():
            print("❌ Database not found. Run setup first.")
            return False

        try:
            print("🔄 Updating ModelSEED database...")

            # Pull latest changes
            cmd = ["git", "pull", "origin", self.branch]
            result = subprocess.run(
                cmd, cwd=self.database_dir, capture_output=True, text=True
            )

            if result.returncode == 0:
                print("✅ Database updated successfully")
                return self.test_database_access()
            else:
                print(f"❌ Failed to update: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error updating database: {e}")
            return False


def main():
    """Main entry point for database setup"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup ModelSEED biochemistry database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_biochem_database.py          # Setup database
  python scripts/setup_biochem_database.py --update # Update existing database
  python scripts/setup_biochem_database.py --test   # Test database access
        """,
    )

    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing database to latest version",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test database access without setup"
    )
    parser.add_argument(
        "--data-dir", type=str, help="Custom data directory (default: repo/data)"
    )

    args = parser.parse_args()

    # Create setup instance
    setup = ModelSEEDDatabaseSetup(args.data_dir)

    # Execute requested action
    if args.test:
        if setup.is_database_available():
            success = setup.test_database_access()
        else:
            print("❌ Database not found. Run setup first.")
            success = False
    elif args.update:
        success = setup.update_database()
    else:
        success = setup.setup_database()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
