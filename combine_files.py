#!/usr/bin/env python3
import os
from pathlib import Path


def combine_files(output_file="combined_output.txt"):
    """
    Recursively finds all .py and .yaml files in the current directory and its subdirectories,
    then combines their contents into a single text file with file location indicators.
    """
    # Get the repository root directory
    repo_root = os.getcwd()

    # Get the name of this script
    current_script = os.path.basename(__file__)

    # Initialize list to store file paths
    target_files = []

    # Walk through all directories
    for root, _, files in os.walk(repo_root):
        # Skip virtual environment and ipynb checkpoints directories
        if any(
            ignore_dir in root.split(os.sep)
            for ignore_dir in ["venv", ".ipynb_checkpoints"]
        ):
            continue

        for file in files:
            # Skip this script itself
            if file == current_script:
                continue
            if file.endswith((".py", ".yaml", ".yml")):
                target_files.append(os.path.join(root, file))

    # Sort files for consistent output
    target_files.sort()

    # Process each file and write to output
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_path in target_files:
            # Get relative path from repo root
            rel_path = os.path.relpath(file_path, repo_root)

            # Write file header
            outfile.write(f"\n{'='*80}\n")
            outfile.write(f"FILE: {rel_path}\n")
            outfile.write(f"{'='*80}\n\n")

            # Read and write file contents
            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    outfile.write(content)

                # Add newline if file doesn't end with one
                if content and not content.endswith("\n"):
                    outfile.write("\n")

                # Write file footer
                outfile.write(f"\n{'#'*80}\n")
                outfile.write(f"END OF FILE: {rel_path}\n")
                outfile.write(f"{'#'*80}\n")

            except Exception as e:
                outfile.write(f"Error reading file: {str(e)}\n")


if __name__ == "__main__":
    combine_files()
    print("Files have been combined successfully!")
