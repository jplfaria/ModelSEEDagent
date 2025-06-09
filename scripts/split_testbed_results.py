#!/usr/bin/env python3
"""
Split Testbed Results
====================

This script takes the large combined testbed results JSON file and splits it into
individual JSON files for each tool and model combination for easier analysis.

Output structure:
- testbed_results/individual/
  - e_coli_core/
    - FBA_results.json
    - FluxVariability_results.json
    - etc.
  - iML1515/
    - FBA_results.json
    - FluxVariability_results.json
    - etc.
"""

import json
import os
from datetime import datetime
from pathlib import Path


def split_testbed_results(input_file_path):
    """Split combined testbed results into individual tool/model files"""

    # Load the combined results
    with open(input_file_path, "r") as f:
        data = json.load(f)

    # Create output directory structure
    base_output_dir = Path("testbed_results/individual")
    base_output_dir.mkdir(exist_ok=True)

    timestamp = data["metadata"]["timestamp"]
    models_tested = data["metadata"]["models_tested"]
    tools_tested = data["metadata"]["tools_tested"]

    print(f"🔄 Splitting testbed results from {timestamp}")
    print(
        f"📊 Processing {len(models_tested)} models × {len(tools_tested)} tools = {len(models_tested) * len(tools_tested)} files"
    )
    print("=" * 60)

    files_created = []

    # Process each model
    for model_name in models_tested:
        model_dir = base_output_dir / model_name
        model_dir.mkdir(exist_ok=True)

        print(f"\n🧬 Processing {model_name.upper()} model:")

        model_results = data["results"][model_name]

        # Process each tool for this model
        for tool_name, tool_result in model_results.items():

            # Create individual tool result file
            output_filename = f"{tool_name}_results.json"
            output_path = model_dir / output_filename

            # Create enhanced individual result with metadata
            individual_result = {
                "metadata": {
                    "original_timestamp": timestamp,
                    "model_name": model_name,
                    "tool_name": tool_name,
                    "split_timestamp": datetime.now().isoformat(),
                    "success": tool_result["success"],
                    "execution_time": tool_result["execution_time"],
                },
                "test_parameters": {
                    "tool": tool_result["tool"],
                    "model": tool_result["model"],
                    "model_path": tool_result["model_path"],
                },
                "execution_details": {
                    "success": tool_result["success"],
                    "execution_time": tool_result["execution_time"],
                    "error": tool_result["error"],
                    "solver_messages": tool_result["solver_messages"],
                    "warnings": tool_result["warnings"],
                },
                "tool_output": tool_result["output"],
                "debug_info": {
                    "detailed_output": tool_result["detailed_output"],
                    "debug_info": tool_result.get("debug_info", {}),
                },
            }

            # Write individual file
            with open(output_path, "w") as f:
                json.dump(individual_result, f, indent=2, default=str)

            # Track file creation
            files_created.append(str(output_path))

            # Show progress
            status = "✅" if tool_result["success"] else "❌"
            time_str = f"{tool_result['execution_time']:.2f}s"
            file_size = os.path.getsize(output_path) / 1024

            print(
                f"  {status} {tool_name:<18} {time_str:>8} → {output_filename} ({file_size:.1f} KB)"
            )

    # Create summary file
    summary_file = base_output_dir / "split_summary.json"
    summary = {
        "split_metadata": {
            "original_file": str(input_file_path),
            "split_timestamp": datetime.now().isoformat(),
            "original_timestamp": timestamp,
            "models_processed": models_tested,
            "tools_processed": tools_tested,
            "total_files_created": len(files_created),
        },
        "file_structure": {},
    }

    # Organize files by model for summary
    for model_name in models_tested:
        model_files = [f for f in files_created if f"/{model_name}/" in f]
        summary["file_structure"][model_name] = {
            "directory": f"testbed_results/individual/{model_name}/",
            "files": [os.path.basename(f) for f in model_files],
            "file_count": len(model_files),
        }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n📁 Individual results saved in: {base_output_dir}/")
    print(f"📄 Summary saved in: {summary_file}")
    print(f"🎉 Created {len(files_created)} individual tool result files")

    # Show directory structure
    print(f"\n📂 Directory structure:")
    for model_name in models_tested:
        print(f"  📁 {model_name}/")
        model_files = [f for f in files_created if f"/{model_name}/" in f]
        for file_path in sorted(model_files):
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / 1024
            print(f"    📄 {filename} ({file_size:.1f} KB)")

    return files_created, summary_file


def main():
    """Main execution function"""

    print("🔄 Testbed Results Splitter")
    print("=" * 40)

    # Find the latest testbed results file
    testbed_dir = Path("testbed_results")
    result_files = list(testbed_dir.glob("*_tool_testbed_results.json"))

    if not result_files:
        print("❌ No testbed results files found!")
        return

    # Use the latest file
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)

    print(f"📁 Input file: {latest_file}")
    print(f"📏 File size: {latest_file.stat().st_size / 1024:.1f} KB")

    try:
        files_created, summary_file = split_testbed_results(latest_file)

        print(
            f"\n✅ Successfully split results into {len(files_created)} individual files"
        )
        print(f"🔍 Use these individual files for detailed tool analysis")

    except Exception as e:
        print(f"❌ Error splitting results: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
