#!/usr/bin/env python3
"""
Consolidate Incremental Results
==============================

This script consolidates incremental testbed results into one comprehensive file
and then splits them into individual tool result files.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def consolidate_and_split_results():
    """Consolidate incremental results and split into individual files"""
    
    print("🔄 Consolidating incremental testbed results...")
    
    # Find incremental result files
    testbed_dir = Path("testbed_results")
    incremental_files = list(testbed_dir.glob("*_incremental_*_results.json"))
    
    if not incremental_files:
        print("❌ No incremental result files found!")
        return
    
    print(f"📁 Found {len(incremental_files)} incremental files")
    
    # Load and consolidate results
    consolidated_results = {}
    all_metadata = {}
    
    for file_path in sorted(incremental_files):
        print(f"📖 Loading {file_path.name}")
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Extract model name and results
        model_name = data["metadata"]["model_completed"]
        consolidated_results[model_name] = data["results"][model_name]
        
        # Store metadata from the first file
        if not all_metadata:
            all_metadata = data["metadata"].copy()
            all_metadata["models_tested"] = []
        
        all_metadata["models_tested"].append(model_name)
    
    # Update metadata
    all_metadata["timestamp"] = datetime.now().isoformat()
    all_metadata["testbed_version"] = "comprehensive_v1_consolidated"
    
    # Calculate success statistics
    total_tests = 0
    successful_tests = 0
    
    for model_results in consolidated_results.values():
        for tool_result in model_results.values():
            total_tests += 1
            if tool_result.get("success", False):
                successful_tests += 1
    
    all_metadata["total_tests"] = total_tests
    all_metadata["successful_tests"] = successful_tests
    all_metadata["success_rate"] = successful_tests / total_tests if total_tests > 0 else 0
    
    # Create comprehensive results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comprehensive_file = testbed_dir / f"{timestamp}_comprehensive_testbed_results.json"
    
    comprehensive_data = {
        "metadata": all_metadata,
        "results": consolidated_results
    }
    
    with open(comprehensive_file, "w") as f:
        json.dump(comprehensive_data, f, indent=2, default=str)
    
    print(f"💾 Consolidated results saved: {comprehensive_file}")
    print(f"📊 Total tests: {total_tests}, Success rate: {100*all_metadata['success_rate']:.1f}%")
    
    # Now split into individual files
    from split_comprehensive_results import ComprehensiveResultsAnalyzer
    
    print("\n🔄 Splitting into individual tool result files...")
    
    analyzer = ComprehensiveResultsAnalyzer()
    individual_files, analyses, summary_file = analyzer.split_comprehensive_results(str(comprehensive_file))
    
    print(f"✅ Split complete!")
    print(f"📄 {len(individual_files)} individual tool files created")
    print(f"📊 {len(analyses)} analysis files generated")
    print(f"📋 Summary: {summary_file}")
    
    return comprehensive_file, individual_files

if __name__ == "__main__":
    consolidate_and_split_results()