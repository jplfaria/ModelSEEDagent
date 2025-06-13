#!/usr/bin/env python3
"""
Test what happens during actual analysis query execution.
This should reveal if the agent execution itself causes repeated imports.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def test_analysis_imports():
    """Test imports during actual analysis execution with short timeout"""
    print("🧪 Testing Analysis Query Imports (30s timeout)")
    print("=" * 60)

    env = os.environ.copy()
    env["MODELSEED_DEBUG"] = "true"

    # Use actual analysis query but with short timeout to capture beginning
    analysis_query = "I need a comprehensive metabolic analysis of E. coli\nexit\n"

    print("🚀 Running with analysis query (30s timeout to see start of execution)...")
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.cli.main", "interactive"],
            input=analysis_query,
            text=True,
            capture_output=True,
            timeout=30,  # Short timeout to see what happens in first 30s
            env=env,
            cwd=str(Path(__file__).parent),
        )

        duration = time.time() - start_time
        print(f"⏱️ Completed in {duration:.2f}s")
        completed_normally = True

    except subprocess.TimeoutExpired as e:
        duration = 30.0
        print(f"⏱️ Timed out after {duration}s (as expected)")
        result = e
        completed_normally = False

        # Get partial output
        stdout = e.stdout if e.stdout else ""
        stderr = e.stderr if e.stderr else ""
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    if completed_normally:
        stdout = result.stdout

    # Analyze the output
    lines = stdout.split("\n")

    print(f"\n📊 IMPORT ANALYSIS:")

    # Count modelseedpy imports
    modelseedpy_lines = [
        i for i, line in enumerate(lines) if "modelseedpy 0.4.2" in line
    ]
    print(f"   modelseedpy imports: {len(modelseedpy_lines)}")
    if modelseedpy_lines:
        print(f"   Found at lines: {modelseedpy_lines}")

    # Count cobrakbase imports
    cobrakbase_lines = [i for i, line in enumerate(lines) if "cobrakbase 0.4.0" in line]
    print(f"   cobrakbase imports: {len(cobrakbase_lines)}")
    if cobrakbase_lines:
        print(f"   Found at lines: {cobrakbase_lines}")

    # Count analysis messages
    analysis_lines = [
        i for i, line in enumerate(lines) if "AI analyzing your query" in line
    ]
    print(f"   'AI analyzing' messages: {len(analysis_lines)}")
    if analysis_lines:
        print(f"   Found at lines: {analysis_lines}")

    # Look for debug messages
    debug_lines = [i for i, line in enumerate(lines) if "DEBUG MODE" in line]
    print(f"   Debug messages: {len(debug_lines)}")

    # Show sequence around imports to understand timing
    if len(modelseedpy_lines) > 1:
        print(f"\n📄 CONTEXT AROUND REPEATED IMPORTS:")
        for line_num in modelseedpy_lines:
            start_context = max(0, line_num - 2)
            end_context = min(len(lines), line_num + 3)
            print(f"   Around line {line_num}:")
            for i in range(start_context, end_context):
                marker = ">>> " if i == line_num else "    "
                print(f"   {marker}{i:3d}: {lines[i]}")
            print()

    # Show first part of output
    print(f"\n📄 FIRST 20 LINES OF OUTPUT:")
    for i, line in enumerate(lines[:20]):
        print(f"   {i:3d}: {line}")

    # Verdict
    if len(modelseedpy_lines) <= 1 and len(analysis_lines) <= 1:
        print(f"\n✅ IMPORTS LOOK GOOD - No repeated imports detected")
        return True
    else:
        print(f"\n❌ REPEATED IMPORTS DETECTED")
        print(
            f"   This suggests the agent execution is still causing repeated module loads"
        )
        return False


if __name__ == "__main__":
    success = test_analysis_imports()
    if not success:
        print(f"\n💡 Next steps:")
        print(f"   1. The agent execution itself might be triggering repeated imports")
        print(f"   2. Could be tool loading during agent run")
        print(f"   3. Could be async/threading causing module reloads")
        print(f"   4. Need to investigate agent._tools_dict usage during execution")
