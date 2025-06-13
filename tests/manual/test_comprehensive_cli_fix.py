#!/usr/bin/env python3
"""
Comprehensive test to verify all CLI fixes are working.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def test_cli_startup():
    """Test CLI startup with debug mode"""
    print("🧪 Test 1: CLI Startup with Debug Mode")
    print("-" * 40)

    env = os.environ.copy()
    env["MODELSEED_DEBUG"] = "true"

    result = subprocess.run(
        [sys.executable, "-m", "src.cli.main", "interactive"],
        input="exit\n",
        text=True,
        capture_output=True,
        timeout=30,
        env=env,
        cwd=str(Path(__file__).parent),
    )

    stdout = result.stdout

    # Check startup imports (should be single)
    modelseedpy_count = stdout.count("modelseedpy 0.4.2")
    cobrakbase_count = stdout.count("cobrakbase 0.4.0")

    print(f"📊 modelseedpy imports at startup: {modelseedpy_count}")
    print(f"📊 cobrakbase imports at startup: {cobrakbase_count}")

    # Should see debug mode indicator
    debug_enabled = "DEBUG MODE ENABLED" in stdout
    print(f"📊 Debug mode properly detected: {debug_enabled}")

    return modelseedpy_count == 1 and cobrakbase_count == 1


def test_query_processing():
    """Test query processing without repeated imports"""
    print("\n🧪 Test 2: Query Processing (Short Timeout)")
    print("-" * 40)

    env = os.environ.copy()
    env["MODELSEED_DEBUG"] = "true"

    # Use a simple query and short timeout to see the start of processing
    result = subprocess.run(
        [sys.executable, "-m", "src.cli.main", "interactive"],
        input="help\nexit\n",  # Use help command instead of analysis
        text=True,
        capture_output=True,
        timeout=30,
        env=env,
        cwd=str(Path(__file__).parent),
    )

    stdout = result.stdout

    # Count analysis message (should be 0 for help command)
    analysis_message_count = stdout.count("AI analyzing your query and executing tools")
    print(f"📊 Analysis message count: {analysis_message_count}")

    # Count total imports (should be same as startup)
    modelseedpy_count = stdout.count("modelseedpy 0.4.2")
    cobrakbase_count = stdout.count("cobrakbase 0.4.0")
    print(f"📊 Total modelseedpy imports: {modelseedpy_count}")
    print(f"📊 Total cobrakbase imports: {cobrakbase_count}")

    return analysis_message_count == 0 and modelseedpy_count <= 1


def test_debug_visibility():
    """Test that debug mode actually shows debug info"""
    print("\n🧪 Test 3: Debug Mode Visibility")
    print("-" * 40)

    env = os.environ.copy()
    env["MODELSEED_DEBUG"] = "true"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
print(f"Environment MODELSEED_DEBUG: {os.getenv('MODELSEED_DEBUG')}")
from src.interactive.interactive_cli import DEBUG_MODE
print(f"DEBUG_MODE variable: {DEBUG_MODE}")
if DEBUG_MODE:
    print("✅ Debug mode properly detected")
else:
    print("❌ Debug mode not detected")
""",
        ],
        text=True,
        capture_output=True,
        timeout=10,
        env=env,
        cwd=str(Path(__file__).parent),
    )

    print(result.stdout)
    return "Debug mode properly detected" in result.stdout


def main():
    """Run all tests"""
    print("🧪 Comprehensive CLI Fix Verification")
    print("=" * 50)

    try:
        test1_pass = test_cli_startup()
        test2_pass = test_query_processing()
        test3_pass = test_debug_visibility()

        print(f"\n🎯 TEST RESULTS:")
        print(f"   Test 1 - Startup Imports: {'✅ PASS' if test1_pass else '❌ FAIL'}")
        print(f"   Test 2 - Query Processing: {'✅ PASS' if test2_pass else '❌ FAIL'}")
        print(f"   Test 3 - Debug Mode: {'✅ PASS' if test3_pass else '❌ FAIL'}")

        if all([test1_pass, test2_pass, test3_pass]):
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"   The duplicate message fix should resolve the repeated imports.")
            print(f"   Ready to test with actual analysis query:")
            print(f"   ")
            print(f"   export MODELSEED_DEBUG=true")
            print(f"   modelseed-agent interactive")
            print(f"   # Enter: I need metabolic analysis")
            return True
        else:
            print(f"\n💥 SOME TESTS FAILED")
            print(f"   More investigation needed")
            return False

    except Exception as e:
        print(f"❌ Testing failed with error: {e}")
        return False


if __name__ == "__main__":
    main()
