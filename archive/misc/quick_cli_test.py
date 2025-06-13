#!/usr/bin/env python3
"""
Quick test to verify the interactive CLI is working without hanging.
"""

import subprocess
import sys
import time
from pathlib import Path


def quick_test():
    """Quick test of the interactive CLI"""
    print("🧪 Quick Interactive CLI Test")
    print("=" * 30)

    # Test with a simple command that should exit quickly
    test_input = "exit\n"

    cmd = [sys.executable, "-m", "src.cli.main", "interactive"]

    print("🚀 Starting CLI with exit command...")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            input=test_input,
            text=True,
            capture_output=True,
            timeout=30,  # 30 second timeout
            cwd=str(Path(__file__).parent),
        )

        duration = time.time() - start_time
        print(f"⏱️ Completed in {duration:.2f}s")
        print(f"📊 Return code: {result.returncode}")

        # Check if it starts properly
        if "ModelSEEDagent Interactive Analysis" in result.stdout:
            print("✅ CLI starts properly")
        else:
            print("❌ CLI startup issue")

        # Check if it exits cleanly
        if "Thank you for using ModelSEEDagent" in result.stdout:
            print("✅ CLI exits cleanly")
        else:
            print("⚠️ CLI exit might be abrupt")

        # Check for hanging indicators
        hanging_patterns = ["modelseedpy 0.4.2" * 3, "cobrakbase 0.4.0" * 3]
        has_hanging = any(pattern in result.stdout for pattern in hanging_patterns)

        if has_hanging:
            print("❌ Still shows hanging patterns")
        else:
            print("✅ No hanging patterns detected")

        if duration < 10 and not has_hanging:
            print("🎉 QUICK TEST PASSED: CLI appears to be working")
            return True
        else:
            print("❌ QUICK TEST FAILED: CLI has issues")
            return False

    except subprocess.TimeoutExpired:
        print("❌ CLI timed out (still hanging)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n💡 You can now test manually with:")
        print("   export MODELSEED_DEBUG=true  # For detailed logging")
        print("   modelseed-agent interactive")
    else:
        print("\n💡 More debugging needed")
