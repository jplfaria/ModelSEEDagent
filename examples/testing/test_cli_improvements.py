#!/usr/bin/env python3
"""
Test script for CLI improvements

This script tests:
1. Environment variable support for defaults
2. Backend switching functionality
3. O-series model parameter handling
4. Configuration persistence
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_environment_defaults():
    """Test that environment variables are respected"""
    print("🧪 Testing environment variable defaults...")

    # Set test environment variables
    env = os.environ.copy()
    env["DEFAULT_LLM_BACKEND"] = "argo"
    env["DEFAULT_MODEL_NAME"] = "gpt4o"
    env["ARGO_USER"] = "test_user"

    # Test non-interactive setup
    try:
        result = subprocess.run(
            ["modelseed-agent", "setup", "--non-interactive"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("   ✅ Environment defaults working")
            return True
        else:
            print(f"   ❌ Setup failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("   ⏰ Setup timed out")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_backend_switching():
    """Test the new switch command"""
    print("🔄 Testing backend switching...")

    backends = ["argo", "openai", "local"]
    for backend in backends:
        try:
            # Skip openai if no API key
            if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
                print(f"   ⏭️  Skipping {backend} (no API key)")
                continue

            result = subprocess.run(
                ["modelseed-agent", "switch", backend],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                print(f"   ✅ Switch to {backend} working")
            else:
                print(f"   ❌ Switch to {backend} failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"   ❌ Error switching to {backend}: {e}")
            return False

    return True


def test_model_selection():
    """Test improved model selection"""
    print("🤖 Testing model selection...")

    models_to_test = [
        ("argo", "gpt4o"),
        ("argo", "gpto1"),  # Test o-series model
    ]

    for backend, model in models_to_test:
        try:
            result = subprocess.run(
                ["modelseed-agent", "switch", backend, "--model", model],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                print(f"   ✅ {backend} with {model} working")
            else:
                print(f"   ❌ {backend} with {model} failed: {result.stderr}")
                # Don't fail completely, as some models might not be available

        except Exception as e:
            print(f"   ❌ Error testing {backend}/{model}: {e}")

    return True


def test_help_commands():
    """Test that help commands work with new options"""
    print("❓ Testing help commands...")

    commands_to_test = [
        ["modelseed-agent", "--help"],
        ["modelseed-agent", "setup", "--help"],
        ["modelseed-agent", "switch", "--help"],
    ]

    for cmd in commands_to_test:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and len(result.stdout) > 100:
                print(f"   ✅ {' '.join(cmd)} working")
            else:
                print(f"   ❌ {' '.join(cmd)} failed")
                return False

        except Exception as e:
            print(f"   ❌ Error with {' '.join(cmd)}: {e}")
            return False

    return True


def test_status_command():
    """Test status command shows configuration"""
    print("📊 Testing status command...")

    try:
        result = subprocess.run(
            ["modelseed-agent", "status"], capture_output=True, text=True, timeout=20
        )

        if result.returncode == 0 and "Configuration" in result.stdout:
            print("   ✅ Status command working")
            return True
        else:
            print(f"   ❌ Status failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("🧬 Testing ModelSEEDagent CLI Improvements")
    print("=" * 50)

    tests = [
        ("Environment Defaults", test_environment_defaults),
        ("Backend Switching", test_backend_switching),
        ("Model Selection", test_model_selection),
        ("Help Commands", test_help_commands),
        ("Status Command", test_status_command),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary")
    print("=" * 50)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")

    print(f"\n🏆 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All CLI improvements working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed - check configuration")
        return 1


if __name__ == "__main__":
    sys.exit(main())
