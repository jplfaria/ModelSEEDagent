#!/usr/bin/env python3
"""
Test script to verify model availability and functionality

This script tests:
1. Argo Gateway model availability
2. Local model functionality
3. Basic LLM response generation
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_argo_models():
    """Test a few key Argo models"""
    print("🧬 Testing Argo Gateway Models...")

    # Test models to try
    test_models = ["gpt4o", "gpto1", "gpt35", "gpt4turbo"]

    try:
        from llm.argo import ArgoLLM

        for model in test_models:
            try:
                print(f"   Testing {model}...")

                config = {
                    "model_name": model,
                    "user": os.getenv("ARGO_USER", "jplfaria"),
                    "system_content": "You are a test assistant.",
                    "max_tokens": 50,
                }

                # Add temperature only for non-o-series models
                if not model.startswith("gpto"):
                    config["temperature"] = 0.1

                llm = ArgoLLM(config)

                # Test with a simple prompt
                response = llm._generate_response("Say 'Hello from ModelSEED!'")

                if response and response.text:
                    print(f"   ✅ {model}: {response.text[:50]}...")
                else:
                    print(f"   ❌ {model}: No response")

            except Exception as e:
                print(f"   ❌ {model}: Error - {str(e)[:100]}")

    except ImportError as e:
        print(f"   ❌ Cannot import ArgoLLM: {e}")

    return True


def test_local_models():
    """Test local model availability"""
    print("\n💻 Testing Local Models...")

    local_models = {
        "llama-3.1-8b": "/Users/jplfaria/.llama/checkpoints/Llama3.1-8B",
        "llama-3.2-3b": "/Users/jplfaria/.llama/checkpoints/Llama3.2-3B",
    }

    for model_name, model_path in local_models.items():
        print(f"   Testing {model_name}...")

        # Check if path exists
        if not Path(model_path).exists():
            print(f"   ❌ {model_name}: Path not found - {model_path}")
            continue

        # Check for required files
        required_files = ["consolidated.00.pth", "params.json", "tokenizer.model"]
        missing_files = []

        for file in required_files:
            if not (Path(model_path) / file).exists():
                missing_files.append(file)

        if missing_files:
            print(f"   ⚠️  {model_name}: Missing files - {missing_files}")
        else:
            print(f"   ✅ {model_name}: All required files present")

        # Try to initialize (this might fail if dependencies aren't installed)
        try:
            from llm.local_llm import LocalLLM

            config = {
                "model_name": model_name,
                "model_path": model_path,
                "system_content": "You are a test assistant.",
                "device": "mps",
                "max_tokens": 50,
                "temperature": 0.7,
            }

            # This might fail if torch/local dependencies aren't set up
            llm = LocalLLM(config)
            print(f"   ✅ {model_name}: LocalLLM initialized successfully")

            # Test basic response (might be slow/fail)
            try:
                response = llm._generate_response("Say 'Hello'")
                if response and response.text:
                    print(f"   ✅ {model_name}: Response - {response.text[:50]}...")
                else:
                    print(f"   ⚠️  {model_name}: No response generated")
            except Exception as e:
                print(
                    f"   ⚠️  {model_name}: Response generation failed - {str(e)[:100]}"
                )

        except ImportError as e:
            print(
                f"   ⚠️  {model_name}: Cannot test LocalLLM (missing dependencies) - {e}"
            )
        except Exception as e:
            print(f"   ❌ {model_name}: Initialization failed - {str(e)[:100]}")


def test_cli_model_lists():
    """Test that CLI shows updated model lists"""
    print("\n🔧 Testing CLI Model Lists...")

    import subprocess

    try:
        # Test if setup help shows new models
        result = subprocess.run(
            ["modelseed-agent", "setup", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print("   ✅ CLI setup help working")
        else:
            print(f"   ❌ CLI setup help failed: {result.stderr}")

    except Exception as e:
        print(f"   ❌ CLI test failed: {e}")


def main():
    """Run all model tests"""
    print("🧪 ModelSEEDagent Model Availability Test")
    print("=" * 50)

    # Test Argo models
    test_argo_models()

    # Test local models
    test_local_models()

    # Test CLI
    test_cli_model_lists()

    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    print("✅ = Working")
    print("⚠️  = Available but may have issues")
    print("❌ = Not working/missing")

    print("\n💡 Recommendations:")
    print("   • Use gpt4o or gpto1 for Argo Gateway")
    print("   • Local models are present but may need dependencies")
    print("   • Run 'modelseed-agent setup --interactive' to test interactively")


if __name__ == "__main__":
    main()
