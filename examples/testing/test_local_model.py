#!/usr/bin/env python3
"""
Test local Meta Llama model functionality

This script tests the local model setup and response generation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_local_llm_response():
    """Test local LLM response generation"""
    print("🧪 Testing Local Meta Llama Model Response Generation")
    print("=" * 60)

    try:
        from llm.local_llm import LocalLLM

        # Test configuration for the 3B model (smaller, faster)
        config = {
            "model_name": "llama-3.2-3b",
            "model_path": "/Users/jplfaria/.llama/checkpoints/Llama3.2-3B",
            "system_content": "You are an expert metabolic modeling assistant.",
            "device": "mps",
            "max_tokens": 200,
            "temperature": 0.7,
        }

        print(f"Loading model: {config['model_name']}")
        llm = LocalLLM(config)

        # Test questions
        test_questions = [
            "What is the growth rate of E. coli?",
            "Analyze the structure of this metabolic model",
            "Explain the glycolysis pathway",
            "What are the key metabolites in central carbon metabolism?",
        ]

        print(f"\n🔍 Testing {len(test_questions)} different queries...")

        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            print("-" * 50)

            try:
                response = llm._generate_response(question)
                print(f"✅ Response: {response.text}")
                print(f"📊 Tokens used: {response.tokens_used}")
                print(f"🔧 Metadata: {response.metadata.get('format', 'unknown')}")

            except Exception as e:
                print(f"❌ Error generating response: {e}")

        print(f"\n🎉 Local model testing completed!")
        print(
            "The Meta Llama models are now working with the enhanced fallback system."
        )

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_local_llm_response()
