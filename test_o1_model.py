#!/usr/bin/env python3
"""
Direct test of o1 model to verify it's working
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.llm.argo import ArgoLLM

def test_o1_model():
    """Test o1 model directly"""
    print("🧪 Testing o1 model directly...")
    
    # Create o1 configuration
    o1_config = {
        "model_name": "gpto1",
        "user": "jplfaria",
        "system_content": "You are a helpful assistant."
    }
    
    try:
        # Create LLM instance
        print("🔧 Creating ArgoLLM instance...")
        llm = ArgoLLM(o1_config)
        print(f"✅ ArgoLLM created successfully")
        print(f"   Model: {llm.model_name}")
        print(f"   Timeout: {llm._timeout}s")
        
        # Test simple query
        print("\n🤔 Testing simple query...")
        test_query = "What is 2+2? Please answer briefly."
        
        print(f"   Query: {test_query}")
        print("   Waiting for response...")
        
        response = llm._generate_response(test_query)
        
        print(f"✅ Response received!")
        print(f"   Type: {type(response)}")
        print(f"   Content: {response.text[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_o1_with_metabolic_query():
    """Test o1 model with a metabolic query"""
    print("\n🧪 Testing o1 model with metabolic query...")
    
    o1_config = {
        "model_name": "gpto1",
        "user": "jplfaria",
        "system_content": "You are an expert metabolic modeling assistant."
    }
    
    try:
        llm = ArgoLLM(o1_config)
        
        metabolic_query = "What tools would you use to analyze E. coli metabolism? List 3 tools."
        
        print(f"   Query: {metabolic_query}")
        print("   Waiting for response...")
        
        response = llm._generate_response(metabolic_query)
        
        print(f"✅ Metabolic response received!")
        print(f"   Content: {response.text[:300]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Metabolic test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting o1 Model Direct Tests\n")
    
    # Test 1: Basic functionality
    basic_success = test_o1_model()
    
    # Test 2: Metabolic query
    metabolic_success = test_o1_with_metabolic_query()
    
    print("\n📋 Test Results:")
    print(f"   Basic test: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"   Metabolic test: {'✅ PASS' if metabolic_success else '❌ FAIL'}")
    
    if basic_success and metabolic_success:
        print("\n🎉 o1 model is working correctly!")
        print("💡 The CLI issue is definitely in the tool input preparation.")
    else:
        print("\n⚠️ o1 model has issues - need to fix LLM connection first!")