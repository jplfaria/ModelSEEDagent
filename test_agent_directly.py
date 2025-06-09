#!/usr/bin/env python3
"""
Direct test of agent to isolate the tool input issue
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

from src.llm.argo import ArgoLLM
from src.agents.factory import create_real_time_agent
from src.tools.cobra.fba import FBATool

def test_agent_directly():
    """Test the agent with the exact same setup as the CLI"""
    print("🧪 Testing agent directly with minimal setup...")
    
    # Create o1 configuration - same as CLI
    o1_config = {
        "model_name": "gpto1",
        "user": "jplfaria",
        "system_content": "You are an expert metabolic modeling assistant."
    }
    
    try:
        # Create LLM - same as CLI
        print("🔧 Creating LLM...")
        llm = ArgoLLM(o1_config)
        print(f"✅ LLM created: {llm.model_name}")
        
        # Create single tool for testing
        print("🔧 Creating FBA tool...")
        fba_tool = FBATool({
            "name": "run_metabolic_fba",
            "description": "Run FBA analysis"
        })
        tools = [fba_tool]
        print(f"✅ Created {len(tools)} tools")
        
        # Create agent - same as CLI
        print("🔧 Creating real-time agent...")
        agent_config = {"max_iterations": 6}
        agent = create_real_time_agent(llm, tools, agent_config)
        print(f"✅ Agent created: {type(agent).__name__}")
        
        # Test the exact query from CLI
        print("\n🧠 Testing with CLI query...")
        query = "I need a comprehensive metabolic analysis of E. coli"
        
        # Run the agent
        print(f"🚀 Running agent with query: '{query}'")
        print("⏳ This will take some time with o1 model...")
        
        # Run synchronously 
        import asyncio
        result = asyncio.run(agent.run({"query": query}))
        
        print(f"\n✅ Agent completed!")
        print(f"   Success: {result.success}")
        print(f"   Message: {result.message[:200]}...")
        
        if not result.success:
            print(f"   Error: {result.error}")
            
        return result.success
        
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Direct Agent Test\n")
    success = test_agent_directly()
    print(f"\n📋 Result: {'✅ SUCCESS' if success else '❌ FAILED'}")