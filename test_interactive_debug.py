#!/usr/bin/env python3
"""
Test the interactive CLI programmatically to debug the hanging issue
"""

import asyncio
import os
import sys
import time

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.interactive.conversation_engine import DynamicAIConversationEngine
from src.interactive.session_manager import AnalysisSession


def test_conversation_engine():
    """Test the conversation engine directly"""
    print("🧪 Testing conversation engine directly...")
    
    # Create session
    from datetime import datetime
    session = AnalysisSession(
        id="test_session",
        name="Debug Session", 
        created_at=datetime.now(),
        last_active=datetime.now()
    )
    
    # Create conversation engine
    engine = DynamicAIConversationEngine(session)
    
    print(f"🤖 AI Agent Available: {engine.ai_agent is not None}")
    
    if not engine.ai_agent:
        print("⚠️ No AI agent - testing will be limited")
        return
    
    # Test query that should trigger timeout
    query = "analyze the model at data/examples/e_coli_core.xml"
    
    print(f"\n🔍 Testing query: '{query}'")
    print("This should:")
    print("1. Start AI processing")
    print("2. Timeout after 30s")
    print("3. Use fallback logic")
    print("4. Execute tools")
    
    start_time = time.time()
    
    try:
        # This should trigger the same behavior as the interactive CLI
        response = engine.process_user_input(query)
        
        elapsed = time.time() - start_time
        
        print(f"\n⏱️ Completed in {elapsed:.1f} seconds")
        print(f"📝 Response type: {response.response_type}")
        print(f"📊 Processing time: {response.processing_time:.1f}s")
        print(f"✅ Success: {not hasattr(response, 'metadata') or not response.metadata.get('error')}")
        
        if hasattr(response, 'metadata') and response.metadata.get('tools_executed'):
            print(f"🔧 Tools executed: {response.metadata['tools_executed']}")
        
        print(f"\n📄 Response content (first 300 chars):")
        print(response.content[:300] + "..." if len(response.content) > 300 else response.content)
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n💥 Exception after {elapsed:.1f} seconds: {e}")
        import traceback
        traceback.print_exc()


def test_streaming_workflow():
    """Test if the issue is in the streaming workflow"""
    print("\n🧪 Testing streaming workflow directly...")
    
    from src.agents import create_real_time_agent
    from src.llm.factory import LLMFactory
    from src.tools import ToolRegistry
    
    # Try to create LLM (this might timeout/fail and that's OK)
    llm_config = {
        "model_name": "gpt-4o-mini",
        "system_content": "You are an expert metabolic modeling AI agent.",
        "temperature": 0.7,
        "max_tokens": 4000,
    }
    
    llm = None
    try:
        llm = LLMFactory.create("argo", llm_config)
        print("✅ Argo LLM created")
    except Exception:
        try:
            llm = LLMFactory.create("openai", llm_config)
            print("✅ OpenAI LLM created")
        except Exception:
            print("⚠️ No LLM available")
            return
    
    # Get tools
    tool_names = ["run_metabolic_fba", "find_minimal_media", "analyze_essentiality"]
    tools = []
    for tool_name in tool_names:
        try:
            tool = ToolRegistry.create_tool(tool_name, {})
            tools.append(tool)
        except Exception as e:
            print(f"⚠️ Could not load tool {tool_name}: {e}")
    
    if not tools:
        print("❌ No tools available")
        return
    
    # Create agent
    config = {"max_iterations": 3}
    agent = create_real_time_agent(llm, tools, config)
    
    print(f"🤖 Agent created with {len(tools)} tools")
    
    # Test query
    query = "analyze the model at data/examples/e_coli_core.xml"
    
    print(f"\n🔍 Testing agent run directly: '{query}'")
    
    async def run_agent():
        start_time = time.time()
        try:
            result = await agent.run({"query": query})
            elapsed = time.time() - start_time
            
            print(f"\n⏱️ Agent completed in {elapsed:.1f} seconds")
            print(f"✅ Success: {result.success}")
            
            if result.success:
                print(f"📝 Message: {result.message[:200]}...")
                tools_executed = result.metadata.get("tools_executed", [])
                if tools_executed:
                    print(f"🔧 Tools executed: {tools_executed}")
                else:
                    print("⚠️ No tools executed")
            else:
                print(f"❌ Error: {result.error}")
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n💥 Exception after {elapsed:.1f} seconds: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the async test
    asyncio.run(run_agent())


def main():
    """Run debug tests"""
    print("🔧 Interactive CLI Debug Test")
    print("=" * 50)
    
    # Test conversation engine (this is what interactive CLI uses)
    test_conversation_engine()
    
    print("\n" + "=" * 50)
    
    # Test agent directly
    test_streaming_workflow()
    
    print("\n✅ Debug tests complete!")


if __name__ == "__main__":
    main()