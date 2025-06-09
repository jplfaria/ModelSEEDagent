#!/usr/bin/env python3
"""
Test script to verify the fixed interactive CLI.
This simulates the interactive CLI logic without the full interface.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.interactive.conversation_engine import DynamicAIConversationEngine
from src.interactive.session_manager import AnalysisSession


async def test_fixed_interactive():
    """Test the fixed interactive CLI logic."""
    print("🧪 Testing Fixed Interactive CLI Logic")
    print("=" * 50)
    
    try:
        # Create session with required parameters
        from datetime import datetime
        import uuid
        
        session = AnalysisSession(
            id=str(uuid.uuid4())[:8],
            name="test_session",
            created_at=datetime.now(),
            last_active=datetime.now(),
            description="Test session for verifying fixes"
        )
        print("✅ Created test session")
        
        # Create conversation engine
        engine = DynamicAIConversationEngine(session)
        print("✅ Created conversation engine")
        
        # Start conversation
        greeting = engine.start_conversation()
        print("✅ Got greeting response")
        print(f"   Preview: {greeting.content[:100]}...")
        
        # Test a query
        query = "I need a comprehensive metabolic analysis of E. coli"
        print(f"\n🔍 Testing query: '{query}'")
        
        response = engine.process_user_input(query)
        print(f"✅ Got response: success={response.response_type}")
        print(f"   Processing time: {response.processing_time:.2f}s")
        print(f"   AI reasoning steps: {response.ai_reasoning_steps}")
        print(f"   Content preview: {response.content[:150]}...")
        
        if response.metadata.get("ai_agent_result"):
            tools_executed = response.metadata.get("tools_executed", [])
            print(f"   Tools executed: {tools_executed}")
        
        print("\n🎉 Interactive CLI test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_fixed_interactive())