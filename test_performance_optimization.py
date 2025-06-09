#!/usr/bin/env python3
"""
Test script to verify performance optimizations
"""
import time
import asyncio
import logging
from src.agents.real_time_metabolic import RealTimeMetabolicAgent
from src.agents.langgraph_metabolic import LangGraphMetabolicAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_real_time_agent():
    """Test RealTimeMetabolicAgent performance"""
    print("\n=== Testing RealTimeMetabolicAgent ===")
    
    config = {
        "llm_backend": "openai",
        "llm_config": {
            "model_name": "gpt-4o-mini",
            "max_tokens": 1000,
            "temperature": 0.3
        }
    }
    
    agent = RealTimeMetabolicAgent(config)
    
    # Test simple query
    start = time.time()
    result = await agent.run({"query": "What is the growth rate of E. coli?"})
    duration = time.time() - start
    
    print(f"✅ Simple query completed in {duration:.2f}s")
    print(f"   Success: {result.success}")
    print(f"   Tools executed: {len(agent.tool_execution_history)}")
    
    # Test complex query 
    agent2 = RealTimeMetabolicAgent(config)
    start = time.time()
    result2 = await agent2.run({"query": "I need a comprehensive metabolic analysis of E. coli"})
    duration2 = time.time() - start
    
    print(f"\n✅ Complex query completed in {duration2:.2f}s")
    print(f"   Success: {result2.success}")
    print(f"   Tools executed: {len(agent2.tool_execution_history)}")
    
def test_langgraph_agent():
    """Test LangGraphMetabolicAgent performance"""
    print("\n=== Testing LangGraphMetabolicAgent ===")
    
    config = {
        "llm_backend": "openai",
        "llm_config": {
            "model_name": "gpt-4o-mini",
            "max_tokens": 1000,
            "temperature": 0.3
        }
    }
    
    agent = LangGraphMetabolicAgent(config)
    
    # Test simple query
    start = time.time()
    result = agent.run({"query": "What is the growth rate of E. coli?"})
    duration = time.time() - start
    
    print(f"✅ Simple query completed in {duration:.2f}s")
    print(f"   Success: {result.success}")
    
    # Test complex query
    agent2 = LangGraphMetabolicAgent(config) 
    start = time.time()
    result2 = agent2.run({"query": "I need a comprehensive metabolic analysis of E. coli"})
    duration2 = time.time() - start
    
    print(f"\n✅ Complex query completed in {duration2:.2f}s")
    print(f"   Success: {result2.success}")

if __name__ == "__main__":
    # Test both agents
    asyncio.run(test_real_time_agent())
    test_langgraph_agent()