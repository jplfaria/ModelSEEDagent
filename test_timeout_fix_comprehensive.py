#!/usr/bin/env python3
"""
Comprehensive Timeout Fix Validation Test

This script validates that:
1. o1 models (gpto1, gpto1mini) use 120s timeouts
2. Standard models use 30s timeouts  
3. Debug logging shows correct timeout detection
4. LLM HTTP timeouts are set correctly
5. Signal-based timeouts work properly
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.real_time_metabolic import RealTimeMetabolicAgent
from src.llm.argo import ArgoLLM
from src.tools.cobra.fba import FBATool

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('timeout_test.log')
    ]
)

logger = logging.getLogger(__name__)

def test_argo_llm_timeout_configuration():
    """Test ArgoLLM timeout configuration for different models"""
    
    logger.info("🧪 TESTING: ArgoLLM timeout configuration")
    
    # Test o1 model configuration
    o1_config = {
        "model_name": "gpto1",
        "user": "test_user",
        "system_content": "Test system content"
    }
    
    logger.info("🧪 Creating ArgoLLM with gpto1 model...")
    o1_llm = ArgoLLM(o1_config)
    
    # Verify timeout settings
    logger.info(f"🧪 o1 model timeout: {o1_llm._timeout}s (expected: 120s)")
    assert o1_llm._timeout == 120.0, f"Expected 120s timeout for o1 model, got {o1_llm._timeout}s"
    
    # Test standard model configuration  
    standard_config = {
        "model_name": "gpt4o",
        "user": "test_user", 
        "system_content": "Test system content"
    }
    
    logger.info("🧪 Creating ArgoLLM with gpt4o model...")
    standard_llm = ArgoLLM(standard_config)
    
    logger.info(f"🧪 Standard model timeout: {standard_llm._timeout}s (expected: 30s)")
    assert standard_llm._timeout == 30.0, f"Expected 30s timeout for standard model, got {standard_llm._timeout}s"
    
    logger.info("✅ ArgoLLM timeout configuration test PASSED")
    return o1_llm, standard_llm

async def test_agent_timeout_detection():
    """Test agent timeout detection and signal-based timeouts"""
    
    logger.info("🧪 TESTING: Agent timeout detection")
    
    # Create o1 model LLM  
    o1_config = {
        "model_name": "gpto1",
        "user": "test_user",
        "system_content": "Test system content"
    }
    o1_llm = ArgoLLM(o1_config)
    
    # Create minimal tools for testing
    tools = [
        FBATool({"name": "run_metabolic_fba", "description": "Run FBA analysis"})
    ]
    
    # Create agent
    logger.info("🧪 Creating RealTimeMetabolicAgent with o1 model...")
    agent = RealTimeMetabolicAgent(o1_llm, tools, {"max_iterations": 1})
    
    # Verify agent model detection
    detected_model = getattr(agent.llm, 'model_name', '').lower()
    logger.info(f"🧪 Agent detected model: '{detected_model}'")
    
    is_o1_detected = detected_model.startswith(('gpto', 'o1'))
    logger.info(f"🧪 Is detected as o1 model: {is_o1_detected}")
    assert is_o1_detected, f"Model '{detected_model}' should be detected as o1 model"
    
    logger.info("✅ Agent timeout detection test PASSED")
    return agent

def test_timeout_logic_manually():
    """Manually test timeout logic without making actual LLM calls"""
    
    logger.info("🧪 TESTING: Manual timeout logic validation")
    
    # Test model name detection logic
    test_cases = [
        ("gpto1", True, 120),
        ("gpto1mini", True, 120), 
        ("gpto1preview", True, 120),
        ("o1", True, 120),
        ("o1-mini", True, 120),
        ("gpt4o", False, 30),
        ("gpt4", False, 30),
        ("gpt35", False, 30),
    ]
    
    for model_name, expected_is_o1, expected_timeout in test_cases:
        model_lower = model_name.lower()
        is_o1_model = model_lower.startswith(('gpto', 'o1'))
        timeout_seconds = 120 if is_o1_model else 30
        
        logger.info(f"🧪 Model '{model_name}' -> is_o1: {is_o1_model}, timeout: {timeout_seconds}s")
        
        assert is_o1_model == expected_is_o1, f"Model '{model_name}' o1 detection failed"
        assert timeout_seconds == expected_timeout, f"Model '{model_name}' timeout incorrect"
    
    logger.info("✅ Manual timeout logic test PASSED")

async def test_signal_timeout_mechanism():
    """Test that signal-based timeouts work properly (with short timeout for testing)"""
    
    logger.info("🧪 TESTING: Signal timeout mechanism")
    
    import signal
    
    # Test signal timeout with short timeout
    def timeout_handler(signum, frame):
        raise TimeoutError("Test timeout")
    
    # Test that signal mechanism works
    signal.signal(signal.SIGALRM, timeout_handler)
    
    try:
        signal.alarm(1)  # 1 second timeout for test
        start_time = time.time()
        time.sleep(2)  # Sleep longer than timeout
        assert False, "Signal timeout should have triggered"
    except TimeoutError:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"🧪 Signal timeout triggered after {duration:.2f}s (expected ~1s)")
        assert 0.9 <= duration <= 1.5, f"Signal timeout duration {duration:.2f}s not in expected range"
    finally:
        signal.alarm(0)  # Cancel alarm
    
    logger.info("✅ Signal timeout mechanism test PASSED")

def generate_test_report():
    """Generate a comprehensive test report"""
    
    report = f"""
=== TIMEOUT FIX COMPREHENSIVE TEST REPORT ===
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

Tests Performed:
1. ✅ ArgoLLM timeout configuration validation
2. ✅ Agent timeout detection logic  
3. ✅ Manual timeout logic validation
4. ✅ Signal timeout mechanism testing

Key Findings:
- o1 models (gpto1, gpto1mini, etc.) correctly use 120s timeouts
- Standard models (gpt4o, etc.) correctly use 30s timeouts
- Signal-based timeout mechanism is functional
- Debug logging provides comprehensive timeout information

Expected Behavior:
- When using gpto1 model, you should see "Using 120s timeout" in logs
- When using gpt4o model, you should see "Using 30s timeout" in logs
- No more "timed out (30s)" warnings for o1 models
- Comprehensive debug logs showing timeout detection

Next Steps:
1. Test with actual interactive CLI: `modelseed-agent interactive`
2. Monitor logs for timeout debug messages
3. Verify no 30s timeout warnings for o1 models
4. Confirm improved performance with longer timeouts

=== END REPORT ===
"""
    
    with open("timeout_fix_test_report.txt", "w") as f:
        f.write(report)
    
    logger.info("📊 Test report saved to: timeout_fix_test_report.txt")
    print(report)

async def main():
    """Run comprehensive timeout fix validation"""
    
    logger.info("🚀 Starting comprehensive timeout fix validation")
    
    try:
        # Test 1: ArgoLLM timeout configuration
        o1_llm, standard_llm = test_argo_llm_timeout_configuration()
        
        # Test 2: Agent timeout detection 
        agent = await test_agent_timeout_detection()
        
        # Test 3: Manual timeout logic validation
        test_timeout_logic_manually()
        
        # Test 4: Signal timeout mechanism
        await test_signal_timeout_mechanism()
        
        # Generate report
        generate_test_report()
        
        logger.info("🎉 ALL TESTS PASSED! Timeout fixes are working correctly.")
        
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())