#!/usr/bin/env python3
"""
Automated verification system for the interactive CLI.
This provides a comprehensive test to verify the CLI actually works end-to-end.
"""

import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

def test_interactive_cli_verification():
    """
    Comprehensive verification that the interactive CLI works correctly.
    Returns True if verification passes, False if it fails.
    """
    
    print("🧪 Starting Interactive CLI Verification")
    print("=" * 60)
    
    # Enable debug mode for detailed logging
    env = os.environ.copy()
    env["MODELSEED_DEBUG"] = "true"
    
    # Create a test input script
    test_commands = [
        "I need a comprehensive metabolic analysis of E. coli",  # Test query
        "exit"  # Exit command
    ]
    
    # Write commands to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for cmd in test_commands:
            f.write(cmd + '\n')
        input_file = f.name
    
    try:
        print(f"📝 Created test input: {input_file}")
        print(f"🔍 Test commands: {test_commands}")
        
        # Run the interactive CLI with our test input
        print(f"🚀 Running interactive CLI with test input...")
        
        cmd = [
            sys.executable, "-m", "src.cli.main", "interactive"
        ]
        
        start_time = time.time()
        
        # Use shorter timeout for verification
        result = subprocess.run(
            cmd,
            input='\n'.join(test_commands),
            text=True,
            capture_output=True,
            timeout=120,  # 2 minute timeout for verification
            env=env,
            cwd=str(Path(__file__).parent)
        )
        
        duration = time.time() - start_time
        
        print(f"⏱️ CLI process completed in {duration:.2f}s")
        print(f"📊 Return code: {result.returncode}")
        
        # Analyze the output
        stdout = result.stdout
        stderr = result.stderr
        
        print(f"📄 STDOUT length: {len(stdout)} characters")
        print(f"📄 STDERR length: {len(stderr)} characters")
        
        # Check for success indicators
        success_indicators = [
            "Welcome to ModelSEEDagent",
            "Dynamic AI Agent initialized",
            "AI analyzing your query",
            "Analysis completed",
            "Thank you for using ModelSEEDagent"
        ]
        
        failure_indicators = [
            "modelseedpy 0.4.2" * 5,  # Repeated imports (indicates hanging)
            "cobrakbase 0.4.0" * 5,   # Repeated imports (indicates hanging)
            "Traceback",
            "Error:",
            "Failed",
            "Exception"
        ]
        
        # Count successes and failures
        successes = []
        failures = []
        
        for indicator in success_indicators:
            if indicator in stdout or indicator in stderr:
                successes.append(indicator)
                print(f"✅ Found success indicator: '{indicator}'")
        
        for indicator in failure_indicators:
            if indicator in stdout or indicator in stderr:
                failures.append(indicator)
                print(f"❌ Found failure indicator: '{indicator}'")
        
        # Check for hanging pattern (repeated module imports)
        module_import_count = stdout.count("modelseedpy 0.4.2") + stdout.count("cobrakbase 0.4.0")
        if module_import_count > 10:  # More than 10 imports suggests hanging
            failures.append(f"Excessive module imports: {module_import_count}")
            print(f"❌ Excessive module imports detected: {module_import_count}")
        
        # Determine overall success
        has_critical_success = any(indicator in successes for indicator in [
            "Welcome to ModelSEEDagent",
            "AI analyzing your query"
        ])
        
        has_critical_failure = len(failures) > 0 or not has_critical_success
        
        print(f"\n📊 VERIFICATION RESULTS:")
        print(f"   ✅ Success indicators: {len(successes)}")
        print(f"   ❌ Failure indicators: {len(failures)}")
        print(f"   ⏱️ Execution time: {duration:.2f}s")
        print(f"   📊 Return code: {result.returncode}")
        
        if not has_critical_failure and duration < 60:  # Should complete in under 60s
            print(f"\n🎉 VERIFICATION PASSED: Interactive CLI working correctly!")
            return True
        else:
            print(f"\n💥 VERIFICATION FAILED: Interactive CLI has issues")
            
            # Show detailed output for debugging
            print(f"\n📄 STDOUT (first 1000 chars):")
            print(stdout[:1000])
            if len(stdout) > 1000:
                print(f"... [truncated {len(stdout) - 1000} more characters]")
            
            print(f"\n📄 STDERR (first 1000 chars):")
            print(stderr[:1000])
            if len(stderr) > 1000:
                print(f"... [truncated {len(stderr) - 1000} more characters]")
            
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ VERIFICATION FAILED: CLI timed out after 120s (indicates hanging)")
        return False
    except Exception as e:
        print(f"❌ VERIFICATION FAILED: Exception during testing: {e}")
        return False
    finally:
        # Clean up temp file
        try:
            os.unlink(input_file)
        except:
            pass

def test_conversation_engine_directly():
    """
    Test the conversation engine directly without the full CLI.
    This helps isolate whether the issue is in the CLI or the engine.
    """
    print(f"\n🔬 Testing Conversation Engine Directly")
    print("=" * 50)
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        from src.interactive.conversation_engine import DynamicAIConversationEngine
        from src.interactive.session_manager import AnalysisSession
        from datetime import datetime
        import uuid
        
        # Create test session
        session = AnalysisSession(
            id=str(uuid.uuid4())[:8],
            name="verification_test",
            created_at=datetime.now(),
            last_active=datetime.now(),
            description="Automated verification test"
        )
        
        print("✅ Created test session")
        
        # Create conversation engine
        engine = DynamicAIConversationEngine(session)
        print("✅ Created conversation engine")
        
        # Test conversation flow
        greeting = engine.start_conversation()
        print("✅ Got greeting response")
        
        # Test analysis query
        query = "I need a comprehensive metabolic analysis of E. coli"
        print(f"🔍 Testing query: '{query}'")
        
        start_time = time.time()
        response = engine.process_user_input(query)
        duration = time.time() - start_time
        
        print(f"✅ Got response in {duration:.2f}s")
        print(f"   Response type: {response.response_type}")
        print(f"   AI reasoning steps: {response.ai_reasoning_steps}")
        print(f"   Processing time: {response.processing_time:.2f}s")
        
        if response.metadata.get("ai_agent_result"):
            tools_executed = response.metadata.get("tools_executed", [])
            print(f"   Tools executed: {tools_executed}")
            print(f"🎉 CONVERSATION ENGINE TEST PASSED")
            return True
        else:
            print(f"❌ CONVERSATION ENGINE TEST FAILED: No AI agent result")
            return False
            
    except Exception as e:
        print(f"❌ CONVERSATION ENGINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 ModelSEEDagent Interactive CLI Verification System")
    print("=" * 60)
    
    # Test 1: Conversation engine directly
    engine_success = test_conversation_engine_directly()
    
    # Test 2: Full interactive CLI
    cli_success = test_interactive_cli_verification()
    
    print(f"\n🎯 FINAL VERIFICATION RESULTS:")
    print(f"   🔬 Conversation Engine: {'✅ PASS' if engine_success else '❌ FAIL'}")
    print(f"   🖥️ Interactive CLI: {'✅ PASS' if cli_success else '❌ FAIL'}")
    
    if engine_success and cli_success:
        print(f"\n🎉 ALL TESTS PASSED: Interactive CLI is working correctly!")
        sys.exit(0)
    else:
        print(f"\n💥 TESTS FAILED: Interactive CLI needs more work")
        print(f"\nTo enable debug mode, run:")
        print(f"   export MODELSEED_DEBUG=true")
        print(f"   modelseed-agent interactive")
        sys.exit(1)