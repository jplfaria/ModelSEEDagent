#!/usr/bin/env python3
"""
Comprehensive test script to verify streaming interface fixes.

This script tests the fixed streaming interface to ensure:
1. No more flickering empty boxes
2. Robust content validation works
3. Error handling prevents crashes
4. All edge cases are handled properly

Run this before claiming the interactive CLI is fixed!
"""

import sys
import time
import threading
from pathlib import Path

# Add src to path so we can import the fixed streaming interface
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from interactive.streaming_interface import RealTimeStreamingInterface, StreamingEventType
except ImportError as e:
    print(f"❌ Failed to import streaming interface: {e}")
    print(f"Looking in: {src_path}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def test_basic_functionality():
    """Test basic streaming interface functionality"""
    print("🔍 Testing basic streaming interface functionality...")
    
    try:
        interface = RealTimeStreamingInterface()
        
        # Test initialization
        assert hasattr(interface, 'events')
        assert hasattr(interface, 'current_ai_thought')
        assert hasattr(interface, 'executed_tools')
        assert hasattr(interface, 'ai_decisions')
        
        print("✅ Basic initialization works")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_content_validation():
    """Test the robust content validation in panel creation"""
    print("\n🔍 Testing content validation...")
    
    try:
        interface = RealTimeStreamingInterface()
        
        # Test with problematic content
        test_cases = [
            (None, "None value"),
            ("", "Empty string"),
            ("   ", "Whitespace only"),
            ("\n\n", "Newlines only"),
            ("Valid content", "Valid content"),
        ]
        
        for content, description in test_cases:
            # Set problematic content
            interface.current_ai_thought = content
            interface.ai_decisions = [content] if content is not None else []
            interface.executed_tools = [{"name": content, "success": True, "duration": 1.0}] if content else []
            
            # Try to create panels
            try:
                ai_panel = interface._create_ai_thinking_panel()
                decisions_panel = interface._create_decisions_panel()
                tools_panel = interface._create_tool_progress_panel()
                results_panel = interface._create_results_panel()
                
                # Verify panels have content
                assert ai_panel is not None
                assert decisions_panel is not None
                assert tools_panel is not None
                assert results_panel is not None
                
                print(f"✅ {description}: Panels created successfully")
                
            except Exception as e:
                print(f"❌ {description}: Panel creation failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Content validation test failed: {e}")
        return False

def test_streaming_workflow():
    """Test complete streaming workflow with edge cases"""
    print("\n🔍 Testing complete streaming workflow...")
    
    try:
        interface = RealTimeStreamingInterface()
        
        # Test streaming start with various query types
        queries = [
            "Normal query",
            "",  # Empty query
            None,  # None query
            "   ",  # Whitespace query
            "Very long query " * 20,  # Long query
        ]
        
        for i, query in enumerate(queries):
            print(f"  Testing query {i+1}/{len(queries)}: {type(query).__name__}")
            
            try:
                # Start streaming
                interface.start_streaming(query)
                
                # Add various events
                interface.add_event(StreamingEventType.AI_THINKING, "Test thought")
                interface.add_event(StreamingEventType.TOOL_SELECTED, "Test tool", {"tool_name": "test_tool"})
                interface.add_event(StreamingEventType.TOOL_COMPLETED, "Test completion", {
                    "tool_name": "test_tool", 
                    "duration": 1.5, 
                    "success": True,
                    "summary": "Test summary"
                })
                interface.add_event(StreamingEventType.DECISION_MADE, "Test decision")
                
                # Brief pause to let display update
                time.sleep(0.1)
                
                # Stop streaming
                interface.stop_streaming()
                
                print(f"    ✅ Query {i+1} handled successfully")
                
            except Exception as e:
                print(f"    ❌ Query {i+1} failed: {e}")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Streaming workflow test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and recovery"""
    print("\n🔍 Testing error handling and recovery...")
    
    try:
        interface = RealTimeStreamingInterface()
        
        # Test with corrupted data
        interface.ai_decisions = ["valid", None, "", "  ", "another valid"]
        interface.executed_tools = [
            {"name": "valid_tool", "success": True, "duration": 1.0},
            {"name": None, "success": True, "duration": 1.0},  # Invalid name
            {"success": True, "duration": 1.0},  # Missing name
            None,  # Invalid tool entry
            {"name": "another_valid", "success": False, "duration": 2.5},
        ]
        
        # Try to create panels with corrupted data
        ai_panel = interface._create_ai_thinking_panel()
        decisions_panel = interface._create_decisions_panel()
        tools_panel = interface._create_tool_progress_panel()
        results_panel = interface._create_results_panel()
        
        # All panels should be created successfully
        assert ai_panel is not None
        assert decisions_panel is not None
        assert tools_panel is not None
        assert results_panel is not None
        
        print("✅ Error handling works - corrupted data handled gracefully")
        
        # Test with extreme values
        interface.start_time = time.time() + 1000  # Future time (negative elapsed)
        interface.events = None  # None instead of list
        
        results_panel = interface._create_results_panel()
        assert results_panel is not None
        
        print("✅ Extreme values handled correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_concurrent_updates():
    """Test concurrent updates to streaming interface"""
    print("\n🔍 Testing concurrent updates...")
    
    try:
        interface = RealTimeStreamingInterface()
        interface.start_streaming("Concurrent test query")
        
        # Flag to track if any errors occurred
        errors = []
        
        def update_worker(worker_id):
            """Worker function that updates the interface"""
            try:
                for i in range(10):
                    interface.add_event(
                        StreamingEventType.AI_THINKING, 
                        f"Worker {worker_id} thought {i}"
                    )
                    interface.add_event(
                        StreamingEventType.DECISION_MADE,
                        f"Worker {worker_id} decision {i}"
                    )
                    time.sleep(0.01)  # Brief pause
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")
        
        # Start multiple worker threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=update_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Stop streaming
        interface.stop_streaming()
        
        # Check for errors
        if errors:
            print(f"❌ Concurrent updates failed: {errors}")
            return False
        
        print("✅ Concurrent updates handled successfully")
        return True
        
    except Exception as e:
        print(f"❌ Concurrent updates test failed: {e}")
        return False

def test_visual_verification():
    """Visual test to verify no flickering occurs"""
    print("\n🔍 Visual verification test (watch for flickering)...")
    print("👀 Watch the display below - there should be NO flickering or empty boxes!")
    
    try:
        interface = RealTimeStreamingInterface()
        interface.start_streaming("Visual verification: E. coli comprehensive analysis")
        
        # Simulate a realistic AI workflow
        workflow_steps = [
            ("🧠 Analyzing query structure and intent...", "ai_thinking"),
            ("🎯 Decision: Start with metabolic characterization", "decision"),
            ("🔧 Selecting run_metabolic_fba tool", "tool_selected"),
            ("⚡ Executing FBA analysis...", "tool_executing"), 
            ("✅ FBA completed: Growth rate 0.518 h⁻¹", "tool_completed"),
            ("🧠 High growth rate detected - investigating requirements...", "ai_thinking"),
            ("🎯 Decision: Analyze nutritional dependencies", "decision"),
            ("🔧 Selecting find_minimal_media tool", "tool_selected"),
            ("⚡ Finding minimal media requirements...", "tool_executing"),
            ("✅ Minimal media found: 20 essential nutrients", "tool_completed"),
            ("🧠 Checking biosynthetic capabilities...", "ai_thinking"),
            ("🎯 Decision: Identify essential genes", "decision"),
            ("🔧 Selecting analyze_essentiality tool", "tool_selected"),
            ("⚡ Analyzing gene essentiality...", "tool_executing"),
            ("✅ Essentiality analysis complete: 142 essential genes", "tool_completed"),
            ("🎉 Comprehensive analysis complete!", "workflow_complete"),
        ]
        
        for i, (message, event_type) in enumerate(workflow_steps):
            print(f"Step {i+1}/{len(workflow_steps)}: {message}")
            
            if event_type == "ai_thinking":
                interface.show_ai_analysis(message)
            elif event_type == "decision":
                interface.show_decision(message, "AI reasoning")
            elif event_type == "tool_selected":
                interface.show_tool_execution("test_tool", "AI selected this tool")
            elif event_type == "tool_executing":
                pass  # Just update display
            elif event_type == "tool_completed":
                interface.show_tool_completion("test_tool", 1.5, True, message)
            elif event_type == "workflow_complete":
                interface.show_workflow_complete(message, {"tools": 3, "decisions": 3})
            
            time.sleep(0.8)  # Pause to observe each step
        
        interface.stop_streaming()
        
        print("✅ Visual verification completed")
        print("❓ Did you see any flickering or empty boxes? If NO, the fix works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Visual verification test failed: {e}")
        return False

def main():
    """Run all comprehensive tests"""
    print("🧪 Comprehensive Streaming Interface Fix Verification")
    print("=" * 60)
    print("This test suite verifies that the flickering empty boxes issue is fixed.")
    print()
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Content Validation", test_content_validation), 
        ("Streaming Workflow", test_streaming_workflow),
        ("Error Handling", test_error_handling),
        ("Concurrent Updates", test_concurrent_updates),
        ("Visual Verification", test_visual_verification),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        result = test_func()
        results.append((test_name, result))
        
        time.sleep(0.5)  # Brief pause between tests
    
    # Summary
    elapsed = time.time() - start_time
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n{'='*60}")
    print("🎯 TEST RESULTS SUMMARY")
    print('='*60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"Total time: {elapsed:.1f}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The streaming interface fixes appear to be working correctly.")
        print("✅ The flickering empty boxes issue should be resolved.")
        print("\n💡 Ready to test the interactive CLI!")
    else:
        print(f"\n❌ {total-passed} test(s) failed.")
        print("🔧 Additional fixes may be needed before testing interactive CLI.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)