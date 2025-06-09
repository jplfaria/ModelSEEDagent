#!/usr/bin/env python3
"""
Test the fix for duplicate messages in the interactive CLI.
"""

import subprocess
import sys
import time
from pathlib import Path

def test_duplicate_message_fix():
    """Test if the duplicate message issue is resolved"""
    print("🧪 Testing Duplicate Message Fix")
    print("=" * 40)
    
    # Test with exit command to see if duplicate message still appears
    test_input = "exit\n"
    
    import os
    env = os.environ.copy()
    env["MODELSEED_DEBUG"] = "true"
    cmd = [sys.executable, "-m", "src.cli.main", "interactive"]
    
    print("🚀 Running CLI with debug mode and exit command...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            input=test_input,
            text=True,
            capture_output=True,
            timeout=30,
            env=env,
            cwd=str(Path(__file__).parent)
        )
        
        duration = time.time() - start_time
        print(f"⏱️ Completed in {duration:.2f}s")
        
        # Analyze output for duplicate messages
        stdout = result.stdout
        
        # Count occurrences of the analysis message
        analysis_message_count = stdout.count("AI analyzing your query and executing tools")
        print(f"📊 'AI analyzing' message count: {analysis_message_count}")
        
        # Look for debug mode indicators
        debug_indicators = stdout.count("DEBUG MODE")
        print(f"📊 Debug mode indicators: {debug_indicators}")
        
        # Look for module imports (should only be at startup)
        modelseedpy_count = stdout.count("modelseedpy 0.4.2")
        cobrakbase_count = stdout.count("cobrakbase 0.4.0")
        print(f"📊 modelseedpy imports: {modelseedpy_count}")
        print(f"📊 cobrakbase imports: {cobrakbase_count}")
        
        print(f"\n📄 First 500 chars of output:")
        print(stdout[:500])
        
        if analysis_message_count <= 1 and modelseedpy_count <= 2:
            print("✅ DUPLICATE MESSAGE FIX APPEARS TO WORK")
            return True
        else:
            print("❌ DUPLICATE MESSAGE ISSUE STILL EXISTS")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ CLI timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_duplicate_message_fix()
    if success:
        print("\n💡 Now test with actual query:")
        print("   export MODELSEED_DEBUG=true")
        print("   modelseed-agent interactive")
        print("   # Enter: I need metabolic analysis")
    else:
        print("\n💡 More investigation needed")