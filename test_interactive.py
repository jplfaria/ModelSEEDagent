#!/usr/bin/env python3
"""
Interactive Analysis Interface Test with GPT-4o Latest via Argo
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_interactive_interface():
    """Test the interactive interface with Argo GPT-4o"""
    print("🧬 ModelSEED Interactive Analysis Interface")
    print("📡 Using GPT-4o Latest via Argo")
    print("=" * 60)
    
    try:
        # Check if we have interactive components
        print("🔍 Checking interactive components...")
        
        # Import interactive components
        from src.interactive.interactive_cli import InteractiveCLI
        from src.interactive.session_manager import SessionManager
        from src.interactive.query_processor import QueryProcessor
        from src.interactive.conversation_engine import ConversationEngine
        from src.interactive.live_visualizer import LiveVisualizer
        
        print("✅ All interactive components available!")
        
        # Initialize interactive CLI
        print("\n🚀 Initializing Interactive CLI...")
        cli = InteractiveCLI()  # No config parameter needed
        
        print("✅ Interactive CLI ready!")
        print("\n" + "=" * 60)
        print("🎉 You can now launch the interactive interface!")
        print("\n📋 How to start:")
        print("   python test_interactive.py --launch")
        print("   OR run the components directly:")
        print("   python src/interactive/interactive_cli.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📁 Available interactive files:")
        if os.path.exists("src/interactive"):
            for file in os.listdir("src/interactive"):
                if file.endswith(".py"):
                    print(f"   ✓ {file}")
        return False
        
    except Exception as e:
        print(f"❌ Setup error: {e}")
        import traceback
        traceback.print_exc()
        return False

def launch_interactive():
    """Launch the interactive interface"""
    print("🚀 Launching Interactive Analysis Interface...")
    
    try:
        # Set up path
        sys.path.insert(0, os.path.abspath('.'))
        
        # Try to launch the interactive CLI
        from src.interactive.interactive_cli import main
        main()
        
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        print("\n🔧 Manual launch options:")
        print("1. python src/interactive/interactive_cli.py")
        print("2. python -c \"import sys; sys.path.insert(0, '.'); from src.interactive.interactive_cli import main; main()\"")

def show_setup_guide():
    """Show the complete setup guide"""
    print("\n🔧 Complete Setup Guide for GPT-4o Latest via Argo")
    print("=" * 60)
    
    print("\n1️⃣ Set Environment Variables:")
    print("   export ARGO_USER='your_argo_username'")
    print("   export ARGO_API_KEY='your_argo_api_key'")
    print("   export DEFAULT_MODEL_NAME='gpt-4o-2024-11-20'")
    
    print("\n2️⃣ Launch Interactive Interface:")
    print("   python test_interactive.py --launch")
    print("   OR:")
    print("   python src/interactive/interactive_cli.py")
    
    print("\n3️⃣ Example Natural Language Queries:")
    print("   • 'Load and analyze the E. coli model'")
    print("   • 'What is the growth rate on glucose?'")
    print("   • 'Run flux balance analysis'")
    print("   • 'Create a network visualization'")
    print("   • 'Compare growth on glucose vs acetate'")
    
    print("\n4️⃣ Available Test Models:")
    if os.path.exists("data/models"):
        for file in os.listdir("data/models"):
            if file.endswith(".xml"):
                print(f"   📁 {file}")
    
    print("\n5️⃣ Session Features:")
    print("   • Persistent session storage")
    print("   • Real-time visualizations")
    print("   • Natural language understanding")
    print("   • Context-aware conversations")
    print("   • Automatic browser integration")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--launch":
        launch_interactive()
    else:
        success = test_interactive_interface()
        if success:
            show_setup_guide() 