#!/usr/bin/env python3
"""
ModelSEED Interactive Interface with GPT-4o Latest via Argo
Configured to match Concordia project setup
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))


def setup_argo_config():
    """Setup Argo configuration matching Concordia project"""
    print("🔧 Setting up Argo Configuration")
    print("=" * 50)

    # Check environment variables
    argo_user = os.getenv("ARGO_USER", "jplfaria")
    model_name = os.getenv("DEFAULT_MODEL_NAME", "gpt4olatest")
    backend = os.getenv("DEFAULT_LLM_BACKEND", "argo")

    print(f"📡 Argo User: {argo_user}")
    print(f"🤖 Model: {model_name}")
    print(f"🔌 Backend: {backend}")
    print(f"🌍 Environment: dev (auto-selected for gpt4olatest)")

    # Verify API key (optional)
    api_key = os.getenv("ARGO_API_KEY")
    if api_key:
        print("🔑 API Key: ✅ configured")
    else:
        print("🔑 API Key: ⚠️  not set (may work without it)")

    return {
        "model_name": model_name,
        "api_base": "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource",  # Use dev environment
        "user": argo_user,
        "system_content": "You are a metabolic modeling expert using ModelSEED tools.",
        "max_tokens": 4000,
        "temperature": 0.1,
        "env": "dev",  # Force dev environment for gpt4olatest
    }


def launch_interactive():
    """Launch the interactive interface with Argo configuration"""
    print("🧬 ModelSEED Interactive Analysis with GPT-4o Latest")
    print("📡 Using Argo Gateway (same as Concordia)")
    print("=" * 60)

    try:
        # Setup configuration
        config = setup_argo_config()

        # Import and initialize components
        print("\n🔍 Loading ModelSEED components...")
        from src.agents.metabolic import MetabolicAgent
        from src.interactive.interactive_cli import InteractiveCLI
        from src.llm.argo import ArgoLLM
        from src.tools.cobra.analysis import ModelAnalysisTool
        from src.tools.cobra.fba import FBATool

        # Initialize LLM with Argo
        print("🤖 Initializing GPT-4o Latest via Argo...")
        llm = ArgoLLM(config)
        print(f"✅ LLM ready: {llm.config.model_name}")

        # Initialize tools
        print("🔧 Setting up metabolic tools...")
        tool_config = {"name": "metabolic_tools", "verbose": True}
        tools = [FBATool(tool_config), ModelAnalysisTool(tool_config)]
        print(f"✅ {len(tools)} tools ready")

        # Initialize agent
        print("🧠 Creating MetabolicAgent...")
        agent_config = {
            "name": "ModelSEED_Agent",
            "description": "AI agent for metabolic model analysis with ModelSEED",
            "verbose": True,
            "max_iterations": 10,
        }
        agent = MetabolicAgent(llm, tools, agent_config)
        print("✅ Agent ready")

        # Start interactive CLI
        print("\n🚀 Launching Interactive Interface...")
        print("=" * 60)
        print("💡 Available commands:")
        print("   • 'Load model data/models/iML1515.xml'")
        print("   • 'What is the growth rate on glucose minimal media?'")
        print("   • 'Run flux balance analysis'")
        print("   • 'Show me the central carbon metabolism'")
        print("   • 'Compare growth on different carbon sources'")
        print("   • 'Create a metabolic network visualization'")
        print("   • 'exit' or 'quit' to stop")
        print("=" * 60)

        # Create a simple interactive loop
        print("\n🎯 Ready for analysis! Type your questions...")

        while True:
            try:
                query = input("\n🧬 ModelSEED> ").strip()

                if query.lower() in ["exit", "quit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not query:
                    continue

                print(f"\n🔄 Processing: {query}")
                print("⏳ Thinking... (GPT-4o Latest)")

                # Run the query through the agent
                result = agent.analyze_model(query)

                print("\n✨ Result:")
                print("-" * 40)
                print(result.message if hasattr(result, "message") else str(result))
                if hasattr(result, "data") and result.data:
                    print(f"\n📊 Generated data: {len(result.data)} items")
                print("-" * 40)

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("💡 Try a simpler query or 'exit' to quit")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📋 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        return False

    except Exception as e:
        print(f"❌ Setup error: {e}")
        import traceback

        traceback.print_exc()
        return False


def show_quick_start():
    """Show quick start instructions"""
    print("\n🚀 Quick Start Guide")
    print("=" * 40)
    print("1. Set environment variables:")
    print('   export ARGO_USER="jplfaria"')
    print('   export DEFAULT_MODEL_NAME="gpt4olatest"')
    print('   export DEFAULT_LLM_BACKEND="argo"')
    print("")
    print("2. Launch interactive interface:")
    print("   python launch_interactive_argo.py")
    print("")
    print("3. Try example queries:")
    print("   • Load model data/models/iML1515.xml")
    print("   • What pathways are most active?")
    print("   • Run FBA and show me the results")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_quick_start()
    else:
        success = launch_interactive()
        if not success:
            print("\n" + "=" * 60)
            show_quick_start()
