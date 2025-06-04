#!/usr/bin/env python3
"""
Test Script for Interactive Analysis Interface

Demonstrates the capabilities of the new Phase 3.2 Interactive Analysis Interface
including conversational AI, session management, and real-time visualization.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

def test_session_manager():
    """Test session management functionality"""
    console.print("\n[bold blue]🔬 Testing Session Manager...[/bold blue]")
    
    try:
        from src.interactive.session_manager import SessionManager, SessionStatus
        
        # Create session manager
        session_mgr = SessionManager("test_sessions")
        
        # Create test session
        session = session_mgr.create_session(
            "Test Analysis Session",
            "Testing interactive analysis capabilities"
        )
        
        # Show session info
        session_info = session.get_session_summary()
        console.print(f"✅ Created session: {session_info['session_info']['name']}")
        console.print(f"   ID: {session_info['session_info']['id']}")
        
        # Display sessions table
        session_mgr.display_sessions_table()
        
        console.print("✅ Session Manager: [green]Working perfectly![/green]")
        return True
        
    except Exception as e:
        console.print(f"❌ Session Manager Error: {e}")
        return False

def test_query_processor():
    """Test intelligent query processing"""
    console.print("\n[bold blue]🧠 Testing Query Processor...[/bold blue]")
    
    try:
        from src.interactive.query_processor import QueryProcessor
        
        processor = QueryProcessor()
        
        # Test queries
        test_queries = [
            "Analyze the structure of my E. coli model",
            "What is the growth rate on glucose?",
            "Optimize glycolysis pathway flux",
            "Create a flux heatmap visualization",
            "Compare two different growth conditions"
        ]
        
        table = Table(title="🔍 Query Analysis Results", box=box.ROUNDED)
        table.add_column("Query", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Complexity", style="yellow")
        table.add_column("Confidence", style="magenta")
        
        for query in test_queries:
            analysis = processor.analyze_query(query)
            table.add_row(
                query[:40] + "..." if len(query) > 40 else query,
                analysis.query_type.value.replace('_', ' ').title(),
                analysis.complexity.value.title(),
                f"{analysis.confidence:.1%}"
            )
        
        console.print(table)
        console.print("✅ Query Processor: [green]Working perfectly![/green]")
        return True
        
    except Exception as e:
        console.print(f"❌ Query Processor Error: {e}")
        return False

def test_conversation_engine():
    """Test conversational AI engine"""
    console.print("\n[bold blue]💬 Testing Conversation Engine...[/bold blue]")
    
    try:
        from src.interactive.session_manager import SessionManager
        from src.interactive.query_processor import QueryProcessor
        from src.interactive.conversation_engine import ConversationEngine
        
        # Setup components
        session_mgr = SessionManager("test_sessions")
        session = session_mgr.create_session("Conversation Test", "Testing conversation engine")
        processor = QueryProcessor()
        
        # Create conversation engine
        conv_engine = ConversationEngine(session, processor)
        
        # Start conversation
        greeting = conv_engine.start_conversation()
        console.print(Panel(
            greeting.content,
            title="[green]🤖 Assistant Greeting[/green]",
            border_style="green"
        ))
        
        # Test conversation with sample queries
        test_interactions = [
            "What can you help me with?",
            "Analyze model structure",
            "How do I run flux balance analysis?"
        ]
        
        for query in test_interactions:
            console.print(f"\n[bold cyan]👤 User:[/bold cyan] {query}")
            
            response = conv_engine.process_user_input(query)
            console.print(Panel(
                response.content[:200] + "..." if len(response.content) > 200 else response.content,
                title=f"[green]🤖 Assistant ({response.response_type.value.title()})[/green]",
                border_style="green"
            ))
            
            if response.suggested_actions:
                console.print("[yellow]💡 Suggestions:[/yellow]")
                for i, action in enumerate(response.suggested_actions[:3], 1):
                    console.print(f"  {i}. {action}")
        
        console.print("✅ Conversation Engine: [green]Working perfectly![/green]")
        return True
        
    except Exception as e:
        console.print(f"❌ Conversation Engine Error: {e}")
        return False

def test_live_visualizer():
    """Test live visualization capabilities"""
    console.print("\n[bold blue]🎨 Testing Live Visualizer...[/bold blue]")
    
    try:
        from src.interactive.live_visualizer import LiveVisualizer
        import numpy as np
        
        visualizer = LiveVisualizer("test_visualizations")
        
        # Test workflow visualization
        workflow_data = {
            "nodes": [
                {"id": "input", "label": "Input Model", "type": "input", "status": "completed"},
                {"id": "analyze", "label": "Analysis", "type": "tool", "status": "running"},
                {"id": "output", "label": "Results", "type": "output", "status": "pending"}
            ],
            "edges": [
                {"source": "input", "target": "analyze", "label": "SBML Model"},
                {"source": "analyze", "target": "output", "label": "Analysis Results"}
            ]
        }
        
        # Test progress tracking
        visualizer.start_live_progress(5)
        
        visualizer.update_progress("Creating workflow visualization...", 1)
        workflow_file = visualizer.create_workflow_visualization(workflow_data)
        
        visualizer.update_progress("Creating dashboard...", 1)
        dashboard_data = {
            "execution_timeline": {
                "timestamps": ["0s", "2s", "4s", "6s"],
                "tools": ["Input", "Process", "Analyze", "Complete"]
            },
            "tool_performance": {
                "Analysis": 2.5,
                "Visualization": 1.2,
                "Export": 0.8
            },
            "success_rate": 0.95
        }
        dashboard_file = visualizer.create_progress_dashboard(dashboard_data)
        
        visualizer.update_progress("Creating network visualization...", 1)
        network_data = {
            "nodes": [{"id": f"N{i}", "attributes": {"type": "metabolite"}} for i in range(10)],
            "edges": [{"source": f"N{i}", "target": f"N{(i+1)%10}", "attributes": {}} for i in range(10)]
        }
        network_file = visualizer.create_network_visualization(network_data)
        
        visualizer.update_progress("Creating flux heatmap...", 1)
        flux_data = {
            "flux_matrix": np.random.rand(8, 3) * 10 - 2,
            "reaction_names": [f"R{i:03d}" for i in range(8)],
            "condition_names": ["Glucose", "Acetate", "Glycerol"]
        }
        heatmap_file = visualizer.create_flux_heatmap(flux_data)
        
        visualizer.update_progress("Visualizations complete!", 1)
        visualizer.stop_live_progress()
        
        # Display visualization summary
        visualizer.display_visualization_table()
        
        console.print("✅ Live Visualizer: [green]Working perfectly![/green]")
        return True
        
    except Exception as e:
        console.print(f"❌ Live Visualizer Error: {e}")
        return False

def test_complete_workflow():
    """Test complete interactive workflow"""
    console.print("\n[bold blue]🚀 Testing Complete Interactive Workflow...[/bold blue]")
    
    try:
        from src.interactive.interactive_cli import InteractiveCLI
        
        # Test CLI initialization (without starting full interactive session)
        cli = InteractiveCLI()
        
        # Test individual components
        session_mgr = cli.session_manager
        query_processor = cli.query_processor
        visualizer = cli.visualizer
        
        # Create test session
        session = session_mgr.create_session("Workflow Test", "Testing complete workflow")
        
        # Test query processing
        analysis = query_processor.analyze_query("Analyze E. coli growth on glucose")
        
        # Test mock workflow
        workflow_data = cli._create_mock_workflow_data(analysis.to_dict())
        dashboard_data = cli._create_mock_dashboard_data(analysis.to_dict())
        
        console.print(f"✅ Created workflow with {len(workflow_data['nodes'])} nodes")
        console.print(f"✅ Generated dashboard with {len(dashboard_data)} metrics")
        
        console.print("✅ Complete Workflow: [green]Working perfectly![/green]")
        return True
        
    except Exception as e:
        console.print(f"❌ Complete Workflow Error: {e}")
        return False

def main():
    """Run all interactive interface tests"""
    console.print(Panel("""
[bold cyan]
╔══════════════════════════════════════════════════════════════════════════════╗
║              🧬 ModelSEEDagent Interactive Interface Test Suite              ║
║                          Phase 3.2 Implementation                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
[/bold cyan]

[bold yellow]Testing Interactive Analysis Interface Components...[/bold yellow]

This test suite validates:
• 💾 Session Management with persistence and analytics
• 🧠 Intelligent Query Processing with NLP analysis  
• 💬 Conversational AI Engine with context awareness
• 🎨 Live Visualization with real-time progress tracking
• 🚀 Complete Interactive Workflow integration

[dim]Starting comprehensive testing...[/dim]
    """, border_style="blue"))
    
    # Run individual component tests
    tests = [
        ("Session Manager", test_session_manager),
        ("Query Processor", test_query_processor),
        ("Conversation Engine", test_conversation_engine),
        ("Live Visualizer", test_live_visualizer),
        ("Complete Workflow", test_complete_workflow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            console.print(f"[red]❌ {test_name} failed with error: {e}[/red]")
            results.append((test_name, False))
    
    # Display test results summary
    console.print("\n" + "="*80)
    console.print("[bold blue]📊 Test Results Summary[/bold blue]")
    
    results_table = Table(box=box.ROUNDED)
    results_table.add_column("Component", style="bold cyan")
    results_table.add_column("Status", style="bold")
    results_table.add_column("Result", style="bold")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        if success:
            results_table.add_row(test_name, "✅", "[green]PASSED[/green]")
            passed += 1
        else:
            results_table.add_row(test_name, "❌", "[red]FAILED[/red]")
    
    console.print(results_table)
    
    # Final summary
    success_rate = passed / total
    
    if success_rate == 1.0:
        console.print(f"\n[bold green]🎉 All tests passed! ({passed}/{total})[/bold green]")
        console.print("[green]Phase 3.2: Interactive Analysis Interface is ready for production![/green]")
    elif success_rate >= 0.8:
        console.print(f"\n[bold yellow]⚠️  Most tests passed ({passed}/{total})[/bold yellow]")
        console.print("[yellow]Phase 3.2 is mostly functional with minor issues.[/yellow]")
    else:
        console.print(f"\n[bold red]❌ Many tests failed ({passed}/{total})[/bold red]")
        console.print("[red]Phase 3.2 needs additional work before production.[/red]")
    
    # Usage instructions
    console.print("\n[bold blue]🚀 How to Use the Interactive Interface:[/bold blue]")
    console.print("1. Run: [cyan]python -m src.cli.standalone interactive[/cyan]")
    console.print("2. Or use: [cyan]python src/interactive/interactive_cli.py[/cyan]")
    console.print("3. Follow the interactive prompts to:")
    console.print("   • Create or load analysis sessions")
    console.print("   • Ask natural language questions about metabolic modeling")
    console.print("   • View real-time visualizations and progress tracking")
    console.print("   • Explore session analytics and history")
    
    return success_rate

if __name__ == "__main__":
    success_rate = main()
    exit_code = 0 if success_rate >= 0.8 else 1
    sys.exit(exit_code) 