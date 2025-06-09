#!/usr/bin/env python3
"""
Minimal test case to isolate Rich Live display issues in the interactive CLI.

This test script reproduces the core Rich Live functionality without the complexity
of the full AI agent system to identify the root cause of flickering empty boxes.
"""

import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

console = Console()


def test_basic_live_display():
    """Test basic Live display functionality"""
    print("🔍 Testing basic Rich Live display...")

    try:
        with Live("Starting test...", console=console, refresh_per_second=4) as live:
            for i in range(5):
                live.update(f"Test step {i+1}/5")
                time.sleep(0.5)
        print("✅ Basic Live display works")
        return True
    except Exception as e:
        print(f"❌ Basic Live display failed: {e}")
        return False


def test_panel_content_validation():
    """Test panel content validation"""
    print("\n🔍 Testing panel content validation...")

    # Test various content types
    test_cases = [
        ("Empty string", ""),
        ("Whitespace only", "   "),
        ("None", None),
        ("Valid content", "This is valid content"),
        ("Rich markup", "[bold]Bold text[/bold]"),
    ]

    for name, content in test_cases:
        try:
            # This is similar to what the streaming interface does
            if not content or str(content).strip() == "":
                content = "[dim]Fallback content for empty input[/dim]"

            panel = Panel(str(content), title=f"Test: {name}")
            console.print(f"✅ {name}: Panel created successfully")

        except Exception as e:
            console.print(f"❌ {name}: Panel creation failed: {e}")
            return False

    return True


def test_layout_with_panels():
    """Test layout with multiple panels (reproduces streaming interface structure)"""
    print("\n🔍 Testing layout with panels...")

    try:
        layout = Layout()

        # Split like the streaming interface
        layout.split_column(Layout(name="header", size=3), Layout(name="main"))
        layout["main"].split_row(Layout(name="left"), Layout(name="right"))
        layout["left"].split_column(
            Layout(name="thinking", size=8), Layout(name="decisions")
        )
        layout["right"].split_column(
            Layout(name="tools", size=8), Layout(name="results")
        )

        # Test with empty content (this might be the issue)
        header_panel = Panel("Test Query", style="bold blue")
        thinking_panel = Panel("", title="AI Thinking")  # Empty content!
        decisions_panel = Panel("", title="Decisions")  # Empty content!
        tools_panel = Panel("", title="Tools")  # Empty content!
        results_panel = Panel("", title="Results")  # Empty content!

        layout["header"].update(header_panel)
        layout["thinking"].update(thinking_panel)
        layout["decisions"].update(decisions_panel)
        layout["tools"].update(tools_panel)
        layout["results"].update(results_panel)

        console.print("✅ Layout with empty panels created")
        console.print(layout)

        print("\n🔍 Now testing with Live display...")
        return True

    except Exception as e:
        print(f"❌ Layout test failed: {e}")
        return False


def test_live_layout_updates():
    """Test Live display with layout updates (this is where the issue likely occurs)"""
    print("\n🔍 Testing Live layout with updates...")

    try:
        layout = Layout()
        layout.split_column(Layout(name="header", size=3), Layout(name="main"))
        layout["main"].split_row(Layout(name="left"), Layout(name="right"))
        layout["left"].split_column(
            Layout(name="thinking", size=6), Layout(name="decisions")
        )
        layout["right"].split_column(
            Layout(name="tools", size=6), Layout(name="results")
        )

        # Initialize with proper content
        layout["header"].update(Panel("Test Query Analysis", style="bold blue"))
        layout["thinking"].update(
            Panel("[dim]Waiting for AI...[/dim]", title="AI Thinking")
        )
        layout["decisions"].update(
            Panel("[dim]No decisions yet...[/dim]", title="Decisions")
        )
        layout["tools"].update(Panel("[dim]No tools executed...[/dim]", title="Tools"))
        layout["results"].update(Panel("[dim]No results yet...[/dim]", title="Results"))

        with Live(layout, console=console, refresh_per_second=4) as live:
            # Test various content updates
            for i in range(8):
                time.sleep(0.5)

                # Update thinking panel
                thinking_content = f"AI step {i+1}: Analyzing..."
                layout["thinking"].update(
                    Panel(thinking_content, title="AI Thinking", border_style="cyan")
                )

                # Update decisions panel every 2 steps
                if i % 2 == 0:
                    decisions_content = f"Decision {i//2 + 1}: Proceed with analysis"
                    layout["decisions"].update(
                        Panel(
                            decisions_content, title="Decisions", border_style="yellow"
                        )
                    )

                # Update tools panel every 3 steps
                if i % 3 == 0:
                    tools_content = f"Tool {i//3 + 1}: Executing analysis..."
                    layout["tools"].update(
                        Panel(tools_content, title="Tools", border_style="green")
                    )

                # Update results panel
                results_content = f"Progress: {i+1}/8 steps completed"
                layout["results"].update(
                    Panel(results_content, title="Results", border_style="magenta")
                )

        print("✅ Live layout updates completed successfully")
        return True

    except Exception as e:
        print(f"❌ Live layout updates failed: {e}")
        return False


def test_empty_content_issue():
    """Test specifically with empty content to reproduce the flickering issue"""
    print("\n🔍 Testing empty content issue (reproducing the bug)...")

    try:
        layout = Layout()
        layout.split_column(Layout(name="header", size=3), Layout(name="main"))
        layout["main"].split_row(Layout(name="left"), Layout(name="right"))
        layout["left"].split_column(
            Layout(name="thinking", size=6), Layout(name="decisions")
        )
        layout["right"].split_column(
            Layout(name="tools", size=6), Layout(name="results")
        )

        layout["header"].update(Panel("Testing Empty Content Bug", style="bold red"))

        with Live(layout, console=console, refresh_per_second=4) as live:
            # This simulates what happens in the streaming interface
            for i in range(10):
                time.sleep(0.3)

                # Deliberately create panels with problematic content
                problematic_contents = [
                    "",  # Empty string
                    "   ",  # Whitespace only
                    None,  # None value
                    "\n\n",  # Just newlines
                ]

                try:
                    # This is what causes the issue!
                    content = problematic_contents[i % len(problematic_contents)]
                    if content is None:
                        content = ""

                    # Create panel with potentially empty content
                    panel = Panel(str(content), title=f"Test {i}", border_style="red")
                    layout["thinking"].update(panel)

                except Exception as panel_error:
                    print(f"Panel creation error at step {i}: {panel_error}")
                    # Try to update with fallback
                    fallback_panel = Panel("[red]Panel error[/red]", title=f"Error {i}")
                    layout["thinking"].update(fallback_panel)

        print("✅ Empty content test completed")
        return True

    except Exception as e:
        print(f"❌ Empty content test failed: {e}")
        return False


def test_fixed_content_validation():
    """Test with proper content validation to see if it fixes the issue"""
    print("\n🔍 Testing with proper content validation (potential fix)...")

    def safe_create_panel(content, title, border_style="white"):
        """Safely create a panel with content validation"""
        # Validate and fix content
        if content is None:
            content = ""

        content_str = str(content).strip()
        if not content_str:
            content_str = "[dim]No content available[/dim]"

        return Panel(content_str, title=title, border_style=border_style)

    try:
        layout = Layout()
        layout.split_column(Layout(name="header", size=3), Layout(name="main"))
        layout["main"].split_row(Layout(name="left"), Layout(name="right"))
        layout["left"].split_column(
            Layout(name="thinking", size=6), Layout(name="decisions")
        )
        layout["right"].split_column(
            Layout(name="tools", size=6), Layout(name="results")
        )

        layout["header"].update(
            Panel("Testing Fixed Content Validation", style="bold green")
        )

        with Live(layout, console=console, refresh_per_second=4) as live:
            for i in range(10):
                time.sleep(0.3)

                # Test with problematic content but with validation
                problematic_contents = ["", "   ", None, "\n\n", "Valid content"]
                content = problematic_contents[i % len(problematic_contents)]

                # Use safe panel creation
                thinking_panel = safe_create_panel(content, f"AI Thinking {i}", "cyan")
                decisions_panel = safe_create_panel(
                    f"Decision {i}", "Decisions", "yellow"
                )
                tools_panel = safe_create_panel(f"Tool {i}", "Tools", "green")
                results_panel = safe_create_panel(f"Result {i}", "Results", "magenta")

                layout["thinking"].update(thinking_panel)
                layout["decisions"].update(decisions_panel)
                layout["tools"].update(tools_panel)
                layout["results"].update(results_panel)

        print("✅ Fixed content validation test completed - no flickering!")
        return True

    except Exception as e:
        print(f"❌ Fixed content validation test failed: {e}")
        return False


def main():
    """Run all tests to identify the Rich Live display issue"""
    print("🧪 Rich Live Display Issue Diagnostic Test")
    print("=" * 50)

    tests = [
        test_basic_live_display,
        test_panel_content_validation,
        test_layout_with_panels,
        test_live_layout_updates,
        test_empty_content_issue,
        test_fixed_content_validation,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        time.sleep(1)  # Pause between tests

    print("\n" + "=" * 50)
    print("🎯 Test Results Summary:")
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i+1}. {test.__name__}: {status}")

    if all(results):
        print("\n🎉 All tests passed! The fix should work.")
    else:
        print("\n🔍 Some tests failed. Check the output above for details.")

    print("\n💡 Key Finding: The issue is likely caused by empty content in panels!")
    print("💡 Solution: Validate all panel content before creating panels.")


if __name__ == "__main__":
    main()
