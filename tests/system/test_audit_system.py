#!/usr/bin/env python3
"""
Test script to verify the Tool Execution Audit System implementation
"""
import json
import os
import tempfile
from pathlib import Path

# Set up environment for testing
os.environ["AUDIT_ENABLED"] = "true"

from src.tools.biochem.resolver import BiochemEntityResolverTool

# Import after setting environment
from src.tools.cobra.fba import FBATool


def test_audit_system():
    """Test the audit system with a simple tool execution"""
    print("Testing Tool Execution Audit System...")

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temp directory: {temp_dir}")

        # Set audit directory
        os.environ["AUDIT_WATCH_DIRS"] = temp_dir

        try:
            # Test 1: FBA Tool (requires a model file)
            print("\n1. Testing FBA Tool audit...")
            fba_tool = FBATool({})

            # Check if e_coli_core model exists
            model_path = "data/models/e_coli_core.xml"
            if os.path.exists(model_path):
                print(f"Running FBA on {model_path}")
                result = fba_tool.run({"model_path": model_path})
                print(f"FBA Result: {result.success}")

                # Check if audit file was created
                if hasattr(result, "metadata") and "audit_file" in result.metadata:
                    audit_file = result.metadata["audit_file"]
                    print(f"Audit file created: {audit_file}")

                    # Read and verify audit content
                    if os.path.exists(audit_file):
                        with open(audit_file, "r") as f:
                            audit_data = json.load(f)
                        print(f"Audit record keys: {list(audit_data.keys())}")
                        print(f"Tool name: {audit_data.get('tool_name')}")
                        print(f"Success: {audit_data.get('output', {}).get('success')}")
                    else:
                        print("Warning: Audit file path provided but file not found")
                else:
                    print("Info: No audit file metadata found in result")
            else:
                print(f"Skipping FBA test - model file not found: {model_path}")

            # Test 2: Biochem Resolver Tool (doesn't need external files)
            print("\n2. Testing Biochem Resolver Tool audit...")
            try:
                resolver_tool = BiochemEntityResolverTool({})
                result = resolver_tool.run({"entity_id": "cpd00001"})
                print(f"Resolver Result: {result.success}")

                if hasattr(result, "metadata") and "audit_file" in result.metadata:
                    audit_file = result.metadata["audit_file"]
                    print(f"Audit file created: {audit_file}")

                    if os.path.exists(audit_file):
                        with open(audit_file, "r") as f:
                            audit_data = json.load(f)
                        print(f"Tool name: {audit_data.get('tool_name')}")
                        print(
                            f"Input entity_id: {audit_data.get('input', {}).get('entity_id')}"
                        )
                        print(f"Success: {audit_data.get('output', {}).get('success')}")
                    else:
                        print("Warning: Audit file path provided but file not found")
                else:
                    print("Info: No audit file metadata found in result")

            except Exception as e:
                print(f"Biochem resolver test failed (expected if no database): {e}")

        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback

            traceback.print_exc()


def test_console_capture():
    """Test console output capture"""
    print("\n3. Testing console capture...")

    from src.tools.audit import ConsoleCapture

    with ConsoleCapture() as capture:
        print("This should be captured")
        print("So should this", file=capture._stderr_capture)

    stdout, stderr = capture.get_output()
    print(f"Captured stdout: {repr(stdout)}")
    print(f"Captured stderr: {repr(stderr)}")


def test_file_tracker():
    """Test file tracking"""
    print("\n4. Testing file tracker...")

    from src.tools.audit import FileTracker

    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = FileTracker([temp_dir])

        # Create some files
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content")

        modified_files = tracker.get_modified_files()
        print(f"Tracked files: {modified_files}")


if __name__ == "__main__":
    test_console_capture()
    test_file_tracker()
    test_audit_system()
    print("\nAudit system test completed!")
