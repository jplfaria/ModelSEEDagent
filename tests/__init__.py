import os
import sys
from pathlib import Path

# Add the src directory to the Python path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

# Set up test configuration
TEST_CONFIG_PATH = Path(__file__).parent / "data" / "test_config.yaml"
TEST_MODEL_PATH = Path(__file__).parent / "data" / "models" / "test_model.xml"
TEST_PROMPTS_DIR = Path(__file__).parent / "data" / "prompts"
