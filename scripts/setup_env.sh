#!/bin/bash

# Exit on error
set -e

# Script configuration
PYTHON_VERSION="3.11"
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"  # Get project root directory
VENV_NAME="venv"
VENV_PATH="$PROJECT_ROOT/$VENV_NAME"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
CONFIG_DIR="$PROJECT_ROOT/config"
DATA_DIR="$PROJECT_ROOT/data/models"

# Text formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}Setting up environment for Metabolic Modeling AI Agent${NC}\n"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo -e "${YELLOW}Checking Python installation...${NC}"
if ! command_exists python3; then
    echo -e "${RED}Python 3 is not installed. Please install Python $PYTHON_VERSION or later.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist, or update if it does
if [ -d "$VENV_PATH" ]; then
    echo -e "\n${YELLOW}Existing virtual environment found. Updating...${NC}"
    source "$VENV_PATH/bin/activate"
else
    echo -e "\n${YELLOW}Creating new virtual environment...${NC}"
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
python -m pip install --upgrade pip

# Install/upgrade requirements
echo -e "\n${YELLOW}Installing/updating requirements...${NC}"
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE" --upgrade
else
    echo -e "${RED}Requirements file not found at $REQUIREMENTS_FILE${NC}"
    exit 1
fi

# Create necessary directories if they don't exist
echo -e "\n${YELLOW}Checking necessary directories...${NC}"
mkdir -p "$CONFIG_DIR"
mkdir -p "$DATA_DIR"

# Check for config file
if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    echo -e "${YELLOW}Config file not found. Creating example config...${NC}"
    cat > "$CONFIG_DIR/config.yaml" <<EOL
# Example configuration file
llm:
  llm_backend: "argo"
  safety_settings:
    enabled: true
    max_api_calls: 100
    max_tokens: 50000

argo:
  user: "your_username"
  system_content: "You are an AI assistant specialized in metabolic modeling."
  models:
    gpt4:
      api_base: "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/"
      model_name: "gpt4"
  default_model: "gpt4"

local:
  model_name: "llama-3.1-8b"
  model_path: "/path/to/your/local/model"
  system_content: "You are an AI assistant specialized in metabolic modeling."

tools:
  configs:
    fba_config:
      default_objective: "biomass_reaction"
      solver: "glpk"
      tolerance: 1e-6
EOL
fi

# Create prompts directory and example files if they don't exist
echo -e "\n${YELLOW}Checking prompt templates...${NC}"
mkdir -p "$CONFIG_DIR/prompts"
for prompt_file in "metabolic.yaml" "rast.yaml"; do
    if [ ! -f "$CONFIG_DIR/prompts/$prompt_file" ]; then
        touch "$CONFIG_DIR/prompts/$prompt_file"
    fi
done

# Set up pre-commit hooks if git is available
if command_exists git; then
    echo -e "\n${YELLOW}Setting up git hooks...${NC}"
    if [ -d "$PROJECT_ROOT/.git" ]; then
        pip install pre-commit
        pre-commit install
    fi
fi

# Print success message
echo -e "\n${GREEN}Setup completed successfully!${NC}"
echo -e "Virtual environment location: $VENV_PATH"
echo -e "\nTo activate the virtual environment, run:"
echo -e "${BOLD}source $VENV_PATH/bin/activate${NC}"

# Optional: Check for GPU support
if pip list | grep -q "torch"; then
    echo -e "\n${YELLOW}Checking GPU support...${NC}"
    python -c "import torch; print('GPU available:', torch.cuda.is_available())"
fi
