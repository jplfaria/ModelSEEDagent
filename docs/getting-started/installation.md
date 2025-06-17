# Installation Guide

ModelSEEDagent is a sophisticated AI-powered metabolic modeling platform that combines the ModelSEED and COBRApy ecosystems with advanced AI reasoning capabilities.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ModelSEED/ModelSEEDagent.git
cd ModelSEEDagent

# Install in development mode with all optional dependencies
pip install -e .[all]

# Verify installation
modelseed-agent --help
```

## Prerequisites

### Python Requirements

- **Python 3.8+** (recommended: 3.9 or 3.10)
- **pip** (latest version)
- **Virtual environment** (recommended)

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev libxml2-dev libxslt-dev
```

#### macOS
```bash
# Using Homebrew
brew install libxml2 libxslt

# Using MacPorts
sudo port install libxml2 libxslt
```

#### Windows
```bash
# Windows Subsystem for Linux (WSL) is recommended
# Or use Anaconda/Miniconda for easier dependency management
```

## Installation Methods

### Method 1: Development Installation (Recommended)

For development and contributing:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Clone repository
git clone https://github.com/ModelSEED/ModelSEEDagent.git
cd ModelSEEDagent

# Install in development mode with all dependencies
pip install -e .[all]

# Install additional development dependencies (if needed)
# pip install -r requirements-dev.txt
```

### Method 2: Production Installation

For production use:

```bash
# Install from PyPI (when available)
pip install modelseed-agent

# Or install from GitHub
pip install git+https://github.com/ModelSEED/ModelSEEDagent.git
```

### Method 3: Conda Installation

Using conda/mamba for better dependency management:

```bash
# Create conda environment
conda create -n modelseed-agent python=3.9
conda activate modelseed-agent

# Install core dependencies
conda install -c bioconda cobra
conda install -c conda-forge requests python-dotenv rich

# Install ModelSEEDagent
pip install -e .[all]
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_key_here

# Argo Gateway (if using)
ARGO_GATEWAY_URL=https://your-argo-gateway.com
ARGO_API_KEY=your_argo_key_here

# Debug Configuration
MODELSEED_DEBUG_LEVEL=INFO
MODELSEED_DEBUG_COBRAKBASE=false
MODELSEED_DEBUG_LANGGRAPH=false
MODELSEED_DEBUG_HTTP=false
MODELSEED_DEBUG_TOOLS=true
MODELSEED_DEBUG_LLM=false
MODELSEED_LOG_LLM_INPUTS=false

# Optional: Custom paths
MODELSEED_DATA_DIR=/path/to/data
MODELSEED_LOG_DIR=/path/to/logs
```

### API Keys Setup

#### OpenAI (Experimental)
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an API key
3. Add to `.env`: `OPENAI_API_KEY=your_key_here`

#### Argo Gateway (Recommended)
1. Contact your Argo administrator for access
2. Add credentials to `.env`

## Verification

### Basic Installation Check

```bash
# Check version
modelseed-agent --version

# Check available commands
modelseed-agent --help

# Test basic functionality
modelseed-agent debug
```

### Comprehensive Test

```bash
# Run test suite
pytest tests/

# Run functional tests
python tests/run_all_functional_tests.py

# Test specific functionality
python -m src.cli.main analyze --help
```

### Example Analysis

```bash
# Test with example model
modelseed-agent analyze data/examples/e_coli_core.xml

# Interactive mode
modelseed-agent analyze

# Advanced AI features
modelseed-agent phase8
```

## Dependency Details

### Core Dependencies

- **cobra**: Constraint-based modeling
- **modelseedpy**: ModelSEED Python library
- **requests**: HTTP client
- **python-dotenv**: Environment variable management
- **rich**: Rich terminal formatting
- **click**: Command-line interface framework

### AI/LLM Dependencies

- **openai**: OpenAI API client
- **langgraph**: Workflow orchestration
- **langchain**: LLM framework components

### Analysis Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **matplotlib**: Plotting (optional)
- **plotly**: Interactive plots (optional)

### Development Dependencies

- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# ModuleNotFoundError
pip install -e .  # Reinstall in development mode

# Missing dependencies
pip install -r requirements.txt
```

#### Permission Issues
```bash
# Use virtual environment
python -m venv venv
source venv/bin/activate
pip install -e .
```

#### COBRApy Installation Issues
```bash
# Use conda for better dependency management
conda install -c bioconda cobra
```

#### Network/Proxy Issues
```bash
# Configure pip for proxy
pip install --proxy http://proxy.server:port -e .

# Or set environment variables
export HTTP_PROXY=http://proxy.server:port
export HTTPS_PROXY=https://proxy.server:port
```

### Performance Optimization

#### Speed up installation
```bash
# Use faster dependency resolver
pip install --use-feature=2020-resolver -e .

# Parallel installation
pip install --upgrade pip
pip install -e . --no-deps
pip install -r requirements.txt
```

#### Memory optimization
```bash
# For systems with limited memory
export PYTHONHASHSEED=0
pip install --no-cache-dir -e .
```

## Platform-Specific Notes

### macOS
- Install Xcode command line tools: `xcode-select --install`
- Use Homebrew for system dependencies
- Consider using pyenv for Python version management

### Linux
- Ensure build tools are installed
- Use system package manager for dependencies
- Consider using conda for scientific computing stack

### Windows
- Windows Subsystem for Linux (WSL) is recommended
- Use Anaconda/Miniconda for easier dependency management
- PowerShell may require execution policy changes

## Next Steps

After installation:

1. **[User Guide](user/README.md)**: Learn basic usage
2. **[API Documentation](api/overview.md)**: Explore programmatic access
3. **[Architecture Guide](ARCHITECTURE.md)**: Understand system design
4. **[Tool Reference](TOOL_REFERENCE.md)**: Detailed tool documentation

For issues or questions, see the [Troubleshooting Guide](troubleshooting.md) or open an issue on GitHub.
