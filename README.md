# Metabolic Modeling AI Agent Framework

## Overview

A sophisticated Python framework that integrates metabolic modeling tools with Large Language Models (LLMs) to automate and enhance metabolic modeling workflows. The framework provides a modular, extensible architecture supporting multiple LLM backends and various metabolic analysis tools.

## Features

- **Multiple LLM Backend Support**:
  - Argonne's Internal Argo Gateway API
  - OpenAI API
  - Local LLM Models (e.g., LLaMA)

- **Core Metabolic Modeling Tools**:
  - Flux Balance Analysis (FBA)
  - Model Analysis
  - Pathway Analysis
  - Model Validation

- **Advanced Modeling Capabilities**:
  - RAST Genome Annotation Integration
  - ModelSEED Model Building
  - Automated Gap Filling
  - Model Comparison

- **Robust Architecture**:
  - Modular Design
  - Type-Safe with Pydantic
  - Comprehensive Testing
  - Async Support
  - Extensive Error Handling

## Project Structure

```
metabolic_agent/
├── src/
│   ├── llm/                    # LLM Infrastructure
│   │   ├── base.py            # Base LLM interface
│   │   ├── argo.py            # Argo implementation
│   │   ├── openai_llm.py      # OpenAI implementation
│   │   └── local_llm.py       # Local LLM implementation
│   │
│   ├── tools/                 # Modeling Tools
│   │   ├── base.py           # Base tool interface
│   │   ├── cobra/            # COBRA tools
│   │   │   ├── fba.py       # FBA analysis
│   │   │   ├── analysis.py  # Model analysis
│   │   │   └── utils.py     # Utilities
│   │   ├── rast/            # RAST tools
│   │   │   └── annotation.py # Genome annotation
│   │   └── modelseed/       # ModelSEED tools
│   │       ├── builder.py   # Model building
│   │       └── gapfill.py   # Gap filling
│   │
│   ├── agents/               # Agent Implementation
│   │   ├── base.py          # Base agent class
│   │   ├── metabolic.py     # Metabolic agent
│   │   └── factory.py       # Agent factory
│   │
│   └── config/              # Configuration
│       ├── settings.py      # Config management
│       └── prompts.py       # Prompt templates
│
├── tests/                    # Test Suite
│   ├── test_llm/           # LLM tests
│   ├── test_tools/         # Tool tests
│   └── test_agents/        # Agent tests
│
├── config/                  # Configuration Files
│   ├── config.yaml         # Main configuration
│   └── prompts/            # Prompt templates
│       ├── metabolic.yaml
│       └── rast.yaml
│
├── notebooks/              # Jupyter Notebooks
│   ├── examples/          # Usage examples
│   └── tutorials/         # Step-by-step guides
│
├── scripts/               # Utility Scripts
│   ├── setup_env.sh      # Unix setup
│   └── setup_env.bat     # Windows setup
│
└── data/                 # Data Directory
    └── models/          # Metabolic models
```

## Prerequisites

- Python 3.11 or higher
- Virtual Environment (venv or conda)
- Git
- Access credentials for desired LLM backends

### Optional Requirements
- GPU for local LLM models
- CUDA support for PyTorch
- Access to Argo Gateway API
- OpenAI API key

## Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/metabolic_agent.git
cd metabolic_agent
```

2. **Set Up Environment**

For Unix-like systems:
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

For Windows:
```batch
scripts\setup_env.bat
```

3. **Configure the Framework**
- Update `config/config.yaml` with your settings
- Configure LLM backend credentials
- Adjust tool-specific parameters

## Usage

### Basic Usage

```python
from src.llm import LLMFactory
from src.tools import ToolRegistry
from src.agents import AgentFactory

# Initialize components
llm = LLMFactory.create("argo", config["llm"])
tools = [
    ToolRegistry.create_tool(name, config["tools"][name])
    for name in ["run_metabolic_fba", "analyze_metabolic_model"]
]

# Create agent
agent = AgentFactory.create_agent(
    agent_type="metabolic",
    llm=llm,
    tools=tools,
    config=config["agent"]
)

# Run analysis
result = agent.analyze_model("path/to/model.xml")
```

### Advanced Usage

#### Running FBA Analysis
```python
# Create specific tool instance
fba_tool = ToolRegistry.create_tool("run_metabolic_fba", config["tools"]["fba"])

# Run analysis
result = fba_tool.run({
    "model_path": "path/to/model.xml",
    "objective": "BIOMASS_reaction"
})
```

#### Genome Annotation Integration
```python
# Create RAST tool
rast_tool = ToolRegistry.create_tool("run_rast_annotation", config["tools"]["rast"])

# Run annotation
result = rast_tool.run({
    "sequence_file": "path/to/genome.fasta",
    "file_type": "fasta"
})
```

#### Model Building and Gap Filling
```python
# Create ModelSEED tools
builder = ToolRegistry.create_tool("build_metabolic_model", config["tools"]["modelseed"])
gapfiller = ToolRegistry.create_tool("gapfill_model", config["tools"]["gapfill"])

# Build and gap fill model
model_result = builder.run({
    "annotation_file": "path/to/annotations.json",
    "output_path": "path/to/output.xml"
})

gapfill_result = gapfiller.run({
    "model_path": model_result.data["model_path"],
    "media_condition": "Complete"
})
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run specific test categories:
```bash
pytest tests/test_llm/         # Test LLM functionality
pytest tests/test_tools/       # Test modeling tools
pytest tests/test_agents/      # Test agents
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation as needed
- Use type hints
- Handle errors appropriately

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ModelSEED team
- COBRA toolbox developers
- Argonne National Laboratory for Argo API access
- OpenAI for API access

## Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **Organization**: Your Organization

## Disclaimer

This software is provided as-is. Always verify results independently for critical applications.