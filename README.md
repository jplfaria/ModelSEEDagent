# ModelSEEDagent

🧬 **AI-Powered Metabolic Modeling with LangGraph Workflows**

ModelSEEDagent is a sophisticated AI agent system for metabolic modeling that combines the power of large language models with specialized bioinformatics tools. Built on LangGraph for intelligent workflow orchestration, it provides both command-line and interactive interfaces for seamless metabolic model analysis.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)
- Access to Argo Gateway or OpenAI API

### Installation

```bash
# Clone the repository
git clone https://github.com/jplfaria/ModelSEEDagent.git
cd ModelSEEDagent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Environment Setup

```bash
# For Argo Gateway (recommended)
export ARGO_USER="your_anl_username"
export DEFAULT_MODEL_NAME="gpt4olatest"
export DEFAULT_LLM_BACKEND="argo"

# For OpenAI (alternative)
export OPENAI_API_KEY="your_openai_key"
export DEFAULT_LLM_BACKEND="openai"
```

## 🎯 Usage

### Interactive Analysis Interface

Launch the natural language interactive interface:

```bash
python examples/launch_with_argo.py
```

**Example interactions:**
- *"Load and analyze the E. coli model"*
- *"What is the growth rate on glucose minimal media?"*
- *"Run flux balance analysis and show results"*
- *"Create a network visualization of central carbon metabolism"*

### Command Line Interface

Use the professional CLI for scripted analysis:

```bash
# Show version and help
./modelseed-agent --version
./modelseed-agent --help

# Interactive setup
./modelseed-agent setup

# Analyze a model
./modelseed-agent analyze data/models/iML1515.xml

# Check system status
./modelseed-agent status
```

### Python API

```python
from src.agents.metabolic import MetabolicAgent
from src.llm.argo import ArgoLLM
from src.tools.cobra.fba import FBATool

# Initialize components
llm = ArgoLLM(config)
tools = [FBATool(config)]
agent = MetabolicAgent(llm, tools, config)

# Run analysis
result = agent.analyze_model("What pathways are most active?")
print(result.message)
```

## 📁 Repository Structure

```
ModelSEEDagent/
├── 📚 docs/                          # Documentation
│   ├── IMPLEMENTATION_PLAN.md        # Development roadmap
│   ├── INTERACTIVE_GUIDE.md         # User guide
│   └── REPOSITORY_CLEANUP_PLAN.md   # Cleanup documentation
├── 🎯 examples/                      # Usage examples
│   └── launch_with_argo.py          # Interactive interface demo
├── 🧪 tests/                         # Test suite
│   ├── integration/                  # End-to-end tests
│   ├── fixtures/                     # Test data
│   ├── test_agents.py               # Agent tests
│   ├── test_llm.py                  # LLM tests
│   └── test_tools.py                # Tool tests
├── 🎮 src/                           # Main source code
│   ├── agents/                       # AI agents
│   │   ├── base.py                  # Base agent class
│   │   ├── metabolic.py             # Metabolic modeling agent
│   │   └── langgraph_metabolic.py   # LangGraph workflow agent
│   ├── cli/                         # Command line interfaces
│   │   ├── main.py                  # Professional CLI
│   │   └── standalone.py            # Standalone CLI
│   ├── interactive/                 # Interactive interface
│   │   ├── interactive_cli.py       # Main interactive CLI
│   │   ├── conversation.py          # Natural language processing
│   │   ├── session_manager.py       # Session management
│   │   └── live_visualizer.py       # Real-time visualizations
│   ├── llm/                         # Large language models
│   │   ├── argo.py                  # Argo Gateway client
│   │   ├── openai_llm.py           # OpenAI client
│   │   └── base.py                  # LLM base class
│   ├── tools/                       # Specialized tools
│   │   ├── cobra/                   # COBRApy integration
│   │   ├── modelseed/              # ModelSEED tools
│   │   └── rast/                   # RAST annotation tools
│   ├── workflow/                    # Workflow automation
│   │   ├── workflow_engine.py      # Core workflow engine
│   │   ├── batch_processor.py      # Batch processing
│   │   └── scheduler.py            # Advanced scheduling
│   └── config/                      # Configuration management
├── 📊 data/                          # Sample data and models
├── 📝 config/                        # Configuration templates
├── 🚀 modelseed-agent               # CLI entry point
├── ⚙️ setup.py                      # Package configuration
└── 🔧 requirements.txt              # Dependencies
```

## 🧬 Features

### 🤖 **AI-Powered Analysis**
- **Natural Language Interface**: Ask questions in plain English
- **GPT-4o Latest Integration**: State-of-the-art language model via Argo
- **Intelligent Tool Selection**: Automatically chooses appropriate analysis tools
- **Context-Aware Conversations**: Maintains conversation history and context

### 🛠️ **Metabolic Modeling Tools**
- **Flux Balance Analysis (FBA)**: Growth rate optimization and flux calculations
- **Model Analysis**: Comprehensive model structure and property analysis
- **Pathway Analysis**: Metabolic pathway connectivity and flux analysis
- **Network Visualization**: Interactive metabolic network diagrams
- **Comparative Analysis**: Multi-model and multi-condition comparisons

### 🎨 **Advanced Visualizations**
- **Real-time Dashboards**: Live progress tracking and performance monitoring
- **Interactive Networks**: Explorable metabolic network visualizations
- **Flux Heatmaps**: Visual representation of metabolic flux distributions
- **Workflow Diagrams**: Analysis pipeline visualization

### ⚡ **Workflow Automation**
- **LangGraph Integration**: Sophisticated workflow orchestration
- **Parallel Processing**: Concurrent execution of independent analysis steps
- **Batch Operations**: Process multiple models or conditions simultaneously
- **Smart Scheduling**: Resource-aware task scheduling and optimization

### 🎯 **Professional Interfaces**
- **Rich CLI**: Beautiful terminal interface with progress bars and formatting
- **Interactive Sessions**: Persistent analysis sessions with full history
- **Session Management**: Save, resume, and share analysis sessions
- **Multiple Output Formats**: Rich text, JSON, and structured data export

## 🧪 Testing

### Run All Tests
```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Structure
- **Unit Tests**: `tests/` - Individual component testing
- **Integration Tests**: `tests/integration/` - End-to-end workflow testing
- **Test Fixtures**: `tests/fixtures/` - Sample data and configurations

## 🔧 Development

### Code Quality
Pre-commit hooks ensure consistent code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### Architecture

ModelSEEDagent follows a modular architecture:

1. **Agents** (`src/agents/`) - High-level AI orchestrators
2. **Tools** (`src/tools/`) - Specialized analysis capabilities
3. **LLMs** (`src/llm/`) - Language model integrations
4. **Workflows** (`src/workflow/`) - Process automation
5. **Interfaces** (`src/cli/`, `src/interactive/`) - User interaction

### Adding New Tools

```python
from src.tools.base import BaseTool

class MyTool(BaseTool):
    def _run(self, query: str) -> Dict[str, Any]:
        # Implementation here
        return {"result": "analysis_output"}
```

## 📚 Documentation

- **[Implementation Plan](docs/IMPLEMENTATION_PLAN.md)** - Development roadmap and progress
- **[Interactive Guide](docs/INTERACTIVE_GUIDE.md)** - Comprehensive user guide
- **[Repository Cleanup](docs/REPOSITORY_CLEANUP_PLAN.md)** - Cleanup and organization plan

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run pre-commit hooks (`pre-commit run --all-files`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ModelSEED** - Metabolic model reconstruction platform
- **COBRApy** - Constraint-based metabolic modeling
- **LangGraph** - Workflow orchestration framework
- **LangChain** - LLM application framework
- **Argo Gateway** - LLM access infrastructure

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/jplfaria/ModelSEEDagent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jplfaria/ModelSEEDagent/discussions)
- **Documentation**: [docs/](docs/)

---

🧬 **Ready for professional metabolic modeling analysis!** 🤖
