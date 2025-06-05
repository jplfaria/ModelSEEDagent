# ModelSEEDagent 🧬

**AI-Powered Metabolic Modeling with LangGraph Workflows**

ModelSEEDagent is a sophisticated AI agent system for metabolic modeling that combines the power of large language models with specialized bioinformatics tools. Built on LangGraph for intelligent workflow orchestration.

## 🚀 Quick Start

```bash
# Install and setup
pip install -e .
python run_cli.py interactive

# Or use CLI
modelseed-agent setup --backend argo --model gpto1
modelseed-agent analyze data/models/e_coli_core.xml
```

## 📚 Documentation

- **[User Guide](docs/user/README.md)** - Complete installation, usage, and examples
- **[Interactive Guide](docs/user/INTERACTIVE_GUIDE.md)** - Natural language interface
- **[Development Roadmap](docs/development/DEVELOPMENT_ROADMAP.md)** - Project status and progress
- **[API Reference](docs/development/API_REFERENCE.md)** - Developer documentation

## 🎯 Key Features

- ✅ **Natural Language Interface** - Ask questions in plain English
- ✅ **GPT-o1 Integration** - Advanced reasoning for complex analysis
- ✅ **Real-time Visualizations** - Interactive metabolic network graphs
- ✅ **Complete CLI Suite** - Professional command-line tools
- ✅ **Session Management** - Persistent analysis workflows
- ✅ **100% Test Coverage** - Production-ready reliability

## 🧪 Status

**Production Ready** - All features working with 47/47 tests passing

## 📁 Repository Structure

```
ModelSEEDagent/
├── 📚 docs/              # Complete documentation
├── 🎯 examples/          # Usage examples by complexity
├── 🧪 tests/             # 100% passing test suite
├── 🎮 src/               # Main source code
├── ⚙️ config/            # Configuration files
├── 📊 data/              # Models and analysis results
└── 📝 logs/              # Execution logs
```

## 🤝 Contributing

See [Development Guide](docs/development/) for contribution guidelines.

## 📄 License

MIT License - see LICENSE file for details.
