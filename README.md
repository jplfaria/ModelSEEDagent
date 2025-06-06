# ModelSEEDagent 🧬

**Advanced AI-Powered Metabolic Modeling Platform**

ModelSEEDagent is a production-ready AI-powered metabolic modeling platform that combines the power of large language models with **17 specialized bioinformatics tools**. Built on LangGraph for intelligent workflow orchestration, it provides the most comprehensive metabolic modeling AI assistant available.

## 🚀 Quick Start

```bash
# Install with all dependencies
pip install .[all]

# Setup and launch interactive interface
modelseed-agent setup --backend argo --model gpt4o
modelseed-agent interactive

# Or analyze a model directly
modelseed-agent analyze data/models/e_coli_core.xml --query "Find essential genes"
```

## 📚 Documentation

- **[User Guide](docs/user/README.md)** - Complete installation, usage, and examples
- **[Interactive Tutorial](notebooks/comprehensive_tutorial.ipynb)** - Hands-on showcase of all capabilities
- **[Interactive Guide](docs/user/INTERACTIVE_GUIDE.md)** - Natural language interface
- **[Development Roadmap](docs/development/DEVELOPMENT_ROADMAP.md)** - Project status and progress

## 🎯 Core Capabilities

- 🧬 **Complete Genome-to-Model Pipeline** - RAST annotation → Model building → Gapfilling
- 📊 **Advanced COBRA Analysis** - 11 tools covering 60% of COBRApy capabilities
- 🔄 **Universal Compatibility** - Perfect ModelSEED ↔ COBRApy integration
- 🧪 **Biochemistry Intelligence** - Universal ID resolution across 45K+ compounds and 55K+ reactions
- 🕵️ **AI Transparency** - Advanced hallucination detection and audit system
- 💬 **Natural Language Interface** - Conversational AI for complex metabolic analysis
- 📈 **Real-time Visualizations** - Interactive metabolic network graphs
- 🛠️ **Professional CLI** - Complete command-line interface with audit capabilities

## 🛠️ Specialized Tools (17 Total)

### ModelSEED Integration (4 tools)
- **Genome Annotation** (`annotate_genome_rast`) - BV-BRC RAST service integration
- **Model Building** (`build_metabolic_model`) - MSBuilder with template selection
- **Gapfilling** (`gapfill_model`) - Advanced MSGapfill algorithms
- **Protein Annotation** (`annotate_proteins_rast`) - Individual protein sequence annotation

### Advanced COBRA Analysis (11 tools)
- **Basic Analysis**: FBA, minimal media, auxotrophy analysis
- **Advanced Analysis**: Flux variability, gene deletion, essentiality analysis
- **Statistical Methods**: Flux sampling, production envelope analysis
- **Specialized Tools**: Reaction expression, missing media analysis

### Biochemistry Database (2 tools)
- **Universal ID Resolution** (`resolve_biochem_entity`) - ModelSEED ↔ BiGG ↔ KEGG mapping
- **Biochemistry Search** (`search_biochem`) - Compound/reaction discovery by name

## 🧪 Production Status

**✅ Production Ready** - All phases complete with comprehensive testing

| Component | Status | Coverage |
|-----------|--------|----------|
| **Tool Suite** | ✅ 17/17 tools operational | 100% |
| **Test Coverage** | ✅ All tests passing | 100% |
| **Compatibility** | ✅ Perfect ModelSEED-COBRApy integration | 100% |
| **Audit System** | ✅ Hallucination detection active | A+ reliability |
| **Biochemistry DB** | ✅ 45K+ compounds, 55K+ reactions | Universal coverage |

## 📁 Repository Structure

```
ModelSEEDagent/
├── 📚 docs/                          # Complete documentation
│   ├── user/INTERACTIVE_GUIDE.md    # Interactive interface guide
│   └── development/                  # Development documentation
├── 📖 notebooks/                     # Interactive tutorials
│   ├── comprehensive_tutorial.ipynb # Complete capability showcase
│   ├── argo.ipynb                   # Argo Gateway testing
│   ├── cobrapy_testing.ipynb        # COBRA analysis examples
│   └── local_llm.ipynb              # Local LLM integration
├── 🎯 examples/                      # Usage examples by complexity
├── 🧪 tests/                         # Comprehensive test suite
├── 🎮 src/                           # Main source code
│   ├── agents/                       # AI agents and LangGraph workflows
│   ├── tools/                        # 17 specialized metabolic tools
│   │   ├── cobra/                    # 11 COBRApy analysis tools
│   │   ├── modelseed/                # 4 ModelSEED integration tools
│   │   ├── biochem/                  # 2 biochemistry database tools
│   │   └── audit.py                  # Tool execution audit system
│   ├── cli/                          # Command-line interfaces
│   ├── interactive/                  # Natural language interface
│   └── llm/                          # Multi-LLM backend support
├── ⚙️ config/                        # Configuration and prompts
├── 📊 data/                          # Models, biochem database, examples
│   ├── biochem.db                    # 45K+ compounds, 55K+ reactions
│   └── examples/                     # Test models and datasets
└── 📝 logs/                          # Session logs and audit trails
```

## 🕵️ AI Transparency Features

ModelSEEDagent includes advanced **hallucination detection** and audit capabilities:

```bash
# View recent tool executions with audit data
modelseed-agent audit list

# Analyze specific execution for hallucinations
modelseed-agent audit verify <audit_id>

# Show all tools used in a session
modelseed-agent audit session <session_id>
```

**Verification Capabilities:**
- 🔍 **Tool Claims Verification** - Compare AI statements vs actual results
- 📁 **File Output Validation** - Verify claimed files exist and have correct format
- 💻 **Console Cross-Reference** - Cross-check console output vs structured data
- 📊 **Statistical Analysis** - Pattern detection across multiple runs with confidence scoring

## 🌟 Example Workflows

**Complete Genome-to-Model Pipeline:**
```bash
# 1. Annotate genome with RAST
modelseed-agent run-tool annotate_genome_rast --genome-file pputida.fna

# 2. Build draft model
modelseed-agent run-tool build_metabolic_model --genome-object <result>

# 3. Gapfill for growth
modelseed-agent run-tool gapfill_model --model-object <result>

# 4. Analyze essential genes
modelseed-agent run-tool analyze_essentiality --model-file <result>
```

**Interactive Analysis:**
```bash
# Natural language queries
modelseed-agent interactive

# Example queries:
# "Load E. coli core model and find essential genes"
# "What is cpd00027 and how does it relate to energy metabolism?"
# "Run flux variability analysis and explain the results"
```

## 🤝 Contributing

See [Development Guide](docs/development/) for contribution guidelines.

## 📄 License

MIT License - see LICENSE file for details.
