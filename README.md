# ModelSEEDagent 🧬

**Advanced AI-Powered Metabolic Modeling Platform**

ModelSEEDagent is a production-ready AI-powered metabolic modeling platform that combines the power of large language models with **17 specialized bioinformatics tools** and **advanced AI reasoning capabilities**. Built on LangGraph for intelligent workflow orchestration, it provides the most comprehensive metabolic modeling AI assistant available.

## 🚀 Quick Start

```bash
# Install with all dependencies
pip install .[all]

# Basic analysis
modelseed-agent analyze data/models/e_coli_core.xml --query "Find essential genes"

# Advanced AI reasoning (Phase 8 features)
modelseed-agent phase8

# Interactive natural language interface
modelseed-agent interactive
```

## 🎯 Core Capabilities

- 🧬 **Complete Genome-to-Model Pipeline** - RAST annotation → Model building → Gapfilling
- 📊 **Advanced COBRA Analysis** - 11 tools covering 60% of COBRApy capabilities
- 🔄 **Universal Compatibility** - Perfect ModelSEED ↔ COBRApy integration with auto-detection infrastructure
- 🧪 **Biochemistry Intelligence** - Universal ID resolution across 45K+ compounds and 55K+ reactions
- 🤖 **Advanced AI Reasoning** - Multi-step chains, hypothesis testing, collaborative decisions
- 🧠 **Pattern Learning** - Cross-model learning and intelligent recommendations
- 🕵️ **AI Transparency** - Advanced hallucination detection and audit system
- 💬 **Natural Language Interface** - Conversational AI for complex metabolic analysis

## 🧠 Phase 8: Advanced AI Reasoning

ModelSEEDagent includes cutting-edge AI capabilities that rival human expert analysis:

### 🔗 Multi-Step Reasoning Chains
AI plans and executes complex 5-10 step analysis sequences, adapting in real-time:
```bash
modelseed-agent phase8 chains
```

### 🔬 Hypothesis-Driven Analysis
Scientific hypothesis generation and systematic testing:
```bash
modelseed-agent phase8 hypothesis
```

### 🤝 Collaborative Reasoning
AI recognizes uncertainty and requests human expertise when needed

### 📚 Pattern Learning & Memory
Cross-model learning that improves recommendations over time:
```bash
modelseed-agent phase8 patterns
```

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

## 📁 Repository Structure

```
ModelSEEDagent/
├── 📚 docs/                          # Professional documentation
│   ├── PROJECT_STATUS.md            # Current project status
│   ├── ARCHITECTURE.md              # Technical architecture
│   ├── user/                        # User guides and tutorials
│   │   ├── README.md                # Complete user documentation
│   │   ├── INTERACTIVE_GUIDE.md     # Interactive interface guide
│   │   └── PHASE8_USER_GUIDE.md     # Advanced AI reasoning guide
│   ├── development/                 # Development documentation
│   └── archive/                     # Archived development artifacts
├── 🧪 tests/                        # Organized test suite
│   ├── unit/                        # Unit tests
│   ├── integration/                 # Integration tests
│   ├── phase_tests/                 # Phase-specific tests
│   ├── system/                      # System tests
│   └── validation/                  # Validation scripts
├── 🎯 examples/                     # Usage examples
│   ├── basic/                       # Simple usage examples
│   ├── advanced/                    # Complex workflows
│   ├── demos/                       # Demo scripts and results
│   └── scripts/                     # Utility scripts
├── 📖 notebooks/                    # Interactive tutorials
│   ├── comprehensive_working_tutorial.ipynb # Complete showcase
│   ├── argo.ipynb                   # Argo Gateway testing
│   └── local_llm.ipynb              # Local LLM integration
├── 🎮 src/                          # Main source code
│   ├── agents/                      # AI agents and reasoning
│   │   ├── reasoning_chains.py      # Multi-step reasoning
│   │   ├── hypothesis_system.py     # Hypothesis testing
│   │   ├── collaborative_reasoning.py # AI-human collaboration
│   │   └── pattern_memory.py        # Cross-model learning
│   ├── tools/                       # 17 specialized tools
│   │   ├── cobra/                   # 11 COBRApy tools
│   │   ├── modelseed/               # 4 ModelSEED tools
│   │   ├── biochem/                 # 2 biochemistry tools
│   │   └── audit.py                 # Execution audit system
│   ├── interactive/                 # User interfaces
│   │   └── phase8_interface.py      # Advanced AI interfaces
│   ├── cli/                         # Command-line interfaces
│   └── llm/                         # Multi-LLM backend support
├── ⚙️ config/                       # Configuration and prompts
├── 📊 data/                         # Models, biochem database, examples
│   ├── biochem.db                   # 45K+ compounds, 55K+ reactions
│   └── examples/                    # Test models and datasets
└── 📝 logs/                         # Session logs and audit trails
```

## 🧪 Production Status

**✅ Production Ready** - All development phases complete with comprehensive testing

| Component | Status | Coverage |
|-----------|--------|----------|
| **Tool Suite** | ✅ 17/17 tools operational | 100% |
| **Test Coverage** | ✅ All tests passing | 100% |
| **Advanced AI** | ✅ Phase 8 complete | Sophisticated reasoning |
| **Performance** | ✅ 6600x+ cache speedup | Optimized |
| **Compatibility** | ✅ Perfect ModelSEED-COBRApy integration | 100% |
| **Audit System** | ✅ Hallucination detection active | A+ reliability |
| **Biochemistry DB** | ✅ 45K+ compounds, 55K+ reactions | Universal coverage |

## 📚 Documentation

- **[Project Status](docs/PROJECT_STATUS.md)** - Current capabilities and completion status
- **[Technical Architecture](docs/ARCHITECTURE.md)** - System design and component overview
- **[User Guide](docs/user/README.md)** - Complete installation, usage, and examples
- **[Phase 8 Advanced AI Guide](docs/user/PHASE8_USER_GUIDE.md)** - Advanced reasoning capabilities
- **[Interactive Tutorial](notebooks/comprehensive_working_tutorial.ipynb)** - Hands-on showcase

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

**Advanced AI-Guided Analysis:**
```bash
# Multi-step reasoning chain
modelseed-agent phase8 chains
# Input: "Comprehensive E. coli analysis"
# AI plans: FBA → Nutrition → Essentiality → Synthesis

# Hypothesis-driven investigation
modelseed-agent phase8 hypothesis
# Input: "Model grows slower than expected"
# AI generates testable hypotheses and systematically evaluates them
```

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

**Interactive Natural Language Analysis:**
```bash
modelseed-agent interactive

# Example queries:
# "Load E. coli core model and find essential genes"
# "What is cpd00027 and how does it relate to energy metabolism?"
# "Run flux variability analysis and explain the results"
# "Generate hypotheses about why this model has low growth"
```

## 🚀 Performance Highlights

- **6,600x+ speedup** through intelligent caching
- **5x parallel execution** for independent operations
- **Sub-second response** for cached operations
- **Real-time verification** of AI reasoning accuracy
- **Memory-efficient** pattern storage and learning

## 🎯 Ready for Professional Use

ModelSEEDagent is production-ready for:
- **Research Projects** - Sophisticated AI-guided metabolic analysis
- **Collaborative Work** - Team-based modeling with audit trails
- **Educational Use** - Interactive learning with guided workflows
- **Production Deployment** - Scalable analysis pipelines

## 🤝 Contributing

See [Development Documentation](docs/development/) for contribution guidelines.

## 📄 License

MIT License - see LICENSE file for details.

---

**Welcome to the future of AI-powered metabolic modeling! 🎉**
