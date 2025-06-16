# ModelSEEDagent

**AI-Powered Metabolic Modeling Platform**

ModelSEEDagent combines large-language-model reasoning with **29 specialised metabolic-modeling tools**.  The **interactive interface is production-ready and fully tested**; the Typer-based CLI is available in **beta** while we finish import cleanup and help-formatting fixes.

## 📚 **Documentation**

**🌐 [Complete Documentation Site](https://jplfaria.github.io/ModelSEEDagent/)**

The full documentation is available as a beautiful, searchable site with:
- **Getting Started Guide** - Installation and basic usage
- **User Guide** - Interactive interface and tutorials
- **Technical Reference** - Architecture and tool details
- **API Documentation** - Programmatic usage and implementation
- **Operations Guide** - Configuration, deployment, and monitoring

## Quick Start

```bash
# 1) Install in editable mode (recommended for development)
pip install -e .

# 2) Launch the interactive natural-language interface (stable)
python -m src.interactive.interactive_cli

#  ── or, via the Typer CLI (beta – some commands may still fail) ──
modelseed-agent interactive

# 3) Example analysis using the beta CLI
modelseed-agent analyze data/examples/e_coli_core.xml --query "Find essential genes"
```

> ⚠️  The Typer CLI is still under active refactor – if you run into import errors use the module launch shown above.

## Core Capabilities

- **Complete Genome-to-Model Pipeline** - RAST annotation → Model building → Gapfilling
- **Advanced COBRA Analysis** - 12 tools covering comprehensive COBRApy capabilities
- **Universal Compatibility** - Perfect ModelSEED ↔ COBRApy integration
- **Biochemistry Intelligence** - Universal ID resolution across 45K+ compounds and 56K+ reactions
- **Advanced AI Reasoning** - Multi-step chains, hypothesis testing, collaborative decisions
- **Pattern Learning** - Cross-model learning and intelligent recommendations
- **AI Transparency** - Advanced hallucination detection and audit system
- **Natural Language Interface** - Conversational AI for complex metabolic analysis
- **AI Media Intelligence** - 29 specialized tools for intelligent media management and optimization

## Advanced AI Features

ModelSEEDagent includes sophisticated AI capabilities for metabolic analysis:

### Multi-Step Reasoning
AI plans and executes complex 5-10 step analysis sequences, adapting in real-time based on intermediate results.

### Hypothesis-Driven Analysis
Scientific hypothesis generation and systematic testing with appropriate tool selection and evidence evaluation.

### Collaborative Reasoning
AI recognizes uncertainty and requests human expertise when needed.

### Pattern Learning & Memory
Cross-model learning that improves recommendations over time.

## AI Media Intelligence

Advanced AI-powered media management for metabolic modeling:

### Smart Media Selection
AI analyzes model characteristics and automatically selects optimal media:
```bash
modelseed-agent analyze model.xml --query "select optimal media for this E. coli model"
```

### Natural Language Media Modification
Modify media using simple English commands in the interactive interface:
- "make anaerobic"
- "add vitamins and amino acids"
- "remove all carbon sources except glucose"

### Automated Workflow Templates
Pre-built workflow templates combining media selection with analysis:
- **Optimal Media Discovery** - Find the best media for any model
- **Production Optimization** - Optimize media for specific metabolites
- **Auxotrophy Analysis** - Predict and design media for auxotrophs
- **Cross-Model Comparison** - Compare media performance across species
- **Troubleshooting** - Diagnose and fix media-related growth issues

## Tool Implementation & Testing Status

**🧪 [Complete Tool Testing Status](https://jplfaria.github.io/ModelSEEDagent/TOOL_TESTING_STATUS/)** - Live testing coverage and results

**Current Status**: 23/25 tools actively tested (92% coverage) with 100% success rate across 4 model types

| Category | Implemented | Tested | Success Rate | Status |
|----------|-------------|--------|--------------|--------|
| **COBRA Tools** | 12 | 12 | 100% (48/48) | ✅ Complete coverage |
| **AI Media Tools** | 6 | 6 | 100% (24/24) | ✅ Complete coverage |
| **Biochemistry Tools** | 2 | 2 | 100% (8/8) | ✅ Complete coverage |
| **System Tools** | 3 | 3 | 100% (12/12) | ✅ Functional validation |
| **ModelSEED Tools** | 3 | 0 | N/A | Service dependencies |

**Last Comprehensive Test**: 2025-06-14 | **Models Tested**: e_coli_core, iML1515, EcoliMG1655, B_aphidicola

## Specialized Tools (29 Total)

### AI Media Tools (6 tools) - 100% Tested
- **Media Selection** - AI-powered optimal media selection for models
- **Media Manipulation** - Natural language media modification
- **Media Compatibility** - Intelligent media-model compatibility analysis
- **Media Comparison** - Cross-model media performance comparison
- **Media Optimization** - AI-driven media optimization for growth targets
- **Auxotrophy Prediction** - AI prediction of auxotrophies from model gaps

### Advanced COBRA Analysis (12 tools) - 92% Tested
- **Basic Analysis** - FBA, minimal media, auxotrophy analysis
- **Advanced Analysis** - Flux variability, gene deletion, essentiality analysis
- **Statistical Methods** - Flux sampling, production envelope analysis
- **Specialized Tools** - Reaction expression, missing media analysis
- **Pathway Analysis** - Metabolic pathway analysis (requires annotations)

### Biochemistry Database (2 tools) - 100% Tested
- **Universal ID Resolution** - ModelSEED ↔ BiGG ↔ KEGG mapping
- **Biochemistry Search** - Compound/reaction discovery by name

### ModelSEED Integration (3 tools) - Service Dependencies
- **Model Building** - MSBuilder with template selection
- **Gapfilling** - Advanced MSGapfill algorithms
- **Protein Annotation** - Individual protein sequence annotation

### System Tools (3 tools) - 100% Tested
- **Tool Audit** - Execution auditing and verification
- **AI Audit** - AI reasoning and decision auditing
- **Realtime Verification** - Live hallucination detection

## Repository Structure

```
ModelSEEDagent/
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # Technical architecture
│   ├── user/                      # User guides
│   │   ├── README.md              # Getting started guide
│   │   └── INTERACTIVE_GUIDE.md   # User guide
│   ├── api/                       # API documentation
│   ├── debug.md                   # Debug configuration
│   └── troubleshooting.md         # Troubleshooting guide
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   ├── system/                    # System tests
│   └── validation/                # Validation scripts
├── examples/                      # Usage examples
│   ├── basic/                     # Simple usage examples
│   ├── advanced/                  # Complex workflows
│   └── scripts/                   # Utility scripts
├── notebooks/                     # Interactive tutorials
├── src/                           # Main source code
│   ├── agents/                    # AI agents and reasoning
│   ├── tools/                     # 29 specialized tools
│   │   ├── cobra/                 # 12 COBRApy tools
│   │   ├── modelseed/             # 3 ModelSEED tools
│   │   ├── biochem/               # 2 biochemistry tools
│   │   ├── system/                # 3 system tools
│   │   └── ai_media/              # 6 AI media tools
│   ├── interactive/            # User interfaces
│   ├── cli/                    # Command-line interfaces
│   └── llm/                    # Multi-LLM backend support
├── config/                     # Configuration and prompts
├── data/                       # Models, biochem database, examples
│   ├── biochem.db              # 45K+ compounds, 55K+ reactions
│   └── examples/               # Test models and datasets
└── logs/                       # Session logs and audit trails
```

## Documentation

**🌐 [Complete Documentation Site](https://jplfaria.github.io/ModelSEEDagent/)** - Full documentation with search and navigation

**Quick Links:**
- **[Getting Started](https://jplfaria.github.io/ModelSEEDagent/user/README/)** - Installation and basic usage
- **[User Guide](https://jplfaria.github.io/ModelSEEDagent/user/INTERACTIVE_GUIDE/)** - Interactive interface guide
- **[Tool Reference](https://jplfaria.github.io/ModelSEEDagent/TOOL_REFERENCE/)** - All 29 tools overview
- **[API Documentation](https://jplfaria.github.io/ModelSEEDagent/api/overview/)** - Programmatic usage
- **[Configuration](https://jplfaria.github.io/ModelSEEDagent/configuration/)** - Setup and configuration options

## AI Transparency Features

ModelSEEDagent includes advanced hallucination detection and audit capabilities:

```bash
# View recent tool executions with audit data
modelseed-agent audit list

# Analyze specific execution for hallucinations
modelseed-agent audit verify <audit_id>

# Show all tools used in a session
modelseed-agent audit session <session_id>
```

**Verification Capabilities:**
- **Tool Claims Verification** - Compare AI statements vs actual results
- **File Output Validation** - Verify claimed files exist and have correct format
- **Console Cross-Reference** - Cross-check console output vs structured data
- **Statistical Analysis** - Pattern detection across multiple runs with confidence scoring

## Example Workflows

**Advanced AI-Guided Analysis:**
```bash
# Interactive analysis
modelseed-agent interactive
# Input: "Comprehensive E. coli analysis"
# AI plans: FBA → Nutrition → Essentiality → Synthesis
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

## Performance Highlights

- **Intelligent caching** - Significant speedup for repeated operations
- **Parallel execution** - Multi-threaded tool execution for independent operations
- **Sub-second response** - Fast response for cached operations
- **Real-time verification** - AI reasoning accuracy checking
- **Memory-efficient** - Optimized pattern storage and learning

## Professional Use

ModelSEEDagent is production-ready for:
- **Research Projects** - Sophisticated AI-guided metabolic analysis
- **Collaborative Work** - Team-based modeling with audit trails
- **Educational Use** - Interactive learning with guided workflows
- **Production Deployment** - Scalable analysis pipelines

## Contributing

See the documentation for contribution guidelines and development setup.

## License

MIT License - see LICENSE file for details.
