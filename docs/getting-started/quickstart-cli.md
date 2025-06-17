# Getting Started

**AI-Powered Metabolic Modeling Platform**

ModelSEEDagent combines large language models with 27 specialized bioinformatics tools to provide intelligent metabolic modeling assistance. The platform integrates ModelSEED and COBRApy capabilities with natural language interfaces for comprehensive analysis workflows.

## Installation

See the [Installation Guide](../installation.md) for detailed setup instructions.

## Basic Usage

### Interactive Interface

Launch the natural language interface for conversational analysis:

```bash
# Start interactive session
modelseed-agent interactive

# Or check available examples
ls examples/basic/
```

**Example queries:**
- "Load and analyze the E. coli model"
- "What is the growth rate on glucose minimal media?"
- "Run flux balance analysis and show results"
- "Find essential genes in this model"

### Command Line Interface

Use the CLI for direct analysis commands:

```bash
# Configure the system
modelseed-agent setup

# Analyze a metabolic model
modelseed-agent analyze data/examples/e_coli_core.xml --query "Analyze structure"

# Check system status
modelseed-agent status

# View help
modelseed-agent --help
```

### Basic Commands

```bash
# Model analysis
modelseed-agent analyze path/to/model.xml

# Interactive session
modelseed-agent interactive

# System configuration
modelseed-agent setup

# View execution logs
modelseed-agent logs

# Debug configuration
modelseed-agent debug
```

### Python API

For programmatic access, see the [API Documentation](../api/overview.md) for detailed examples and usage patterns.

## Core Capabilities

ModelSEEDagent provides 27 specialized tools organized into several categories:

### ModelSEED Integration (5 tools)
- **Genome Annotation** - RAST-based automated annotation
- **Model Building** - Template-based metabolic model construction
- **Gapfilling** - Pathway completion algorithms
- **Protein Annotation** - Sequence-based functional annotation
- **Model Compatibility** - ModelSEED ↔ COBRApy compatibility testing

### COBRApy Analysis (12 tools)
- **Flux Balance Analysis** - Growth rate and flux predictions
- **Flux Variability Analysis** - Solution space exploration
- **Gene Deletion Analysis** - Knockout effect studies
- **Essentiality Analysis** - Essential gene identification
- **Flux Sampling** - Unbiased solution space sampling
- **Production Envelope** - Phenotype phase plane analysis
- **Reaction Expression** - Gene expression integration
- **Model Analysis** - Comprehensive model statistics
- **Pathway Analysis** - Metabolic pathway insights
- **Auxotrophy Prediction** - Growth requirement analysis
- **Minimal Media Finding** - Essential media component identification
- **Missing Media Detection** - Media gap identification
- **Reaction Expression** - Gene expression integration

### Biochemistry Tools (2 tools)
- **Universal ID Resolution** - Cross-database compound/reaction mapping
- **Biochemistry Search** - Intelligent metabolite discovery

### AI Media Tools (6 tools)
- **Media Selection** - Intelligent media recommendation
- **Media Manipulation** - Dynamic media modification
- **Media Compatibility** - Cross-model media validation
- **Media Comparison** - Comprehensive media analysis
- **Media Optimization** - AI-driven media improvement
- **Auxotrophy Prediction** - AI-powered auxotrophy prediction and validation

## Key Features

- **Natural Language Interface** - Ask questions in plain English about your models
- **AI Transparency** - Complete audit trails and verification of all analysis
- **Universal Model Compatibility** - Seamless ModelSEED ↔ COBRApy integration
- **Biochemistry Intelligence** - 50,000+ compound/reaction database with real-time resolution

## Quick Examples

### Interactive Analysis

```bash
modelseed-agent interactive

# Try these example queries:
# "Load E. coli core model and find essential genes"
# "What is cpd00027 and how does it relate to energy metabolism?"
# "Run flux variability analysis and explain the results"
```

### Command Line Analysis

```bash
# Analyze a specific model
modelseed-agent analyze data/examples/e_coli_core.xml --query "Find essential genes"

# System status and help
modelseed-agent status
modelseed-agent --help
```

## Verification

Test your installation:

```bash
# Check system status
modelseed-agent status

# Run a simple test
modelseed-agent analyze --help
```

## Next Steps

1. **Configure API access** - Set up your AI model credentials (see [Installation Guide](../installation.md))
2. **Try interactive mode** - Run `modelseed-agent interactive`
3. **Explore documentation** - See [API Documentation](../api/overview.md) for programmatic usage

## Additional Resources

- **[Architecture Guide](../ARCHITECTURE.md)** - System design and components
- **[Tool Reference](../TOOL_REFERENCE.md)** - Complete tool documentation
- **[Troubleshooting Guide](../troubleshooting.md)** - Common issues and solutions
