# Getting Started

**AI-Powered Metabolic Modeling Platform**

ModelSEEDagent combines large language models with 29 specialized bioinformatics tools to provide intelligent metabolic modeling assistance. The platform integrates ModelSEED and COBRApy capabilities with natural language interfaces for comprehensive analysis workflows.

## Installation

### Prerequisites
- Python 3.9 or higher
- Virtual environment (recommended)
- API access to Claude, OpenAI, or Argo Gateway

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/ModelSEED/ModelSEEDagent.git
cd ModelSEEDagent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all dependencies
pip install -e .[all]
```

### API Configuration

Configure your AI model access by setting one of these options:

```bash
# Option 1: Anthropic Claude (recommended)
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Option 2: OpenAI
export OPENAI_API_KEY="your_openai_api_key"

# Option 3: Argo Gateway (if available)
export ARGO_USER="your_username"
export ARGO_GATEWAY_URL="your_gateway_url"
```

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

```python
from src.agents.langgraph_metabolic import LangGraphMetabolicAgent
from src.llm.anthropic import AnthropicLLM
from src.tools.cobra.fba import FBATool

# Initialize components
llm_config = {
    "model_name": "claude-3-sonnet-20240229",
    "system_content": "You are an expert metabolic modeling assistant.",
    "max_tokens": 1000,
    "temperature": 0.1,
}

llm = AnthropicLLM(llm_config)
tools = [FBATool({"name": "run_fba", "description": "Run FBA analysis"})]
agent = LangGraphMetabolicAgent(llm, tools, {"name": "metabolic_agent"})

# Run analysis
result = agent.run({
    "query": "Analyze the metabolic model structure",
    "model_path": "path/to/model.xml"
})
print(result.message)
```

## Core Capabilities

ModelSEEDagent provides 23 specialized tools organized into several categories:

### ModelSEED Integration (4 tools)
- **Genome Annotation** - RAST-based automated annotation
- **Model Building** - Template-based metabolic model construction
- **Gapfilling** - Pathway completion algorithms
- **Protein Annotation** - Sequence-based functional annotation

### COBRApy Analysis (11 tools)
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
- **Media Analysis** - Media optimization and troubleshooting

### Biochemistry Tools (2 tools)
- **Universal ID Resolution** - Cross-database compound/reaction mapping
- **Biochemistry Search** - Intelligent metabolite discovery

### AI Media Tools (6 tools)
- **Media Selection** - Intelligent media recommendation
- **Media Manipulation** - Dynamic media modification
- **Media Compatibility** - Cross-model media validation
- **Media Comparison** - Comprehensive media analysis
- **Media Optimization** - AI-driven media improvement
- **Dynamic Media Management** - Adaptive media systems

## Advanced Features

### Natural Language Interface

ModelSEEDagent provides an intuitive natural language interface for metabolic analysis:

- **Conversational Analysis** - Ask questions in plain English about your models
- **Session Management** - Persistent analysis sessions with history
- **Real-time Visualizations** - Interactive dashboards and graphs
- **Context Awareness** - Maintains conversation context across interactions

### AI Transparency and Audit

The platform includes comprehensive verification capabilities:

- **Tool Execution Capture** - Automatic audit of all tool executions
- **Hallucination Detection** - Advanced verification with confidence scoring
- **Statistical Analysis** - Pattern detection across multiple runs
- **Audit Commands** - CLI tools for reviewing analysis history

### Biochemistry Intelligence

Built-in biochemistry database provides universal compound and reaction resolution:

- **Universal Database** - 45,000+ compounds and 55,000+ reactions
- **Cross-Database Mapping** - ModelSEED, BiGG, KEGG, MetaCyc, ChEBI
- **Fast Resolution** - Sub-millisecond query performance
- **Human-Readable Outputs** - All results include biochemistry names

### Universal Model Compatibility

Seamless integration between ModelSEED and COBRApy ecosystems:

- **Perfect Round-Trip Conversion** - Identical results between formats
- **SBML Compatibility** - Standard model format support
- **Growth Rate Preservation** - Maintains model predictions
- **Structure Preservation** - Reactions, metabolites, and genes identical

## Example Workflows

### Complete Genome-to-Model Pipeline

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

### Interactive Natural Language Analysis

```bash
modelseed-agent interactive

# Example queries:
# "Load E. coli core model and find essential genes"
# "What is cpd00027 and how does it relate to energy metabolism?"
# "Run flux variability analysis and explain the results"
# "Generate hypotheses about why this model has low growth"
```

## Testing

Run the test suite to verify your installation:

```bash
# Activate virtual environment
source venv/bin/activate

# Run test suite
pytest -v

# Run functional tests
python tests/run_all_functional_tests.py
```

## Documentation

- **[Architecture Guide](../ARCHITECTURE.md)** - System design and components
- **[Installation Guide](../installation.md)** - Detailed setup instructions
- **[API Documentation](../api/overview.md)** - Programmatic usage
- **[Debug Configuration](../debug.md)** - Troubleshooting and debugging

## Next Steps

After installation:

1. **Configure API access** - Set up your AI model credentials
2. **Try examples** - Run basic analysis commands
3. **Interactive session** - Launch the natural language interface
4. **Explore tools** - Review the 23 available analysis tools

For additional help, see the [Troubleshooting Guide](../troubleshooting.md).
