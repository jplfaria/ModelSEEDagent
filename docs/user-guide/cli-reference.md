# CLI Reference

ModelSEEDagent provides a comprehensive command-line interface for metabolic modeling tasks.

## Overview

```bash
modelseed-agent [OPTIONS] COMMAND [ARGS]...
```

### Global Options
- `--help`: Show help message and exit
- `--version`: Show version and exit
- `--debug`: Enable debug mode
- `--quiet`: Suppress informational output

## Main Commands

### `interactive`
Launch the interactive natural language interface for conversational metabolic modeling.

```bash
modelseed-agent interactive
```

**Example Session:**
```
> Load the E. coli model and analyze growth
> Find essential genes
> What media components are required for growth?
```

### `analyze`
Run AI-powered analysis on a metabolic model with natural language queries.

```bash
modelseed-agent analyze MODEL_PATH --query "YOUR QUESTION"
```

**Parameters:**
- `MODEL_PATH`: Path to the metabolic model file (SBML/XML format)
- `--query`, `-q`: Natural language query for analysis

**Examples:**
```bash
# Basic analysis
modelseed-agent analyze data/examples/e_coli_core.xml --query "What is the growth rate?"

# Complex analysis
modelseed-agent analyze model.xml --query "Find essential genes and suggest media optimizations"

# Multi-step analysis
modelseed-agent analyze model.xml --query "Compare growth on glucose vs pyruvate"
```

### `setup`
Configure ModelSEEDagent with LLM backend and credentials.

```bash
modelseed-agent setup [--backend BACKEND] [--model MODEL]
```

**Parameters:**
- `--backend`: LLM backend to use (`argo`, `openai`, `local`)
- `--model`: Specific model to use (e.g., `claude-3-sonnet`, `gpt-4`)

**Examples:**
```bash
# Interactive setup
modelseed-agent setup

# Direct configuration
modelseed-agent setup --backend argo --model claude-3-sonnet
```

### `status`
Check system status and configuration.

```bash
modelseed-agent status
```

Shows:
- Current LLM configuration
- Available tools and their status
- System health checks
- Recent activity summary

### `debug`
Run diagnostic tests and troubleshooting utilities.

```bash
modelseed-agent debug [--test-llm] [--test-tools] [--check-env]
```

**Options:**
- `--test-llm`: Test LLM connection and response
- `--test-tools`: Verify tool availability and functionality
- `--check-env`: Check environment variables and configuration

## Audit Commands

### `audit list`
List recent tool execution audits.

```bash
modelseed-agent audit list [--limit N] [--session-id ID] [--tool-name NAME]
```

**Parameters:**
- `--limit`: Number of audits to show (default: 10)
- `--session-id`: Filter by session ID
- `--tool-name`: Filter by tool name

### `audit show`
Display detailed audit information for a specific execution.

```bash
modelseed-agent audit show AUDIT_ID [--show-console] [--show-files]
```

**Parameters:**
- `AUDIT_ID`: The audit ID to inspect
- `--show-console`: Include console output
- `--show-files`: Show file operations

### `audit session`
Analyze all audits from a specific session.

```bash
modelseed-agent audit session SESSION_ID [--summary]
```

**Parameters:**
- `SESSION_ID`: Session ID to analyze
- `--summary`: Show summary statistics only

### `audit verify`
Verify audit integrity and claims.

```bash
modelseed-agent audit verify AUDIT_ID [--check-files] [--check-claims]
```

**Parameters:**
- `AUDIT_ID`: Audit ID to verify
- `--check-files`: Verify file operations
- `--check-claims`: Validate AI reasoning claims

## AI Audit Commands

### `ai-audit`
Analyze AI reasoning and decision-making patterns.

```bash
modelseed-agent ai-audit [--logs-dir DIR]
```

**Parameters:**
- `--logs-dir`: Directory containing logs to analyze

Shows:
- Reasoning chain analysis
- Tool selection patterns
- Decision confidence scores
- Potential hallucination detection

## Environment Variables

ModelSEEDagent respects the following environment variables:

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_key
ARGO_GATEWAY_URL=https://your-argo-gateway.com
ARGO_API_KEY=your_argo_key

# Debug Settings
MODELSEED_DEBUG_LEVEL=INFO
MODELSEED_DEBUG_TOOLS=true
MODELSEED_LOG_LLM_INPUTS=false

# Performance
MODELSEED_CACHE_ENABLED=true
MODELSEED_PARALLEL_TOOLS=true
```

## Configuration File

ModelSEEDagent looks for configuration in `~/.modelseed/config.yaml`:

```yaml
llm:
  backend: argo
  model: claude-3-sonnet
  temperature: 0.1

tools:
  cobra:
    solver: glpk
    tolerance: 1e-9

performance:
  cache_enabled: true
  parallel_execution: true
```

## Examples

### Basic Workflow
```bash
# 1. Setup
modelseed-agent setup --backend argo

# 2. Analyze a model
modelseed-agent analyze data/examples/e_coli_core.xml \
  --query "What are the essential genes?"

# 3. Interactive exploration
modelseed-agent interactive
```

### Advanced Analysis
```bash
# Multi-step analysis with specific questions
modelseed-agent analyze model.xml --query \
  "1. Find the growth rate on glucose
   2. Identify essential genes
   3. Suggest minimal media composition
   4. Compare with pyruvate growth"

# Debug mode for troubleshooting
modelseed-agent --debug analyze model.xml \
  --query "Why is growth rate zero?"
```

### Audit and Verification
```bash
# List recent activities
modelseed-agent audit list --limit 20

# Inspect specific execution
modelseed-agent audit show abc123 --show-console

# Verify AI reasoning
modelseed-agent ai-audit
```

## Troubleshooting

### Common Issues

**LLM Connection Failed:**
```bash
modelseed-agent debug --test-llm
modelseed-agent setup --backend argo
```

**Tool Not Found:**
```bash
modelseed-agent status
modelseed-agent debug --test-tools
```

**Import Errors:**
```bash
modelseed-agent debug --check-env
pip install -e .[all]
```

For more help, see the [Troubleshooting Guide](../troubleshooting.md) or run:
```bash
modelseed-agent --help
```
