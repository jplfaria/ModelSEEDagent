# Configuration Guide

ModelSEEDagent provides flexible configuration options to customize behavior, performance, and integration with external services.

## Configuration Methods

### 1. Environment Variables (.env file)

Create a `.env` file in the project root:

```bash
# Core LLM Configuration
OPENAI_API_KEY=your_openai_key_here

# Argo Gateway Configuration (Recommended)
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

# Console Output Capture (Phase 1 CLI Debug Capture)
MODELSEED_CAPTURE_CONSOLE_DEBUG=false      # Capture console debug output
MODELSEED_CAPTURE_AI_REASONING_FLOW=false  # Capture AI reasoning steps
MODELSEED_CAPTURE_FORMATTED_RESULTS=false  # Capture final formatted results

# Directory Configuration
MODELSEED_DATA_DIR=/path/to/data
MODELSEED_LOG_DIR=/path/to/logs
MODELSEED_SESSION_DIR=/path/to/sessions

# Performance Configuration
MODELSEED_CACHE_ENABLED=true
MODELSEED_PARALLEL_TOOLS=true
MODELSEED_MAX_WORKERS=4
```

### 2. Command Line Arguments

```bash
# Override LLM provider
modelseed-agent --llm argo analyze

# Set debug level
modelseed-agent --debug-level DEBUG analyze

# Custom data directory
modelseed-agent --data-dir /custom/path analyze
```

### 3. Configuration Files

Create `config/config.yaml`:

```yaml
llm:
  default_provider: argo
  temperature: 0.1
  max_tokens: 4000
  timeout: 30

agents:
  metabolic:
    max_iterations: 10
    reasoning_depth: 3
  langgraph:
    visualization: true
    save_graphs: true

tools:
  cobra:
    default_solver: glpk
    tolerance: 1e-9
    timeout: 300
  modelseed:
    template_version: v5
    gapfill_mode: comprehensive

performance:
  cache_ttl: 3600
  max_memory_gb: 8
  parallel_execution: true
```

## LLM Provider Configuration

### OpenAI (Experimental)

```bash
# Required
OPENAI_API_KEY=your_key_here

# Optional
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.1
OPENAI_ORGANIZATION=your_org_id
```

### Argo Gateway

```bash
# Required
ARGO_GATEWAY_URL=https://gateway.argos.anl.gov
ARGO_API_KEY=your_key_here

# Optional
ARGO_MODEL=claude-3-sonnet
ARGO_TIMEOUT=60
ARGO_MAX_RETRIES=3
```

### Local LLM

```bash
# Local model configuration
LOCAL_LLM_ENABLED=true
LOCAL_LLM_MODEL_PATH=/path/to/model
LOCAL_LLM_DEVICE=cuda  # or cpu
LOCAL_LLM_MAX_LENGTH=2048
```

## Debug Configuration

### Granular Debug Control

```bash
# Overall debug level (TRACE, DEBUG, INFO, WARNING, ERROR)
MODELSEED_DEBUG_LEVEL=INFO

# Component-specific debugging
MODELSEED_DEBUG_COBRAKBASE=false    # COBRApy/cobrakbase messages
MODELSEED_DEBUG_LANGGRAPH=false     # LangGraph workflow messages
MODELSEED_DEBUG_HTTP=false          # HTTP/SSL debug from httpx
MODELSEED_DEBUG_TOOLS=true          # Tool execution details
MODELSEED_DEBUG_LLM=false           # LLM interaction details

# Special logging
MODELSEED_LOG_LLM_INPUTS=false      # Log LLM prompts and responses
```

### Debug Profiles

#### Developer Profile
```bash
MODELSEED_DEBUG_LEVEL=DEBUG
MODELSEED_DEBUG_TOOLS=true
MODELSEED_DEBUG_LLM=true
MODELSEED_LOG_LLM_INPUTS=true
```

#### Production Profile
```bash
MODELSEED_DEBUG_LEVEL=WARNING
MODELSEED_DEBUG_COBRAKBASE=false
MODELSEED_DEBUG_LANGGRAPH=false
MODELSEED_DEBUG_HTTP=false
MODELSEED_DEBUG_TOOLS=false
MODELSEED_DEBUG_LLM=false
```

#### Silent Profile
```bash
MODELSEED_DEBUG_LEVEL=ERROR
# All other debug flags default to false
```

### Checking Debug Configuration

```bash
# View current debug settings
modelseed-agent debug

# Test debug levels
modelseed-agent --debug-level DEBUG debug
```

## Performance Configuration

### Caching

```bash
# Enable/disable caching
MODELSEED_CACHE_ENABLED=true

# Cache settings
MODELSEED_CACHE_TTL=3600           # Cache TTL in seconds
MODELSEED_CACHE_DIR=/path/to/cache # Custom cache directory
MODELSEED_CACHE_MAX_SIZE=1000      # Max cache entries
```

### Parallel Execution

```bash
# Enable parallel tool execution
MODELSEED_PARALLEL_TOOLS=true

# Control number of workers
MODELSEED_MAX_WORKERS=4

# Tool-specific parallelization
COBRA_PARALLEL_FBA=true
COBRA_MAX_PARALLEL_JOBS=2
```

### Memory Management

```bash
# Memory limits
MODELSEED_MAX_MEMORY_GB=8
MODELSEED_MEMORY_WARNING_THRESHOLD=0.8

# Cleanup settings
MODELSEED_AUTO_CLEANUP=true
MODELSEED_TEMP_DIR_CLEANUP=true
```

## Directory Configuration

### Default Directories

```bash
# Data directory (models, examples, databases)
MODELSEED_DATA_DIR=./data

# Log directory
MODELSEED_LOG_DIR=./logs

# Session directory
MODELSEED_SESSION_DIR=./sessions

# Cache directory
MODELSEED_CACHE_DIR=./cache

# Temporary directory
MODELSEED_TEMP_DIR=/tmp/modelseed
```

### Custom Directories

```bash
# Example: Network storage setup
MODELSEED_DATA_DIR=/shared/modelseed/data
MODELSEED_LOG_DIR=/shared/modelseed/logs
MODELSEED_SESSION_DIR=/local/sessions
MODELSEED_CACHE_DIR=/local/cache
```

## Tool-Specific Configuration

### COBRApy Tools

```bash
# Solver configuration
COBRA_DEFAULT_SOLVER=glpk           # glpk, cplex, gurobi
COBRA_SOLVER_TIMEOUT=300            # seconds
COBRA_SOLVER_TOLERANCE=1e-9

# FBA configuration
COBRA_FBA_THREADS=1
COBRA_FBA_PRESOLVE=true

# Precision configuration
COBRA_FLUX_THRESHOLD=1e-6
COBRA_GROWTH_THRESHOLD=1e-3
```

### ModelSEED Tools

```bash
# Template configuration
MODELSEED_TEMPLATE_VERSION=v5
MODELSEED_TEMPLATE_PATH=/path/to/templates

# Gapfilling configuration
MODELSEED_GAPFILL_MODE=comprehensive  # fast, comprehensive
MODELSEED_GAPFILL_TIMEOUT=1800        # seconds
MODELSEED_MAX_GAPFILL_REACTIONS=50

# Annotation configuration
RAST_SERVER_URL=https://rast.nmpdr.org
RAST_TIMEOUT=3600
```

## Advanced Configuration

### Custom Configuration Classes

```python
# config/custom_settings.py
from src.config.settings import Settings

class CustomSettings(Settings):
    def __init__(self):
        super().__init__()
        self.custom_parameter = "value"

    def validate_custom(self):
        # Custom validation logic
        pass
```

### Runtime Configuration

```python
from src.config.settings import get_settings

# Get current settings
settings = get_settings()

# Override at runtime
settings.llm.temperature = 0.2
settings.debug.level = "DEBUG"
```

## Configuration Validation

### Validate Configuration

```bash
# Check configuration validity
modelseed-agent validate-config

# Verbose validation
modelseed-agent validate-config --verbose

# Check specific components
modelseed-agent validate-config --llm --tools
```

### Configuration Testing

```python
# Test configuration in Python
from src.config.settings import validate_configuration

# Validate current configuration
is_valid, errors = validate_configuration()

if not is_valid:
    for error in errors:
        print(f"Configuration error: {error}")
```

## Security Considerations

### API Key Security

```bash
# Use environment variables, not config files
export OPENAI_API_KEY="sk-..."

# Use key management services
OPENAI_API_KEY=$(aws secretsmanager get-secret-value --secret-id openai-key --query SecretString --output text)

# Rotate keys regularly
# Monitor key usage
# Use restricted permissions
```

### Network Security

```bash
# Proxy configuration
HTTP_PROXY=http://proxy.company.com:8080
HTTPS_PROXY=https://proxy.company.com:8080
NO_PROXY=localhost,127.0.0.1,.company.com

# SSL verification
SSL_VERIFY=true
SSL_CERT_PATH=/path/to/certificates

# Timeout configuration
REQUEST_TIMEOUT=30
CONNECTION_TIMEOUT=10
```

## Troubleshooting Configuration

### Common Issues

#### API Key Issues
```bash
# Test API key validity
modelseed-agent test-llm-connection

# Check environment variables
env | grep -E "(OPENAI|ARGO)"
```

#### Path Issues
```bash
# Check directory permissions
ls -la $MODELSEED_DATA_DIR

# Create missing directories
mkdir -p $MODELSEED_LOG_DIR $MODELSEED_CACHE_DIR
```

#### Configuration Conflicts
```bash
# Show effective configuration
modelseed-agent show-config

# Show configuration sources
modelseed-agent show-config --sources
```

## Next Steps

- **[Interactive Guide](getting-started/interactive-guide.md)**: Learn how to use ModelSEEDagent
- **[API Documentation](api/overview.md)**: Explore programmatic usage
- **[Troubleshooting](troubleshooting.md)**: Solve common issues
- **[Development](archive/development/DEVELOPMENT_ROADMAP.md)**: Contribute to the project


### Connection Pooling Configuration

ModelSEEDagent automatically manages HTTP connection pooling for optimal performance.

#### LLM Connection Pooling

**Automatic Configuration**:
- HTTP clients are pooled per configuration key
- Connections are reused across tool executions
- Timeout and connection limits are automatically managed

**Benefits**:
- Eliminates redundant connection setup overhead
- Reduces memory usage for LLM communications
- Improves overall session performance

**Monitoring**:
Connection pool statistics are logged at DEBUG level:
```
LLM Connection Pool initialized
Created new HTTP client for config: dev_120.0
Reusing existing LLM instance: argo|gpto1|prod|jplfaria|30.0
```

*Configuration is automatic and requires no user intervention.*

*Last updated: 3003b76c - Connection pooling implementation detected*


### COBRA Multiprocessing Configuration

COBRA tools support both single-process and multiprocess execution modes.

#### Default Behavior

**Single Process Mode** (Default):
- All COBRA tools default to `processes=1`
- Prevents connection pool fragmentation
- Recommended for most use cases

#### Multiprocessing Control

**Global Environment Variables**:
```bash
# Disable multiprocessing for all COBRA tools
export COBRA_DISABLE_MULTIPROCESSING=1

# Set process count for all COBRA tools
export COBRA_PROCESSES=4
```

**Tool-Specific Environment Variables**:
```bash
# Flux Variability Analysis
export COBRA_FVA_PROCESSES=8

# Flux Sampling
export COBRA_SAMPLING_PROCESSES=4

# Gene Deletion Analysis
export COBRA_GENE_DELETION_PROCESSES=2

# Essentiality Analysis
export COBRA_ESSENTIALITY_PROCESSES=2
```

#### Performance Considerations

**Single Process (Default)**:
- No connection pool fragmentation
- Lower memory usage
- Simpler debugging
- Slower for large analyses

**Multiprocess**:
- Faster for large-scale analyses
- Higher memory usage
- Connection pool overhead
- Complex error handling

*Last updated: 3003b76c - COBRA multiprocessing changes detected*
