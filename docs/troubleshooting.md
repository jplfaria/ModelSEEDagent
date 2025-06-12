# Troubleshooting Guide

This guide covers common issues and their solutions when using ModelSEEDagent.

## Quick Diagnostics

### System Health Check

```bash
# Check overall system status
modelseed-agent debug

# Test LLM connectivity
modelseed-agent test-llm-connection

# Validate configuration
modelseed-agent validate-config

# Check tool availability
modelseed-agent check-tools
```

### Environment Verification

```bash
# Check Python environment
python --version
pip list | grep -E "(cobra|modelseed|anthropic|openai)"

# Check environment variables
env | grep -E "(MODELSEED|ANTHROPIC|OPENAI|ARGO)"

# Check file permissions
ls -la .env
ls -la data/ logs/ cache/
```

## Installation Issues

### Common Installation Problems

#### ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'modelseed_agent'`

**Solutions**:
```bash
# Reinstall in development mode
pip install -e .

# Check PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Verify installation
pip show modelseed-agent
```

#### Dependency Conflicts

**Problem**: Package version conflicts during installation

**Solutions**:
```bash
# Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install -e .

# Use conda for better dependency management
conda create -n modelseed python=3.9
conda activate modelseed
conda install -c bioconda cobra
pip install -e .

# Force reinstall problematic packages
pip install --force-reinstall cobra modelseedpy
```

#### COBRApy Installation Issues

**Problem**: COBRApy fails to install or import

**Solutions**:
```bash
# Install from conda-forge
conda install -c bioconda cobra

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install build-essential python3-dev libxml2-dev libxslt-dev

# Install system dependencies (macOS)
brew install libxml2 libxslt
```

## Configuration Issues

### API Key Problems

#### Invalid API Key

**Problem**: "Invalid API key" or authentication errors

**Solutions**:
```bash
# Check API key format
echo $ANTHROPIC_API_KEY | head -c 20  # Should start with "sk-ant-"
echo $OPENAI_API_KEY | head -c 20     # Should start with "sk-"

# Test API key validity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# Check for trailing whitespace
ANTHROPIC_API_KEY=$(echo $ANTHROPIC_API_KEY | tr -d '[:space:]')
```

#### Environment Variable Issues

**Problem**: Environment variables not loaded

**Solutions**:
```bash
# Check .env file location and format
cat .env | head -5

# Load environment manually
source .env
export $(grep -v '^#' .env | xargs)

# Check variable expansion
echo "Key starts with: ${ANTHROPIC_API_KEY:0:10}"

# Use explicit export
export ANTHROPIC_API_KEY="your-key-here"
```

### Debug Configuration Issues

#### Excessive Logging

**Problem**: Too much debug output flooding console

**Solutions**:
```bash
# Reduce debug level
export MODELSEED_DEBUG_LEVEL=WARNING

# Disable specific components
export MODELSEED_DEBUG_COBRAKBASE=false
export MODELSEED_DEBUG_LANGGRAPH=false
export MODELSEED_DEBUG_HTTP=false

# Silent mode
export MODELSEED_DEBUG_LEVEL=ERROR
```

#### Missing Debug Information

**Problem**: Not enough information for debugging

**Solutions**:
```bash
# Enable detailed debugging
export MODELSEED_DEBUG_LEVEL=DEBUG
export MODELSEED_DEBUG_TOOLS=true
export MODELSEED_DEBUG_LLM=true
export MODELSEED_LOG_LLM_INPUTS=true

# Check specific component
modelseed-agent --debug-level DEBUG analyze model.xml
```

## Runtime Issues

### LLM Connection Problems

#### Timeout Errors

**Problem**: LLM requests timing out

**Solutions**:
```bash
# Increase timeout
export LLM_TIMEOUT=120

# Check network connectivity
curl -w "%{time_total}" https://api.anthropic.com/v1/health
ping -c 4 api.openai.com

# Use different LLM provider
modelseed-agent --llm openai analyze
```

#### Rate Limiting

**Problem**: "Rate limit exceeded" errors

**Solutions**:
```bash
# Add delays between requests
export LLM_REQUEST_DELAY=1

# Use different API key tier
# Upgrade your API plan

# Implement exponential backoff
export LLM_MAX_RETRIES=5
export LLM_RETRY_DELAY=2
```

#### SSL/Certificate Issues

**Problem**: SSL certificate verification failures

**Solutions**:
```bash
# Update certificates
pip install --upgrade certifi

# Temporary: disable SSL verification (not recommended for production)
export SSL_VERIFY=false

# Use specific certificate bundle
export SSL_CERT_FILE=/path/to/cacert.pem
```

### Tool Execution Issues

#### COBRApy Solver Problems

**Problem**: "No solver available" or solver errors

**Solutions**:
```bash
# Check available solvers
python -c "import cobra; print(cobra.Configuration().solver)"

# Install GLPK solver
# Ubuntu/Debian
sudo apt-get install glpk-utils

# macOS
brew install glpk

# Configure solver explicitly
export COBRA_DEFAULT_SOLVER=glpk
```

#### Memory Issues

**Problem**: Out of memory errors during large model analysis

**Solutions**:
```bash
# Increase memory limits
export MODELSEED_MAX_MEMORY_GB=16

# Enable memory monitoring
export MODELSEED_MEMORY_WARNING_THRESHOLD=0.8

# Use memory-efficient options
export COBRA_MEMORY_EFFICIENT=true

# Reduce parallel workers
export MODELSEED_MAX_WORKERS=2
```

#### File Permission Issues

**Problem**: Permission denied errors accessing files

**Solutions**:
```bash
# Check file permissions
ls -la data/ logs/ cache/

# Fix permissions
chmod -R 755 data/
chmod -R 755 logs/
chmod -R 755 cache/

# Check directory ownership
sudo chown -R $USER:$USER data/ logs/ cache/
```

### Performance Issues

#### Slow Analysis Performance

**Problem**: Analysis taking too long

**Solutions**:
```bash
# Enable caching
export MODELSEED_CACHE_ENABLED=true

# Increase parallel workers
export MODELSEED_MAX_WORKERS=8
export MODELSEED_PARALLEL_TOOLS=true

# Use faster solver
export COBRA_DEFAULT_SOLVER=cplex  # if available

# Monitor performance
modelseed-agent monitor --performance
```

#### Cache Issues

**Problem**: Cache not working or corrupted

**Solutions**:
```bash
# Clear cache
rm -rf cache/
mkdir cache/

# Disable cache temporarily
export MODELSEED_CACHE_ENABLED=false

# Check cache permissions
ls -la cache/

# Rebuild cache
modelseed-agent rebuild-cache
```

## Data Issues

### Model Loading Problems

#### Invalid Model Format

**Problem**: "Cannot load model" or format errors

**Solutions**:
```bash
# Validate model format
python -c "import cobra; model = cobra.io.read_sbml_model('model.xml'); print(f'Model loaded: {len(model.reactions)} reactions')"

# Convert model format
cobra-convert -i model.json -o model.xml

# Check model integrity
modelseed-agent validate-model model.xml
```

#### Missing Model Files

**Problem**: Model files not found

**Solutions**:
```bash
# Check file paths
ls -la data/examples/
find . -name "*.xml" -o -name "*.json"

# Use absolute paths
modelseed-agent analyze /full/path/to/model.xml

# Download example models
modelseed-agent download-examples
```

### Database Issues

#### Biochemistry Database Problems

**Problem**: Biochemistry database not available

**Solutions**:
```bash
# Check database location
ls -la data/biochem.db

# Rebuild database
python scripts/build_biochem_db.py

# Use remote database
export MODELSEED_USE_REMOTE_BIOCHEM=true

# Check database integrity
modelseed-agent validate-biochem-db
```

## Network Issues

### Proxy Configuration

#### Corporate Proxy Issues

**Problem**: Cannot connect through corporate proxy

**Solutions**:
```bash
# Configure proxy
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=https://proxy.company.com:8080
export NO_PROXY=localhost,127.0.0.1,.company.com

# Test proxy connectivity
curl --proxy $HTTP_PROXY https://api.anthropic.com/v1/health

# Use proxy with authentication
export HTTP_PROXY=http://username:password@proxy.company.com:8080
```

#### Firewall Issues

**Problem**: Firewall blocking connections

**Solutions**:
```bash
# Check required domains
# api.anthropic.com
# api.openai.com
# your-argo-gateway.com

# Test connectivity
telnet api.anthropic.com 443
nc -zv api.openai.com 443

# Use alternative ports/endpoints if available
```

## Development Issues

### Import Errors

#### Circular Import Issues

**Problem**: Circular import errors in development

**Solutions**:
```python
# Use lazy imports
def get_agent():
    from src.agents.metabolic import MetabolicAgent
    return MetabolicAgent()

# Restructure imports
# Move shared code to separate modules
# Use dependency injection
```

#### Module Path Issues

**Problem**: Modules not found in development

**Solutions**:
```bash
# Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Use relative imports correctly
from ..agents import MetabolicAgent  # Instead of absolute imports

# Install in development mode
pip install -e .
```

### Testing Issues

#### Test Failures

**Problem**: Tests failing in development

**Solutions**:
```bash
# Run specific test
pytest tests/test_specific.py -v

# Skip slow tests
pytest -m "not slow"

# Run with debug output
pytest -s --log-cli-level=DEBUG

# Update test fixtures
pytest --fixtures-update
```

## Error Code Reference

### Common Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| E001 | Invalid API key | Check API key format and validity |
| E002 | LLM timeout | Increase timeout or check network |
| E003 | Model format error | Validate and convert model format |
| E004 | Solver not available | Install and configure solver |
| E005 | Memory limit exceeded | Increase memory limit or reduce model size |
| E006 | Cache corruption | Clear and rebuild cache |
| E007 | Database error | Rebuild biochemistry database |
| E008 | Permission denied | Fix file/directory permissions |
| E009 | Network error | Check connectivity and proxy settings |
| E010 | Configuration error | Validate configuration settings |

### Warning Codes

| Code | Meaning | Action |
|------|---------|--------|
| W001 | Low memory warning | Monitor memory usage |
| W002 | Cache miss | Normal, will populate cache |
| W003 | Solver suboptimal | Consider using different solver |
| W004 | Model quality warning | Review model validation results |
| W005 | Rate limit warning | Slow down request rate |

## Getting Help

### Diagnostic Information

When reporting issues, include:

```bash
# System information
modelseed-agent debug > debug_info.txt

# Version information
modelseed-agent --version
python --version
pip list > package_list.txt

# Log files
tail -100 logs/current/default/tool_audits/*.json

# Configuration (remove sensitive info)
modelseed-agent show-config --anonymize
```

### Support Channels

1. **GitHub Issues**: Report bugs and feature requests
2. **Documentation**: Check this troubleshooting guide
3. **Community Forum**: Ask questions and share solutions
4. **Email Support**: For enterprise customers

### Before Reporting Issues

1. Check this troubleshooting guide
2. Search existing GitHub issues
3. Test with minimal example
4. Gather diagnostic information
5. Include reproduction steps

## Advanced Troubleshooting

### Debug Mode Analysis

```bash
# Enable comprehensive debugging
export MODELSEED_DEBUG_LEVEL=TRACE
export MODELSEED_DEBUG_ALL=true

# Profile performance
modelseed-agent profile analyze model.xml

# Memory profiling
modelseed-agent memory-profile analyze model.xml
```

### Custom Debugging

```python
# Add custom debug points
import logging
logger = logging.getLogger(__name__)

def debug_analysis(model, step):
    logger.debug(f"Analysis step {step}: {len(model.reactions)} reactions")
    
# Use with breakpoints
import pdb; pdb.set_trace()
```

### Log Analysis

```bash
# Search for errors
grep -r "ERROR" logs/

# Analyze performance
grep -r "duration" logs/ | sort -k3 -n

# Find memory issues
grep -r "memory" logs/ | grep -i warning
```

This troubleshooting guide covers most common issues. For specific problems not covered here, please consult the [API Documentation](api/overview.md) or open an issue on GitHub.