# Debug Configuration

ModelSEEDagent provides comprehensive debug configuration options for troubleshooting and development. The debug system allows fine-grained control over logging verbosity for different components.

## Debug Levels

### Overall Debug Level

Set the general verbosity level using `MODELSEED_DEBUG_LEVEL`:

- **`quiet`** - Minimal output, errors only
- **`normal`** - Standard information messages (default)
- **`verbose`** - Detailed debugging information
- **`trace`** - Maximum verbosity with all component debugging enabled

## Component-Specific Debug Controls

### Tool Execution Debugging
**`MODELSEED_DEBUG_TOOLS`** - Controls tool execution information
- `true` - Show detailed tool execution information
- `false` - Standard tool execution messages (default)

### LLM Interaction Debugging
**`MODELSEED_DEBUG_LLM`** - Controls language model interaction details
- `true` - Show LLM request/response details
- `false` - Standard LLM interaction messages (default)

### Agent System Debugging
**`MODELSEED_DEBUG_LANGGRAPH`** - Controls agent initialization debugging
- `true` - Show agent creation and initialization messages
- `false` - Suppress agent initialization details (default)

### Network Debugging
**`MODELSEED_DEBUG_HTTP`** - Controls HTTP/SSL connection debugging
- `true` - Show HTTP requests and SSL connection details
- `false` - Suppress HTTP connection details (default)

### Integration Debugging
**`MODELSEED_DEBUG_COBRAKBASE`** - Controls COBRApy integration messages
- `true` - Show cobrakbase availability and fallback messages
- `false` - Suppress cobrakbase messages (default)

## Special Logging Options

### Complete LLM Input Logging
**`MODELSEED_LOG_LLM_INPUTS`** - Log complete prompts for AI analysis
- `true` - Log complete prompts and tool data sent to LLM
- `false` - Standard LLM logging (default)

**Note**: When enabled, this logs all prompts sent to language models, which is useful for debugging AI decisions but may produce large log files.

## Configuration Examples

### Development Mode
For comprehensive debugging during development:
```bash
export MODELSEED_DEBUG_LEVEL=verbose
export MODELSEED_DEBUG_TOOLS=true
export MODELSEED_DEBUG_LLM=true
export MODELSEED_LOG_LLM_INPUTS=true
```

### Production Mode
For minimal logging in production:
```bash
export MODELSEED_DEBUG_LEVEL=quiet
export MODELSEED_DEBUG_TOOLS=false
export MODELSEED_DEBUG_LLM=false
export MODELSEED_DEBUG_HTTP=false
```

### Tool-Focused Debugging
For debugging tool execution issues:
```bash
export MODELSEED_DEBUG_LEVEL=normal
export MODELSEED_DEBUG_TOOLS=true
export MODELSEED_DEBUG_COBRAKBASE=true
```

### LLM-Focused Debugging
For debugging AI reasoning issues:
```bash
export MODELSEED_DEBUG_LEVEL=verbose
export MODELSEED_DEBUG_LLM=true
export MODELSEED_DEBUG_LANGGRAPH=true
export MODELSEED_LOG_LLM_INPUTS=true
```

## Using Debug Configuration

### Command Line
Check your current debug configuration:
```bash
modelseed-agent debug
```

### Environment File
Create a `.env` file with your debug settings:
```bash
# .env file
MODELSEED_DEBUG_LEVEL=verbose
MODELSEED_DEBUG_TOOLS=true
MODELSEED_DEBUG_LLM=false
```

### Runtime Configuration
Set debug options for a single command:
```bash
MODELSEED_DEBUG_TOOLS=true modelseed-agent analyze model.xml
```

## Debug Output Interpretation

### Tool Execution Messages
When `MODELSEED_DEBUG_TOOLS=true`:
- Tool selection and parameter information
- Execution timing and performance metrics
- Input validation and output processing details

### LLM Interaction Messages  
When `MODELSEED_DEBUG_LLM=true`:
- Model selection and configuration
- Request timing and response processing
- Error handling and retry logic

### Agent System Messages
When `MODELSEED_DEBUG_LANGGRAPH=true`:
- Agent initialization and configuration
- Workflow orchestration details
- Inter-agent communication

## Performance Considerations

- **Trace Level**: Can significantly impact performance due to extensive logging
- **LLM Input Logging**: May produce large log files when analyzing complex models
- **HTTP Debugging**: Adds overhead to network operations

## Log File Locations

Debug logs are stored in the `logs/` directory:
- **Current logs**: `logs/current/`
- **Archived logs**: `logs/archive/`
- **Tool-specific logs**: `logs/current/default/tool_audits/`

For log management and retention policies, see the `logs/README.md` file.

## Common Debug Scenarios

### Model Loading Issues
```bash
export MODELSEED_DEBUG_TOOLS=true
export MODELSEED_DEBUG_COBRAKBASE=true
modelseed-agent analyze problematic_model.xml
```

### LLM Connection Problems
```bash
export MODELSEED_DEBUG_LLM=true
export MODELSEED_DEBUG_HTTP=true
modelseed-agent analyze model.xml
```

### Performance Analysis
```bash
export MODELSEED_DEBUG_LEVEL=verbose
export MODELSEED_DEBUG_TOOLS=true
modelseed-agent analyze model.xml --performance-metrics
```

### AI Reasoning Debugging
```bash
export MODELSEED_DEBUG_LLM=true
export MODELSEED_LOG_LLM_INPUTS=true
modelseed-agent analyze model.xml --mode advanced
```

This debug configuration system provides the flexibility needed for both development and production environments while maintaining system performance when debug features are disabled.