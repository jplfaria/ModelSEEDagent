# Debug Configuration System

This document describes the comprehensive debug configuration system that allows fine-grained control over logging verbosity for different components of ModelSEEDagent.

## Overview

The debug configuration system supports multiple levels of debug verbosity and component-specific logging control through environment variables. This allows users to see exactly the level of detail they need while suppressing noise from other components.

## Environment Variables

### Overall Debug Level

**`MODELSEED_DEBUG_LEVEL`** - Controls overall verbosity level
- `quiet` - Minimal output, errors only
- `normal` - Standard info messages (default)
- `verbose` - Detailed debugging
- `trace` - Maximum verbosity (enables all component debugging)

### Component-Specific Debug Flags

**`MODELSEED_DEBUG_COBRAKBASE`** - Controls cobrakbase integration messages
- `true` - Show cobrakbase availability and fallback messages
- `false` - Suppress cobrakbase messages (default)

**`MODELSEED_DEBUG_LANGGRAPH`** - Controls LangGraph initialization debugging
- `true` - Show LangGraph agent creation and initialization messages
- `false` - Suppress LangGraph initialization spam (default)

**`MODELSEED_DEBUG_HTTP`** - Controls HTTP/SSL connection debugging
- `true` - Show HTTP requests and SSL connection details
- `false` - Suppress HTTP connection noise (default)

**`MODELSEED_DEBUG_TOOLS`** - Controls tool execution debugging
- `true` - Show detailed tool execution information
- `false` - Standard tool execution messages (default)

**`MODELSEED_DEBUG_LLM`** - Controls LLM interaction debugging
- `true` - Show LLM request/response details
- `false` - Standard LLM interaction messages (default)

### Special Logging Flags

**`MODELSEED_LOG_LLM_INPUTS`** - Complete LLM input logging for analysis
- `true` - Log complete prompts and tool data sent to LLM (for debugging AI decisions)
- `false` - Standard LLM logging (default)

## Usage Examples

### Quiet Mode (Minimal Output)
```bash
export MODELSEED_DEBUG_LEVEL=quiet
modelseed-agent interactive
```

### Verbose Mode with Specific Component Control
```bash
export MODELSEED_DEBUG_LEVEL=verbose
export MODELSEED_DEBUG_COBRAKBASE=true
export MODELSEED_DEBUG_LANGGRAPH=false
modelseed-agent analyze model.xml
```

### Trace Mode (Maximum Debugging)
```bash
export MODELSEED_DEBUG_LEVEL=trace
modelseed-agent interactive
```

### LLM Analysis Mode (For AI Decision Debugging)
```bash
export MODELSEED_LOG_LLM_INPUTS=true
export MODELSEED_DEBUG_LLM=true
modelseed-agent interactive
```

### Suppress Specific Noise Sources
```bash
# Suppress cobrakbase and HTTP noise while keeping other debug info
export MODELSEED_DEBUG_LEVEL=verbose
export MODELSEED_DEBUG_COBRAKBASE=false
export MODELSEED_DEBUG_HTTP=false
modelseed-agent analyze model.xml
```

## Debug Configuration Commands

### Check Current Debug Status
```bash
modelseed-agent debug
```

This command shows:
- Current debug level
- Status of all component flags
- Tips for debug control
- Example usage patterns

### View System Configuration
```bash
modelseed-agent status
```

Shows overall system status including debug configuration.

## Implementation Details

### Auto-Configuration Rules

1. **Quiet Mode**: Overrides all component flags to `false`
2. **Trace Mode**: Enables all component debugging unless explicitly disabled
3. **Normal/Verbose Modes**: Use explicit component flag values

### Logging Levels

- **QUIET**: Python logging.WARNING level
- **NORMAL**: Python logging.INFO level  
- **VERBOSE**: Python logging.DEBUG level
- **TRACE**: Python logging level 5 (very detailed)

### Component Integration

The debug system integrates with:
- **cobrakbase**: Controls availability and fallback messages
- **LangGraph**: Controls agent initialization spam
- **HTTP libraries**: Controls httpx/httpcore debug noise
- **Tool execution**: Controls detailed tool execution logs
- **LLM interactions**: Controls AI decision debugging

## Benefits

1. **Reduces Noise**: Users can suppress repetitive or irrelevant debug messages
2. **Targeted Debugging**: Enable debugging only for components of interest
3. **Preserves Detail**: When needed, full verbosity is available
4. **Flexible Control**: Mix and match different debug levels per component
5. **Easy Management**: Single environment variables control complex logging behavior

## Examples of Resolved Issues

### Before (Excessive cobrakbase Messages)
```
2025-01-01 10:00:01 - DEBUG - cobrakbase.core not available - using standard COBRApy methods (fallback)
2025-01-01 10:00:02 - DEBUG - cobrakbase.core not available - using standard COBRApy methods (fallback)
2025-01-01 10:00:03 - DEBUG - cobrakbase.core not available - using standard COBRApy methods (fallback)
```

### After (Controlled Messaging)
```bash
# Default: Message appears once only at very low log level
export MODELSEED_DEBUG_COBRAKBASE=false
# Result: Clean logs with no repetitive cobrakbase spam

# When needed: Explicit cobrakbase debugging
export MODELSEED_DEBUG_COBRAKBASE=true
# Result: cobrakbase messages visible for debugging integration issues
```

This system provides the level of detail users want while maintaining clean, readable logs for standard operation.