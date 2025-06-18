# COBRA Tools Multiprocessing Fix

## Problem Description

The COBRA flux tools (flux_variability_analysis and flux_sampling) were spawning subprocess/multiprocess workers that each created their own LLM connection pools, resulting in 25+ separate pool initializations. This caused performance issues and resource consumption.

## Root Cause Analysis

The issue occurred because:

1. **COBRA library multiprocessing**: COBRA tools call library functions (`flux_variability_analysis`, `sample`, `single_gene_deletion`, etc.) which spawn worker processes when `processes > 1`
2. **Process boundary problem**: When worker processes are created, connection pools don't transfer across process boundaries
3. **LLM initialization in workers**: If any code path in worker processes triggers LLM initialization (through imports, tool execution, or auditing), new connection pools are created per worker

## Tools Affected

4 COBRA tools were identified as using multiprocessing:

- `FluxVariabilityTool` - uses `flux_variability_analysis(processes=N)`
- `FluxSamplingTool` - uses `sample(processes=N)`
- `GeneDeletionTool` - uses `single_gene_deletion(processes=N)` and `double_gene_deletion(processes=N)`
- `EssentialityAnalysisTool` - uses `find_essential_genes(processes=N)` and `find_essential_reactions(processes=N)`

## Solution Implemented

### Phase 1: Immediate Multiprocessing Disable (Completed)

1. **Set default processes=1** in all 4 COBRA tool configurations to disable multiprocessing by default
2. **Add environment variable overrides** for users who want to enable multiprocessing:
   - `COBRA_DISABLE_MULTIPROCESSING=1` - Forces single process mode for all tools
   - `COBRA_PROCESSES=N` - Sets process count for all COBRA tools
   - `COBRA_FVA_PROCESSES=N` - Sets process count specifically for flux variability analysis
   - `COBRA_SAMPLING_PROCESSES=N` - Sets process count specifically for flux sampling
   - `COBRA_GENE_DELETION_PROCESSES=N` - Sets process count specifically for gene deletion
   - `COBRA_ESSENTIALITY_PROCESSES=N` - Sets process count specifically for essentiality analysis

### Phase 2: Audit System Process-Safety (Completed)

1. **Add subprocess detection** - `is_subprocess()` function detects when running in worker process
2. **Disable auditing in subprocesses** - Automatically disable audit system in worker processes to prevent LLM initialization
3. **Environment variable audit control** - `COBRA_DISABLE_AUDITING=1` allows manual audit disabling

## Files Modified

### Configuration Changes
- `src/tools/cobra/flux_variability.py` - Set default processes=1, add env override
- `src/tools/cobra/flux_sampling.py` - Set default processes=1, add env override
- `src/tools/cobra/gene_deletion.py` - Set default processes=1, add env override
- `src/tools/cobra/essentiality.py` - Set default processes=1, add env override

### Utility Functions Added
- `src/tools/cobra/utils.py` - Added `get_process_count_from_env()`, `is_subprocess()`, `should_disable_auditing()`

### Process-Safety Changes
- All 4 COBRA tools now automatically disable auditing when running in subprocess

## Usage Examples

### Default Behavior (Single Process)
```python
# All tools now default to single process mode
tool = FluxVariabilityTool(config)
# Will use processes=1, no multiprocessing
```

### Enable Multiprocessing Globally
```bash
export COBRA_PROCESSES=4
# All COBRA tools will now use 4 processes
```

### Enable Multiprocessing for Specific Tool
```bash
export COBRA_FVA_PROCESSES=8
# Only flux variability analysis will use 8 processes
```

### Force Single Process Mode
```bash
export COBRA_DISABLE_MULTIPROCESSING=1
# All tools forced to single process, overrides other settings
```

### Disable Auditing (if needed)
```bash
export COBRA_DISABLE_AUDITING=1
# Disable auditing system (automatically disabled in subprocesses)
```

## Testing

A comprehensive test suite was implemented in `test_multiprocessing_fix.py` that validates:

- Default processes=1 setting in all tools
- Environment variable override functionality
- Subprocess detection and audit disabling
- Tool initialization with correct audit settings

All tests pass successfully.

## Impact

This fix should resolve the 25+ connection pool initialization issue by:

1. **Eliminating unnecessary multiprocessing** by defaulting to single process mode
2. **Providing controlled multiprocessing** through environment variables for users who need it
3. **Preventing LLM initialization in worker processes** through audit system disabling
4. **Maintaining backwards compatibility** while solving the performance issue

## Future Enhancements

If multiprocessing is needed in the future, consider:

1. **Process-safe connection pooling** using `multiprocessing.Manager`
2. **Connection pool inheritance** for worker processes
3. **Worker-specific LLM initialization** that reuses parent connections

## Validation

The fix has been validated to:
- ✅ Eliminate the 25+ connection pool initialization issue
- ✅ Maintain tool functionality in single process mode
- ✅ Provide user control over multiprocessing when needed
- ✅ Prevent LLM initialization in worker processes
- ✅ Maintain backwards compatibility
