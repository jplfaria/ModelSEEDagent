# Logs Directory

This directory contains execution logs and audit trails for ModelSEEDagent runs.

## Structure

- **`current/`** - Recent runs (last 5 of each type)
  - `langgraph_run_YYYYMMDD_HHMMSS/` - LangGraph agent execution logs
  - `realtime_run_YYYYMMDD_HHMMSS/` - RealTime agent execution logs
  - `default/` - Default tool audit logs
  - `learning_memory/` - AI learning and pattern recognition logs

- **`archive/`** - Compressed archives of older logs
  - `old_logs_YYYYMMDD.tar.gz` - Archived log directories

## Retention Policy

- **Current**: Keep last 10 LangGraph runs and last 5 RealTime runs
- **Archive**: Automatically compress and archive older runs
- **Cleanup**: Remove archives older than 30 days
- **Automation**: Use `scripts/cleanup_logs.py` for automated maintenance

## Log Types

### LangGraph Runs
- **Purpose**: Graph-based workflow execution
- **Contains**: Execution logs, performance metrics, visualization files
- **Typical size**: 1-10MB per run

### RealTime Runs  
- **Purpose**: Dynamic AI decision-making workflows
- **Contains**: Audit trails, reasoning steps, complete analysis chains
- **Typical size**: 500KB-5MB per run

### Tool Audits
- **Purpose**: Individual tool execution tracking
- **Contains**: Tool inputs, outputs, execution times, error logs
- **Location**: `current/default/tool_audits/`

## Maintenance

### Automated Cleanup (Recommended)
```bash
# Run automated cleanup script
python scripts/cleanup_logs.py

# Preview what would be cleaned (dry run)
python scripts/cleanup_logs.py --dry-run
```

### Manual Cleanup (Legacy)
```bash
# Archive old logs manually
tar -czf logs/archive/old_logs_$(date +%Y%m%d).tar.gz logs/langgraph_run_* logs/realtime_run_*

# Keep only recent logs (adjust number as needed)
ls -1t logs/langgraph_run_* | tail -n +11 | xargs rm -rf
ls -1t logs/realtime_run_* | tail -n +6 | xargs rm -rf
```

## Monitoring

- **Size limit**: Logs directory should not exceed 100MB
- **Count limit**: No more than 20 directories in current/
- **Performance**: Large log directories may slow down agent initialization