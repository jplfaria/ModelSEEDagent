# Sessions Directory

This directory contains interactive CLI session data and analysis results.

## Structure

- **`*.json`** - Recent interactive session files
- **`archive/`** - Older session files (automatically archived)

## Session Files

### Format
Each session file contains:
```json
{
  "session_id": "unique_identifier",
  "timestamp": "2025-06-09T10:30:00",
  "user_queries": ["list of user inputs"],
  "agent_responses": ["list of agent outputs"],
  "tools_used": ["list of tools executed"],
  "performance_metrics": {...},
  "metadata": {...}
}
```

### Purpose
- **User Experience Analysis**: Track interaction patterns
- **Performance Monitoring**: Measure response times and success rates
- **Feature Usage**: Understand which tools are most valuable
- **Error Analysis**: Debug failed sessions and improve reliability

## Management

### Automatic Archival
- Sessions older than 7 days are automatically moved to `archive/`
- Archive files are compressed monthly

### Manual Cleanup
```bash
# Archive old sessions
mkdir -p sessions/archive
mv sessions/*.json sessions/archive/

# Compress monthly archives
tar -czf sessions/archive/sessions_$(date +%Y%m).tar.gz sessions/archive/*.json
rm sessions/archive/*.json
```

### Privacy
- Session files may contain user queries and model paths
- Do not commit session files to public repositories
- Sessions are gitignored by default

## File Naming Convention

- **Format**: `{session_id}.json`
- **Session ID**: 8-character hexadecimal identifier
- **Example**: `a1fcb3c7.json`

## Monitoring

- **Size**: Individual sessions typically 10-100KB
- **Count**: Keep no more than 50 sessions in main directory
- **Performance**: Large session files may slow CLI startup