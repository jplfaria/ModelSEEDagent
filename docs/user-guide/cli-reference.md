# CLI Reference

This page is auto-generated from the Typer app; do not edit manually.

## `None`

### `None`

🔄 Quick switch between LLM backends

    Examples:
      modelseed-agent switch argo              # Switch to Argo with default gpt4o
      modelseed-agent switch argo --model gpto1  # Switch to Argo with gpt-o1
      modelseed-agent switch openai           # Switch to OpenAI with default
      modelseed-agent switch local            # Switch to local LLM

| Parameter | Type | Default |
| --- | --- | --- |
| `backend` | str | Ellipsis |
| `model` | str | — |

## `ai-audit None`

### `None`

🎮 Launch interactive audit exploration mode

    Interactive interface for browsing AI workflows, analyzing reasoning
    patterns, and performing verification across multiple workflows.

| Parameter | Type | Default |
| --- | --- | --- |
| `logs_dir` | str | logs |

## `audit list`

### `list`

📋 List recent tool executions

    Shows recent tool audit records with execution details and success status.

| Parameter | Type | Default |
| --- | --- | --- |
| `limit` | int | 10 |
| `session_id` | Optional | — |
| `tool_name` | Optional | — |

## `audit session`

### `session`

📊 Show all tool executions for a specific session

    Displays comprehensive session-level audit information for workflow analysis
    and hallucination pattern detection.

| Parameter | Type | Default |
| --- | --- | --- |
| `session_id` | str | Ellipsis |
| `summary` | bool | false |

## `audit show`

### `show`

🔍 Show detailed audit information for a specific tool execution

    Displays comprehensive execution details including inputs, outputs, console logs,
    and file artifacts for hallucination detection analysis.

| Parameter | Type | Default |
| --- | --- | --- |
| `audit_id` | str | Ellipsis |
| `show_console` | bool | true |
| `show_files` | bool | true |

## `audit verify`

### `verify`

🔍 Verify tool execution for potential hallucinations

    Performs automated checks to detect discrepancies between tool claims
    and actual execution results, helping identify AI hallucinations.

| Parameter | Type | Default |
| --- | --- | --- |
| `audit_id` | str | Ellipsis |
| `check_files` | bool | true |
| `check_claims` | bool | true |
