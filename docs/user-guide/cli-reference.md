# CLI Reference

This page is auto-generated from the Typer app; do not edit manually.

## `None`

### `None`

| Parameter | Type | Default |
| --- | --- | --- |
| `backend` | str | <typer.models.ArgumentInfo object at 0x36285da10> |
| `model` | str | <typer.models.OptionInfo object at 0x36285f250> |

## `ai-audit None`

### `None`

| Parameter | Type | Default |
| --- | --- | --- |
| `logs_dir` | str | <typer.models.OptionInfo object at 0x3589ae6d0> |

## `audit list`

### `list`

| Parameter | Type | Default |
| --- | --- | --- |
| `limit` | int | <typer.models.OptionInfo object at 0x362851c50> |
| `session_id` | Optional | <typer.models.OptionInfo object at 0x362851f10> |
| `tool_name` | Optional | <typer.models.OptionInfo object at 0x362851990> |

## `audit session`

### `session`

| Parameter | Type | Default |
| --- | --- | --- |
| `session_id` | str | <typer.models.ArgumentInfo object at 0x362986e10> |
| `summary` | bool | <typer.models.OptionInfo object at 0x362987fd0> |

## `audit show`

### `show`

| Parameter | Type | Default |
| --- | --- | --- |
| `audit_id` | str | <typer.models.ArgumentInfo object at 0x3628532d0> |
| `show_console` | bool | <typer.models.OptionInfo object at 0x359cb6210> |
| `show_files` | bool | <typer.models.OptionInfo object at 0x359cb6150> |

## `audit verify`

### `verify`

| Parameter | Type | Default |
| --- | --- | --- |
| `audit_id` | str | <typer.models.ArgumentInfo object at 0x362987d90> |
| `check_files` | bool | <typer.models.OptionInfo object at 0x362984fd0> |
| `check_claims` | bool | <typer.models.OptionInfo object at 0x362984d50> |
