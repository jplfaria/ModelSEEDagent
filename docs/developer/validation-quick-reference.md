# Validation System Quick Reference

## Quick Commands

```bash
# Development workflow
python scripts/dev_validate.py --quick         # Quick validation (5-10s)
python scripts/dev_validate.py --status       # Show current status
python scripts/dev_validate.py --compare      # Compare with previous

# Before commit
python scripts/dev_validate.py --full         # Full validation (15-30s)

# Component testing
python scripts/dev_validate.py --component prompts      # Test prompts
python scripts/dev_validate.py --component quality      # Test quality system

# Advanced comparison
python scripts/validation_comparison.py --mode=trend    # Show trends over time
```

## Understanding Output Files

| File | Purpose | When Populated |
|------|---------|----------------|
| `latest_validation_summary.json` | Current test results | Always |
| `reasoning_metrics.json` | Performance metrics | Always |
| `improvement_patterns.json` | Pattern analysis | After ≥10 runs |
| `learning_insights.json` | Learning insights | After ≥25 runs |

## Key Metrics to Watch

| Metric | Target | Alert Level | Current |
|--------|--------|-------------|---------|
| Success Rate | 100% | <100% | 100% PASS |
| Quality Score | ≥85% | <80% | 88.5% PASS |
| Execution Time | ≤30s | >45s | 25.0s PASS |
| Biological Accuracy | ≥90% | <85% | 92% PASS |

## Empty Files (Normal Behavior)

- **`improvement_patterns.json`**: Empty until 10+ validation runs
- **`learning_insights.json`**: Empty until 25+ validation runs

This is expected behavior - the system needs enough data to identify patterns and generate insights.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Components not available" | `pip install -r requirements.txt` |
| No result files | Run validation first: `python scripts/dev_validate.py --quick` |
| Test failures | Check specific error in `latest_validation_summary.json` |
| Performance regression | Compare trends: `python scripts/validation_comparison.py --mode=trend` |

## Development Workflow

1. **Make changes** (code, prompts, config)
2. **Quick validation**: `python scripts/dev_validate.py --quick`
3. **Check status**: Look for PASS or FAIL in output
4. **Compare changes**: `python scripts/dev_validate.py --compare`
5. **Before commit**: `python scripts/dev_validate.py --full`

## Available Components

- `prompts` - Prompt management system
- `context` - Context enhancement
- `quality` - Quality validation
- `intelligence` - Artifact intelligence
- `integration` - Cross-component integration
