# Intelligence Validation System - Developer Guide

**ModelSEEDagent Intelligence Enhancement Framework**
**Version**: 1.0
**Last Updated**: June 18, 2025

## Overview

The Intelligence Validation System provides comprehensive testing and continuous improvement tracking for the ModelSEEDagent's enhanced reasoning capabilities. This guide explains how to use the validation system during development and understand its outputs.

## Quick Start

### Running Validation Tests

```bash
# Full comprehensive validation (recommended for releases)
python scripts/integrated_intelligence_validator.py --mode=full

# Quick validation (recommended for development)
python scripts/integrated_intelligence_validator.py --mode=quick

# Component-specific validation
python scripts/integrated_intelligence_validator.py --mode=component --component=prompts
```

### Understanding Results

After running validation, check these key files:
- `results/reasoning_validation/latest_validation_summary.json` - Latest test results
- `results/reasoning_validation/performance_summary.md` - Human-readable performance report
- `results/reasoning_validation/reasoning_metrics.json` - Detailed performance metrics

## CLI Interface Reference

### Command Line Options

```bash
python scripts/integrated_intelligence_validator.py [OPTIONS]
```

#### Mode Options

| Mode | Description | Use Case | Duration |
|------|-------------|----------|----------|
| `--mode=full` | Complete validation suite (default) | Release validation, comprehensive testing | ~15-30 seconds |
| `--mode=quick` | Essential tests only (high priority) | Development cycles, rapid feedback | ~5-10 seconds |
| `--mode=component` | Validate specific component | Debugging, focused testing | ~3-8 seconds |

#### Additional Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--component=<name>` | Component to validate (requires --mode=component) | None | `--component=prompts` |
| `--output=<path>` | Output directory for results | `results/reasoning_validation` | `--output=my_test_results` |

### Usage Examples

```bash
# Development workflow - quick validation after changes
python scripts/integrated_intelligence_validator.py --mode=quick

# Test prompt changes specifically
python scripts/integrated_intelligence_validator.py --mode=component --component=prompts

# Full validation before commit
python scripts/integrated_intelligence_validator.py --mode=full

# Custom output location
python scripts/integrated_intelligence_validator.py --mode=quick --output=results/my_validation_run
```

## Understanding Output Files

### Core Result Files

#### `latest_validation_summary.json`
**Purpose**: Latest test results summary
**Key Metrics**:
```json
{
  "total_tests": 6,
  "passed_tests": 6,
  "average_quality_score": 0.885,
  "average_execution_time": 25.0,
  "system_performance": {
    "overall_success_rate": 1.0,
    "total_artifacts_generated": 10,
    "total_hypotheses_generated": 10
  }
}
```

#### `reasoning_metrics.json`
**Purpose**: Detailed performance metrics for trend analysis
**Key Metrics**:
```json
[
  {
    "overall_quality": 0.89,
    "biological_accuracy": 0.92,
    "artifact_usage_rate": 0.76,
    "hypothesis_count": 3,
    "execution_time": 28.0,
    "user_satisfaction": 0.94,
    "timestamp": "2025-06-18T19:42:52.987974"
  }
]
```

### Learning and Improvement Files

#### `improvement_patterns.json`
**Purpose**: Identified improvement patterns over time
**Behavior**:
- **Empty until ≥10 validation runs** - This is normal and expected
- Automatically populates as you run more validations
- Tracks patterns like quality improvements, efficiency gains, error reductions

#### `learning_insights.json`
**Purpose**: High-level learning insights from accumulated data
**Behavior**:
- **Empty until ≥25 validation runs** - This is normal and expected
- Generates insights about system evolution, user satisfaction correlations, trade-offs
- Provides actionable recommendations for improvements

### Historical Files

#### `validation_summary_YYYYMMDD_HHMMSS.json`
**Purpose**: Timestamped validation summaries for comparison
**Usage**: Compare performance across different development iterations

#### `validation_results_YYYYMMDD_HHMMSS.json`
**Purpose**: Detailed test results with individual test data
**Usage**: Debug specific test failures or performance issues

## Development Workflow

### After Making Changes

1. **Quick Validation** (recommended for iterative development):
   ```bash
   python scripts/integrated_intelligence_validator.py --mode=quick
   ```

2. **Check Key Metrics**:
   ```bash
   # Quick quality check
   cat results/reasoning_validation/latest_validation_summary.json | grep average_quality_score

   # Quick success rate check
   cat results/reasoning_validation/latest_validation_summary.json | grep overall_success_rate
   ```

3. **Review Changes**: Compare with previous run results

### Before Committing

1. **Full Validation**:
   ```bash
   python scripts/integrated_intelligence_validator.py --mode=full
   ```

2. **Verify All Tests Pass**:
   ```bash
   # Should show 0 failed tests
   cat results/reasoning_validation/latest_validation_summary.json | grep failed_tests
   ```

3. **Check Performance Regression**:
   - Compare `average_quality_score` with previous runs
   - Ensure `overall_success_rate` remains 1.0

### Comparing Results Over Time

#### Manual Comparison
```bash
# Compare latest vs specific previous run
diff results/reasoning_validation/latest_validation_summary.json \
     results/reasoning_validation/validation_summary_20250618_195304.json
```

#### Key Metrics to Track
- **Quality Score**: Should remain ≥0.85 (target: >0.90)
- **Success Rate**: Should remain 1.0 (100%)
- **Execution Time**: Should not significantly increase
- **Artifact/Hypothesis Generation**: Should remain consistent

## Troubleshooting

### Common Issues

#### "Components not available" Error
**Cause**: Import errors in intelligence components
**Solution**:
```bash
# Check if all required packages are installed
pip install -r requirements.txt

# Verify Python path
export PYTHONPATH="/path/to/ModelSEEDagent:$PYTHONPATH"
```

#### Empty or Missing Result Files
**Cause**: Insufficient metrics for pattern/insight generation
**Expected Behavior**:
- `improvement_patterns.json` empty until 10+ runs ✓
- `learning_insights.json` empty until 25+ runs ✓
- Other files should always contain data

#### Test Failures
**Investigation Steps**:
1. Check `latest_validation_summary.json` for specific failed tests
2. Review `error_message` and `error_details` in test results
3. Run component-specific validation: `--mode=component --component=<failing_component>`

### Performance Benchmarks

#### Target Metrics (Minimum Acceptable)
- **Overall Success Rate**: 100% (1.0)
- **Average Quality Score**: ≥85% (0.85)
- **Average Execution Time**: ≤30 seconds
- **Biological Accuracy**: ≥90% (0.90)

#### Excellence Targets
- **Average Quality Score**: ≥90% (0.90)
- **Average Execution Time**: ≤25 seconds
- **Artifact Usage Rate**: ≥75% (0.75)
- **User Satisfaction**: ≥90% (0.90)

## Integration with Development Tools

### Git Hooks (Recommended)

Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
echo "Running intelligence validation..."
python scripts/integrated_intelligence_validator.py --mode=quick

# Check if validation passed
if [ $? -ne 0 ]; then
    echo "FAIL Validation failed - commit aborted"
    exit 1
fi

echo "PASS: Validation passed"
```

### IDE Integration

#### VS Code Tasks (`.vscode/tasks.json`)
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Quick Validation",
            "type": "shell",
            "command": "python",
            "args": ["scripts/integrated_intelligence_validator.py", "--mode=quick"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        }
    ]
}
```

## Advanced Usage

### Custom Validation Scenarios

#### Testing Specific Queries
Modify test cases in `scripts/integrated_intelligence_validator.py`:
```python
# Add custom test case
ValidationTestCase(
    test_id="CUSTOM_001",
    name="My Custom Test",
    query="Your specific test query here",
    validation_criteria={"min_quality_score": 0.8}
)
```

#### Component-Specific Testing
Available components for `--mode=component`:
- `prompts` - Test prompt management system
- `context` - Test context enhancement
- `quality` - Test quality validation
- `intelligence` - Test artifact intelligence
- `integration` - Test cross-component integration

### Performance Monitoring

#### Long-term Tracking
```bash
# Create daily validation reports
python scripts/integrated_intelligence_validator.py --mode=full \
  --output="results/daily_validation/$(date +%Y%m%d)"
```

#### Automated Monitoring
```bash
# Add to cron for daily monitoring
0 9 * * * cd /path/to/ModelSEEDagent && python scripts/integrated_intelligence_validator.py --mode=full
```

## Best Practices

### Development Cycle
1. **Make changes** (code, prompts, configuration)
2. **Run quick validation** (`--mode=quick`)
3. **Check key metrics** (quality score, success rate)
4. **Iterate** until satisfied
5. **Run full validation** before commit (`--mode=full`)
6. **Track trends** over multiple development cycles

### Performance Optimization
- Use `--mode=quick` for rapid iteration
- Use `--mode=component` when working on specific areas
- Save `--mode=full` for comprehensive testing before releases
- Monitor trends in `reasoning_metrics.json` for performance regression

### Data Management
- Keep historical validation files for trend analysis
- Archive old results periodically to manage disk space
- Back up validation results before major system changes

## Support and Troubleshooting

For additional help:
- Check the validation system logs for detailed error information
- Review the intelligence framework documentation
- Examine specific test failures in detailed result files
- Use component-specific validation to isolate issues

---

*Intelligence Validation System Guide*
*Part of the ModelSEEDagent Intelligence Enhancement Framework v1.0*
