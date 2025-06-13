# Tests Directory

This directory contains all test files for ModelSEEDagent.

## Structure

### **Automated Tests**
- **`functional/`** - Core functionality tests
- **`integration/`** - Cross-component integration tests
- **`phase_tests/`** - Phase-specific development tests
- **`system/`** - System-level tests
- **`validation/`** - Data validation and verification tests

### **Manual Tests**
- **`manual/`** - Manual testing scripts and debugging tools
  - `test_*.py` - Various manual test scenarios
  - Quick debugging and validation scripts
  - One-off test files for specific issues

### **Test Artifacts**
- **`artifacts/`** - Test output files, logs, and generated data
- **`fixtures/`** - Test data and model files

## Running Tests

### Functional Tests (Recommended)
```bash
# Run all functional tests
python run_functional_tests.py

# Run specific test category
pytest tests/functional/test_metabolic_tools_correctness.py -v
```

### Integration Tests
```bash
# Run integration test suite
pytest tests/integration/ -v
```

### Phase Tests
```bash
# Run specific phase tests
pytest tests/phase_tests/test_phase8_integration.py -v
```

### Manual Tests
```bash
# Run individual manual tests
python tests/manual/test_comprehensive_cli_fix.py
python tests/manual/test_timeout_fix_comprehensive.py
```

## Test Categories

### **Functional Tests** (High Confidence)
- Core tool functionality
- Agent behavior validation
- Data processing correctness
- Algorithm implementation verification

### **Integration Tests** (Medium Confidence)
- LangGraph workflow testing
- Agent orchestration
- Tool integration
- End-to-end workflows

### **Manual Tests** (Development Only)
- Quick debugging scripts
- Issue reproduction
- Performance testing
- Interactive validation

## Test Data

- **Model Files**: `fixtures/test_model.xml`
- **Expected Outputs**: Stored in individual test files
- **Generated Data**: `artifacts/` directory

## Maintenance

- **Archive**: Move old manual tests to `manual/archive/` periodically
- **Cleanup**: Remove obsolete test artifacts
- **Update**: Keep functional tests current with main code changes
