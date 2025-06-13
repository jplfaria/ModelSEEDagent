# Tests Directory

This directory contains ModelSEEDagent's organized testing infrastructure. For comprehensive biological validation, see also the [Testbed System](../scripts/) and [Testing Strategy Roadmap](../docs/development/testing-infrastructure-roadmap.md).

## Testing Strategy Overview

**When to Use Each Testing Approach**:

- **Pytest Tests** (this directory): Unit/integration testing during development
- **Testbed System** (`scripts/`): Comprehensive biological validation with real models
- **CI Tests** (GitHub Actions): Essential validation for PRs and releases

## Structure

### **Core Test Categories**

#### **Functional Tests** (`functional/`) - High Priority
- **Purpose**: Validate core tool functionality and biological correctness
- **Scope**: Individual tool behavior, algorithm correctness
- **When to Run**: During development, before commits
- **Files**: 5 test modules covering all tool categories

#### **Integration Tests** (`integration/`) - Medium Priority
- **Purpose**: Validate system workflows and component interaction
- **Scope**: LangGraph workflows, agent orchestration, tool chains
- **When to Run**: Before releases, after major changes
- **Files**: Workflow and agent integration validation

#### **System Tests** (`system/`) - Medium Priority
- **Purpose**: End-to-end system validation
- **Scope**: Audit systems, dynamic agents, full workflows
- **When to Run**: Release validation, system updates

### **Development Tests**

#### **Manual Tests** (`manual/`) - Development Only
- **Purpose**: Interactive debugging and validation scripts
- **Organization**:
  - **Active**: Current debug/validation scripts
  - **Archive**: Historical fix scripts and outdated tests
- **When to Use**: Debugging issues, testing specific scenarios

#### **Phase Tests** (`phase_tests/`) - Historical
- **Purpose**: Validation of specific development phases
- **Status**: Maintained for historical reference
- **Usage**: Validate specific feature implementations

### **Test Support**
- **`artifacts/`** - Test outputs, logs, visualizations
- **`fixtures/`** - Test data and model files

## Running Tests

### Quick Decision Guide

```
Need biological validation?
├─ Yes → Use Testbed System
│   ├─ Full validation needed? → python scripts/comprehensive_tool_testbed.py
│   └─ Quick check needed? → CI automatically runs FBA on e_coli_core
└─ No → Use Pytest System
    ├─ Integration testing? → pytest tests/integration/ -v
    ├─ Unit testing? → pytest tests/functional/ -v
    └─ Debugging? → python tests/manual/test_*.py
```

### Functional Tests (Recommended for Development)
```bash
# Run all functional tests (organized by capability)
python tests/run_all_functional_tests.py

# Run specific test category
pytest tests/functional/test_metabolic_tools_correctness.py -v
pytest tests/functional/test_ai_reasoning_correctness.py -v
```

### Integration Tests (Before Major Changes)
```bash
# Run integration test suite
pytest tests/integration/ -v

# Specific workflow testing
pytest tests/integration/test_langgraph_workflow.py -v
```

### Comprehensive Biological Validation (Release Preparation)
```bash
# Full testbed validation (comprehensive)
python scripts/comprehensive_tool_testbed.py

# Split results for analysis
python scripts/split_testbed_results.py

# View results
ls testbed_results/comprehensive/
```

### Manual Testing (Debugging)
```bash
# Active debugging scripts
python tests/manual/test_comprehensive_cli_fix.py
python tests/manual/test_timeout_fix_comprehensive.py

# See archived tests
ls tests/manual/archive/
```

## Test Organization Details

### **Current Test Counts**
- **Functional Tests**: 5 modules, 2,300+ lines of validation code
- **Integration Tests**: 4 workflow validation modules
- **Manual Tests**: 15+ active scripts, 6 archived
- **Testbed Coverage**: 19 tools × 4 models = 76 biological validations

### **CI/CD Integration**
- **Pull Requests**: Essential testbed (FBA on e_coli_core) + full pytest suite
- **Dev Branch**: Testbed validation + functional tests
- **Release**: Comprehensive validation including security scans

### **Biological Validation Standards**
- **Growth Rates**: 0.1-1.0 h⁻¹ for bacterial models
- **Essential Genes**: 10-20% of total genes expected
- **Flux Consistency**: Carbon balance and ATP production validation
- **Model Compatibility**: Format consistency across ModelSEED ↔ COBRApy

## Test Data and Fixtures

### **Model Files**
- **`fixtures/test_model.xml`**: Basic test model for unit tests
- **Testbed Models**: 4 comprehensive models in `data/examples/`
  - `e_coli_core.xml`: BiGG core model (CI validation)
  - `iML1515.xml`: Genome-scale E. coli model
  - `EcoliMG1655.xml`: ModelSEED E. coli model
  - `Mycoplasma_G37.GMM.mdl.xml`: Minimal organism model

### **Generated Data**
- **`artifacts/sessions/`**: Test session data
- **`artifacts/visualizations/`**: Test result visualizations
- **`testbed_results/`**: Comprehensive biological validation results

## Maintenance and Best Practices

### **Regular Maintenance**
- **Monthly**: Archive outdated manual tests to `manual/archive/`
- **Quarterly**: Review functional test coverage and update
- **Release**: Run full testbed validation and update baselines

### **Adding New Tests**
1. **Functional Tests**: For core tool functionality and biological validation
2. **Integration Tests**: For multi-component workflows
3. **Manual Tests**: For debugging specific issues (archive when resolved)

### **Test Selection Guidelines**
- **Development**: Use functional tests for rapid feedback
- **Integration**: Use integration tests before major merges
- **Biological Accuracy**: Use testbed system for scientific validation
- **Release**: Use comprehensive validation (pytest + testbed)

## Performance Expectations

- **Functional Tests**: <5 minutes total execution
- **Integration Tests**: <10 minutes for full suite
- **Essential Testbed (CI)**: <3 minutes for FBA validation
- **Full Testbed**: 15-30 minutes for comprehensive validation

## Troubleshooting

### **Common Issues**
- **Model Loading Failures**: Check `data/examples/` directory and file permissions
- **Import Errors**: Ensure all dependencies installed with `poetry install --with dev`
- **Biological Validation Failures**: Compare against known good results in `testbed_results/`

### **Getting Help**
- **Testing Strategy**: See [Testing Infrastructure Roadmap](../docs/development/testing-infrastructure-roadmap.md)
- **Testbed System**: Documentation in `scripts/` directory
- **Functional Tests**: Each test file contains detailed validation criteria
