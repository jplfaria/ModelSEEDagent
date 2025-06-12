# Contributing to ModelSEEDagent

Thank you for your interest in contributing to ModelSEEDagent! This guide will help you get started with contributing to the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of metabolic modeling concepts
- Familiarity with COBRApy and ModelSEED (helpful but not required)

### Development Setup

1. **Fork and Clone the Repository**

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/ModelSEEDagent.git
cd ModelSEEDagent

# Add upstream remote
git remote add upstream https://github.com/ModelSEED/ModelSEEDagent.git
```

2. **Set Up Development Environment**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

3. **Verify Installation**

```bash
# Run tests
pytest tests/

# Check code style
black --check src/
flake8 src/

# Verify basic functionality
modelseed-agent debug
```

## Development Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for new features
- `feature/feature-name`: Feature development
- `bugfix/issue-description`: Bug fixes
- `docs/topic`: Documentation updates

### Creating a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout develop
git merge upstream/develop

# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes...

# Commit and push
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(tools): add new flux sampling tool
fix(agents): resolve memory leak in workflow execution
docs(api): update tool reference documentation
test(cobra): add integration tests for FBA tools
```

## Code Standards

### Python Style

We use [Black](https://black.readthedocs.io/) for code formatting and [flake8](https://flake8.pycqa.org/) for linting:

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

### Code Structure

```
src/
├── agents/          # AI agent implementations
├── tools/           # Analysis tool implementations
├── llm/             # LLM integration
├── cli/             # Command-line interface
├── config/          # Configuration management
├── interactive/     # Interactive interfaces
└── workflow/        # Workflow management
```

### Naming Conventions

- **Classes**: PascalCase (`MetabolicAgent`)
- **Functions/Methods**: snake_case (`analyze_model`)
- **Variables**: snake_case (`model_path`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_TIMEOUT`)
- **Files/Modules**: snake_case (`metabolic_agent.py`)

### Documentation Standards

#### Docstrings

Use Google-style docstrings:

```python
def analyze_model(model_path: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """Analyze a metabolic model using AI-powered workflows.
    
    Args:
        model_path: Path to the model file (SBML, JSON, or MAT format)
        analysis_type: Type of analysis to perform ("basic", "comprehensive", "custom")
    
    Returns:
        Dictionary containing analysis results with keys:
        - "model_info": Basic model information
        - "analysis_results": Detailed analysis output
        - "recommendations": AI-generated recommendations
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If analysis_type is not supported
        ModelAnalysisError: If analysis fails
    
    Example:
        >>> results = analyze_model("data/models/e_coli.xml", "comprehensive")
        >>> print(f"Model has {results['model_info']['reactions']} reactions")
    """
```

#### Type Hints

Use type hints throughout the codebase:

```python
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

def process_results(
    results: List[Dict[str, Any]], 
    output_path: Optional[Path] = None
) -> Dict[str, Union[str, int, float]]:
    """Process analysis results."""
    pass
```

### Error Handling

Use specific exception classes and proper error handling:

```python
# Custom exceptions
class ModelSeedAgentError(Exception):
    """Base exception for ModelSEEDagent."""
    pass

class ModelAnalysisError(ModelSeedAgentError):
    """Raised when model analysis fails."""
    pass

class LLMConnectionError(ModelSeedAgentError):
    """Raised when LLM connection fails."""
    pass

# Usage
def analyze_model(model_path: str) -> Dict[str, Any]:
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        raise ModelAnalysisError(f"Model file not found: {model_path}")
    except Exception as e:
        raise ModelAnalysisError(f"Failed to load model: {e}") from e
    
    return perform_analysis(model)
```

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── functional/     # Functional tests
├── fixtures/       # Test fixtures and data
└── conftest.py     # Pytest configuration
```

### Writing Tests

Use pytest with descriptive test names:

```python
# tests/unit/test_metabolic_agent.py
import pytest
from src.agents.metabolic import MetabolicAgent
from src.llm.factory import LLMFactory

class TestMetabolicAgent:
    """Test cases for MetabolicAgent class."""
    
    @pytest.fixture
    def agent(self):
        """Create a test agent instance."""
        llm = LLMFactory.create_llm("mock")
        return MetabolicAgent(llm)
    
    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent is not None
        assert len(agent.tools) > 0
    
    def test_analyze_model_with_valid_input(self, agent, sample_model):
        """Test model analysis with valid input."""
        result = agent.analyze(sample_model)
        
        assert "analysis_results" in result
        assert "recommendations" in result
        assert result["success"] is True
    
    def test_analyze_model_with_invalid_input(self, agent):
        """Test model analysis with invalid input."""
        with pytest.raises(ModelAnalysisError):
            agent.analyze("nonexistent_model.xml")
    
    @pytest.mark.slow
    def test_comprehensive_analysis(self, agent, complex_model):
        """Test comprehensive analysis with complex model."""
        # This test takes longer to run
        result = agent.analyze(complex_model, analysis_type="comprehensive")
        assert len(result["analysis_results"]) > 10
```

### Test Data and Fixtures

```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def sample_model(test_data_dir):
    """Return path to sample model file."""
    return test_data_dir / "e_coli_core.xml"

@pytest.fixture
def mock_llm_response():
    """Return mock LLM response for testing."""
    return {
        "analysis": "This is a test response",
        "recommendations": ["Use glucose minimal medium", "Check for gene essentiality"]
    }
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_metabolic_agent.py

# Run with coverage
pytest --cov=src

# Run only fast tests
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run specific test
pytest tests/unit/test_metabolic_agent.py::TestMetabolicAgent::test_agent_initialization
```

## Adding New Features

### Tool Development

To add a new analysis tool:

1. **Create the tool class**:

```python
# src/tools/cobra/new_tool.py
from typing import Dict, Any
from .base import CobrapyTool

class NewAnalysisTool(CobrapyTool):
    """New analysis tool for metabolic models."""
    
    name = "new_analysis"
    description = "Performs new type of analysis on metabolic models"
    
    def execute(self, model_path: str, **kwargs) -> Dict[str, Any]:
        """Execute the new analysis.
        
        Args:
            model_path: Path to the model file
            **kwargs: Additional parameters
            
        Returns:
            Analysis results dictionary
        """
        model = self.load_model(model_path)
        
        # Implement your analysis logic here
        results = self._perform_analysis(model, **kwargs)
        
        return {
            "tool_name": self.name,
            "model_id": model.id,
            "results": results,
            "success": True
        }
    
    def _perform_analysis(self, model, **kwargs):
        """Implement the core analysis logic."""
        # Your implementation here
        pass
```

2. **Register the tool**:

```python
# src/tools/cobra/__init__.py
from .new_tool import NewAnalysisTool

COBRA_TOOLS = [
    # ... existing tools ...
    NewAnalysisTool,
]
```

3. **Add tests**:

```python
# tests/unit/tools/test_new_tool.py
import pytest
from src.tools.cobra.new_tool import NewAnalysisTool

class TestNewAnalysisTool:
    def test_tool_execution(self, sample_model):
        tool = NewAnalysisTool()
        result = tool.execute(sample_model)
        
        assert result["success"] is True
        assert "results" in result
```

### Agent Development

To add a new agent type:

1. **Inherit from base agent**:

```python
# src/agents/new_agent.py
from typing import Dict, Any, List
from .base import BaseAgent

class NewAgent(BaseAgent):
    """New specialized agent for specific workflows."""
    
    def __init__(self, llm, tools: List[Any], config: Dict[str, Any] = None):
        super().__init__(llm, tools, config)
        self.specialized_config = config.get("specialized", {})
    
    def analyze(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform specialized analysis."""
        # Implement specialized logic
        pass
```

2. **Add to agent factory**:

```python
# src/agents/factory.py
from .new_agent import NewAgent

def create_agent(agent_type: str, llm, tools, config=None):
    if agent_type == "new":
        return NewAgent(llm, tools, config)
    # ... existing agent types ...
```

## Documentation

### API Documentation

Use mkdocstrings for automatic API documentation:

```python
def analyze_model(model_path: str) -> Dict[str, Any]:
    """Analyze a metabolic model.
    
    This function performs comprehensive analysis of a metabolic model
    using AI-powered workflows.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Analysis results dictionary
        
    Example:
        ```python
        from modelseed_agent import analyze_model
        
        results = analyze_model("data/models/e_coli.xml")
        print(results["summary"])
        ```
    """
```

### User Documentation

For user-facing documentation:

1. Update relevant .md files in `docs/`
2. Add examples to `examples/`
3. Create Jupyter notebook tutorials in `notebooks/`

### Changelog

Update `CHANGELOG.md` with your changes:

```markdown
## [Unreleased]

### Added
- New flux sampling tool for enhanced metabolic analysis
- Support for custom solver configurations

### Changed
- Improved performance of FBA calculations
- Updated LLM integration for better error handling

### Fixed
- Resolved memory leak in long-running workflows
- Fixed issue with model loading from remote URLs
```

## Pull Request Process

### Before Submitting

1. **Run the complete test suite**:
```bash
pytest
black --check src/
flake8 src/
mypy src/
```

2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Update changelog** if applicable

### Pull Request Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
```

### Review Process

1. **Automated checks** must pass
2. **At least one reviewer** approval required
3. **No merge conflicts** with target branch
4. **All conversations resolved**

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and questions
- Focus on constructive feedback
- Assume good intentions

### Getting Help

- **Documentation**: Check existing docs first
- **GitHub Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive issues

### Issue Reporting

Use the issue templates:

- **Bug Report**: Include reproduction steps, environment details
- **Feature Request**: Describe the problem and proposed solution
- **Documentation**: Identify what's missing or unclear

## Advanced Topics

### Performance Optimization

When optimizing code:

1. **Profile first**: Use `cProfile` or `line_profiler`
2. **Measure impact**: Benchmark before and after
3. **Consider memory**: Use `memory_profiler` for memory-intensive operations
4. **Cache appropriately**: Implement caching for expensive operations

### Security Considerations

- **Never commit secrets**: Use environment variables
- **Validate inputs**: Sanitize all user inputs
- **Handle errors gracefully**: Don't expose internal details
- **Follow principle of least privilege**: Minimal required permissions

### Release Process

For maintainers:

1. **Update version numbers** in `pyproject.toml`
2. **Update changelog** with release notes
3. **Tag release** with semantic versioning
4. **Build and publish** to PyPI
5. **Update documentation** site

## Resources

- **Project Documentation**: [User Guide](../user/README.md)
- **API Reference**: [API Documentation](../api/overview.md)
- **Examples**: See examples/ directory in the repository
- **GitHub Repository**: https://github.com/ModelSEED/ModelSEEDagent
- **Issue Tracker**: https://github.com/ModelSEED/ModelSEEDagent/issues

Thank you for contributing to ModelSEEDagent! Your contributions help make metabolic modeling more accessible and powerful for researchers worldwide.