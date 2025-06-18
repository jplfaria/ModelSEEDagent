# Changelog

## [0.2.0] - 2025-06-18

## ✨ New Features


## 🐛 Bug Fixes


## 📚 Documentation


## 🔧 Other Changes




All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Smart Summarization Framework**: Three-tier information hierarchy for optimal LLM performance
  - Achieves 95-99.9% size reduction on tool outputs (up to 99.998% for FluxSampling)
  - Automatic summarization for FBA, FluxVariability, FluxSampling, and GeneDeletion tools
  - Preserves full data access via artifact storage system
- **Query-Aware Stopping Criteria**: Dynamic analysis depth based on query intent
  - Prevents premature tool chain termination
  - Adapts to query indicators like "detailed", "comprehensive", "quick"
- **Learning Memory Enhancement**: Tracks Smart Summarization effectiveness metrics
- Intelligent release automation system with semantic versioning
- Automated PR creation for dev-to-main releases
- Comprehensive release validation workflow
- Release notes templates and configurations

### Changed
- Real-time agent now uses proper tool execution path for Smart Summarization
- Tool validation suite updated to use correct execution interface
- Improved documentation deployment workflow using gh-pages

### Fixed
- Tool execution bypassing Smart Summarization by using internal methods
- GitHub Actions deployment protection rule issues
- Documentation not updating from dev branch pushes

## [0.1.0] - 2025-06-13

### Added
- First official release of ModelSEEDagent
- AI-powered metabolic modeling with LangGraph workflows
- 30 specialized tools for metabolic analysis
- Comprehensive documentation system
- GitHub Pages documentation deployment

### Features
- **AI Media Tools (6 tools)**: Intelligent media management and optimization
- **COBRApy Tools (12 tools)**: Comprehensive metabolic modeling analysis
- **ModelSEED Tools (5 tools)**: Genome annotation and model building (in development)
- **Biochemistry Tools (2 tools)**: Universal compound and reaction resolution
- **RAST Tools (2 tools)**: Genome annotation and analysis (in development)

### Infrastructure
- Poetry-based dependency management
- Pre-commit hooks for code quality
- Automated documentation review system
- GitHub Actions workflows for CI/CD

---

**Legend:**
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` for vulnerability fixes
