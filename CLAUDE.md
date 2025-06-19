# Claude Code Guidelines for ModelSEEDagent

This file contains important guidelines for AI assistance on this project.

## Documentation Style Guidelines

### CRITICAL: No Emojis in Documentation Files

**DO NOT use emojis in any `.md` files in the `docs/` directory.**

This project maintains professional documentation standards:
- Documentation files must be emoji-free
- Professional, technical writing style
- Clear, concise language without decorative elements

### Where Emojis ARE Allowed:
- CLI tool output (Python scripts showing user interface elements)
- Code comments (sparingly)
- Commit messages (when appropriate)

### Where Emojis are FORBIDDEN:
- Any `.md` file in `docs/` directory
- README.md files
- Technical documentation
- API documentation
- User guides

### Pre-commit Hook Enforcement

The project has a pre-commit hook that will FAIL commits containing emojis in documentation:
```yaml
- id: no-emojis-in-docs
  name: No Emojis in Documentation
  entry: python -c "import re,sys,glob; files=[f for f in glob.glob('docs/**/*.md',recursive=True) if '/archive/' not in f]; content=''.join(open(f).read() for f in files); emojis=re.findall(r'[🔧🎯✅🟡🟢🔴⚠️📝📊🚀🛡️📋💡🔍⏳]',content); sys.exit(1 if emojis else 0)"
```

## Writing Guidelines

1. **Professional Tone**: Technical documentation should be clear, concise, and professional
2. **Code Examples**: When showing CLI output, format as code blocks - emojis in examples are acceptable as they show actual tool output
3. **Consistency**: Follow existing documentation patterns and styles
4. **Clarity**: Focus on clear explanations over decorative elements

## Common Mistakes to Avoid

❌ **DON'T**: Put emojis directly in documentation
```markdown
## 🔧 Tool Configuration
✅ This system works great!
```

✅ **DO**: Use professional language
```markdown
## Tool Configuration
This system provides reliable operation.
```

✅ **DO**: Show emojis in code examples when demonstrating CLI output
```markdown
Example output:
```
🔍 Starting documentation review...
✅ Updated 2 files
```

## Remember

- The goal is professional, maintainable documentation
- Emojis in CLI tools enhance user experience
- Emojis in documentation detract from professionalism
- Pre-commit hooks enforce these standards for good reason

**When in doubt, leave emojis out of documentation files.**

## Validation and Testing Guidelines

### CRITICAL: Regular Checkpoint Validation

**ALWAYS run validation before major changes, commits, or feature implementations.**

### Development Validation Workflow

1. **During Development** - Quick validation after code changes:
   ```bash
   python scripts/dev_validate.py --quick
   ```
   - Validates core Intelligence Framework functionality
   - Checks for regressions in key components
   - Target: All tests passing, quality score ≥85%

2. **Before Commits** - Full validation:
   ```bash
   python scripts/dev_validate.py --full
   ```
   - Comprehensive validation across all components
   - Performance benchmarking
   - Integration testing

3. **Component-Specific Testing** - When working on specific areas:
   ```bash
   python scripts/dev_validate.py --component [component_name]
   ```

### Intelligence Framework Validation

**Required validation checkpoints:**
- Before implementing new Intelligence features
- After modifying prompt registry
- Before merging interface changes
- After updating reasoning components

**Performance Benchmarks:**
- Quality scores must maintain ≥85% average
- Execution time should stay within reasonable limits
- All integration tests must pass
- No critical component failures

### Validation Results Interpretation

**Success Criteria:**
- Tests Passed: 100% for critical components
- Quality Score: ≥85% (Good), ≥90% (Excellent)
- No error states in core Intelligence Framework

**When Validation Fails:**
1. Review specific test failures
2. Check component integration
3. Validate prompt registry integrity
4. Ensure method signatures are compatible

### Regular Validation Schedule

- **Every major feature addition**: Full validation required
- **Interface modifications**: Intelligence Framework validation
- **Before Phase transitions**: Comprehensive validation
- **Weekly development cycles**: Quick validation minimum

**Remember: Validation prevents regressions and ensures Intelligence Framework maintains high performance standards.**

## Interface Consistency Guidelines

### CRITICAL: Multi-Interface Feature Implementation

**ALWAYS ensure new features are implemented consistently across all user-facing interfaces.**

### Interface Consistency Requirements

When implementing new features, particularly Intelligence Framework enhancements, ensure consistent availability across:

1. **Direct Agent Access** - Core agent classes with full feature access
2. **Interactive CLI** - Conversational interface with natural language input
3. **Regular CLI** - Command-line interface with structured commands

### Feature Implementation Workflow

1. **Feature Design Phase** - Consider interface implications:
   ```
   - Will this feature benefit all interfaces?
   - Are there interface-specific adaptations needed?
   - How should the feature be exposed in each interface?
   ```

2. **Implementation Phase** - Multi-interface integration:
   ```
   - Implement in core agent classes first
   - Integrate into Interactive CLI conversation flow
   - Ensure Regular CLI uses enhanced agents
   - Validate factory creates Intelligence-enabled agents
   ```

3. **Validation Phase** - Cross-interface testing:
   ```bash
   # Test all three interface paths
   python tests/test_intelligence_interface_consistency.py
   ```

### Interface Capability Matrix

| Feature Category | Direct Agent | Interactive CLI | Regular CLI |
|-----------------|--------------|----------------|-------------|
| Intelligence Framework | ✅ Full | ✅ Full | ✅ Full |
| Enhanced Prompts | ✅ Yes | ✅ Yes | ✅ Yes |
| Context Enhancement | ✅ Yes | ✅ Yes | ✅ Yes |
| Quality Assessment | ✅ Yes | ✅ Yes | ✅ Yes |
| Artifact Intelligence | ✅ Yes | ✅ Yes | ✅ Yes |
| Self Reflection | ✅ Yes | ✅ Yes | ✅ Yes |

### Exception Guidelines

**When features might appropriately differ between interfaces:**

1. **UI-Specific Features** - Interactive elements only relevant to CLI interfaces
2. **Format-Specific Outputs** - Different presentation for different interaction modes
3. **Context-Dependent Behavior** - Features that adapt based on interface capabilities

**NOT Acceptable Exceptions:**
- Core Intelligence Framework availability
- Essential analysis capabilities
- Quality assessment features
- Fundamental tool access

### Implementation Checklist

Before marking a feature "complete":

- [ ] Feature implemented in core agent classes
- [ ] Interactive CLI integration verified
- [ ] Regular CLI uses Intelligence-enhanced agents
- [ ] Agent factory creates proper agent types
- [ ] Cross-interface consistency tests pass
- [ ] Documentation updated for all interfaces

### Validation Commands

```bash
# Quick interface consistency check
python tests/test_intelligence_interface_consistency.py

# Full interface validation
python scripts/dev_validate.py --interfaces

# Specific agent type validation
python scripts/dev_validate.py --agent-type langgraph
```

**Remember: Consistent user experience across interfaces ensures feature accessibility and reduces user confusion.**
