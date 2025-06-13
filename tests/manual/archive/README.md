# Manual Tests Archive

This directory contains archived manual test scripts that are no longer actively used but preserved for historical reference.

## Archived Tests

### **Fix and Debug Scripts**
These scripts were created to address specific issues during development and are now resolved:

- `test_all_fixes.py` - General fixes validation script
- `test_duplicate_message_fix.py` - Fix for duplicate message handling
- `test_fixed_interactive.py` - Interactive interface fixes
- `test_interactive_cli_fix.py` - CLI interaction improvements
- `test_streaming_interface_fix.py` - Streaming interface corrections

## Archive Policy

**Tests are archived when**:
- The specific issue they address has been resolved
- They haven't been modified in 30+ days
- They are superseded by functional tests or testbed validation
- They are one-off debug scripts no longer needed

## Accessing Archived Tests

If you need to reference or restore an archived test:

1. **Review the script**: Check if the functionality is still relevant
2. **Check current tests**: See if equivalent validation exists in functional tests
3. **Consider testbed**: Use the comprehensive testbed system for biological validation
4. **Restore if needed**: Move back to `tests/manual/` if still useful

## Current Active Tests

For currently active manual tests, see the main `tests/manual/` directory and the [Testing Strategy Guide](../../README.md).
