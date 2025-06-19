# Interactive CLI Bug Report: Missing Method

## Issue Description

The Interactive CLI has a bug where it calls a non-existent method `_process_with_simple_ai()` in the `DynamicAIConversationEngine` class.

## Location

File: `/src/interactive/conversation_engine.py`
Line: 298

## Current Code (Buggy)
```python
def process_user_input(self, user_input: str) -> ConversationResponse:
    """Process user input using real AI agent"""
    start_time = time.time()

    # Update conversation context
    self.context.interaction_count += 1

    if not self.ai_agent:
        return self._handle_no_ai_fallback(user_input)

    # Use real AI agent for processing - disable streaming for now to fix display issues
    return self._process_with_simple_ai(user_input, start_time)  # ← BUG: Method doesn't exist
```

## Problem

The method `_process_with_simple_ai` is called but never defined in the class. Looking at the available methods, there are:
- `_process_with_real_ai` (defined at line 300)
- `_process_with_streaming_ai` (defined at line 357)

## Proposed Fix

Replace line 298 with:
```python
return self._process_with_real_ai(user_input, start_time)
```

## Impact on Intelligence Framework Testing

This bug prevents proper testing of the Intelligence Framework through the Interactive CLI because:

1. The Interactive CLI would crash when trying to process user input
2. The Intelligence Framework integration in the agent would never be reached
3. Interface consistency tests would fail for the Interactive CLI path

## Verification

To verify this bug:
1. Run the Interactive CLI
2. Enter any query
3. The application will crash with: `AttributeError: 'DynamicAIConversationEngine' object has no attribute '_process_with_simple_ai'`

## Testing Recommendation

After fixing this bug, the interface consistency tests should be run to ensure:
1. Both CLI interfaces call the Intelligence Framework correctly
2. Results are consistent between regular CLI and Interactive CLI
3. The Intelligence Framework's `execute_comprehensive_workflow` is invoked in both paths
