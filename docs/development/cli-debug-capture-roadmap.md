# CLI Debug Capture Implementation Roadmap

## Overview

This roadmap outlines the implementation of a comprehensive CLI debug information capture system for ModelSEEDagent. The goal is to capture valuable console output that includes AI reasoning flows and formatted results without breaking existing functionality.

## Problem Statement

Currently, valuable debug information is displayed in the CLI console but not captured in persistent logs:

1. **AI Reasoning Flow**: Step-by-step decision-making process showing tool selection rationale
2. **Formatted Final Results**: Comprehensive analysis output with structured presentation
3. **Intermediate Decision Points**: Real-time AI analysis of tool results and next steps

### Current Architecture

**Two-Agent System:**
- **RealTimeMetabolicAgent** (`/logs/realtime_run_*`): High-level workflow coordination
- **LangGraphMetabolicAgent** (`/logs/langgraph_run_*`): Detailed execution with planning steps

**Missing Information:**
- Console debug output showing AI reasoning
- Formatted analysis results displayed to users
- Cross-agent execution correlation

## Implementation Phases

### Phase 1: Foundation (Safe) - IN PROGRESS

**Objective**: Create basic capture infrastructure without affecting existing functionality.

**Deliverables**:
- `ConsoleOutputCapture` class for structured logging
- Configuration flags for enabling/disabling capture
- Basic file structure for captured data
- Integration hooks in existing logging system

**Implementation Details**:

```python
# New component: src/utils/console_output_capture.py
class ConsoleOutputCapture:
    def __init__(self, run_dir: Path, enabled: bool = False):
        self.run_dir = run_dir
        self.enabled = enabled
        self.console_log_file = run_dir / "console_debug_output.jsonl"
        self.reasoning_flow_file = run_dir / "ai_reasoning_flow.json"
        self.formatted_results_file = run_dir / "formatted_results.json"

    def capture_reasoning_step(self, step_type: str, content: str, metadata: dict):
        """Capture AI reasoning steps as they occur"""
        if not self.enabled:
            return

        reasoning_entry = {
            "timestamp": datetime.now().isoformat(),
            "step_type": step_type,  # "tool_selection", "decision_analysis", "conclusion"
            "content": content,
            "metadata": metadata
        }

        # Append to JSONL file for streaming
        with open(self.console_log_file, "a") as f:
            f.write(json.dumps(reasoning_entry) + "\n")

    def capture_formatted_output(self, output_type: str, content: str):
        """Capture final formatted results"""
        if not self.enabled:
            return

        # Store formatted output for easy retrieval

    def get_complete_reasoning_flow(self) -> List[dict]:
        """Retrieve complete reasoning flow for analysis"""
```

**Configuration Addition**:
```python
# Add to agent configuration
config = {
    "capture_console_debug": False,  # Default disabled for safety
    "capture_ai_reasoning_flow": False,
    "capture_formatted_results": False,
    "console_output_max_size_mb": 50,
    "debug_capture_level": "basic"  # "basic", "detailed", "comprehensive"
}
```

**Integration Points**:
- Enhance existing `_log_llm_input()` method (line 1680 in `real_time_metabolic.py`)
- Add capture hooks in AI decision methods
- Integrate with existing audit trail system

**Risk Level**: **LOW** - Purely additive functionality
**Testing Requirements**:
- [ ] Existing workflows run unchanged
- [ ] New capture works when enabled
- [ ] Performance impact < 2%
- [ ] Memory usage reasonable

**Success Criteria**:
- [ ] ConsoleOutputCapture class created and tested
- [ ] Configuration options functional
- [ ] Basic reasoning capture working
- [ ] No impact on existing functionality

---

### Phase 2: Integration (Moderate) - PLANNED

**Objective**: Integrate capture system with existing logging infrastructure.

**Deliverables**:
- Enhanced session file structure
- Cross-agent log correlation
- Real-time reasoning flow capture
- Console output preservation

**Implementation Details**:

```python
# Enhanced LLM input logging
def _log_llm_input_and_console(self, phase: str, step: int, prompt: str,
                               knowledge_base: Dict[str, Any],
                               console_output: Optional[str] = None,
                               reasoning_context: Optional[dict] = None):
    """Enhanced version of existing _log_llm_input with console capture"""

    # Existing LLM input logging
    self._log_llm_input(phase, step, prompt, knowledge_base)

    # New console output capture
    if self.console_capture and self.console_capture.enabled:
        self.console_capture.capture_reasoning_step(
            step_type=phase,
            content=console_output or "",
            metadata={
                "step": step,
                "prompt_length": len(prompt),
                "knowledge_base_size": len(knowledge_base),
                "reasoning_context": reasoning_context
            }
        )
```

**Session Enhancement**:
```json
{
  "detailed_execution_trace": {
    "ai_reasoning_steps_file": "logs/realtime_run_*/ai_reasoning_flow.json",
    "console_output_file": "logs/realtime_run_*/console_debug_output.jsonl",
    "formatted_results_file": "logs/realtime_run_*/formatted_results.json",
    "cross_agent_correlation": {
      "realtime_run_id": "20250613_010716",
      "langgraph_run_id": "20250613_010725"
    }
  }
}
```

**Specific Capture Points**:
1. `_ai_analyze_query_for_first_tool()` (line 291) - Initial tool selection
2. `_ai_analyze_results_and_decide_next_step()` (line 481) - Iterative decisions
3. `_ai_generate_final_conclusions()` (line 783) - Final analysis
4. `_execute_tool_with_audit()` (line 686) - Tool execution context

**Risk Level**: **MEDIUM** - Touches core execution flow
**Testing Requirements**:
- [ ] Session loading/saving unchanged
- [ ] Agent delegation works normally
- [ ] Log folder structure preserved
- [ ] New debug info captured correctly

---

### Phase 3: Enhancement (Higher) - PLANNED

**Objective**: Add advanced features for comprehensive debug capture.

**Deliverables**:
- Real-time console streaming capture
- Advanced cross-agent correlation
- Formatted output preservation with styling
- Debug replay functionality

**Implementation Details**:

```python
# Advanced capture features
class AdvancedConsoleCapture(ConsoleOutputCapture):
    def __init__(self, run_dir: Path, enabled: bool = False):
        super().__init__(run_dir, enabled)
        self.stream_capture = enabled
        self.correlation_engine = CrossAgentCorrelationEngine()

    def start_streaming_capture(self):
        """Capture console output in real-time"""

    def correlate_with_langgraph(self, langgraph_run_id: str):
        """Create correlation between realtime and langgraph logs"""

    def generate_debug_replay(self) -> str:
        """Generate a replay of the complete debugging session"""
```

**Advanced Features**:
- Console output streaming with timestamps
- Cross-agent execution timeline
- Interactive debug replay viewer
- Formatted output with preserved styling
- Debug session comparison tools

**Risk Level**: **HIGH** - Complex integrations
**Testing Requirements**:
- [ ] Comprehensive regression testing
- [ ] Performance impact analysis
- [ ] Memory and storage optimization
- [ ] Cross-platform compatibility

---

## Risk Mitigation Strategy

### Checkpoint Creation
```bash
# Before any implementation
git checkout -b feature/cli-debug-capture
git tag checkpoint-before-logging-changes
```

### Non-Breaking Implementation Guidelines
1. **Optional Feature**: All capture functionality disabled by default
2. **Backward Compatibility**: Existing API unchanged
3. **Graceful Degradation**: System works without capture features
4. **Configuration Control**: Fine-grained enable/disable options

### Rollback Plan
- Revert to checkpoint tag: `git reset --hard checkpoint-before-logging-changes`
- Disable features via configuration
- Remove new log files if needed
- Restore original session format

## Validation Framework

### Functionality Tests
- [ ] All existing CLI workflows work unchanged
- [ ] Agent delegation functions normally
- [ ] Session persistence works correctly
- [ ] Tool execution completes successfully

### Performance Tests
- [ ] Execution time impact < 5%
- [ ] Memory usage increase < 10%
- [ ] Log file size manageable
- [ ] No resource leaks

### Integration Tests
- [ ] RealTime + LangGraph coordination
- [ ] Session file compatibility
- [ ] Cross-agent log correlation
- [ ] Configuration system integration

## Configuration Reference

```python
# Complete configuration options
CLI_DEBUG_CAPTURE_CONFIG = {
    # Phase 1 - Foundation
    "capture_console_debug": False,
    "capture_ai_reasoning_flow": False,
    "capture_formatted_results": False,
    "console_output_max_size_mb": 50,

    # Phase 2 - Integration
    "enable_cross_agent_correlation": False,
    "session_trace_enhancement": False,
    "real_time_reasoning_capture": False,

    # Phase 3 - Enhancement
    "streaming_console_capture": False,
    "debug_replay_generation": False,
    "advanced_formatting_preservation": False,

    # Global settings
    "debug_capture_level": "basic",  # "basic", "detailed", "comprehensive"
    "auto_cleanup_debug_files": True,
    "max_debug_sessions_retained": 10
}
```

## Progress Tracking

### Phase 1 Status: COMPLETED
- [x] Foundation planning complete
- [x] ConsoleOutputCapture class implemented
- [x] Configuration system added
- [x] Basic integration hooks created
- [x] Testing and validation complete

**Implementation Details:**
- Created `src/utils/console_output_capture.py` with full functionality
- Integrated into `RealTimeMetabolicAgent` class
- Added configuration options for all capture types
- Implemented capture at key decision points:
  - LLM input logging (enhanced existing `_log_llm_input` method)
  - Tool selection reasoning (`_ai_analyze_query_for_first_tool`)
  - Decision analysis reasoning (`_ai_analyze_results_and_decide_next_step`)
  - Final conclusions reasoning (`_ai_generate_final_conclusions`)
  - Formatted result capture (final AgentResult)
- All tests passing with no impact on existing functionality
- Safe rollback checkpoint created: `checkpoint-before-logging-changes`

**Usage Example:**
```python
# Enable console capture in agent configuration
config = {
    "capture_console_debug": True,
    "capture_ai_reasoning_flow": True,
    "capture_formatted_results": True,
    "debug_capture_level": "detailed"
}

# Console capture will automatically create files in the run directory:
# - console_debug_output.jsonl (streaming reasoning steps)
# - ai_reasoning_flow.json (complete reasoning flow)
# - formatted_results.json (final formatted outputs)
```

### Phase 2 Status: PLANNED
- [ ] Enhanced session structure designed
- [ ] Cross-agent correlation implemented
- [ ] Real-time capture integration
- [ ] Advanced testing complete

### Phase 3 Status: PLANNED
- [ ] Streaming capture implemented
- [ ] Debug replay functionality
- [ ] Advanced correlation features
- [ ] Performance optimization complete

## Future Considerations

### Potential Extensions
- Integration with external debugging tools
- Export to standard debugging formats
- Machine learning analysis of debug patterns
- Automated debugging recommendations

### Monitoring and Observability
- Debug capture performance metrics
- Storage usage monitoring
- Capture success/failure rates
- User adoption analytics

---

## Conclusion

This roadmap provides a structured approach to implementing comprehensive CLI debug capture while maintaining system stability. The phased approach allows for incremental delivery and risk mitigation, ensuring that valuable debug information can be captured without compromising existing functionality.

**Next Steps**: Begin Phase 1 implementation with foundation components and basic capture infrastructure.
