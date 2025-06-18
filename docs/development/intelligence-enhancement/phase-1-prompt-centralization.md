# Phase 1: Centralized Prompt Management + Reasoning Traces

**Duration**: 3 days (June 19-21, 2025)
**Objective**: Consolidate scattered prompts and implement transparent reasoning traces
**Priority**: Critical - Foundation for all subsequent intelligence enhancements

## Overview

Phase 1 establishes the fundamental infrastructure for intelligent reasoning by:
1. Centralizing 27+ scattered prompts into a unified registry
2. Implementing transparent reasoning trace logging
3. Creating prompt version control and impact tracking
4. Establishing the foundation for dynamic prompt optimization

## Current State Analysis

### Scattered Prompt Distribution
- **Total Prompts**: 27+ distinct prompt templates
- **File Distribution**: 8 major files across the codebase
- **Management Issues**: No version control, impact tracking, or consistency standards

### Key Prompt Locations to Migrate
1. `config/prompts/metabolic.yaml` (2 prompts)
2. `src/agents/real_time_metabolic.py` (3 prompts)
3. `src/agents/collaborative_reasoning.py` (3 prompts)
4. `src/agents/reasoning_chains.py` (6 prompts)
5. `src/agents/hypothesis_system.py` (5 prompts)
6. `src/agents/pattern_memory.py` (1 prompt)
7. `src/agents/langgraph_metabolic.py` (2 prompts)
8. `src/agents/performance_optimizer.py` (1 prompt)

## Implementation Plan

### Day 1: Centralized Prompt Registry

#### 1.1 Create Prompt Registry Infrastructure

**File**: `src/prompts/prompt_registry.py`

```python
class PromptRegistry:
    """Centralized prompt management system"""

    def __init__(self):
        self.prompts = {}
        self.versions = {}
        self.usage_tracking = {}
        self.a_b_tests = {}

    def register_prompt(self, prompt_id: str, template: str,
                       category: str, version: str = "1.0"):
        """Register a new prompt with versioning"""

    def get_prompt(self, prompt_id: str, variables: Dict[str, Any]):
        """Retrieve and format prompt with tracking"""

    def track_usage(self, prompt_id: str, context: Dict[str, Any]):
        """Track prompt usage for optimization"""

    def compare_versions(self, prompt_id: str, version_a: str, version_b: str):
        """A/B testing infrastructure"""
```

#### 1.2 Prompt Categories and Standards

**Categories**:
- `tool_selection`: AI tool choice reasoning
- `result_analysis`: Result interpretation and insight extraction
- `workflow_planning`: Multi-step analysis planning
- `hypothesis_generation`: Scientific hypothesis formation
- `synthesis`: Cross-tool result integration
- `quality_assessment`: Self-evaluation and validation

**Standard Template Format**:
```yaml
prompt_id: "tool_selection_initial"
category: "tool_selection"
version: "1.0"
description: "Initial tool selection based on query analysis"
template: |
  You are an expert metabolic modeling AI agent. Analyze this query and select the BEST first tool.

  Query: "{query}"
  Available tools: {available_tools}

  Consider:
  1. What type of analysis is being requested?
  2. Which tool provides the most informative starting point?

  Respond with:
  SELECTED_TOOL: tool_name
  REASONING: detailed explanation
variables:
  - query
  - available_tools
validation_rules:
  - must_select_valid_tool
  - reasoning_min_length: 50
```

### Day 2: Reasoning Trace Implementation

#### 2.1 Reasoning Trace Logger

**File**: `src/reasoning/trace_logger.py`

```python
class ReasoningTraceLogger:
    """Captures and logs step-by-step AI reasoning"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.trace_steps = []
        self.decision_points = []
        self.hypothesis_trail = []

    def log_decision(self, decision_type: str, reasoning: str,
                    confidence: float, alternatives: List[str]):
        """Log a reasoning decision with full context"""

    def log_tool_selection(self, query: str, selected_tool: str,
                          reasoning: str, available_tools: List[str]):
        """Log tool selection reasoning"""

    def log_insight_extraction(self, tool_result: Dict,
                              extracted_insights: List[str],
                              reasoning: str):
        """Log insight extraction from tool results"""

    def log_hypothesis_formation(self, observation: str,
                               hypothesis: str, rationale: str,
                               testable_predictions: List[str]):
        """Log scientific hypothesis generation"""
```

#### 2.2 Transparent Decision Framework

**Core Principles**:
1. **Every Decision Logged**: No black box reasoning
2. **Rationale Required**: All choices must include explanation
3. **Alternative Consideration**: Document why other options rejected
4. **Confidence Tracking**: Quantify certainty in decisions
5. **Hypothesis Traceability**: Link conclusions to evidence

**Implementation Pattern**:
```python
# Before (opaque decision)
selected_tool = ai_agent.select_tool(query, available_tools)

# After (transparent decision with reasoning trace)
decision_context = {
    "query": query,
    "available_tools": available_tools,
    "previous_results": context.get_results(),
    "analysis_goal": context.get_goal()
}

decision_result = reasoning_tracer.make_decision(
    decision_type="tool_selection",
    context=decision_context,
    prompt_id="tool_selection_with_context"
)

selected_tool = decision_result.choice
reasoning_trace.log_decision(
    decision_type="tool_selection",
    reasoning=decision_result.reasoning,
    confidence=decision_result.confidence,
    alternatives=decision_result.alternatives_considered
)
```

### Day 3: Prompt Migration and Integration

#### 3.1 Systematic Prompt Migration

**Migration Process**:
1. **Extract Current Prompts**: Identify all existing prompt templates
2. **Categorize by Function**: Group prompts by reasoning purpose
3. **Standardize Format**: Convert to new registry format
4. **Add Reasoning Traces**: Integrate trace logging into prompt usage
5. **Validate Migration**: Ensure no functionality loss

**Example Migration**:

**Before** (in `src/agents/real_time_metabolic.py`):
```python
prompt = f"""You are an expert metabolic modeling AI agent. Analyze this query and select the BEST first tool to start with.

Query: "{query}"
Available tools: {available_tools}

Based on the query, what tool should you start with and why?"""
```

**After** (using centralized registry):
```python
from src.prompts.prompt_registry import get_prompt_registry
from src.reasoning.trace_logger import ReasoningTraceLogger

registry = get_prompt_registry()
tracer = ReasoningTraceLogger(session_id)

decision_result = registry.execute_prompt(
    prompt_id="tool_selection_initial",
    variables={
        "query": query,
        "available_tools": available_tools
    },
    trace_logger=tracer
)

selected_tool = decision_result.extracted_choice
reasoning = decision_result.reasoning_trace
```

#### 3.2 Agent Integration Updates

**Files to Update**:
- `src/agents/real_time_metabolic.py`
- `src/agents/collaborative_reasoning.py`
- `src/agents/reasoning_chains.py`
- `src/agents/hypothesis_system.py`
- `src/agents/langgraph_metabolic.py`

**Integration Pattern**:
```python
class EnhancedAgent(BaseAgent):
    def __init__(self, llm, tools, config):
        super().__init__(llm, tools, config)
        self.prompt_registry = get_prompt_registry()
        self.reasoning_tracer = ReasoningTraceLogger(self.session_id)

    def _make_reasoning_decision(self, decision_type: str, context: Dict):
        """Unified reasoning decision with transparent traces"""
        return self.prompt_registry.execute_prompt(
            prompt_id=f"{decision_type}_reasoning",
            variables=context,
            trace_logger=self.reasoning_tracer
        )
```

## Deliverables

### Core Infrastructure
1. **`src/prompts/prompt_registry.py`**: Centralized prompt management
2. **`src/reasoning/trace_logger.py`**: Reasoning trace capture
3. **`src/reasoning/trace_analyzer.py`**: Trace quality assessment
4. **`config/prompts/centralized/`**: Migrated prompt templates

### Enhanced Agents
1. **Updated Agent Classes**: All agents using centralized prompts
2. **Reasoning Integration**: Transparent decision-making in all workflows
3. **Trace Persistence**: Session-based reasoning history storage
4. **Quality Monitoring**: Automated reasoning quality assessment

### Documentation
1. **Prompt Development Guide**: Standards for creating new prompts
2. **Reasoning Trace Schema**: Format for decision logging
3. **Migration Documentation**: Complete migration process and validation
4. **Usage Examples**: Practical demonstrations of enhanced reasoning

## Success Metrics

### Quantitative Targets
- **Prompt Centralization**: 27+ prompts migrated (100%)
- **Reasoning Transparency**: 90%+ decisions logged with rationale
- **Decision Quality**: Reasoning explanations >50 characters average
- **Performance Impact**: <20% increase in response time

### Qualitative Improvements
- **Consistency**: Standardized reasoning quality across all agents
- **Debuggability**: Full visibility into AI decision-making process
- **Optimization**: Foundation for A/B testing and prompt improvement
- **Maintainability**: Single source of truth for all AI reasoning

## Testing Strategy

### Unit Testing
- Prompt registry functionality
- Reasoning trace capture accuracy
- Decision logging completeness
- Migration validation

### Integration Testing
- Agent workflow continuity
- Cross-agent reasoning consistency
- Trace persistence reliability
- Performance impact assessment

### Quality Validation
- Reasoning trace quality assessment
- Decision rationale completeness
- Hypothesis formation transparency
- Scientific rigor maintenance

## Risk Mitigation

### Identified Risks
1. **Performance Degradation**: Additional logging overhead
2. **Complexity Increase**: More sophisticated reasoning infrastructure
3. **Migration Errors**: Prompt functionality changes during migration
4. **Integration Issues**: Agent compatibility with new framework

### Mitigation Strategies
1. **Asynchronous Logging**: Minimize performance impact through async trace capture
2. **Gradual Migration**: Phase migration to validate each component
3. **Regression Testing**: Comprehensive validation of migrated functionality
4. **Rollback Capability**: Maintain ability to revert to pre-migration state

## Phase 1 Completion Criteria

### Technical Requirements
- [ ] All 27+ prompts migrated to centralized registry
- [ ] Reasoning trace logging operational across all agents
- [ ] Decision transparency >90% for all AI choices
- [ ] Performance impact <20% increase in response time

### Quality Gates
- [ ] No regression in existing functionality
- [ ] Improved reasoning consistency across agents
- [ ] Enhanced debuggability of AI decisions
- [ ] Foundation ready for Phase 2 context enhancement

### Documentation Complete
- [ ] Centralized prompt development standards
- [ ] Reasoning trace schema documented
- [ ] Migration process fully documented
- [ ] Usage examples and best practices

**Phase 1 Success**: Complete centralization with transparent reasoning traces ready for dynamic context enhancement in Phase 2.
