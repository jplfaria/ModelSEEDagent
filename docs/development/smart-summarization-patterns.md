# Smart Summarization Framework - Patterns & Guidelines

This document provides comprehensive patterns and guidelines for implementing summarizers in the Smart Summarization Framework. Use these patterns to create effective summarizers for new tools.

## Framework Overview

The Smart Summarization Framework implements a three-tier information hierarchy:

1. **key_findings** (≤2KB): Critical insights optimized for LLM consumption
2. **summary_dict** (≤5KB): Structured data for follow-up analysis
3. **full_data_path**: Complete raw data stored on disk (unlimited size)

**Target Results**: 95-99.9% size reduction while preserving critical scientific insights.

## Proven Patterns

### Pattern 1: Statistical Tool Summarization (FluxSampling)
**Use Case**: Tools generating massive datasets with statistical distributions
**Example**: FluxSampling (39MB → 2KB, 99.99% reduction)

```python
def _generate_key_findings(self, n_samples, n_reactions, analysis, model_id):
    """Focus on distribution insights and variability patterns"""
    key_findings = [
        f"Tool analysis of {model_id}: {n_samples} samples across {n_reactions} items",
        f"Distribution characteristics and key patterns identified"
    ]

    # Extract critical patterns from analysis
    if 'patterns' in analysis:
        patterns = analysis['patterns']
        # Summarize top patterns with percentages
        # Include variability insights
        # Highlight optimization opportunities

    return key_findings

def _generate_summary_dict(self, analysis_data, model_stats):
    """Structured statistical summary with key metrics"""
    return {
        "statistics": {
            "total_items": total_count,
            "data_reduction_achieved": "99.9%",
            "coverage_metrics": coverage_data
        },
        "pattern_summary": {
            # Top patterns with counts and percentages
            # Distribution characteristics
        },
        "optimization_insights": {
            # Key targets for optimization
            # Flexibility metrics
        },
        "model_context": model_stats,
        "analysis_metadata": {
            "method": "statistical_analysis",
            "framework_version": "1.0"
        }
    }
```

### Pattern 2: Categorical Analysis (GeneDeletion)
**Use Case**: Tools categorizing items by functional impact
**Example**: GeneDeletion (130KB → 3KB, 97.6% reduction)

```python
def _generate_key_findings(self, total_items, analysis, model_id):
    """Focus on critical categories and essential items"""
    key_findings = [
        f"Analysis of {model_id}: {total_items} items tested"
    ]

    # Extract category statistics
    summary = analysis.get('summary', {})
    if summary:
        # Calculate percentages for each category
        # Highlight critical categories (essential, impaired, etc.)
        # Include rate assessments and warnings
        # Show examples of critical items

    return key_findings

def _generate_summary_dict(self, analysis, model_stats):
    """Category-focused summary with examples"""
    return {
        "analysis_statistics": {
            "total_items_tested": total_count,
            "model_coverage": coverage_ratio
        },
        "item_categories": {
            # Each category with count, percentage, examples
            "critical_category": {
                "count": count,
                "percentage": percentage,
                "examples": top_examples[:10]  # Limit examples
            }
        },
        "criticality_analysis": {
            # Assess overall criticality
            # Rate comparisons
            # Robustness metrics
        },
        "model_context": model_stats
    }
```

### Pattern 3: Variability Analysis (FluxVariability + Smart Bucketing)
**Use Case**: Tools analyzing ranges and variability
**Example**: FluxVariability (170KB → 2KB, 98.6% reduction)

```python
def _generate_key_findings(self, total_items, categories, model_id):
    """Focus on variability insights and network properties"""
    key_findings = [
        f"Variability analysis of {model_id}: {total_items} items analyzed"
    ]

    # Categorize by variability levels
    # Calculate percentages and network properties
    # Include flexibility assessments
    # Add optimization insights from smart bucketing

    return key_findings

def _smart_bucketing(self, data, variable_items):
    """Intelligent categorization by variability levels"""
    # Calculate dynamic thresholds from data distribution
    # Categorize into high/medium/low/minimal variability
    # Provide optimization potential metrics
    # Include distribution insights

    return {
        "bucketing_thresholds": thresholds,
        "variability_categories": categories,
        "insights": optimization_insights
    }
```

## Implementation Guidelines

### 1. Summarizer Class Structure

```python
class NewToolSummarizer(BaseSummarizer):
    """Smart summarizer for NewTool

    Brief description of what the tool does and the summarization approach.
    Target reduction: X% (original_size → target_size)
    """

    def get_tool_name(self) -> str:
        return "tool_name_here"

    def summarize(self, raw_output, artifact_path, model_stats=None):
        """Main summarization method"""
        # Extract data from raw_output
        # Generate key_findings (≤2KB)
        # Generate summary_dict (≤5KB)
        # Validate size limits
        # Return ToolResult

    def _generate_key_findings(self, ...):
        """Generate LLM-optimized insights"""
        # Focus on actionable insights
        # Include percentages and critical numbers
        # Add warnings and optimization opportunities
        # Keep language concise and scientific

    def _generate_summary_dict(self, ...):
        """Generate structured analysis data"""
        # Organize data hierarchically
        # Include metadata and context
        # Limit examples and details
        # Preserve critical information for follow-up analysis
```

### 2. Key Findings Best Practices

**DO:**
- Start with tool and model identification
- Include critical percentages and counts
- Use scientific terminology
- Add warning indicators (WARNING:) for critical issues
- Include optimization opportunities
- Show top examples (3-5 items max)

**DON'T:**
- Include verbose descriptions
- Repeat information
- Use overly technical jargon
- Include raw data or long lists
- Exceed 2KB limit

**Example Structure:**
```python
key_findings = [
    f"Tool analysis of {model_id}: {total_count} items processed",
    f"Critical category: {count} ({percentage:.1f}%) - impact description",
    f"Secondary category: {count} ({percentage:.1f}%) - brief description",
    f"WARNING: Warning condition if applicable",
    f"Success: Positive insight if applicable",
    f"Top examples: {', '.join(examples[:3])}"
]
```

### 3. Summary Dict Architecture

```python
summary_dict = {
    "tool_statistics": {
        # Basic counts and coverage metrics
        "total_items": total_count,
        "model_coverage": coverage_ratio,
        "data_reduction_achieved": "XX.X%"
    },
    "primary_analysis": {
        # Main analysis results organized by category
        # Include counts, percentages, and limited examples
    },
    "secondary_insights": {
        # Additional insights and patterns
        # Optimization opportunities
    },
    "model_context": model_stats or {},
    "analysis_metadata": {
        "method": "tool_method_name",
        "framework_version": "1.0",
        "key_thresholds": threshold_info
    }
}
```

## Size Optimization Strategies

### 1. Data Reduction Techniques

**Remove Redundancy:**
- Eliminate duplicate data storage
- Replace verbose objects with IDs
- Use counts instead of full lists where possible

**Smart Sampling:**
- Limit examples to top N items (5-15 typically)
- Use representative samples for large datasets
- Focus on extreme cases (highest/lowest values)

**Efficient Encoding:**
- Round numerical values appropriately
- Use compact data structures
- Avoid unnecessary nesting

### 2. Information Preservation

**Critical Information (Always Preserve):**
- Essential items/categories for safety
- Negative evidence (blocked reactions, failed genes)
- Statistical summaries and distributions
- Optimization opportunities

**Optional Information (Can Summarize):**
- Detailed metadata
- Intermediate calculations
- Verbose descriptions
- Full correlation matrices

## Testing Guidelines

### 1. Size Validation

```python
def test_new_tool_summarizer():
    # Create realistic mock data at scale
    mock_output = create_large_mock_output(realistic_size=True)

    # Test summarization
    result = summarizer.summarize(mock_output, artifact_path, model_stats)

    # Validate size limits
    key_findings_size = len(json.dumps(result.key_findings))
    summary_dict_size = len(json.dumps(result.summary_dict))

    assert key_findings_size <= 2000
    assert summary_dict_size <= 5000

    # Calculate reduction
    original_size = len(json.dumps(mock_output))
    reduction_percentage = (1 - total_summary_size / original_size) * 100

    # Validate target achievement (tool-specific)
    assert reduction_percentage >= target_reduction
```

### 2. Information Validation

```python
def validate_information_preservation():
    # Ensure critical information is preserved
    assert "essential_category" in result.summary_dict
    assert len(result.key_findings) >= minimum_insights

    # Check for key scientific insights
    findings_text = " ".join(result.key_findings)
    assert "critical_keyword" in findings_text

    # Validate examples are included
    if original_has_examples:
        assert "examples" in result.summary_dict["primary_analysis"]
```

## Target Reduction Rates by Tool Type

| Tool Type | Original Size | Target Reduction | Rationale |
|-----------|---------------|------------------|-----------|
| Statistical Sampling | 25MB+ | 99.9% | Massive redundant data |
| Categorical Analysis | 100KB+ | 95-98% | Detailed categorization |
| Variability Analysis | 200KB+ | 95-99% | Range data and statistics |
| Network Analysis | 500KB+ | 90-95% | Complex graph structures |
| Optimization Results | 50KB+ | 90-95% | Solution details |

## Registration and Integration

### 1. Summarizer Registration

```python
# At end of summarizer file
new_tool_summarizer = NewToolSummarizer()
summarizer_registry.register(new_tool_summarizer)
```

### 2. Module Integration

```python
# In __init__.py
from .new_tool_summarizer import new_tool_summarizer

__all__ = [
    'existing_summarizers',
    'new_tool_summarizer',
]
```

### 3. Tool Integration

```python
# In tool configuration
class NewTool(BaseTool):
    def __init__(self, config):
        # Enable smart summarization
        config["smart_summarization_enabled"] = True
        super().__init__(config)
```

## Future Enhancement Opportunities

### 1. Advanced Bucketing
- Implement adaptive thresholds based on data distribution
- Add multi-dimensional categorization
- Include temporal analysis for time-series data

### 2. Cross-Tool Insights
- Compare results across related tools
- Identify patterns across different analysis types
- Provide meta-analysis insights

### 3. Dynamic Summarization
- Adjust summarization based on model size
- Context-aware insight generation
- User-customizable detail levels

## References and Examples

- **FluxSamplingSummarizer**: Statistical pattern for massive datasets
- **GeneDeletionSummarizer**: Categorical pattern for functional analysis
- **FluxVariabilitySummarizer**: Variability pattern with smart bucketing
- **Smart Summarization Framework**: Core framework documentation
- **Validation Suite**: Comprehensive testing examples

---

*This documentation evolves with new patterns and tools. Contribute improvements and new patterns as you implement additional summarizers.*
