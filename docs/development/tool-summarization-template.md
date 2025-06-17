# Tool Summarization Documentation Template

This template should be used for every tool that implements smart summarization. Copy this template and fill in the specific details for each tool.

---

## Tool: [ToolName]

**Implementation Status**: [ ] Not Started / [ ] In Progress / [ ] Complete  
**Priority**: High / Medium / Low  
**Estimated Size Reduction**: [X]% ([Before] → [After])

### **Raw Output Assessment**

**Model**: iML1515 (2,712 reactions) / EcoliMG1655 (1,867 reactions)
- **Current Output Size**: [X] KB/MB
- **Data Structure**: DataFrame / Dict / List / etc.
- **Key Components**: [describe what makes up the output]
- **Growth Pattern**: Linear/Quadratic with model size

### **Three-Tier Implementation**

#### **Tier 1: key_findings (≤2KB)**
*What the LLM sees by default - critical insights only*

**Example output**:
```
• [Key finding 1 with counts/percentages]
• [Key finding 2 highlighting negative evidence]  
• [Key finding 3 with specific examples]
• [Key finding 4 with critical warnings/alerts]
• [Key finding 5 with scientific interpretation]
```

**Content Strategy**:
- [ ] Preserve negative evidence (blocked reactions, missing compounds, failed tests)
- [ ] Include statistical summary (counts, percentages, ranges)
- [ ] Highlight unexpected/critical results  
- [ ] Provide specific examples of top/bottom items
- [ ] Scientific interpretation ready for LLM reasoning

#### **Tier 2: summary_dict (≤5KB)**  
*Structured data for follow-up analysis*

**Example structure**:
```python
{
    "statistics": {
        "total_count": 2712,
        "category_counts": {"active": 2180, "blocked": 98, "variable": 434},
        "ranges": {"min": -45.2, "max": 50.1, "mean": 0.12}
    },
    "top_items": [
        {"name": "SUCDi", "value": 45.2, "category": "variable"},
        {"name": "PFK", "value": 23.1, "category": "variable"}
    ],
    "critical_items": ["PFL", "ACALD", "EDD"],  # blocked reactions
    "metadata": {
        "model_id": "iML1515",
        "analysis_time": "2025-06-17T10:30:00Z",
        "parameters": {"epsilon": 1e-6, "solver": "glpk"}
    }
}
```

#### **Tier 3: full_data_path**
*Complete raw data on disk*

**File Format**: CSV / JSON / Pickle / NPZ  
**Path Pattern**: `/data/artifacts/{tool_name}_{model_id}_{timestamp}.{ext}`  
**Access Pattern**: On-demand via FetchArtifact tool

### **Summarization Logic**

#### **Smart Bucketing Strategy**
*How to categorize data to preserve meaning*

```python
def categorize_results(raw_data, epsilon=1e-6):
    # Define categories based on scientific meaning
    categories = {
        "category_1": raw_data[condition_1],  # e.g., variable reactions
        "category_2": raw_data[condition_2],  # e.g., fixed reactions  
        "category_3": raw_data[condition_3],  # e.g., blocked reactions
    }
    return categories
```

#### **Negative Evidence Preservation**
*Critical: What "didn't happen" that should be reported*

- [ ] Zero/blocked values with biological significance
- [ ] Missing expected results  
- [ ] Failed validations or constraints
- [ ] Inactive pathways in expected conditions

#### **Statistical Summarization**
*How to compress large datasets meaningfully*

- [ ] Count-based summaries (N out of total)
- [ ] Distribution statistics (mean, std, percentiles)
- [ ] Top/bottom N items with biological relevance
- [ ] Correlation patterns or clustering results

### **Implementation Code**

```python
def summarize_[tool_name](raw_output: [DataType], artifact_path: str) -> ToolResult:
    """
    Summarize [ToolName] output for LLM consumption
    
    Args:
        raw_output: [Description of input data structure]
        artifact_path: Path where full data is stored
        
    Returns:
        ToolResult with three-tier information hierarchy
    """
    
    # 1. Analyze and categorize raw data
    categories = categorize_results(raw_output)
    
    # 2. Generate key findings (≤2KB)
    key_findings = [
        f"• Category 1: {len(categories['category_1'])}/{len(raw_output)} items",
        f"• Category 2: {len(categories['category_2'])}/{len(raw_output)} items",
        f"• Critical items: {_get_critical_examples(categories)}",
        f"• Biological insight: {_interpret_results(categories)}"
    ]
    
    # 3. Create structured summary (≤5KB)
    summary_dict = {
        "statistics": _compute_statistics(raw_output),
        "categories": {k: len(v) for k, v in categories.items()},
        "top_items": _get_top_items(raw_output),
        "critical_alerts": _identify_critical_results(raw_output)
    }
    
    # 4. Validate size limits
    _validate_size_limits(key_findings, summary_dict)
    
    return ToolResult(
        full_data_path=artifact_path,
        summary_dict=summary_dict,
        key_findings=key_findings,
        tool_name="[ToolName]"
    )
```

### **Validation & Testing**

#### **Size Validation**
```python
def test_[tool_name]_size_limits():
    """Test that summarization respects size limits"""
    result = summarize_[tool_name](large_test_data, "/tmp/test.csv")
    
    key_findings_size = len(json.dumps(result.key_findings))
    summary_size = len(json.dumps(result.summary_dict))
    
    assert key_findings_size <= 2000, f"key_findings too large: {key_findings_size}B"
    assert summary_size <= 5000, f"summary_dict too large: {summary_size}B"
```

#### **Information Preservation**
```python
def test_[tool_name]_preserves_critical_info():
    """Test that critical information isn't lost"""
    # Test specific to this tool's domain
    # e.g., ensure blocked reactions are reported
    # e.g., ensure essential genes aren't omitted
```

#### **Scientific Accuracy**
```python  
def test_[tool_name]_scientific_accuracy():
    """Test that summarization maintains scientific validity"""
    # Compare insights from summary vs full data
    # Ensure no misleading conclusions possible
```

### **Integration Points**

#### **Agent Usage Patterns**
*When should the agent fetch full data?*

- [ ] User asks for "exact values" or "complete list"
- [ ] Summary indicates anomalous results requiring investigation
- [ ] Downstream calculations need precise data
- [ ] Agent confidence is low and needs verification

#### **FetchArtifact Queries**
*What queries should be supported?*

```python
# Example queries for this tool
fetch_artifact(path, query={"reaction": "SUCDi"})           # Single item
fetch_artifact(path, query={"category": "blocked"})        # Category
fetch_artifact(path, query={"top": 10, "by": "flux_range"}) # Top N
fetch_artifact(path, query={"subsystem": "Glycolysis"})    # Biological grouping
```

### **Success Metrics**

- **Size Reduction**: [X]% reduction achieved
- **Information Retention**: [Y] critical insights preserved  
- **Agent Performance**: No degradation in reasoning quality
- **Scientific Validity**: Expert review confirms accuracy

### **Documentation Completed**

- [ ] Three-tier examples documented
- [ ] Implementation code provided
- [ ] Validation tests written
- [ ] Integration patterns defined
- [ ] Success metrics measured

---

**Implementation Checklist**:
- [ ] Analyze raw output size and structure
- [ ] Define categorization strategy  
- [ ] Implement summarization function
- [ ] Write validation tests
- [ ] Document examples and patterns
- [ ] Integrate with ToolResult framework
- [ ] Test with large models (iML1515, EcoliMG1655)
- [ ] Measure and validate success metrics