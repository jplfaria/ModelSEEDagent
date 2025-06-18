# Reasoning Framework API Documentation

**ModelSEEDagent Intelligence Enhancement Framework**
**API Version**: 1.0
**Last Updated**: June 18, 2025

## Overview

The Reasoning Framework API provides programmatic access to ModelSEEDagent's enhanced intelligence capabilities. This comprehensive API enables developers to integrate advanced reasoning, quality assessment, and continuous learning features into their applications.

## Architecture Overview

### Core Components

```
Intelligence Enhancement Framework
├── Phase 1: Enhanced Prompt Management
├── Phase 2: Context Enhancement
├── Phase 3: Quality Validation
├── Phase 4: Artifact Intelligence + Self-Reflection
└── Phase 5: Integrated Validation
```

### API Endpoints

#### Base URL
```
https://api.modelseedagent.org/v1/reasoning/
```

#### Authentication
```python
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}
```

## Phase 1: Enhanced Prompt Management

### Enhanced Prompt Provider

#### Get Optimized Prompt
```http
GET /prompts/enhanced/{prompt_type}
```

**Parameters:**
- `prompt_type` (string): Type of analysis prompt
- `context` (object, optional): Additional context for prompt optimization

**Example Request:**
```python
import requests

response = requests.get(
    "https://api.modelseedagent.org/v1/reasoning/prompts/enhanced/fba_analysis",
    headers=headers,
    params={
        "organism": "E. coli",
        "condition": "glucose_limitation",
        "optimization_target": "growth_rate"
    }
)
```

**Response:**
```json
{
    "prompt_id": "fba_analysis_optimized_001",
    "prompt_text": "Analyze the metabolic flux distribution...",
    "optimization_score": 0.94,
    "version": "2.3.1",
    "context_enhancements": [
        "glucose_metabolism_constraints",
        "aerobic_respiration_pathways"
    ]
}
```

#### Reasoning Trace Logger

#### Start Reasoning Trace
```http
POST /traces/start
```

**Request Body:**
```json
{
    "trace_id": "analysis_trace_001",
    "query": "Analyze E. coli growth optimization",
    "analysis_type": "metabolic_flux_analysis",
    "user_id": "user_123"
}
```

**Response:**
```json
{
    "trace_id": "analysis_trace_001",
    "status": "active",
    "start_time": "2025-06-18T10:30:00Z",
    "expected_completion": "2025-06-18T10:31:30Z"
}
```

#### Log Reasoning Step
```http
POST /traces/{trace_id}/steps
```

**Request Body:**
```json
{
    "step_number": 1,
    "step_type": "tool_selection",
    "decision": "selected_fba_analysis",
    "reasoning": "FBA provides baseline growth rate measurements",
    "confidence": 0.92,
    "alternatives_considered": ["flux_sampling", "gene_deletion"],
    "timestamp": "2025-06-18T10:30:15Z"
}
```

#### Get Reasoning Trace
```http
GET /traces/{trace_id}
```

**Response:**
```json
{
    "trace_id": "analysis_trace_001",
    "status": "completed",
    "total_steps": 8,
    "quality_score": 0.91,
    "transparency_score": 0.89,
    "steps": [
        {
            "step_number": 1,
            "step_type": "tool_selection",
            "decision": "selected_fba_analysis",
            "reasoning": "FBA provides baseline growth rate measurements",
            "confidence": 0.92,
            "timestamp": "2025-06-18T10:30:15Z"
        }
    ]
}
```

## Phase 2: Context Enhancement

### Context Enhancer

#### Enhance Query Context
```http
POST /context/enhance
```

**Request Body:**
```json
{
    "query": "Analyze E. coli metabolism",
    "organism": "Escherichia coli K-12",
    "experimental_conditions": {
        "temperature": 37,
        "ph": 7.0,
        "carbon_source": "glucose",
        "oxygen_availability": "aerobic"
    }
}
```

**Response:**
```json
{
    "enhanced_context": {
        "biochemical_pathways": [
            "glycolysis",
            "citric_acid_cycle",
            "electron_transport_chain"
        ],
        "relevant_constraints": [
            "glucose_uptake_rate_limit",
            "oxygen_consumption_constraint"
        ],
        "knowledge_sources": [
            "KEGG_pathways",
            "BioCyc_database",
            "literature_data"
        ]
    },
    "enhancement_score": 0.94,
    "confidence": 0.91
}
```

#### Get Available Context Types
```http
GET /context/types
```

**Response:**
```json
{
    "context_types": [
        {
            "type": "biochemical_pathways",
            "description": "Metabolic pathway information",
            "coverage": "comprehensive"
        },
        {
            "type": "regulatory_networks",
            "description": "Gene regulatory information",
            "coverage": "extensive"
        },
        {
            "type": "experimental_conditions",
            "description": "Growth and environmental constraints",
            "coverage": "standard_conditions"
        }
    ]
}
```

## Phase 3: Quality Validation

### Integrated Quality System

#### Assess Analysis Quality
```http
POST /quality/assess
```

**Request Body:**
```json
{
    "analysis_id": "analysis_001",
    "analysis_results": {
        "growth_rate": 0.87,
        "flux_distribution": {...},
        "gene_essentiality": {...}
    },
    "reasoning_trace": "trace_001",
    "artifacts_generated": ["fba_result.json", "flux_analysis.json"]
}
```

**Response:**
```json
{
    "quality_assessment": {
        "overall_score": 0.924,
        "biological_accuracy": 0.94,
        "reasoning_transparency": 0.89,
        "synthesis_effectiveness": 0.91,
        "artifact_usage_quality": 0.87
    },
    "validation_details": {
        "passed_checks": 15,
        "total_checks": 17,
        "warnings": ["minor_pathway_gaps"],
        "recommendations": ["include_amino_acid_synthesis"]
    },
    "confidence_intervals": {
        "overall_score": [0.91, 0.94],
        "biological_accuracy": [0.92, 0.96]
    }
}
```

### Composite Metrics Calculator

#### Calculate Composite Metrics
```http
POST /metrics/composite
```

**Request Body:**
```json
{
    "metrics": {
        "execution_time": 28.5,
        "quality_score": 0.924,
        "user_satisfaction": 0.94,
        "hypothesis_count": 3,
        "artifact_utilization": 0.78
    },
    "weights": {
        "quality": 0.4,
        "performance": 0.2,
        "user_experience": 0.2,
        "scientific_value": 0.2
    }
}
```

**Response:**
```json
{
    "composite_score": 0.887,
    "component_scores": {
        "quality_component": 0.924,
        "performance_component": 0.82,
        "user_experience_component": 0.94,
        "scientific_value_component": 0.86
    },
    "trend_analysis": {
        "30_day_improvement": 0.12,
        "performance_trend": "improving"
    }
}
```

## Phase 4: Artifact Intelligence + Self-Reflection

### Artifact Intelligence Engine

#### Register Artifact
```http
POST /artifacts/register
```

**Request Body:**
```json
{
    "artifact_path": "/results/fba_analysis_001.json",
    "metadata": {
        "type": "fba_results",
        "source_tool": "cobra_fba",
        "analysis_id": "analysis_001",
        "format": "json",
        "size_bytes": 15420
    }
}
```

**Response:**
```json
{
    "artifact_id": "artifact_12345",
    "registration_status": "success",
    "initial_assessment": {
        "completeness": 0.92,
        "estimated_quality": 0.89,
        "context_relevance": 0.91
    }
}
```

#### Perform Artifact Self-Assessment
```http
POST /artifacts/{artifact_id}/self-assess
```

**Response:**
```json
{
    "assessment_id": "assessment_001",
    "overall_score": 0.918,
    "detailed_scores": {
        "completeness": 0.94,
        "consistency": 0.91,
        "biological_validity": 0.96,
        "methodological_soundness": 0.88,
        "contextual_relevance": 0.92
    },
    "confidence_score": 0.89,
    "uncertainty_sources": [
        "limited_pathway_coverage",
        "missing_regulatory_constraints"
    ],
    "improvement_opportunities": [
        "include_additional_pathways",
        "add_regulatory_validation"
    ]
}
```

#### Analyze Contextual Intelligence
```http
GET /artifacts/{artifact_id}/context-analysis
```

**Response:**
```json
{
    "contextual_intelligence": {
        "experimental_context": "Growth rate optimization under glucose limitation",
        "biological_significance": "Central carbon metabolism efficiency analysis",
        "methodological_implications": "Constraint-based modeling approach",
        "cross_scale_connections": [
            "molecular_level_flux_rates",
            "cellular_growth_phenotype",
            "system_level_optimization"
        ]
    },
    "relevance_score": 0.93,
    "knowledge_gaps": ["regulatory_network_data"],
    "related_artifacts": ["artifact_12344", "artifact_12346"]
}
```

### Self-Reflection Engine

#### Capture Reasoning Trace for Reflection
```http
POST /reflection/capture-trace
```

**Request Body:**
```json
{
    "trace_id": "trace_001",
    "query": "Analyze E. coli growth optimization",
    "response": "Analysis shows glucose uptake limitation...",
    "tools_used": ["fba_analysis", "flux_variability"],
    "reasoning_steps": [...],
    "outcome_quality": 0.92
}
```

#### Perform Meta-Analysis
```http
POST /reflection/meta-analysis
```

**Request Body:**
```json
{
    "trace_ids": ["trace_001", "trace_002", "trace_003"],
    "analysis_window": "7_days",
    "pattern_types": ["success_patterns", "efficiency_patterns", "quality_patterns"]
}
```

**Response:**
```json
{
    "meta_analysis_id": "meta_001",
    "patterns_discovered": [
        {
            "pattern_type": "success_pattern",
            "pattern_id": "pattern_001",
            "description": "FBA followed by flux variability analysis",
            "frequency": 12,
            "success_rate": 0.89,
            "effectiveness_score": 0.91
        }
    ],
    "bias_analysis": {
        "biases_detected": ["tool_selection_bias"],
        "bias_scores": {"confirmation_bias": 0.05, "anchoring_bias": 0.03},
        "mitigation_suggestions": ["diversify_tool_selection"]
    },
    "improvement_recommendations": [
        "increase_flux_sampling_usage",
        "enhance_regulatory_analysis"
    ]
}
```

#### Generate Improvement Plan
```http
POST /reflection/improvement-plan
```

**Response:**
```json
{
    "improvement_plan": {
        "plan_id": "improvement_001",
        "target_areas": [
            "efficiency_optimization",
            "quality_enhancement",
            "pattern_diversification"
        ],
        "specific_actions": [
            {
                "action": "implement_parallel_tool_execution",
                "expected_impact": "15% time reduction",
                "priority": "high"
            },
            {
                "action": "enhance_pathway_validation",
                "expected_impact": "8% quality improvement",
                "priority": "medium"
            }
        ],
        "success_metrics": [
            "execution_time_reduction",
            "quality_score_improvement",
            "user_satisfaction_increase"
        ]
    }
}
```

### Meta-Reasoning Engine

#### Optimize Cognitive Strategy
```http
POST /meta-reasoning/optimize-strategy
```

**Request Body:**
```json
{
    "current_strategy": "analytical",
    "analysis_context": {
        "complexity": "high",
        "time_constraints": "moderate",
        "accuracy_requirements": "high"
    },
    "performance_history": [...]
}
```

**Response:**
```json
{
    "optimized_strategy": {
        "primary_approach": "systematic",
        "secondary_approach": "analytical",
        "cognitive_allocation": {
            "systematic_thinking": 0.6,
            "analytical_reasoning": 0.3,
            "creative_exploration": 0.1
        },
        "expected_performance": {
            "quality_improvement": 0.08,
            "efficiency_gain": 0.05
        }
    }
}
```

## Phase 5: Integrated Validation

### Improvement Tracker

#### Record Analysis Metrics
```http
POST /improvement/record-metrics
```

**Request Body:**
```json
{
    "analysis_id": "analysis_001",
    "metrics": {
        "overall_quality": 0.924,
        "biological_accuracy": 0.94,
        "reasoning_transparency": 0.89,
        "synthesis_effectiveness": 0.91,
        "artifact_usage_rate": 0.78,
        "hypothesis_count": 3,
        "execution_time": 28.5,
        "error_rate": 0.002
    }
}
```

#### Get Quality Trend
```http
GET /improvement/quality-trend
```

**Parameters:**
- `days` (integer): Number of days to analyze (default: 30)

**Response:**
```json
{
    "trend_analysis": {
        "period_days": 30,
        "metrics_count": 156,
        "quality_trend": {
            "current_average": 0.924,
            "period_average": 0.891,
            "improvement": 0.15,
            "stability": 0.94
        },
        "performance_trend": {
            "average_time": 28.5,
            "efficiency_improvement": 0.12,
            "consistency": 0.89
        }
    }
}
```

#### Get Improvement Recommendations
```http
GET /improvement/recommendations
```

**Response:**
```json
{
    "recommendations": [
        {
            "type": "quality_optimization",
            "priority": "high",
            "title": "Enhance Pathway Validation",
            "description": "Strengthen biochemical pathway validation",
            "suggested_actions": [
                "integrate_additional_databases",
                "implement_cross_validation",
                "enhance_constraint_checking"
            ],
            "confidence": 0.87,
            "expected_impact": "8% quality improvement"
        }
    ]
}
```

### Integrated Validator

#### Run Validation Suite
```http
POST /validation/run-suite
```

**Request Body:**
```json
{
    "validation_type": "comprehensive",
    "test_categories": ["integration", "performance", "quality", "regression"],
    "priority_filter": "high"
}
```

**Response:**
```json
{
    "validation_id": "validation_001",
    "status": "running",
    "estimated_completion": "2025-06-18T11:45:00Z",
    "test_count": 25,
    "progress_endpoint": "/validation/validation_001/status"
}
```

#### Get Validation Results
```http
GET /validation/{validation_id}/results
```

**Response:**
```json
{
    "validation_summary": {
        "total_tests": 25,
        "passed_tests": 23,
        "failed_tests": 1,
        "error_tests": 1,
        "success_rate": 0.92,
        "average_quality_score": 0.887,
        "average_execution_time": 31.2
    },
    "detailed_results": [...],
    "recommendations": [
        "investigate_failed_integration_test",
        "optimize_performance_bottleneck"
    ]
}
```

## Data Models

### Core Data Types

#### ReasoningMetrics
```python
class ReasoningMetrics:
    overall_quality: float
    biological_accuracy: float
    reasoning_transparency: float
    synthesis_effectiveness: float
    artifact_usage_rate: float
    hypothesis_count: int
    execution_time: float
    error_rate: float
    timestamp: str
    analysis_id: str
```

#### QualityAssessment
```python
class QualityAssessment:
    overall_score: float
    detailed_scores: Dict[str, float]
    confidence_score: float
    uncertainty_sources: List[str]
    improvement_opportunities: List[str]
    validation_timestamp: str
```

#### ArtifactMetadata
```python
class ArtifactMetadata:
    artifact_id: str
    file_path: str
    artifact_type: str
    source_tool: str
    format: str
    size_bytes: int
    creation_timestamp: str
    analysis_id: str
```

## Error Handling

### Standard Error Responses

#### Authentication Error
```json
{
    "error": {
        "code": "AUTHENTICATION_FAILED",
        "message": "Invalid API key provided",
        "status": 401
    }
}
```

#### Validation Error
```json
{
    "error": {
        "code": "VALIDATION_FAILED",
        "message": "Invalid request parameters",
        "details": {
            "field": "quality_score",
            "issue": "must be between 0 and 1"
        },
        "status": 400
    }
}
```

#### Rate Limit Error
```json
{
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "API rate limit exceeded",
        "retry_after": 60,
        "status": 429
    }
}
```

## SDK Examples

### Python SDK

#### Installation
```bash
pip install modelseed-reasoning-framework
```

#### Basic Usage
```python
from modelseed_reasoning import ReasoningFramework

# Initialize client
client = ReasoningFramework(api_key="your_api_key")

# Enhanced analysis with full intelligence features
result = client.analyze(
    query="Analyze E. coli growth under glucose limitation",
    enable_reasoning_trace=True,
    enable_quality_assessment=True,
    enable_artifact_intelligence=True,
    enable_self_reflection=True
)

# Access results
print(f"Quality Score: {result.quality_score}")
print(f"Reasoning Trace: {result.reasoning_trace}")
print(f"Generated Hypotheses: {result.hypotheses}")
print(f"Improvement Suggestions: {result.improvement_suggestions}")
```

#### Advanced Usage
```python
# Start reasoning trace
trace = client.start_reasoning_trace(
    query="Complex metabolic analysis",
    analysis_type="comprehensive"
)

# Enhance context
enhanced_context = client.enhance_context(
    query="Analyze E. coli metabolism",
    organism="E. coli K-12",
    conditions={"carbon_source": "glucose", "oxygen": "aerobic"}
)

# Perform analysis with enhanced features
analysis = client.analyze_with_intelligence(
    query="Optimized query text",
    context=enhanced_context,
    trace_id=trace.trace_id,
    quality_threshold=0.85
)

# Get self-reflection insights
insights = client.get_self_reflection_insights(
    analysis_id=analysis.analysis_id,
    include_patterns=True,
    include_biases=True
)
```

### JavaScript SDK

#### Installation
```bash
npm install @modelseed/reasoning-framework
```

#### Basic Usage
```javascript
import { ReasoningFramework } from '@modelseed/reasoning-framework';

const client = new ReasoningFramework({
    apiKey: 'your_api_key',
    baseUrl: 'https://api.modelseedagent.org/v1/reasoning'
});

// Enhanced analysis
const result = await client.analyze({
    query: 'Analyze E. coli growth under glucose limitation',
    enableReasoningTrace: true,
    enableQualityAssessment: true,
    enableArtifactIntelligence: true
});

console.log('Quality Score:', result.qualityScore);
console.log('Hypotheses:', result.hypotheses);
```

## Rate Limits and Quotas

### Standard Limits
- **Analysis Requests**: 100 per hour
- **Validation Requests**: 20 per hour
- **Trace Queries**: 500 per hour
- **Quality Assessments**: 200 per hour

### Premium Limits
- **Analysis Requests**: 1000 per hour
- **Validation Requests**: 100 per hour
- **Trace Queries**: 2000 per hour
- **Quality Assessments**: 1000 per hour

## Webhooks

### Event Types
- `analysis.completed`: Analysis finished successfully
- `quality.threshold_exceeded`: Quality score above threshold
- `validation.failed`: Validation test failure
- `improvement.recommendation_available`: New improvement suggestion

### Webhook Configuration
```http
POST /webhooks/configure
```

**Request Body:**
```json
{
    "url": "https://your-app.com/webhooks/reasoning",
    "events": ["analysis.completed", "quality.threshold_exceeded"],
    "secret": "your_webhook_secret"
}
```

## Changelog

### Version 1.0 (June 18, 2025)
- Initial release of complete intelligence enhancement framework
- All Phase 1-5 components available
- Comprehensive API coverage for all features
- Python and JavaScript SDKs released

---

*For additional support, contact the ModelSEEDagent development team*
*API documentation is automatically updated with each framework release*
