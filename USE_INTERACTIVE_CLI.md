# 🚀 Using the Interactive CLI with Multi-Tool Analysis

## Quick Start

```bash
# Activate your virtual environment
source venv/bin/activate

# Start the interactive CLI
python -m src.cli.main interactive
```

## Example Multi-Tool Prompts

### 1. Comprehensive Metabolic Analysis
```
Perform a comprehensive metabolic analysis of the E. coli core model - analyze growth, find essential genes, determine minimal media requirements, and identify any auxotrophies
```

**This will trigger:**
- `run_metabolic_fba` → Check growth rate
- `analyze_essentiality` → Find essential genes/reactions
- `find_minimal_media` → Determine required nutrients
- `identify_auxotrophies` → Find biosynthetic gaps

### 2. Growth Investigation
```
Why might this model be growing slowly? Investigate potential bottlenecks by checking growth rate, essential nutrients, and metabolic efficiency
```

**This triggers adaptive analysis based on findings:**
- FBA to measure growth
- Minimal media analysis if growth is low
- Auxotrophy check for missing pathways
- Essential gene analysis

### 3. Nutritional Requirements
```
What nutrients does the E. coli core model need to grow? Find minimal media and any auxotrophies
```

**This will run:**
- `find_minimal_media` → Essential nutrients
- `identify_auxotrophies` → Missing biosynthetic capabilities
- Additional analysis based on findings

### 4. Essential Component Analysis
```
Find all essential genes and reactions in the E. coli core model and explain their importance
```

**This executes:**
- `analyze_essentiality` → Core essential components
- Additional analysis to understand why they're essential

## Key Features

### 🧠 Dynamic Tool Selection
The AI agent:
- Analyzes your query to understand intent
- Selects the most appropriate first tool
- Examines results and decides next steps
- Chains multiple tools based on discoveries

### 📊 Real-Time Reasoning
Watch the AI:
- Show its thought process
- Explain tool selection rationale
- Adapt workflow based on results
- Provide integrated conclusions

### 🔍 Complete Transparency
- Every tool execution is logged
- AI decisions are recorded
- Full audit trail for verification
- Confidence scoring on results

## What Was Fixed

1. **Async/Await Issues**: Fixed coroutine handling in interactive CLI
2. **Missing Methods**: Added all required helper methods to RealTimeMetabolicAgent
3. **Session Cleanup**: Cleared old non-functional sessions as requested
4. **Method Name Mismatches**: Fixed inconsistent method names

## Tips for Best Results

1. **Be Specific**: Include model paths when asking about specific models
   ```
   Analyze the model at data/examples/e_coli_core.xml
   ```

2. **Use Action Words**: Words like "analyze", "find", "determine", "investigate" help the AI understand intent

3. **Ask Complex Questions**: The system excels at multi-step analysis
   ```
   Compare growth on glucose vs lactose and identify metabolic differences
   ```

4. **Request Explanations**: Ask "why" questions to trigger hypothesis-driven analysis
   ```
   Why does this organism require histidine for growth?
   ```

## Troubleshooting

### "No LLM backend available"
The system needs either:
- Argo Gateway access (internal Argonne network)
- OpenAI API key in environment
- Local LLM model configured

### "Tool execution failed"
- Check model file paths are correct
- Ensure all dependencies are installed
- Review error messages for specific issues

## Example Session

```
🧬 ModelSEEDagent Interactive Analysis
=====================================

You: Perform a comprehensive analysis of E. coli core model
AI: I'll analyze the E. coli core model comprehensively...

🔧 Executing: run_metabolic_fba
   → Growth rate: 0.874 h⁻¹

🧠 AI Decision: Good growth detected, checking nutritional requirements

🔧 Executing: find_minimal_media
   → Requires 4 nutrients: glucose, NH4, phosphate, sulfate

🧠 AI Decision: Checking for auxotrophies to understand biosynthetic capabilities

🔧 Executing: identify_auxotrophies
   → No auxotrophies found - can synthesize all biomass components

📊 Final Analysis:
The E. coli core model shows robust growth (0.874 h⁻¹) with minimal nutritional
requirements (4 basic nutrients) and complete biosynthetic capabilities.
```

## Next Steps

1. Try the example prompts above
2. Experiment with your own complex queries
3. Watch how the AI chains tools dynamically
4. Review audit logs to understand AI reasoning

The system is now ready for comprehensive metabolic modeling analysis with real AI-driven tool selection!
