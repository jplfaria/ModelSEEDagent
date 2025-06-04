# 🧬 ModelSEEDagent Interactive Analysis Interface

## Overview

The Interactive Analysis Interface is a revolutionary conversational AI system for metabolic modeling that transforms complex command-line operations into natural language conversations. Built in Phase 3.2, this interface provides real-time visualization, intelligent query processing, and persistent session management.

## 🚀 Getting Started

### Quick Launch

```bash
# Method 1: Direct launch
python src/interactive/interactive_cli.py

# Method 2: Via CLI integration
python -m src.cli.standalone interactive

# Method 3: Via main CLI (if configured)
python src/cli/main.py interactive
```

### First Time Setup

1. **Launch the interface** - Run one of the commands above
2. **Create a session** - Choose "Create new session" when prompted
3. **Start asking questions** - Use natural language to analyze models

## 💬 Natural Language Queries

The interface understands natural language questions about metabolic modeling:

### Model Analysis
```
"Analyze the structure of my E. coli model"
"What are the basic statistics of this model?"
"Show me the model components"
"Validate the SBML file structure"
```

### Growth Analysis
```
"What is the growth rate on glucose?"
"Calculate biomass production"
"Analyze growth conditions"
"Compare growth on different substrates"
```

### Pathway Analysis
```
"Analyze glycolysis pathway fluxes"
"Show central carbon metabolism"
"Examine pathway connectivity"
"Find bottlenecks in amino acid synthesis"
```

### Flux Analysis
```
"Run flux balance analysis"
"Calculate flux variability"
"Optimize metabolic fluxes"
"Find essential genes"
```

### Visualization Requests
```
"Create a network visualization"
"Generate a flux heatmap"
"Show pathway diagrams"
"Plot flux distributions"
```

### Comparative Analysis
```
"Compare two growth conditions"
"Analyze knockout vs wildtype"
"Compare different carbon sources"
"Evaluate experimental vs predicted data"
```

## 🎨 Interactive Features

### Session Management

The interface automatically manages your analysis sessions:

- **Session Creation**: Interactive prompts to name and describe sessions
- **Session Loading**: Resume previous analyses from any point
- **Session Analytics**: Track progress, success rates, and execution times
- **Auto-Save**: Automatic session persistence after each interaction

### Real-Time Visualization

- **Workflow Graphs**: See your analysis pipeline in real-time
- **Progress Dashboards**: Monitor execution with live metrics
- **Network Visualizations**: Interactive metabolic network graphs
- **Flux Heatmaps**: Dynamic flux distribution displays
- **Automatic Browser Integration**: Visualizations open automatically

### Intelligent Assistance

- **Context Awareness**: Remembers previous questions and responses
- **Smart Suggestions**: Recommends follow-up analyses
- **Error Guidance**: Helpful error messages with suggested fixes
- **Progressive Disclosure**: Complexity adapted to your needs

## 🔧 Advanced Commands

### Session Commands
```
sessions          # List all available sessions
switch <id>       # Switch to a different session
status            # Show current session status
analytics         # View session analytics
```

### Visualization Commands
```
visualizations    # Show available visualizations
viz               # Alias for visualizations
open <type>       # Open specific visualization in browser
```

### Utility Commands
```
help              # Show help information
clear             # Clear the terminal
exit              # Exit the interactive session
```

## 📊 Session Analytics

The interface tracks comprehensive analytics for each session:

### Performance Metrics
- **Total Interactions**: Number of queries processed
- **Success Rate**: Percentage of successful analyses
- **Average Execution Time**: Mean processing time per query
- **Tool Usage**: Statistics on which tools are used most

### Activity Tracking
- **Recent Activity**: Timeline of recent interactions
- **Query Types**: Distribution of query categories
- **Session Duration**: Total time spent in analysis
- **Error Analysis**: Categorization of any issues encountered

## 🎯 Example Workflow

Here's a typical analysis workflow using the interactive interface:

### 1. Launch and Setup
```bash
python src/interactive/interactive_cli.py
```

### 2. Create Session
```
Session name: E_coli_glucose_analysis
Description: Analyzing E. coli growth on glucose media
```

### 3. Load Model
```
"Load and analyze the E. coli core model"
```

### 4. Basic Analysis
```
"What are the model statistics?"
"Show me the growth rate on glucose"
```

### 5. Detailed Analysis
```
"Run flux balance analysis with glucose as carbon source"
"Create a flux heatmap for central carbon metabolism"
```

### 6. Visualization
```
"Generate a network visualization of the metabolic network"
"Open workflow visualization in browser"
```

### 7. Comparative Analysis
```
"Compare growth on glucose vs acetate"
"What happens if I knockout gene XYZ?"
```

## 🔍 Query Processing Intelligence

The interface uses advanced NLP to understand your queries:

### Query Classification
- **Structural Analysis**: Model components, validation, statistics
- **Growth Analysis**: Biomass, growth rates, conditions
- **Pathway Analysis**: Specific pathways, connectivity, bottlenecks
- **Flux Analysis**: FBA, FVA, optimization, essential genes
- **Network Analysis**: Topology, centrality, clustering
- **Optimization**: Parameter tuning, constraint modification
- **Comparison**: Multi-condition, multi-strain analysis

### Confidence Scoring
- **High Confidence (80-100%)**: Direct execution
- **Medium Confidence (50-79%)**: Clarifying questions
- **Low Confidence (<50%)**: Guided assistance

### Context Awareness
- **Previous Queries**: Remembers conversation history
- **Active Model**: Knows what model you're working with
- **Analysis State**: Tracks completed and pending analyses
- **User Preferences**: Learns from your interaction patterns

## 🎨 Visualization Gallery

### Workflow Visualizations
Interactive graphs showing your analysis pipeline with:
- **Real-time status updates** for each step
- **Execution timing** and performance metrics
- **Tool dependencies** and data flow
- **Error highlighting** and success indicators

### Progress Dashboards
Comprehensive monitoring with:
- **Execution timelines** with interactive hover
- **Tool performance** comparisons
- **Success rate** gauges with targets
- **Resource usage** tracking over time

### Network Visualizations
Beautiful metabolic network displays featuring:
- **Node classification** by type (metabolites, reactions, genes)
- **Interactive zoom** and pan capabilities
- **Pathway highlighting** with custom colors
- **Connectivity analysis** with centrality metrics

### Flux Heatmaps
Dynamic flux analysis visualizations with:
- **Condition comparisons** across multiple scenarios
- **Interactive hover** showing exact flux values
- **Color-coded significance** with customizable scales
- **Reaction filtering** and pathway focus

## 🛡️ Error Handling and Recovery

The interface provides intelligent error handling:

### Graceful Degradation
- **Network Issues**: Cached responses and offline mode
- **Model Errors**: Validation guidance and fix suggestions
- **Computation Failures**: Alternative approaches and simplified analyses
- **Visualization Problems**: Fallback to text-based outputs

### Recovery Features
- **Session Persistence**: Automatic save on interruption
- **State Recovery**: Resume from any point in analysis
- **Error Diagnosis**: Detailed error analysis with solutions
- **Retry Mechanisms**: Intelligent retry with parameter adjustment

## 🔧 Configuration Options

### Session Configuration
```json
{
  "auto_visualize": true,
  "default_model_path": "models/",
  "visualization_browser": "default",
  "session_timeout": 3600,
  "max_history": 100
}
```

### Visualization Settings
```json
{
  "theme": "plotly_white",
  "auto_open": true,
  "export_format": "html",
  "figure_size": [800, 600],
  "color_scheme": "viridis"
}
```

### AI Configuration
```json
{
  "confidence_threshold": 0.5,
  "max_suggestions": 3,
  "context_window": 10,
  "response_style": "detailed"
}
```

## 🚀 Tips for Effective Usage

### Best Practices

1. **Be Specific**: "Analyze glycolysis" vs "Show me some pathways"
2. **Build Context**: Start with model loading, then dive into specifics
3. **Use Follow-ups**: Take advantage of suggested next steps
4. **Save Sessions**: Use descriptive names for easy retrieval
5. **Explore Visualizations**: Interactive plots reveal hidden insights

### Common Patterns

- **Exploratory Analysis**: Start broad, then narrow down
- **Comparative Studies**: Use consistent terminology across comparisons
- **Hypothesis Testing**: Frame questions as testable hypotheses
- **Iterative Refinement**: Build on previous results progressively

### Advanced Tips

- **Chaining Queries**: Reference previous results in new questions
- **Batch Operations**: Combine multiple analyses in single requests
- **Custom Visualizations**: Request specific plot types and parameters
- **Export Integration**: Seamlessly move between interface and external tools

## 🐛 Troubleshooting

### Common Issues

**Q: Interface won't start**
A: Check Python environment and dependencies:
```bash
pip install -r requirements.txt
python -c "import rich, questionary, plotly; print('All dependencies available')"
```

**Q: Visualizations don't open**
A: Verify browser configuration and file permissions

**Q: Sessions not saving**
A: Check write permissions in session directory

**Q: Queries not understood**
A: Try more specific terminology or use help command

### Getting Help

- Use `help` command within the interface
- Check the example queries in this guide
- Review session analytics for usage patterns
- Consult the conversation module documentation

## 🎯 What's Next?

The Interactive Analysis Interface sets the foundation for:

- **Phase 3.3**: Advanced workflow automation
- **Phase 3.4**: Multi-user collaboration features
- **Phase 4.1**: Web-based interface integration
- **Phase 4.2**: API-driven workflow orchestration

Experience the future of metabolic modeling - where complex analyses become as simple as asking a question! 🧬✨
