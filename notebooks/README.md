# ModelSEEDagent Notebooks

This directory contains interactive Jupyter notebooks demonstrating ModelSEEDagent capabilities.

## 📖 Available Notebooks

### 🎯 **comprehensive_working_tutorial.ipynb** - Complete Platform Showcase ⭐
**The main tutorial showcasing all ModelSEEDagent capabilities with working examples**

- **Purpose**: Complete interactive tutorial demonstrating all 17+ specialized tools
- **Audience**: New users, demonstrations, training, comprehensive platform overview
- **Content**:
  - ✅ **Complete Environment Setup** with kernel configuration
  - ✅ **All 17+ Tools Testing** - working tools demonstrated, missing tools documented
  - ✅ **4 Implementation Phases** - ModelSEED, COBRA, Compatibility, Biochemistry
  - ✅ **Real Data Analysis** using files from `/data/examples/`
  - ✅ **Organized Outputs** with inspection-ready results
  - ✅ **Complete Workflows** from genome to analysis
- **Status**: **Production-ready with working tool demonstrations**
- **Outputs**: `tutorial_working_outputs/` with organized analysis results

### 🔬 **Testing & Development Notebooks**

#### **argo.ipynb** - Argo Gateway LLM Testing
- **Purpose**: Testing Argo Gateway LLM functionality with agent analysis
- **Content**: Comprehensive metabolic model analysis workflows with various models (gpt4o, gpt4, etc.)
- **Use Case**: LLM backend verification and model comparison

#### **cobrapy_testing.ipynb** - COBRA Analysis Examples
- **Purpose**: Testing metabolic model analysis with E. coli core and iML1515 models
- **Content**: FBA simulations, model statistics, flux analysis with CSV export
- **Use Case**: COBRA tool validation and analysis examples

#### **local_llm.ipynb** - Local LLM Integration
- **Purpose**: Testing local LLM models (Llama 3.1-8B, 3.2-3B)
- **Content**: Model loading, inference testing, HuggingFace integration with GPU/MPS
- **Use Case**: Local LLM backend setup and agent creation

#### **utils.py** - Notebook Utilities
- **Purpose**: Common utility functions for notebook setup
- **Content**: `setup_metabolic_agent()` function for consistent agent configuration
- **Use Case**: Shared functionality across notebooks

### 📁 **archive/** - Development Versions
Contains previous tutorial versions for reference:
- `comprehensive_tutorial_v1.ipynb` - Original tutorial with expected results
- `comprehensive_tutorial_v2.ipynb` - Fixed version with formatting improvements
- `functional_tutorial_v1.ipynb` - Working tutorial with basic functionality
- `tutorial_outputs_v1/` - Results from functional tutorial testing

## 🚀 Getting Started

### For New Users
**Start with `comprehensive_working_tutorial.ipynb`** ⭐ - provides complete platform overview with working examples

### For Developers
Review testing notebooks to understand:
- LLM backend integration patterns
- Tool testing approaches
- Model analysis workflows

## 🛠️ Setup Requirements

```bash
# Install ModelSEEDagent with all dependencies
pip install .[all]

# Launch Jupyter
jupyter notebook

# Open comprehensive_tutorial.ipynb to start
```

## 📊 Notebook Status

| Notebook | Status | Purpose | Last Updated |
|----------|--------|---------|--------------|
| comprehensive_working_tutorial.ipynb | ✅ Active | Complete platform showcase | Current |
| argo.ipynb | ✅ Active | Argo Gateway testing | Current |
| cobrapy_testing.ipynb | ✅ Active | COBRA analysis examples | Current |
| local_llm.ipynb | ✅ Active | Local LLM integration | Current |
| utils.py | ✅ Active | Shared utilities | Current |

## 💡 Usage Tips

1. **Start with comprehensive_working_tutorial.ipynb** for the complete platform overview
2. **All notebooks are self-contained** - no dependencies between them
3. **Expected results shown** - actual execution requires proper configuration
4. **Interactive examples** - modify and experiment with the code
5. **Real-world workflows** - see how tools combine for complete analyses

---

🧬 **Ready to explore ModelSEEDagent's capabilities?** Start with `comprehensive_working_tutorial.ipynb`!
