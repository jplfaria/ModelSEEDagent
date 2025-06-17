# Tool Reference Guide

## Overview

ModelSEEDagent provides **25 specialized metabolic modeling tools** organized into six main categories, enhanced with the **Smart Summarization Framework** for optimal LLM performance. Each tool is designed for specific analysis tasks and integrates seamlessly with the AI reasoning system.

## Tool Categories

1. [AI Media Tools (6 tools)](#ai-media-tools) - Intelligent media management and optimization
2. [COBRApy Tools (12 tools)](#cobrapy-tools) - Comprehensive metabolic modeling analysis
3. [Biochemistry Tools (3 tools)](#biochemistry-tools) - Enhanced compound/reaction resolution and cross-database translation
4. [System Tools (4 tools)](#system-tools) - AI auditing and verification

For detailed technical implementation information, see the [API Tool Implementation Reference](api/tools.md).

## Smart Summarization Framework

All ModelSEEDagent tools integrate with the **Smart Summarization Framework**, which automatically transforms massive tool outputs (up to 138 MB) into LLM-optimized formats while preserving complete data access.

### Three-Tier Information Hierarchy

**Tier 1: key_findings (≤2KB)**
- Critical insights optimized for immediate LLM consumption
- Bullet-point format with percentages and key metrics
- Warnings and success indicators
- Top examples (3-5 items maximum)

**Tier 2: summary_dict (≤5KB)**
- Structured data for follow-up analysis
- Statistical summaries and distributions
- Category counts with limited examples
- Metadata and analysis parameters

**Tier 3: full_data_path**
- Complete raw results stored as JSON artifacts
- Accessible via FetchArtifact tool for detailed analysis
- No size limitations - preserves all original data

### Size Reduction Achievements

| Tool | Original Size | Summarized | Reduction | Status |
|------|--------------|------------|-----------|---------|
| FluxSampling | 138.5 MB | 2.2 KB | 99.998% | Production |
| FluxVariability | 170 KB | 2.4 KB | 98.6% | Production |
| GeneDeletion | 130 KB | 3.1 KB | 97.6% | Production |
| FBA | 48 KB | 1.8 KB | 96.3% | Production |

---

## AI Media Tools

Intelligent media management and optimization tools powered by AI reasoning:

### 1. Media Selector (`select_optimal_media`)
**Purpose**: Automatically find the best growth media for your model
**Usage**: `modelseed-agent analyze model.xml --query "select optimal media"`
**What it does**: Tests multiple media types and recommends the one that gives best growth

### 2. Media Manipulator (`manipulate_media_composition`)
**Purpose**: Modify media using natural language commands
**Usage**: `"make this media anaerobic"` or `"add vitamins to the media"`
**What it does**: Interprets commands like "add amino acids" and applies the changes

### 3. Media Compatibility Checker (`analyze_media_compatibility`)
**Purpose**: Check if your model can grow on specific media types
**Usage**: Automatically runs when testing different media
**What it does**: Identifies missing transporters and suggests improvements

### 4. Media Performance Comparator (`compare_media_performance`)
**Purpose**: Compare growth rates across different media types
**Usage**: `"compare growth on different media types"`
**What it does**: Ranks media by growth performance and provides insights

### 5. Media Optimizer (`optimize_media_composition`)
**Purpose**: Design custom media to achieve target growth rates
**Usage**: `"optimize media for maximum growth"`
**What it does**: Iteratively adds/removes compounds to reach growth targets

### 6. Auxotrophy Predictor (`predict_auxotrophies`)
**Purpose**: Predict which nutrients your model requires
**Usage**: `"predict auxotrophies for this model"`
**What it does**: Identifies essential compounds the model cannot synthesize

---

## COBRApy Tools

Core metabolic modeling analysis capabilities:

### 1. FBA Tool (`run_metabolic_fba`)
**Purpose**: Calculate growth rates and metabolic fluxes
**Usage**: `"run flux balance analysis on this model"`
**What it does**: Predicts growth rate and identifies active metabolic pathways

### 2. Model Analyzer (`analyze_metabolic_model`)
**Purpose**: Analyze the structure and composition of metabolic models
**Usage**: `"analyze the structure of this model"`
**What it does**: Counts reactions, metabolites, genes, and identifies network properties

### 3. Pathway Analyzer (`analyze_pathway`)
**Purpose**: Analyze specific metabolic pathways and subsystems
**Usage**: `"analyze the glycolysis pathway"`
**What it does**: Examines pathway completeness, connectivity, and gene associations

### 4. Flux Variability Analysis (`run_flux_variability_analysis`)
**Purpose**: Determine the range of possible flux values for each reaction
**Usage**: `"run flux variability analysis"`
**What it does**: Calculates min/max flux ranges and identifies flexible vs. fixed reactions

### 5. Gene Deletion Analysis (`run_gene_deletion_analysis`)
**Purpose**: Test the effect of removing genes from the model
**Usage**: `"perform gene knockout analysis"`
**What it does**: Simulates gene deletions and categorizes essentiality

### 6. Essentiality Analysis (`analyze_essentiality`)
**Purpose**: Comprehensive analysis of essential genes and reactions
**Usage**: `"find essential genes and reactions"`
**What it does**: Identifies components critical for growth and survival

### 7. Flux Sampling (`run_flux_sampling`)
**Purpose**: Statistical exploration of the metabolic solution space
**Usage**: `"sample flux distributions"`
**What it does**: Generates thousands of possible flux states to understand variability

### 8. Production Envelope (`run_production_envelope`)
**Purpose**: Analyze trade-offs between growth and product formation
**Usage**: `"analyze production envelope for ethanol"`
**What it does**: Maps the relationship between growth rate and production capacity

### 9. Auxotrophy Identification (`identify_auxotrophies`)
**Purpose**: Find nutrients the model cannot produce
**Usage**: `"identify auxotrophies"`
**What it does**: Tests removal of compounds to find essential nutrients

### 10. Minimal Media Finder (`find_minimal_media`)
**Purpose**: Find the smallest set of nutrients needed for growth
**Usage**: `"find minimal media requirements"`
**What it does**: Systematically removes nutrients to find the minimal viable set

### 11. Missing Media Checker (`check_missing_media`)
**Purpose**: Diagnose media gaps when growth is poor
**Usage**: `"check for missing media components"`
**What it does**: Tests addition of essential nutrients to improve growth

### 12. Reaction Expression (`analyze_reaction_expression`)
**Purpose**: Analyze reaction activity levels across the network
**Usage**: `"analyze reaction expression levels"`
**What it does**: Calculates how active each reaction is under given conditions


---

## Biochemistry Tools

Enhanced universal compound and reaction information tools with pure ModelSEEDpy integration:

### 1. Biochemistry Resolver (`resolve_biochem_entity`) ✨ ENHANCED
**Purpose**: Look up chemical information for metabolites and reactions using official ModelSEED database
**Usage**: `"what is cpd00027?"` or `"resolve this compound ID"`
**What it does**: Provides names, formulas, chemical properties, and comprehensive database cross-references from 45,706+ compounds and 56,009+ reactions

### 2. Biochemistry Search (`search_biochem`) ✨ ENHANCED
**Purpose**: Advanced search across the complete ModelSEED biochemistry database
**Usage**: `"search for glucose compounds"` or `"find reactions containing ATP"`
**What it does**: Intelligent search with match scoring across 45,706+ compounds and 56,009+ reactions by name, formula, aliases, and chemical properties

### 3. Cross-Database ID Translator (`translate_database_ids`) ✨ NEW
**Purpose**: Universal ID translation between biochemical databases using official ModelSEED mappings
**Usage**: `"convert BiGG IDs to ModelSEED format"` or `"translate C00002 to other databases"`
**What it does**: Converts IDs between ModelSEED ↔ BiGG ↔ KEGG ↔ MetaCyc ↔ ChEBI across 55+ databases with automatic compartment handling

---

## ModelSEED Tools

Genome-scale model construction and annotation tools:

### 1. Model Builder (`build_modelseed_model`) (in development - currently not functional)
**Purpose**: Build metabolic models from genome annotations
**Usage**: `"build a model from this genome"`
**What it does**: Creates draft metabolic models from gene annotations

### 2. Model Gapfiller (`gapfill_modelseed_model`) (in development - currently not functional)
**Purpose**: Fill gaps in metabolic networks to enable growth
**Usage**: `"gapfill this model"`
**What it does**: Adds missing reactions needed for biomass production

### 3. Annotation Tool (`annotate_with_modelseed`) (in development - currently not functional)
**Purpose**: Annotate models with ModelSEED database information
**Usage**: Automatically applied during model analysis
**What it does**: Adds standardized biochemistry annotations

### 4. Compatibility Checker (`check_modelseed_compatibility`) (in development - currently not functional)
**Purpose**: Check ModelSEED-COBRApy compatibility
**Usage**: `"check model compatibility"`
**What it does**: Validates model format and suggests conversions

### 5. Protein Annotator (`annotate_proteins_rast`) (in development - currently not functional)
**Purpose**: Functional annotation of protein sequences
**Usage**: `"annotate protein functions"`
**What it does**: Assigns enzyme functions and metabolic roles

---

## RAST Tools

Genome annotation and analysis tools:

### 1. RAST Annotator (`annotate_with_rast`) (in development - currently not functional)
**Purpose**: Genome annotation using RAST server
**Usage**: `"annotate this genome with RAST"`
**What it does**: Automated genome annotation and functional assignment

### 2. Annotation Analyzer (`analyze_rast_annotations`) (in development - currently not functional)
**Purpose**: Quality assessment of genome annotations
**Usage**: `"analyze annotation quality"`
**What it does**: Evaluates completeness and accuracy of genome annotations

---

## Getting Started

All tools are accessible through natural language queries in the interactive interface:

```bash
modelseed-agent interactive
```

**Example queries to try:**
- `"Load E. coli core model and run FBA"`
- `"Find essential genes in this model"`
- `"What is the optimal media for growth?"`
- `"Identify auxotrophies and suggest supplements"`
- `"Compare growth on different media types"`

## Additional Information

For detailed technical implementation information including parameters, precision configurations, and advanced usage patterns, see the [API Tool Implementation Reference](api/tools.md).

---

## System Tools

AI auditing and verification tools for transparency and quality assurance:

### 1. Tool Audit (`tool_audit`)
**Purpose**: Audit and verify tool execution with detailed tracking
**Usage**: Automatically tracks all tool executions during workflows
**What it does**: Records tool inputs, outputs, execution times, and success/failure status

### 2. AI Audit (`ai_audit`)
**Purpose**: Audit AI reasoning and decision-making processes
**Usage**: Monitors AI agent decisions and reasoning chains
**What it does**: Tracks AI model responses, reasoning steps, and decision paths for transparency

### 3. Realtime Verification (`realtime_verification`)
**Purpose**: Live verification of AI statements against actual results
**Usage**: Automatically validates AI claims during execution
**What it does**: Cross-references AI assertions with tool outputs to detect and prevent hallucinations

### 4. FetchArtifact (`fetch_artifact_data`)
**Purpose**: Retrieve complete raw data from Smart Summarization artifacts
**Usage**: `"get the full flux sampling data for detailed analysis"`
**What it does**: Loads complete original tool outputs from storage when detailed analysis is needed beyond summarized results

**When to use FetchArtifact**:
- User asks for "detailed analysis" or "complete results"
- Statistical analysis beyond summary_dict scope is needed
- Debugging scenarios requiring full data inspection
- Cross-model comparisons requiring raw data

## Summary

ModelSEEDagent's 25 tools provide comprehensive metabolic modeling capabilities through an intuitive AI interface enhanced with Smart Summarization. Each tool is designed to work seamlessly with the AI reasoning system, allowing for complex multi-step analyses through simple natural language commands.
