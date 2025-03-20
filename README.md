# Extensive Documentation for Metabolic Modeling Agent Framework

Version: 2.0  
Last Updated: 2025-02-28

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Architecture](#architecture)
    - [Configuration Layer](#configuration-layer)
    - [LLM Integration](#llm-integration)
    - [Agent Layer](#agent-layer)
    - [Tool Layer](#tool-layer)
    - [Utilities](#utilities)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
    - [Interactive Use](#interactive-use)
    - [Command-Line Interface](#command-line-interface)
6. [Detailed Component Documentation](#detailed-component-documentation)
    - [Configuration Files](#configuration-files)
    - [Prompt Templates](#prompt-templates)
    - [Agents](#agents)
    - [LLM Implementations](#llm-implementations)
    - [Tool Implementations](#tool-implementations)
    - [Testing Framework](#testing-framework)
7. [Extending the Framework](#extending-the-framework)
8. [Developer Guidelines](#developer-guidelines)
9. [Contribution Guidelines](#contribution-guidelines)
10. [FAQs](#faqs)
11. [License](#license)
12. [Contact](#contact)

---

## 1. Introduction

The Metabolic Modeling Agent Framework is a modular, extensible platform that integrates advanced language models (LLMs) with specialized metabolic modeling tools. It is designed to support tasks such as Flux Balance Analysis (FBA), model structural analysis, media optimization, auxotrophy detection, and genome annotation integration. This documentation provides a comprehensive guide for developers and users detailing the system’s structure, usage, and guidelines for extending and contributing to the project.

---

## 2. Project Overview

This framework couples LLM reasoning with a suite of metabolic modeling tools using a ReAct (Reasoning + Acting) agent paradigm. The agent iteratively reasons about a given problem, calls specific tools as needed, and logs intermediate steps to eventually provide a final, well-supported answer.

Key functionalities include:

- **Flux Balance Analysis (FBA)**
- **Model Structural and Pathway Analysis**
- **Minimal Media Determination**
- **Auxotrophy Identification**
- **Genome Annotation via RAST Integration**

---

## 3. Architecture

### Configuration Layer

- **Configuration Files:**  
  YAML files (e.g., `config/config.yaml`) define LLM settings, tool parameters, and agent behavior.
- **Prompts:**  
  Prompt templates (stored in `config/prompts/`) standardize the instructions for both agents and tools.

### LLM Integration

- **Abstract LLM Interface:**  
  Provides a unified interface for different LLM backends (Argo, OpenAI, local models).
- **Implementations:**  
  - **ArgoLLM:** Communicates with a hosted API.
  - **OpenAILLM:** Uses OpenAI’s ChatCompletion API.
  - **LocalLLM:** Runs local models (e.g., Llama) via Hugging Face’s transformers.
- **Safety and Token Management:**  
  Implements safety checks, token estimation, and API call limits.

### Agent Layer

- **Core Agent:**  
  The metabolic agent (in `src/agents/metabolic.py`) follows a ReAct-style architecture, alternating between reasoning (“Thoughts”) and performing actions (“Actions”).
- **Output Parsing:**  
  Custom parsers distinguish between tool calls and final answers.
- **Logging:**  
  The agent logs execution steps and tool outputs for traceability.

### Tool Layer

- **Tool Registry:**  
  Tools are registered in a central registry (`src/tools/base.py`) to allow dynamic loading and invocation.
- **Implemented Tools:**  
  - **FBATool:** Runs FBA analysis.
  - **ModelAnalysisTool & PathwayAnalysisTool:** Analyze model structure and network properties.
  - **MissingMediaTool & MinimalMediaTool:** Identify media deficiencies and determine minimal media formulations.
  - **ReactionExpressionTool:** Analyzes flux distributions under specific media conditions.
  - **AuxotrophyTool:** Detects nutrient dependencies.
  - **RAST Tools:** For genome annotation and integration.
- **Extensibility:**  
  New tools can be added by following the BaseTool interface.

### Utilities

- **Model Utilities:**  
  In `src/tools/cobra/utils.py`, functions for loading, saving, verifying, and analyzing COBRA models are provided.
- **General Helpers:**  
  Additional helper functions for configuration management and prompt handling.

---

## 4. Installation and Setup

### Prerequisites

- Python 3.8+
- [COBRApy](https://github.com/opencobra/cobrapy) for metabolic modeling
- [PyYAML](https://pyyaml.org/) for configuration management
- LLM-specific dependencies (e.g., `transformers` for local models, `requests` for API-based models)
- Other dependencies as listed in your `requirements.txt`

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://your-repo-url.git
   cd metabolic-modeling-agent-framework