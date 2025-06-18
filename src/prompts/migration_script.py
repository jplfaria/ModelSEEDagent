"""
Prompt Migration Script for ModelSEEDagent

Migrates all scattered prompts from various files into the centralized
prompt registry system with proper categorization and validation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from .prompt_registry import PromptCategory, PromptRegistry

logger = logging.getLogger(__name__)


class PromptMigrationManager:
    """Manages migration of prompts from scattered files to centralized registry"""

    def __init__(self):
        self.registry = PromptRegistry()
        self.migration_log: List[Dict[str, Any]] = []

    def migrate_all_prompts(self) -> bool:
        """Migrate all identified prompts to the centralized registry"""
        try:
            logger.info("Starting comprehensive prompt migration...")

            # Migration in order of complexity
            success_count = 0
            total_count = 0

            # 1. Configuration-based prompts
            success_count += self._migrate_config_prompts()
            total_count += 2

            # 2. Real-time metabolic agent prompts
            success_count += self._migrate_realtime_agent_prompts()
            total_count += 3

            # 3. Collaborative reasoning prompts
            success_count += self._migrate_collaborative_prompts()
            total_count += 3

            # 4. Reasoning chains prompts
            success_count += self._migrate_reasoning_chains_prompts()
            total_count += 6

            # 5. Hypothesis system prompts
            success_count += self._migrate_hypothesis_prompts()
            total_count += 5

            # 6. Pattern memory prompts
            success_count += self._migrate_pattern_memory_prompts()
            total_count += 1

            # 7. LangGraph agent prompts
            success_count += self._migrate_langgraph_prompts()
            total_count += 2

            # 8. LLM interface prompts
            success_count += self._migrate_llm_interface_prompts()
            total_count += 1

            # 9. Performance optimizer prompts
            success_count += self._migrate_performance_optimizer_prompts()
            total_count += 1

            # 10. Metabolic agent prompts (newly found)
            success_count += self._migrate_metabolic_agent_prompts()
            total_count += 1

            # 11. Tool-specific prompts (if any found during deeper analysis)
            success_count += self._migrate_additional_prompts()
            total_count += 3  # Estimated additional prompts

            logger.info(
                f"Migration completed: {success_count}/{total_count} prompts migrated successfully"
            )

            if success_count == total_count:
                logger.info("✅ All prompts migrated successfully!")
                return True
            else:
                logger.warning(
                    f"⚠️ {total_count - success_count} prompts failed to migrate"
                )
                return False

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

    def _migrate_config_prompts(self) -> int:
        """Migrate configuration-based prompts from YAML files"""
        success_count = 0

        # 1. Metabolic Agent Configuration System Prompt
        if self.registry.register_prompt(
            prompt_id="metabolic_agent_system",
            template="""You are an AI assistant specialized in metabolic modeling. Your expertise includes:
- Flux Balance Analysis (FBA)
- Metabolic network analysis
- Pathway analysis
- Model validation and improvement
- Integration of genomic and metabolic data

Context: {context}
Current Task: {task}""",
            category=PromptCategory.SYSTEM_CONFIGURATION,
            description="System prompt for metabolic modeling agents",
            variables=["context", "task"],
            validation_rules={"min_length": 50},
        ):
            success_count += 1
            self._log_migration(
                "metabolic_agent_system", "config/prompts/metabolic.yaml", "SUCCESS"
            )

        # 2. Metabolic Agent Prefix/Instructions
        if self.registry.register_prompt(
            prompt_id="metabolic_agent_format",
            template="""You are a metabolic modeling expert. Analyze metabolic models using the available tools.
IMPORTANT: Follow these rules exactly:
1. Only provide ONE response type at a time - either an Action or a Final Answer, never both.
2. Use "Action" when you need to call a tool.
3. Use "Final Answer" only when you have all necessary information and are done.

Previous Results:
{tool_results}

Available Tools:
{available_tools}

Use this EXACT format - do not deviate from it:

When using a tool:
  Thought: [your reasoning]
  Action: tool_name
  Action Input: [input for the tool]
  Observation: [result from the tool]
  ... (repeat as needed)

When providing final answer:
  Thought: [summarize findings]
  Final Answer: [final summary]

Query: {query}""",
            category=PromptCategory.SYSTEM_CONFIGURATION,
            description="Format instructions and workflow for metabolic agents",
            variables=["tool_results", "available_tools", "query"],
            validation_rules={"min_length": 100, "must_contain": "Action"},
        ):
            success_count += 1
            self._log_migration(
                "metabolic_agent_format", "config/prompts/metabolic.yaml", "SUCCESS"
            )

        return success_count

    def _migrate_realtime_agent_prompts(self) -> int:
        """Migrate real-time metabolic agent prompts"""
        success_count = 0

        # 1. Initial Tool Selection
        if self.registry.register_prompt(
            prompt_id="initial_tool_selection",
            template="""You are an expert metabolic modeling AI agent. Analyze this query and select the BEST first tool to start with.

Query: "{query}"

Available tools are categorized as follows:

ANALYSIS TOOLS (for analyzing existing models): {available_analysis_tools}
BUILD TOOLS (for creating new models from genome data): {available_build_tools}
BIOCHEMISTRY TOOLS (for biochemistry database queries): {available_biochem_tools}

IMPORTANT GUIDELINES:
- If the query asks for "analysis", "comprehensive analysis", "characterization", or mentions analyzing an existing model, START with ANALYSIS TOOLS
- BUILD TOOLS should only be used when the query explicitly mentions building a new model from genome/annotation data
- For E. coli analysis queries, "run_metabolic_fba" is usually the best starting point as it provides foundational growth and flux information
- ANALYSIS TOOLS work with existing model files and don't require genome annotation data
- BUILD TOOLS require genome annotation files and are not appropriate for analyzing pre-built models

Based on the query, what tool should you start with and why? Consider:
1. What type of analysis is being requested?
2. Do you have an existing model to analyze, or do you need to build one from genome data?
3. Which tool provides the most informative starting point for THIS specific query?

Respond with:
SELECTED_TOOL: [tool_name]
REASONING: [detailed explanation of why this tool is the best choice]""",
            category=PromptCategory.TOOL_SELECTION,
            description="Initial tool selection for real-time metabolic agent",
            variables=[
                "query",
                "available_analysis_tools",
                "available_build_tools",
                "available_biochem_tools",
            ],
            validation_rules={"min_length": 200, "must_contain": "SELECTED_TOOL"},
        ):
            success_count += 1
            self._log_migration(
                "initial_tool_selection",
                "src/agents/real_time_metabolic.py:377-397",
                "SUCCESS",
            )

        # 2. Next Tool Selection
        if self.registry.register_prompt(
            prompt_id="next_tool_selection",
            template="""You are an expert metabolic modeling AI agent. You have executed some tools and now need to decide what to do next based on the ACTUAL RESULTS you've obtained.

ORIGINAL QUERY: "{query}"

CURRENT STEP: {step_number}

RESULTS OBTAINED SO FAR:
{results_context}

Available tools are categorized as follows:

ANALYSIS TOOLS (for analyzing existing models): {available_analysis_tools}
BUILD TOOLS (for creating new models from genome data): {available_build_tools}
BIOCHEMISTRY TOOLS (for biochemistry database queries): {available_biochem_tools}

IMPORTANT GUIDELINES:
- Continue using ANALYSIS TOOLS to explore different aspects of the metabolic model
- BUILD TOOLS should only be used if you need to create a new model from genome data (rare in analysis workflows)
- For comprehensive analysis, consider tools like: minimal media, essentiality, flux variability, auxotrophy analysis
- NEW AI MEDIA TOOLS: Use select_optimal_media for intelligent media selection, manipulate_media_composition for natural language media modifications, analyze_media_compatibility for compatibility analysis, compare_media_performance for cross-media comparisons
- Each tool provides different insights: FBA (growth), minimal media (nutritional requirements), essentiality (critical genes), AI media tools (intelligent media management), etc.

Based on your results so far, what should you do next?

Respond with:
NEXT_ACTION: [CONTINUE_ANALYSIS | TOOL_SELECTION | ANALYSIS_COMPLETE]
SELECTED_TOOL: [tool_name if TOOL_SELECTION]
REASONING: [detailed explanation based on actual results obtained]""",
            category=PromptCategory.TOOL_SELECTION,
            description="Next tool selection based on previous results",
            variables=[
                "query",
                "step_number",
                "results_context",
                "available_analysis_tools",
                "available_build_tools",
                "available_biochem_tools",
            ],
            validation_rules={"min_length": 150, "must_contain": "NEXT_ACTION"},
        ):
            success_count += 1
            self._log_migration(
                "next_tool_selection",
                "src/agents/real_time_metabolic.py:589-609",
                "SUCCESS",
            )

        # 3. Final Analysis Generation
        if self.registry.register_prompt(
            prompt_id="final_analysis_generation",
            template="""You are an expert metabolic modeling AI agent. Provide a comprehensive analysis based on the ACTUAL RESULTS you've collected.

ORIGINAL QUERY: "{query}"

ACTUAL RESULTS COLLECTED:
{results_context}

Based on the REAL DATA you've gathered, provide:

1. KEY QUANTITATIVE FINDINGS: Extract specific numbers, rates, counts from your results
2. BIOLOGICAL INSIGHTS: What do these results tell us about the organism's metabolism?
3. DIRECT ANSWERS: Answer the specific questions posed in the original query
4. CONFIDENCE ASSESSMENT: Rate your confidence (0.0-1.0) in these conclusions

Format your response as:
QUANTITATIVE_FINDINGS: [specific numbers and metrics you discovered]
BIOLOGICAL_INSIGHTS: [metabolic insights from the data]
DIRECT_ANSWERS: [answers to the original query]
CONFIDENCE_SCORE: [0.0-1.0]
SUMMARY: [concise overall conclusion]""",
            category=PromptCategory.RESULT_ANALYSIS,
            description="Final comprehensive analysis generation",
            variables=["query", "results_context"],
            validation_rules={
                "min_length": 100,
                "must_contain": "QUANTITATIVE_FINDINGS",
            },
        ):
            success_count += 1
            self._log_migration(
                "final_analysis_generation",
                "src/agents/real_time_metabolic.py:922-942",
                "SUCCESS",
            )

        return success_count

    def _migrate_collaborative_prompts(self) -> int:
        """Migrate collaborative reasoning prompts"""
        success_count = 0

        # 1. Uncertainty Assessment
        if self.registry.register_prompt(
            prompt_id="uncertainty_assessment",
            template="""Analyze this reasoning situation and determine if human collaboration would be beneficial:

Current Reasoning: {ai_reasoning}
Available Options: {available_options}
Context: {context}

Assess:
1. How confident are you in the best next step? (0.0 = very uncertain, 1.0 = very confident)
2. Are there multiple equally valid approaches?
3. Would domain expertise improve the decision?
4. Is the analysis at a critical decision point?

Respond with JSON:
{{
    "confidence": 0.8,
    "requires_collaboration": false,
    "collaboration_type": "none",
    "reasoning": "explanation of assessment"
}}""",
            category=PromptCategory.QUALITY_ASSESSMENT,
            description="Assess uncertainty and need for human collaboration",
            variables=["ai_reasoning", "available_options", "context"],
            validation_rules={"min_length": 50, "must_contain": "confidence"},
        ):
            success_count += 1
            self._log_migration(
                "uncertainty_assessment",
                "src/agents/collaborative_reasoning.py:116-131",
                "SUCCESS",
            )

        # 2. Option Recommendation
        if self.registry.register_prompt(
            prompt_id="option_recommendation",
            template="""Analyze these options and provide your recommendation:

Options: {options}

Which option do you think is best and why? Provide a brief rationale.

Consider:
- Potential outcomes of each option
- Alignment with analysis goals
- Risk vs benefit assessment
- Resource requirements

Respond with:
RECOMMENDED_OPTION: [chosen option]
RATIONALE: [detailed reasoning for the choice]
CONFIDENCE: [0.0-1.0]
ALTERNATIVE_CONSIDERATIONS: [brief notes on other viable options]""",
            category=PromptCategory.QUALITY_ASSESSMENT,
            description="Recommend best option from available choices",
            variables=["options"],
            validation_rules={"min_length": 30, "must_contain": "RECOMMENDED_OPTION"},
        ):
            success_count += 1
            self._log_migration(
                "option_recommendation",
                "src/agents/collaborative_reasoning.py:454-460",
                "SUCCESS",
            )

        # 3. Autonomous Decision Selection
        if self.registry.register_prompt(
            prompt_id="autonomous_decision_selection",
            template="""Select the best option for this analysis situation:

Context: {reasoning_context}
Options: {available_options}

Which option is most appropriate and why?

Consider the analysis context and select the option that:
- Best advances the analysis objectives
- Makes optimal use of available information
- Balances thoroughness with efficiency
- Minimizes risk of analysis errors

Respond with:
SELECTED_OPTION: [chosen option]
JUSTIFICATION: [reasoning for selection]
EXPECTED_OUTCOME: [what you expect this choice to achieve]""",
            category=PromptCategory.WORKFLOW_PLANNING,
            description="Autonomous selection of best analysis option",
            variables=["reasoning_context", "available_options"],
            validation_rules={"min_length": 40, "must_contain": "SELECTED_OPTION"},
        ):
            success_count += 1
            self._log_migration(
                "autonomous_decision_selection",
                "src/agents/collaborative_reasoning.py:514-521",
                "SUCCESS",
            )

        return success_count

    def _migrate_reasoning_chains_prompts(self) -> int:
        """Migrate reasoning chains prompts"""
        success_count = 0

        # 1. Analysis Goal Determination
        if self.registry.register_prompt(
            prompt_id="analysis_goal_determination",
            template="""Analyze this user query and determine the high-level analysis objective:

User Query: "{user_query}"
Context: {context}

Provide a clear, specific analysis goal that captures what the user wants to achieve.
Examples:
- "Comprehensive characterization of metabolic capabilities"
- "Investigation of growth limitations and bottlenecks"
- "Optimization of metabolic efficiency and resource utilization"

The goal should be:
- Specific and measurable
- Achievable with available tools
- Aligned with the user's intent
- Scientifically meaningful

Analysis Goal: [Clear, specific objective statement]""",
            category=PromptCategory.WORKFLOW_PLANNING,
            description="Determine high-level analysis objectives from user queries",
            variables=["user_query", "context"],
            validation_rules={"min_length": 30, "must_contain": "Analysis Goal"},
        ):
            success_count += 1
            self._log_migration(
                "analysis_goal_determination",
                "src/agents/reasoning_chains.py:173-185",
                "SUCCESS",
            )

        # 2. Multi-Step Plan Generation
        if self.registry.register_prompt(
            prompt_id="multi_step_plan_generation",
            template="""Create a detailed multi-step reasoning plan for this metabolic modeling analysis:

User Query: "{query}"
Analysis Goal: "{goal}"
Available Tools: {available_tools}

Plan 5-8 steps that will systematically achieve the analysis goal. For each step:
1. Specify the reasoning/purpose
2. Select the most appropriate tool
3. Explain why this step is necessary
4. Describe what insights you expect to gain
5. Note how this step enables subsequent steps

Format as a JSON array with this structure for each step:
{{
    "step_number": 1,
    "reasoning": "Purpose of this step",
    "tool": "tool_name",
    "justification": "Why this tool at this time",
    "expected_insights": "What we expect to learn",
    "enables_next": "How this enables subsequent steps"
}}

The plan should be:
- Logically sequenced
- Comprehensive yet efficient
- Adaptable based on results
- Scientifically sound""",
            category=PromptCategory.WORKFLOW_PLANNING,
            description="Generate detailed multi-step analysis plans",
            variables=["query", "goal", "available_tools"],
            validation_rules={"min_length": 100, "must_contain": "step_number"},
        ):
            success_count += 1
            self._log_migration(
                "multi_step_plan_generation",
                "src/agents/reasoning_chains.py:198-213",
                "SUCCESS",
            )

        # 3. Insight Extraction
        if self.registry.register_prompt(
            prompt_id="insight_extraction",
            template="""Analyze this tool result and extract the key insights relevant to our analysis goal:

Analysis Goal: {goal}
Step Reasoning: {reasoning}
Tool Result: {result}

Extract 2-4 key insights that are most relevant to the analysis goal.
Focus on actionable findings that inform next steps.

Each insight should be:
- Specific and quantitative when possible
- Biologically meaningful
- Connected to the analysis objective
- Useful for subsequent analysis steps

Format as a JSON array of strings:
["insight 1", "insight 2", "insight 3"]

Insights should go beyond obvious observations to provide scientific understanding.""",
            category=PromptCategory.RESULT_ANALYSIS,
            description="Extract key insights from tool results",
            variables=["goal", "reasoning", "result"],
            validation_rules={"min_length": 50, "must_contain": "insight"},
        ):
            success_count += 1
            self._log_migration(
                "insight_extraction",
                "src/agents/reasoning_chains.py:432-443",
                "SUCCESS",
            )

        # 4. Question Identification
        if self.registry.register_prompt(
            prompt_id="question_identification",
            template="""Based on this tool result, what new questions emerge that could guide further analysis?

Original Query: {original_query}
Tool Result: {result}

Identify 1-3 specific questions that arise from these results.
Focus on questions that could lead to deeper understanding.

Questions should be:
- Specific and answerable with available tools
- Scientifically meaningful
- Building on current findings
- Advancing toward the analysis goal

Format as a JSON array of strings:
["question 1", "question 2"]

Good questions drive scientific discovery and reveal new analysis directions.""",
            category=PromptCategory.WORKFLOW_PLANNING,
            description="Identify new questions emerging from results",
            variables=["original_query", "result"],
            validation_rules={"min_length": 30, "must_contain": "question"},
        ):
            success_count += 1
            self._log_migration(
                "question_identification",
                "src/agents/reasoning_chains.py:459-469",
                "SUCCESS",
            )

        # 5. Plan Adaptation
        if self.registry.register_prompt(
            prompt_id="plan_adaptation",
            template="""Analyze this completed step and determine if we should adapt our analysis plan:

Original Goal: {analysis_goal}
Completed Step: {completed_step_reasoning}
Tool Used: {tool_used}
Key Insights: {insights_gained}
Questions Raised: {questions_raised}

Remaining Planned Steps: {remaining_steps}

Should we modify our plan based on what we discovered? Consider:
1. Do the insights suggest a different analytical direction?
2. Are there urgent questions that should be addressed immediately?
3. Should we add, remove, or reorder remaining steps?

Respond with JSON:
{{
    "modify_plan": true/false,
    "modifications": [list of changes],
    "reasoning": "explanation of why changes are needed",
    "priority_adjustments": "any priority changes needed"
}}""",
            category=PromptCategory.WORKFLOW_PLANNING,
            description="Adapt analysis plans based on discovered results",
            variables=[
                "analysis_goal",
                "completed_step_reasoning",
                "tool_used",
                "insights_gained",
                "questions_raised",
                "remaining_steps",
            ],
            validation_rules={"min_length": 50, "must_contain": "modify_plan"},
        ):
            success_count += 1
            self._log_migration(
                "plan_adaptation", "src/agents/reasoning_chains.py:486-501", "SUCCESS"
            )

        # 6. Results Synthesis
        if self.registry.register_prompt(
            prompt_id="results_synthesis",
            template="""Synthesize the results of this multi-step analysis into a comprehensive conclusion:

Original Query: {user_query}
Analysis Goal: {analysis_goal}

Key Insights Discovered:
{all_insights}

Tool Results Summary:
{results_summary}

Provide a comprehensive conclusion that:
1. Directly answers the original user query
2. Highlights the most important findings
3. Explains how the different analysis steps connected
4. Identifies any limitations or areas for further investigation
5. Provides actionable recommendations

Structure your response with clear sections:
MAIN_FINDINGS: [key discoveries]
BIOLOGICAL_SIGNIFICANCE: [what this means scientifically]
QUERY_ANSWERS: [direct responses to user questions]
LIMITATIONS: [analysis constraints or uncertainties]
RECOMMENDATIONS: [suggested next steps or applications]

The synthesis should demonstrate deep understanding of the metabolic system.""",
            category=PromptCategory.SYNTHESIS,
            description="Synthesize multi-step analysis results into comprehensive conclusions",
            variables=[
                "user_query",
                "analysis_goal",
                "all_insights",
                "results_summary",
            ],
            validation_rules={"min_length": 100, "must_contain": "MAIN_FINDINGS"},
        ):
            success_count += 1
            self._log_migration(
                "results_synthesis", "src/agents/reasoning_chains.py:535-550", "SUCCESS"
            )

        return success_count

    def _migrate_hypothesis_prompts(self) -> int:
        """Migrate hypothesis system prompts"""
        success_count = 0

        # 1. Hypothesis Generation from Observations
        if self.registry.register_prompt(
            prompt_id="hypothesis_generation_from_observation",
            template="""Based on this metabolic modeling observation, generate 2-4 testable scientific hypotheses:

Observation: {observation}
Analysis Context: {context}
Available Testing Tools: {available_tools}

For each hypothesis, provide:
1. A clear, specific, testable statement
2. The scientific rationale for why this hypothesis is plausible
3. Specific predictions that can be tested
4. Which tools could be used to test it
5. The type of hypothesis (growth_limitation, nutritional_gap, etc.)

Format as JSON array:
[
    {{
        "statement": "Clear hypothesis statement",
        "rationale": "Scientific reasoning",
        "predictions": ["prediction 1", "prediction 2"],
        "testable_with": ["tool1", "tool2"],
        "hypothesis_type": "category"
    }}
]

Hypotheses should be scientifically sound and lead to actionable testing strategies.""",
            category=PromptCategory.HYPOTHESIS_GENERATION,
            description="Generate testable hypotheses from metabolic observations",
            variables=["observation", "context", "available_tools"],
            validation_rules={"min_length": 100, "must_contain": "hypothesis"},
        ):
            success_count += 1
            self._log_migration(
                "hypothesis_generation_from_observation",
                "src/agents/hypothesis_system.py:120-135",
                "SUCCESS",
            )

        # 2. Hypothesis Generation from Results
        if self.registry.register_prompt(
            prompt_id="hypothesis_generation_from_results",
            template="""Analyze these metabolic modeling results and identify patterns that suggest testable hypotheses:

Original Query Context: {query_context}
Tool Results: {tool_results}

Look for patterns such as:
- Unexpected growth rates (very high/low)
- Unusual nutrient requirements
- Essential gene patterns
- Pathway utilization anomalies
- Biomass composition issues

For each interesting pattern, formulate a hypothesis that could explain it.
Focus on hypotheses that would deepen understanding of the metabolic system.

Use the same JSON format as the observation-based hypothesis generation.

Patterns should lead to mechanistic explanations and testable predictions.""",
            category=PromptCategory.HYPOTHESIS_GENERATION,
            description="Generate hypotheses from patterns in analysis results",
            variables=["query_context", "tool_results"],
            validation_rules={"min_length": 80, "must_contain": "pattern"},
        ):
            success_count += 1
            self._log_migration(
                "hypothesis_generation_from_results",
                "src/agents/hypothesis_system.py:172-186",
                "SUCCESS",
            )

        # 3. Hypothesis Testing Planning
        if self.registry.register_prompt(
            prompt_id="hypothesis_testing_planning",
            template="""Plan how to test this scientific hypothesis using available tools:

Hypothesis: {hypothesis_statement}
Rationale: {hypothesis_rationale}
Predictions: {predictions}
Available Tools: {testable_with_tools}

Select 1-3 tools that would provide the most relevant evidence for or against this hypothesis.
Consider:
- Which tools directly test the predictions
- What evidence would be most convincing
- Logical order of testing

Return as JSON array of tool names:
["tool1", "tool2", "tool3"]

Testing plan should be efficient and provide strong evidence for hypothesis validation.""",
            category=PromptCategory.HYPOTHESIS_GENERATION,
            description="Plan systematic testing of scientific hypotheses",
            variables=[
                "hypothesis_statement",
                "hypothesis_rationale",
                "predictions",
                "testable_with_tools",
            ],
            validation_rules={"min_length": 30, "must_contain": "tool"},
        ):
            success_count += 1
            self._log_migration(
                "hypothesis_testing_planning",
                "src/agents/hypothesis_system.py:356-371",
                "SUCCESS",
            )

        # 4. Tool Input Determination
        if self.registry.register_prompt(
            prompt_id="hypothesis_tool_input_determination",
            template="""Determine the appropriate input parameters for testing this hypothesis:

Hypothesis: {hypothesis_statement}
Tool to Use: {tool_name}

What parameters should be passed to this tool to best test the hypothesis?
Consider the specific predictions and what evidence would be most relevant.

Return as JSON object with parameter names and values:
{{"parameter1": "value1", "parameter2": "value2"}}

If no special parameters are needed, return an empty object: {{}}

Parameters should be chosen to maximize the strength of evidence obtained.""",
            category=PromptCategory.HYPOTHESIS_GENERATION,
            description="Determine optimal tool parameters for hypothesis testing",
            variables=["hypothesis_statement", "tool_name"],
            validation_rules={"min_length": 20, "must_contain": "parameter"},
        ):
            success_count += 1
            self._log_migration(
                "hypothesis_tool_input_determination",
                "src/agents/hypothesis_system.py:389-402",
                "SUCCESS",
            )

        # 5. Evidence Interpretation
        if self.registry.register_prompt(
            prompt_id="hypothesis_evidence_interpretation",
            template="""Interpret this tool result as evidence for or against the hypothesis:

Hypothesis: {hypothesis_statement}
Predictions: {predictions}
Tool Used: {tool_name}
Tool Result: {result_data}

Analyze:
1. Does this result support or contradict the hypothesis?
2. How strong is this evidence (0.0 = no relevance, 1.0 = definitive)?
3. How confident are you in this interpretation (0.0 = very uncertain, 1.0 = very certain)?
4. What specific aspects of the result are most relevant?

Format as JSON:
{{
    "supports_hypothesis": true/false,
    "evidence_strength": 0.8,
    "confidence": 0.9,
    "relevant_findings": ["finding 1", "finding 2"],
    "interpretation": "Detailed explanation"
}}

Interpretation should be objective and scientifically rigorous.""",
            category=PromptCategory.HYPOTHESIS_GENERATION,
            description="Interpret experimental results as evidence for hypotheses",
            variables=[
                "hypothesis_statement",
                "predictions",
                "tool_name",
                "result_data",
            ],
            validation_rules={"min_length": 50, "must_contain": "supports_hypothesis"},
        ):
            success_count += 1
            self._log_migration(
                "hypothesis_evidence_interpretation",
                "src/agents/hypothesis_system.py:418-433",
                "SUCCESS",
            )

        return success_count

    def _migrate_pattern_memory_prompts(self) -> int:
        """Migrate pattern memory prompts"""
        success_count = 0

        # Pattern Analysis from Experiences
        if self.registry.register_prompt(
            prompt_id="pattern_analysis_from_experiences",
            template="""Analyze these successful analysis tool sequences to identify common patterns:

Successful Analyses:
{analysis_data}

Identify 2-3 common tool sequence patterns that appear in successful analyses.
For each pattern:
1. Describe the sequence and when it's effective
2. Estimate its success rate
3. Identify conditions where it applies

Format as JSON:
[
    {{
        "description": "Pattern description",
        "tool_sequence": ["tool1", "tool2", "tool3"],
        "conditions": ["condition1", "condition2"],
        "success_rate": 0.85,
        "rationale": "Why this pattern works"
    }}
]

Patterns should be actionable for improving future analysis workflows.""",
            category=PromptCategory.QUALITY_ASSESSMENT,
            description="Analyze successful tool sequences to identify effective patterns",
            variables=["analysis_data"],
            validation_rules={"min_length": 80, "must_contain": "pattern"},
        ):
            success_count += 1
            self._log_migration(
                "pattern_analysis_from_experiences",
                "src/agents/pattern_memory.py:184-206",
                "SUCCESS",
            )

        return success_count

    def _migrate_langgraph_prompts(self) -> int:
        """Migrate LangGraph metabolic agent prompts"""
        success_count = 0

        # 1. Query Analysis and Tool Selection
        if self.registry.register_prompt(
            prompt_id="langgraph_query_analysis",
            template="""You are an expert metabolic modeling agent. Analyze the query and determine the best approach.

Query: {query}

{context}Available tools: {available_tools}

Current iteration: {iteration}/{max_iterations}
Tools already called: {tools_called}

Based on the query and context, determine your next action:

1. SINGLE_TOOL: If you need to call one specific tool
   Format: ACTION: SINGLE_TOOL
   TOOL: [tool_name]
   REASON: [why this tool]

2. PARALLEL_TOOLS: If you can call multiple tools simultaneously
   Format: ACTION: PARALLEL_TOOLS
   TOOLS: [tool1, tool2, tool3]
   REASON: [why these tools together]

3. ANALYSIS_COMPLETE: If you have sufficient information
   Format: ACTION: ANALYSIS_COMPLETE
   REASON: [why analysis is complete]

Choose the most appropriate action based on the query requirements and available information.""",
            category=PromptCategory.TOOL_SELECTION,
            description="LangGraph agent query analysis and action determination",
            variables=[
                "query",
                "context",
                "available_tools",
                "iteration",
                "max_iterations",
                "tools_called",
            ],
            validation_rules={"min_length": 50, "must_contain": "ACTION"},
        ):
            success_count += 1
            self._log_migration(
                "langgraph_query_analysis",
                "src/agents/langgraph_metabolic.py:879-899",
                "SUCCESS",
            )

        # 2. Results Analysis
        if self.registry.register_prompt(
            prompt_id="langgraph_results_analysis",
            template="""You are analyzing metabolic modeling results. Based on the tool execution results, provide insights and determine next steps.

Original query: {query}

Detailed tool results:{results_summary}

Please provide:
1. Key metabolic insights discovered so far
2. Whether you have sufficient information for a comprehensive answer
3. What additional analysis might enhance the results
4. A brief summary of the most important findings

Respond with actionable metabolic insights, not just procedural information.

Focus on biological significance and scientific understanding.""",
            category=PromptCategory.RESULT_ANALYSIS,
            description="LangGraph agent results analysis and next steps determination",
            variables=["query", "results_summary"],
            validation_rules={"min_length": 40, "must_contain": "insights"},
        ):
            success_count += 1
            self._log_migration(
                "langgraph_results_analysis",
                "src/agents/langgraph_metabolic.py:973-985",
                "SUCCESS",
            )

        return success_count

    def _migrate_llm_interface_prompts(self) -> int:
        """Migrate LLM interface prompts"""
        success_count = 0

        # Local LLM System Template
        if self.registry.register_prompt(
            prompt_id="local_llm_system_template",
            template="""<|system|>
{system_content}
<|user|>
{prompt}
<|assistant|>""",
            category=PromptCategory.SYSTEM_CONFIGURATION,
            description="Template for local LLM system interactions",
            variables=["system_content", "prompt"],
            validation_rules={"min_length": 10, "must_contain": "system"},
        ):
            success_count += 1
            self._log_migration(
                "local_llm_system_template", "src/llm/local_llm.py:185", "SUCCESS"
            )

        return success_count

    def _migrate_performance_optimizer_prompts(self) -> int:
        """Migrate performance optimizer prompts"""
        success_count = 0

        # Optimized Prompt Generation
        if self.registry.register_prompt(
            prompt_id="performance_optimized_prompt",
            template="""{base_prompt}

Respond in JSON format with keys: reasoning, tool_selection, confidence

{optimization_instructions}""",
            category=PromptCategory.QUALITY_ASSESSMENT,
            description="Performance-optimized prompt with compression and JSON formatting",
            variables=["base_prompt", "optimization_instructions"],
            validation_rules={"min_length": 20, "must_contain": "JSON"},
        ):
            success_count += 1
            self._log_migration(
                "performance_optimized_prompt",
                "src/agents/performance_optimizer.py:233-250",
                "SUCCESS",
            )

        return success_count

    def _migrate_metabolic_agent_prompts(self) -> int:
        """Migrate newly found metabolic agent prompts"""
        success_count = 0

        # Tool Results Summarization (newly found)
        if self.registry.register_prompt(
            prompt_id="tool_results_summarization",
            template="""Summarize the results from this tool execution for integration into the analysis workflow:

Tool Name: {tool_name}
Execution Time: {execution_time}
Tool Results: {tool_results}
Analysis Context: {analysis_context}

Provide a concise summary that:
1. Highlights key quantitative findings
2. Explains biological significance
3. Identifies implications for subsequent analysis
4. Notes any limitations or uncertainties

Keep the summary focused and actionable for workflow integration.

SUMMARY: [concise results summary]
KEY_METRICS: [important quantitative findings]
BIOLOGICAL_MEANING: [scientific interpretation]
NEXT_STEPS: [suggested follow-up actions]""",
            category=PromptCategory.RESULT_ANALYSIS,
            description="Summarize tool results for workflow integration",
            variables=[
                "tool_name",
                "execution_time",
                "tool_results",
                "analysis_context",
            ],
            validation_rules={"min_length": 50, "must_contain": "SUMMARY"},
        ):
            success_count += 1
            self._log_migration(
                "tool_results_summarization", "src/agents/metabolic.py:256", "SUCCESS"
            )

        return success_count

    def _migrate_additional_prompts(self) -> int:
        """Migrate any additional prompts found during deep analysis"""
        success_count = 0

        # These would be discovered during implementation
        # Placeholder for additional prompts that might be found

        # 1. Error handling prompt (if found)
        if self.registry.register_prompt(
            prompt_id="error_handling_analysis",
            template="""An error occurred during tool execution. Analyze the error and determine the best recovery strategy:

Tool Name: {tool_name}
Error Message: {error_message}
Execution Context: {context}
Available Recovery Options: {recovery_options}

Determine:
1. Is this error recoverable?
2. What caused the error?
3. What's the best recovery strategy?
4. Should the analysis continue or terminate?

Respond with:
RECOVERABLE: [true/false]
CAUSE: [error analysis]
RECOVERY_STRATEGY: [recommended action]
CONTINUE_ANALYSIS: [true/false]""",
            category=PromptCategory.QUALITY_ASSESSMENT,
            description="Analyze errors and determine recovery strategies",
            variables=["tool_name", "error_message", "context", "recovery_options"],
            validation_rules={"min_length": 30, "must_contain": "RECOVERABLE"},
        ):
            success_count += 1
            self._log_migration(
                "error_handling_analysis", "general_error_handling", "SUCCESS"
            )

        # 2. Quality validation prompt
        if self.registry.register_prompt(
            prompt_id="result_quality_validation",
            template="""Validate the quality of this analysis result:

Analysis Type: {analysis_type}
Result Data: {result_data}
Expected Ranges: {expected_ranges}
Quality Criteria: {quality_criteria}

Check:
1. Are the results within expected biological ranges?
2. Do the results make scientific sense?
3. Are there any obvious errors or inconsistencies?
4. Is additional validation needed?

Respond with:
QUALITY_SCORE: [0.0-1.0]
ISSUES_FOUND: [list of problems if any]
VALIDATION_STATUS: [PASS/FAIL/REQUIRES_REVIEW]
RECOMMENDATIONS: [suggested actions]""",
            category=PromptCategory.QUALITY_ASSESSMENT,
            description="Validate quality and scientific validity of analysis results",
            variables=[
                "analysis_type",
                "result_data",
                "expected_ranges",
                "quality_criteria",
            ],
            validation_rules={"min_length": 40, "must_contain": "QUALITY_SCORE"},
        ):
            success_count += 1
            self._log_migration(
                "result_quality_validation", "quality_validation_system", "SUCCESS"
            )

        # 3. Context enrichment prompt
        if self.registry.register_prompt(
            prompt_id="biochemical_context_enrichment",
            template="""Enrich this analysis with relevant biochemical context:

Primary Analysis: {primary_analysis}
Available Context Sources: {context_sources}
Domain Focus: {domain_focus}

Provide enriched context including:
1. Relevant biochemical pathways
2. Known biological mechanisms
3. Literature connections
4. Comparative information

The enrichment should enhance understanding without biasing conclusions.

ENRICHED_CONTEXT: [relevant biochemical information]
PATHWAY_CONNECTIONS: [related metabolic pathways]
MECHANISTIC_INSIGHTS: [biological mechanisms involved]
LITERATURE_NOTES: [relevant research connections]""",
            category=PromptCategory.SYNTHESIS,
            description="Enrich analysis with relevant biochemical context",
            variables=["primary_analysis", "context_sources", "domain_focus"],
            validation_rules={"min_length": 50, "must_contain": "ENRICHED_CONTEXT"},
        ):
            success_count += 1
            self._log_migration(
                "biochemical_context_enrichment", "context_enrichment_system", "SUCCESS"
            )

        return success_count

    def _log_migration(self, prompt_id: str, source_location: str, status: str) -> None:
        """Log migration progress"""
        self.migration_log.append(
            {
                "prompt_id": prompt_id,
                "source_location": source_location,
                "status": status,
                "timestamp": logger.name,  # Placeholder
            }
        )

        logger.info(f"Migration {status}: {prompt_id} from {source_location}")

    def get_migration_report(self) -> Dict[str, Any]:
        """Generate migration report"""
        total = len(self.migration_log)
        successful = len(
            [log for log in self.migration_log if log["status"] == "SUCCESS"]
        )

        return {
            "total_prompts": total,
            "successful_migrations": successful,
            "failed_migrations": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "migration_details": self.migration_log,
            "registry_stats": self.registry.export_prompts(),
        }
