"""
Enhanced Prompt Provider for ModelSEEDagent

Integrates biochemical context enhancement and reasoning frameworks
with the centralized prompt registry for intelligent prompt generation.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..prompts.prompt_registry import PromptCategory, PromptRegistry
    from .context_enhancer import get_context_enhancer, get_context_memory
    from .frameworks import (
        BiochemicalReasoningFramework,
        GrowthAnalysisFramework,
        MediaOptimizationFramework,
        PathwayAnalysisFramework,
    )
    from .frameworks.biochemical_reasoning import (
        BiochemicalQuestionType,
        ReasoningDepth,
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from prompts.prompt_registry import PromptCategory, PromptRegistry
    from reasoning.context_enhancer import get_context_enhancer, get_context_memory
    from reasoning.frameworks import (
        BiochemicalReasoningFramework,
        GrowthAnalysisFramework,
        MediaOptimizationFramework,
        PathwayAnalysisFramework,
    )
    from reasoning.frameworks.biochemical_reasoning import (
        BiochemicalQuestionType,
        ReasoningDepth,
    )

logger = logging.getLogger(__name__)


class EnhancedPromptProvider:
    """
    Enhanced prompt provider that combines context enhancement
    with reasoning frameworks for intelligent prompt generation
    """

    def __init__(self):
        self.prompt_registry = PromptRegistry()
        self.context_enhancer = get_context_enhancer()
        self.context_memory = get_context_memory()

        # Initialize reasoning frameworks
        self.biochemical_framework = BiochemicalReasoningFramework()
        self.growth_framework = GrowthAnalysisFramework()
        self.pathway_framework = PathwayAnalysisFramework()
        self.media_framework = MediaOptimizationFramework()

        # Register enhanced prompts
        self._register_enhanced_prompts()

    def get_enhanced_prompt(
        self,
        prompt_id: str,
        context_data: Dict[str, Any],
        tool_name: Optional[str] = None,
        reasoning_depth: ReasoningDepth = ReasoningDepth.INTERMEDIATE,
    ) -> str:
        """
        Get an enhanced prompt with biochemical context and reasoning guidance

        Args:
            prompt_id: Base prompt identifier
            context_data: Tool results or analysis context
            tool_name: Name of tool that generated context
            reasoning_depth: Desired depth of reasoning

        Returns:
            Enhanced prompt with context and reasoning guidance
        """
        try:
            # Get base prompt from registry
            base_prompt = self.prompt_registry.get_prompt(prompt_id)
            if not base_prompt:
                logger.warning(
                    f"Prompt {prompt_id} not found, using generic analysis prompt"
                )
                base_prompt = self._get_generic_analysis_prompt()

            # Enhance context data
            enhanced_context = self.context_enhancer.enhance_tool_result(
                tool_name or "unknown", context_data
            )

            # Generate reasoning questions and guidance
            reasoning_questions = self._generate_reasoning_questions(
                enhanced_context, reasoning_depth
            )
            reasoning_prompts = self._get_reasoning_prompts(
                enhanced_context, reasoning_depth
            )

            # Build enhanced prompt
            enhanced_prompt = self._build_enhanced_prompt(
                base_prompt,
                enhanced_context,
                reasoning_questions,
                reasoning_prompts,
                reasoning_depth,
            )

            # Update context memory
            self._update_context_memory(enhanced_context)

            # Track usage
            self.prompt_registry.track_prompt_usage(prompt_id, enhanced_prompt)

            return enhanced_prompt

        except Exception as e:
            logger.error(f"Failed to generate enhanced prompt: {e}")
            return base_prompt or "Analyze the provided data and explain your findings."

    def get_tool_specific_prompt(
        self,
        tool_name: str,
        tool_result: Dict[str, Any],
        analysis_goal: str = "comprehensive",
    ) -> str:
        """
        Get tool-specific enhanced prompt

        Args:
            tool_name: Name of the tool
            tool_result: Tool execution result
            analysis_goal: Analysis objective (comprehensive, focused, optimization)

        Returns:
            Tool-specific enhanced prompt
        """
        # Map tool to appropriate prompt category and framework
        prompt_id, framework = self._map_tool_to_framework(tool_name, analysis_goal)

        # Enhance context
        enhanced_context = self.context_enhancer.enhance_tool_result(
            tool_name, tool_result
        )

        # Get framework-specific prompts
        if framework == "growth":
            framework_prompts = self.growth_framework.get_growth_reasoning_prompts(
                enhanced_context
            )
        elif framework == "pathway":
            framework_prompts = self.pathway_framework.get_pathway_reasoning_prompts(
                enhanced_context
            )
        elif framework == "media":
            framework_prompts = self.media_framework.get_media_reasoning_prompts(
                enhanced_context
            )
        else:
            framework_prompts = self.biochemical_framework.get_reasoning_prompts(
                BiochemicalQuestionType.WHAT_IS_HAPPENING, enhanced_context
            )

        # Build specialized prompt
        specialized_prompt = self._build_tool_specific_prompt(
            prompt_id, enhanced_context, framework_prompts, analysis_goal
        )

        return specialized_prompt

    def get_synthesis_prompt(
        self, tool_results: List[Dict[str, Any]], analysis_focus: str = "comprehensive"
    ) -> str:
        """
        Get synthesis prompt for multiple tool results

        Args:
            tool_results: List of tool execution results
            analysis_focus: Focus area for synthesis

        Returns:
            Enhanced synthesis prompt
        """
        # Enhance all tool results
        enhanced_results = []
        for tool_result in tool_results:
            tool_name = tool_result.get("tool_name", "unknown")
            enhanced_result = self.context_enhancer.enhance_tool_result(
                tool_name, tool_result
            )
            enhanced_results.append(enhanced_result)

        # Generate cross-tool reasoning questions
        synthesis_questions = self._generate_synthesis_questions(enhanced_results)

        # Get session context for synthesis
        session_context = self.context_memory.get_context_for_reasoning()

        # Build synthesis prompt
        synthesis_prompt = self._build_synthesis_prompt(
            enhanced_results, synthesis_questions, session_context, analysis_focus
        )

        return synthesis_prompt

    def _register_enhanced_prompts(self):
        """Register enhanced prompt templates"""
        # Context-enhanced analysis prompt
        self.prompt_registry.register_prompt(
            prompt_id="enhanced_analysis",
            template="""
            Analyze the following biochemical data with enhanced context and reasoning:

            {context_data}

            Biochemical Context:
            {biochemical_context}

            Reasoning Guidance:
            {reasoning_questions}

            Provide a detailed analysis following these reasoning patterns:
            {reasoning_prompts}

            Focus on mechanistic insights and biological significance.
            """,
            category=PromptCategory.RESULT_ANALYSIS,
            description="Enhanced analysis prompt with biochemical context",
            variables=[
                "context_data",
                "biochemical_context",
                "reasoning_questions",
                "reasoning_prompts",
            ],
        )

        # Growth analysis specific prompt
        self.prompt_registry.register_prompt(
            prompt_id="growth_analysis_enhanced",
            template="""
            Perform comprehensive growth analysis with the following data:

            {growth_data}

            Enhanced Context:
            {enhanced_context}

            Growth-Specific Analysis Questions:
            {growth_questions}

            Reasoning Framework:
            {growth_prompts}

            Provide insights into:
            1. Growth performance and limitations
            2. Metabolic efficiency and strategy
            3. Nutrient utilization patterns
            4. Optimization opportunities

            Support your analysis with mechanistic reasoning and quantitative evidence.
            """,
            category=PromptCategory.RESULT_ANALYSIS,
            description="Growth analysis with enhanced context and reasoning",
            variables=[
                "growth_data",
                "enhanced_context",
                "growth_questions",
                "growth_prompts",
            ],
        )

        # Pathway analysis specific prompt
        self.prompt_registry.register_prompt(
            prompt_id="pathway_analysis_enhanced",
            template="""
            Analyze metabolic pathway activity with enhanced context:

            {pathway_data}

            Pathway Context Enhancement:
            {pathway_context}

            Analysis Framework:
            {pathway_questions}

            Reasoning Guidance:
            {pathway_prompts}

            Focus on:
            1. Pathway activity patterns and coordination
            2. Regulatory mechanisms and control points
            3. Metabolic strategy and adaptation
            4. Cross-pathway interactions and trade-offs

            Explain the biochemical basis for observed patterns.
            """,
            category=PromptCategory.RESULT_ANALYSIS,
            description="Pathway analysis with enhanced reasoning framework",
            variables=[
                "pathway_data",
                "pathway_context",
                "pathway_questions",
                "pathway_prompts",
            ],
        )

        # Enhanced synthesis prompt
        self.prompt_registry.register_prompt(
            prompt_id="enhanced_synthesis",
            template="""
            Synthesize insights from multiple analyses with enhanced context:

            Multiple Analysis Results:
            {multiple_results}

            Cross-Tool Context:
            {cross_tool_context}

            Session Context:
            {session_context}

            Synthesis Questions:
            {synthesis_questions}

            Integrate findings to provide:
            1. Unified biological interpretation
            2. Cross-tool validation and consistency
            3. Systems-level insights
            4. Testable hypotheses
            5. Practical implications

            Explain how different analyses support or contradict each other.
            """,
            category=PromptCategory.SYNTHESIS,
            description="Enhanced multi-tool synthesis with context integration",
            variables=[
                "multiple_results",
                "cross_tool_context",
                "session_context",
                "synthesis_questions",
            ],
        )

    def _generate_reasoning_questions(
        self, enhanced_context: Dict[str, Any], depth: ReasoningDepth
    ) -> List[str]:
        """Generate reasoning questions based on context and depth"""
        questions = []

        # Get framework-specific questions
        questions.extend(
            self.biochemical_framework.generate_reasoning_questions(
                enhanced_context, depth
            )
        )

        # Add context-specific questions based on data type
        if "growth_rate" in enhanced_context:
            questions.extend(
                self.growth_framework.generate_reasoning_questions(
                    {"growth_data": enhanced_context}
                )
            )

        if "pathway_analysis" in enhanced_context or "fluxes" in enhanced_context:
            questions.extend(
                self.pathway_framework._generate_pathway_questions(enhanced_context)
            )

        if "media_components" in enhanced_context:
            questions.extend(
                self.media_framework._generate_media_questions(enhanced_context)
            )

        return questions[:8]  # Limit to top 8 questions

    def _get_reasoning_prompts(
        self, enhanced_context: Dict[str, Any], depth: ReasoningDepth
    ) -> Dict[str, str]:
        """Get reasoning prompts based on context and depth"""
        prompts = {}

        # Get biochemical reasoning prompts
        if "fluxes" in enhanced_context:
            prompts.update(
                self.biochemical_framework.get_reasoning_prompts(
                    BiochemicalQuestionType.HOW_DOES_IT_WORK, enhanced_context
                )
            )

        # Add framework-specific prompts
        if "growth_rate" in enhanced_context:
            prompts.update(
                self.growth_framework.get_growth_reasoning_prompts(
                    enhanced_context, depth
                )
            )

        return prompts

    def _build_enhanced_prompt(
        self,
        base_prompt: str,
        enhanced_context: Dict[str, Any],
        reasoning_questions: List[str],
        reasoning_prompts: Dict[str, str],
        depth: ReasoningDepth,
    ) -> str:
        """Build the final enhanced prompt"""

        # Extract key context information
        context_summary = enhanced_context.get("_context_summary", {})
        context_notes = context_summary.get("context_notes", [])

        # Format biochemical context
        biochemical_context = "\n".join([f"- {note}" for note in context_notes])

        # Format reasoning questions
        formatted_questions = "\n".join(
            [f"{i+1}. {question}" for i, question in enumerate(reasoning_questions[:5])]
        )

        # Format reasoning prompts
        formatted_prompts = "\n".join(
            [f"**{key}**: {value}" for key, value in reasoning_prompts.items()]
        )

        # Build enhanced prompt
        enhanced_prompt = f"""
{base_prompt}

=== ENHANCED BIOCHEMICAL CONTEXT ===
{biochemical_context}

=== REASONING GUIDANCE (Depth: {depth.value}) ===
Consider these questions in your analysis:
{formatted_questions}

=== ANALYSIS FRAMEWORK ===
{formatted_prompts}

=== CONTEXT DATA ===
{json.dumps(enhanced_context, indent=2, default=str)}

Please provide a comprehensive analysis that addresses the reasoning questions and follows the analysis framework. Focus on mechanistic insights and biological significance.
"""

        return enhanced_prompt

    def _map_tool_to_framework(
        self, tool_name: str, analysis_goal: str
    ) -> Tuple[str, str]:
        """Map tool name to appropriate prompt and framework"""
        if tool_name in ["run_metabolic_fba", "run_cobra_fba"]:
            if "growth" in analysis_goal:
                return "growth_analysis_enhanced", "growth"
            else:
                return "pathway_analysis_enhanced", "pathway"

        elif tool_name in ["run_flux_variability_analysis", "run_flux_sampling"]:
            return "pathway_analysis_enhanced", "pathway"

        elif "media" in tool_name.lower():
            return "media_analysis_enhanced", "media"

        else:
            return "enhanced_analysis", "biochemical"

    def _build_tool_specific_prompt(
        self,
        prompt_id: str,
        enhanced_context: Dict[str, Any],
        framework_prompts: Dict[str, str],
        analysis_goal: str,
    ) -> str:
        """Build tool-specific enhanced prompt"""

        base_prompt = self.prompt_registry.get_prompt(prompt_id)

        # Format framework prompts
        formatted_framework = "\n".join(
            [f"**{key}**: {value}" for key, value in framework_prompts.items()]
        )

        # Build tool-specific prompt
        tool_prompt = f"""
{base_prompt}

=== TOOL-SPECIFIC ANALYSIS FRAMEWORK ===
{formatted_framework}

=== ENHANCED CONTEXT DATA ===
{json.dumps(enhanced_context, indent=2, default=str)}

Analysis Goal: {analysis_goal}

Provide detailed analysis following the framework above, focusing on biological mechanisms and practical insights.
"""

        return tool_prompt

    def _generate_synthesis_questions(
        self, enhanced_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate questions for cross-tool synthesis"""
        questions = [
            "How do the results from different tools support or contradict each other?",
            "What unified biological story emerges from these analyses?",
            "What systems-level insights can be drawn from the integrated results?",
            "What testable hypotheses arise from combining these findings?",
            "How do the different analyses validate or extend each other?",
        ]

        # Add context-specific synthesis questions
        tool_types = set()
        for result in enhanced_results:
            tool_name = result.get("tool_name", "unknown")
            if "fba" in tool_name.lower():
                tool_types.add("growth_analysis")
            elif "flux" in tool_name.lower():
                tool_types.add("flux_analysis")
            elif "media" in tool_name.lower():
                tool_types.add("media_analysis")

        if "growth_analysis" in tool_types and "flux_analysis" in tool_types:
            questions.append(
                "How do flux patterns explain the observed growth performance?"
            )

        if "media_analysis" in tool_types and "growth_analysis" in tool_types:
            questions.append("How does media composition relate to growth limitations?")

        return questions

    def _build_synthesis_prompt(
        self,
        enhanced_results: List[Dict[str, Any]],
        synthesis_questions: List[str],
        session_context: str,
        analysis_focus: str,
    ) -> str:
        """Build comprehensive synthesis prompt"""

        # Format results summary
        results_summary = []
        for i, result in enumerate(enhanced_results[:5]):  # Limit to 5 results
            tool_name = result.get("tool_name", f"Analysis {i+1}")
            results_summary.append(f"**{tool_name}**: {self._summarize_result(result)}")

        formatted_results = "\n".join(results_summary)

        # Format synthesis questions
        formatted_questions = "\n".join(
            [f"{i+1}. {question}" for i, question in enumerate(synthesis_questions)]
        )

        synthesis_prompt = f"""
=== MULTI-TOOL SYNTHESIS ANALYSIS ===

Analysis Focus: {analysis_focus}

=== RESULTS SUMMARY ===
{formatted_results}

=== SESSION CONTEXT ===
{session_context}

=== SYNTHESIS QUESTIONS ===
{formatted_questions}

=== DETAILED RESULTS ===
{json.dumps(enhanced_results, indent=2, default=str)}

Please provide a comprehensive synthesis that:
1. Integrates findings from all analyses
2. Identifies consistent patterns and potential contradictions
3. Explains biological mechanisms underlying the observations
4. Generates testable hypotheses for further investigation
5. Suggests practical applications or next steps

Focus on systems-level understanding and mechanistic insights.
"""

        return synthesis_prompt

    def _summarize_result(self, result: Dict[str, Any]) -> str:
        """Create brief summary of analysis result"""
        tool_name = result.get("tool_name", "Analysis")

        if "growth_rate" in result:
            return f"Growth rate: {result['growth_rate']:.3f} h⁻¹"
        elif "fluxes" in result:
            flux_count = len(result["fluxes"])
            return f"Analyzed {flux_count} metabolic fluxes"
        elif "media_components" in result:
            component_count = len(result["media_components"])
            return f"Media with {component_count} components"
        else:
            return f"{tool_name} completed successfully"

    def _update_context_memory(self, enhanced_context: Dict[str, Any]):
        """Update context memory with important entities"""
        # Extract important biochemical entities
        important_entities = {}

        # Look for enhanced compound/reaction data
        for key, value in enhanced_context.items():
            if isinstance(value, dict) and "context" in value:
                entity_context = value["context"]
                if entity_context and isinstance(entity_context, dict):
                    entity_id = entity_context.get("id", key)
                    # Create simplified context for memory
                    try:
                        from .context_enhancer import BiochemicalContext
                    except ImportError:
                        from reasoning.context_enhancer import BiochemicalContext
                    context = BiochemicalContext(
                        entity_id, entity_context.get("type", "unknown")
                    )
                    context.name = entity_context.get("name")
                    context.formula = entity_context.get("formula")
                    important_entities[entity_id] = context

        # Remember entities with appropriate importance score
        if important_entities:
            self.context_memory.remember_entities(important_entities, importance=0.8)

    def _get_generic_analysis_prompt(self) -> str:
        """Get generic analysis prompt when specific prompt not found"""
        return """
        Analyze the provided biochemical data systematically:

        1. Describe what the data shows
        2. Identify key patterns and trends
        3. Explain the biological significance
        4. Discuss mechanistic insights
        5. Suggest implications and next steps

        Focus on providing scientifically accurate and mechanistically sound interpretations.
        """


# Global enhanced prompt provider instance
_enhanced_prompt_provider = None


def get_enhanced_prompt_provider() -> EnhancedPromptProvider:
    """Get global enhanced prompt provider instance"""
    global _enhanced_prompt_provider
    if _enhanced_prompt_provider is None:
        _enhanced_prompt_provider = EnhancedPromptProvider()
    return _enhanced_prompt_provider
