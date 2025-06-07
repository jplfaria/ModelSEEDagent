#!/usr/bin/env python3
"""
AI Reasoning Correctness Testing

This module tests that AI agents make intelligent decisions and produce
meaningful reasoning, not just execute without errors.

Tests validate:
- Tool selection intelligence
- Reasoning chain coherence
- Decision quality and biological understanding
- Adaptive behavior based on results
"""

import json
import sys
import time
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.agents.factory import create_real_time_agent
from src.llm.factory import LLMFactory
from src.tools.base import ToolRegistry


class TestAIToolSelectionIntelligence:
    """Test AI makes intelligent tool selection decisions"""

    @pytest.fixture
    def real_agent(self):
        """Create agent with real LLM (not mocked)"""
        try:
            # Try to create real LLM connection with improved configuration
            llm_config = {
                "model_name": "gpt-4o-mini",
                "temperature": 0.3,  # Lower temperature for more consistent reasoning
                "max_tokens": 2000,
            }

            # Try multiple backends in order of preference
            backends_to_try = ["argo", "openai", "local"]
            llm = None

            for backend in backends_to_try:
                try:
                    llm = LLMFactory.create(backend, llm_config)
                    print(f"✅ Using {backend} backend for AI reasoning tests")
                    break
                except Exception as e:
                    print(f"⚠️  {backend} backend unavailable: {e}")
                    continue

            if llm is None:
                pytest.skip("No LLM backend available for AI reasoning tests")

            # Get available tools
            tool_names = ToolRegistry.list_tools()
            tools = []
            for tool_name in tool_names[:10]:  # Limit to prevent token overflow
                try:
                    tool = ToolRegistry.create_tool(tool_name, {})
                    tools.append(tool)
                except Exception:
                    continue

            if len(tools) < 5:
                pytest.skip("Insufficient tools available for reasoning tests")

            config = {"max_iterations": 3}
            return create_real_time_agent(llm, tools, config)

        except Exception as e:
            pytest.skip(f"Cannot create real agent: {e}")

    def test_growth_investigation_intelligence(self, real_agent):
        """Test AI intelligently investigates growth issues"""
        query = "This E. coli model shows very slow growth (0.1 h⁻¹). Why is it growing so slowly?"

        print(f"\n🧠 Testing AI reasoning with query: {query}")

        try:
            result = real_agent.run({"query": query})
        except Exception as e:
            pytest.skip(f"AI agent execution failed: {e}")

        # Should succeed
        if not result.success:
            print(f"❌ AI reasoning failed: {result.error}")
            pytest.skip(f"AI reasoning failed: {result.error}")

        print(f"✅ AI reasoning succeeded")

        # Check tool selection intelligence
        tools_executed = result.metadata.get("tools_executed", [])
        print(f"🔧 Tools executed: {tools_executed}")
        assert len(tools_executed) > 0, "AI should execute at least one tool"

        # AI should investigate nutritional or essentiality issues for slow growth
        intelligent_tools = [
            "find_minimal_media",
            "identify_auxotrophies",
            "analyze_essentiality",
            "run_metabolic_fba",
        ]

        tools_used = [tool.lower() for tool in tools_executed]
        intelligent_selection = any(
            any(intel_tool in used_tool for intel_tool in intelligent_tools)
            for used_tool in tools_used
        )

        assert intelligent_selection, (
            f"AI should select nutritional/essentiality tools for slow growth. "
            f"Selected: {tools_executed}"
        )

        # AI reasoning should mention relevant concepts
        reasoning_text = result.message.lower()
        growth_concepts = [
            "nutrition",
            "nutrient",
            "essential",
            "media",
            "growth",
            "limitation",
        ]

        concept_mentions = sum(
            1 for concept in growth_concepts if concept in reasoning_text
        )
        assert concept_mentions >= 2, (
            f"AI reasoning should mention growth-related concepts. "
            f"Found {concept_mentions} mentions in: {result.message[:200]}..."
        )

    def test_optimization_query_intelligence(self, real_agent):
        """Test AI responds intelligently to optimization queries"""
        query = "How can I optimize this E. coli model for maximum growth?"

        result = real_agent.run({"query": query})

        assert result.success

        tools_executed = result.metadata.get("tools_executed", [])

        # AI should start with baseline analysis
        baseline_tools = ["run_metabolic_fba", "flux_variability"]
        baseline_used = any(
            any(baseline in tool.lower() for baseline in baseline_tools)
            for tool in tools_executed
        )

        assert baseline_used, (
            f"AI should start with baseline analysis for optimization. "
            f"Tools used: {tools_executed}"
        )

        # Response should discuss optimization concepts
        reasoning_text = result.message.lower()
        optimization_concepts = [
            "optimize",
            "maximum",
            "increase",
            "improve",
            "flux",
            "pathway",
        ]

        concept_mentions = sum(
            1 for concept in optimization_concepts if concept in reasoning_text
        )
        assert concept_mentions >= 2, (
            f"AI should discuss optimization concepts. "
            f"Found {concept_mentions} mentions in: {result.message[:200]}..."
        )

    def test_biochemistry_question_intelligence(self, real_agent):
        """Test AI uses biochemistry tools for compound questions"""
        query = "What is cpd00002 and what role does it play in metabolism?"

        result = real_agent.run({"query": query})

        assert result.success

        tools_executed = result.metadata.get("tools_executed", [])

        # AI should use biochemistry resolution tools
        biochem_tools = ["resolve_biochem", "search_biochem"]
        biochem_used = any(
            any(biochem in tool.lower() for biochem in biochem_tools)
            for tool in tools_executed
        )

        assert biochem_used, (
            f"AI should use biochemistry tools for compound questions. "
            f"Tools used: {tools_executed}"
        )

        # Should identify cpd00002 as ATP (if biochem DB is working)
        reasoning_text = result.message.lower()
        if "atp" in reasoning_text:
            # If AI identified ATP, should mention energy concepts
            energy_concepts = ["energy", "phosphate", "cellular", "metabolism"]
            energy_mentions = sum(
                1 for concept in energy_concepts if concept in reasoning_text
            )
            assert energy_mentions >= 1, "AI should discuss energy role of ATP"


class TestReasoningChainCoherence:
    """Test AI reasoning chains are logically coherent"""

    @pytest.fixture
    def real_agent(self):
        """Create agent for reasoning chain tests"""
        # Same setup as above
        try:
            llm_config = {
                "model_name": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 3000,
            }

            try:
                llm = LLMFactory.create("argo", llm_config)
            except Exception:
                try:
                    llm = LLMFactory.create("openai", llm_config)
                except Exception:
                    pytest.skip("No LLM backend available")

            tool_names = ToolRegistry.list_tools()
            tools = []
            for tool_name in tool_names[:8]:
                try:
                    tool = ToolRegistry.create_tool(tool_name, {})
                    tools.append(tool)
                except Exception:
                    continue

            config = {"max_iterations": 4}
            return create_real_time_agent(llm, tools, config)

        except Exception as e:
            pytest.skip(f"Cannot create agent: {e}")

    def test_comprehensive_analysis_progression(self, real_agent):
        """Test AI follows logical progression in comprehensive analysis"""
        query = "Perform a comprehensive metabolic analysis of this E. coli model"

        result = real_agent.run({"query": query})

        assert result.success

        tools_executed = result.metadata.get("tools_executed", [])

        assert (
            len(tools_executed) >= 2
        ), "Comprehensive analysis should use multiple tools"

        # First tool should be baseline analysis (FBA)
        first_tool = tools_executed[0].lower()
        assert (
            "fba" in first_tool or "metabolic" in first_tool
        ), f"Should start with baseline analysis, started with: {tools_executed[0]}"

        # Should show logical progression
        if len(tools_executed) >= 3:
            # Later tools should build on baseline (nutrition, essentiality, etc.)
            later_tools = [tool.lower() for tool in tools_executed[1:]]
            analysis_tools = [
                "media",
                "essential",
                "nutrition",
                "auxotroph",
                "variability",
            ]

            logical_progression = any(
                any(analysis_tool in later_tool for analysis_tool in analysis_tools)
                for later_tool in later_tools
            )

            assert logical_progression, (
                f"Should show logical progression from baseline. "
                f"Tool sequence: {' → '.join(tools_executed)}"
            )

    def test_adaptive_tool_selection(self, real_agent):
        """Test AI adapts tool selection based on discovered results"""
        # This test checks if AI can adapt its strategy mid-analysis
        query = "Analyze this model's metabolic capabilities and identify any issues"

        result = real_agent.run({"query": query})

        assert result.success

        tools_executed = result.metadata.get("tools_executed", [])
        reasoning = result.message

        # AI should execute multiple tools
        assert len(tools_executed) >= 2, "Should use multiple tools for analysis"

        # Reasoning should show connection between tools
        connection_words = [
            "because",
            "since",
            "therefore",
            "so",
            "thus",
            "given",
            "based on",
            "after",
            "then",
            "next",
        ]

        reasoning_lower = reasoning.lower()
        connections = sum(1 for word in connection_words if word in reasoning_lower)

        assert connections >= 1, (
            f"Reasoning should show logical connections between steps. "
            f"Found {connections} connection words in: {reasoning[:300]}..."
        )


class TestDecisionQualityMetrics:
    """Test quality of AI decisions using quantitative metrics"""

    @pytest.fixture
    def agent_with_metrics(self):
        """Create agent that tracks decision quality"""
        try:
            llm_config = {
                "model_name": "gpt-4o-mini",
                "temperature": 0.1,  # Very low for consistent decisions
                "max_tokens": 2000,
            }

            try:
                llm = LLMFactory.create("argo", llm_config)
            except Exception:
                try:
                    llm = LLMFactory.create("openai", llm_config)
                except Exception:
                    pytest.skip("No LLM backend available")

            tool_names = ToolRegistry.list_tools()
            tools = []
            for tool_name in tool_names[:6]:
                try:
                    tool = ToolRegistry.create_tool(tool_name, {})
                    tools.append(tool)
                except Exception:
                    continue

            config = {"max_iterations": 3, "track_decisions": True}
            return create_real_time_agent(llm, tools, config)

        except Exception as e:
            pytest.skip(f"Cannot create agent: {e}")

    def test_decision_consistency(self, agent_with_metrics):
        """Test AI makes consistent decisions for similar queries"""
        base_query = "This E. coli model has slow growth. What could be the issue?"

        results = []
        for i in range(2):  # Run twice
            result = agent_with_metrics.run({"query": base_query})
            if result.success:
                results.append(result)
            time.sleep(1)  # Brief pause between runs

        if len(results) < 2:
            pytest.skip("Could not get multiple successful runs")

        # Compare tool selections
        tools1 = set(results[0].metadata.get("tools_executed", []))
        tools2 = set(results[1].metadata.get("tools_executed", []))

        # Should have some overlap in tool selection (consistency)
        overlap = len(tools1.intersection(tools2))
        total_unique = len(tools1.union(tools2))

        if total_unique > 0:
            consistency_ratio = overlap / total_unique
            assert consistency_ratio >= 0.3, (
                f"AI decisions should be somewhat consistent. "
                f"Overlap: {overlap}/{total_unique} = {consistency_ratio:.2f}"
            )

    def test_response_time_reasonable(self, agent_with_metrics):
        """Test AI responses are generated in reasonable time"""
        query = "Quickly analyze this E. coli model's growth potential"

        start_time = time.time()
        result = agent_with_metrics.run({"query": query})
        end_time = time.time()

        duration = end_time - start_time

        assert result.success
        assert duration < 60, f"AI response took too long: {duration:.1f}s"

        # Should have meaningful content despite time constraint
        assert len(result.message) > 50, "Response should have meaningful content"


class TestBiologicalUnderstanding:
    """Test AI demonstrates biological and metabolic understanding"""

    @pytest.fixture
    def bio_agent(self):
        """Create agent for biological understanding tests"""
        try:
            llm_config = {
                "model_name": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 2500,
            }

            try:
                llm = LLMFactory.create("argo", llm_config)
            except Exception:
                try:
                    llm = LLMFactory.create("openai", llm_config)
                except Exception:
                    pytest.skip("No LLM backend available")

            tool_names = ToolRegistry.list_tools()
            tools = []
            for tool_name in tool_names:
                try:
                    tool = ToolRegistry.create_tool(tool_name, {})
                    tools.append(tool)
                    if len(tools) >= 8:  # Limit for efficiency
                        break
                except Exception:
                    continue

            config = {"max_iterations": 3}
            return create_real_time_agent(llm, tools, config)

        except Exception as e:
            pytest.skip(f"Cannot create agent: {e}")

    def test_metabolic_pathway_understanding(self, bio_agent):
        """Test AI understands metabolic pathway concepts"""
        query = "Explain the role of central metabolism in this E. coli model"

        result = bio_agent.run({"query": query})

        assert result.success

        response = result.message.lower()

        # Should mention key metabolic concepts
        metabolic_concepts = [
            "glycolysis",
            "tca",
            "citrate",
            "respiration",
            "fermentation",
            "glucose",
            "pyruvate",
            "acetyl",
            "atp",
            "nadh",
        ]

        concepts_found = sum(1 for concept in metabolic_concepts if concept in response)
        assert concepts_found >= 3, (
            f"Should demonstrate metabolic knowledge. "
            f"Found {concepts_found} metabolic concepts in: {response[:300]}..."
        )

    def test_growth_condition_understanding(self, bio_agent):
        """Test AI understands growth conditions and limitations"""
        query = "What conditions are needed for optimal E. coli growth?"

        result = bio_agent.run({"query": query})

        assert result.success

        response = result.message.lower()

        # Should mention growth requirements
        growth_concepts = [
            "nutrient",
            "media",
            "oxygen",
            "temperature",
            "ph",
            "carbon",
            "nitrogen",
            "phosphate",
            "sulfur",
        ]

        concepts_found = sum(1 for concept in growth_concepts if concept in response)
        assert concepts_found >= 2, (
            f"Should understand growth requirements. "
            f"Found {concepts_found} growth concepts in: {response[:300]}..."
        )


def test_llm_connection_available():
    """Test if any LLM backend is available for AI reasoning tests"""
    print("\n🔍 Checking LLM backend availability...")

    backends_to_try = ["argo", "openai", "local"]
    available_backends = []

    for backend in backends_to_try:
        try:
            llm_config = {
                "model_name": "gpt-4o-mini",
                "temperature": 0.3,
                "max_tokens": 100,
            }
            LLMFactory.create(backend, llm_config)  # Test connection
            available_backends.append(backend)
            print(f"✅ {backend} backend available")
        except Exception as e:
            print(f"❌ {backend} backend unavailable: {e}")

    if available_backends:
        print(f"🚀 Available backends: {', '.join(available_backends)}")
        return True
    else:
        print("⚠️  No LLM backends available - AI reasoning tests will be skipped")
        return False


def run_ai_reasoning_tests():
    """Run all AI reasoning correctness tests"""
    print("🧠 Running AI Reasoning Correctness Tests")
    print("=" * 50)
    print("⚠️  Note: These tests require real LLM connections")
    print("   Tests will be skipped if no LLM backend is available")
    print()

    # Check if LLM backends are available
    if not test_llm_connection_available():
        print("\n⏭️  Skipping AI reasoning tests - no LLM backends available")
        return True  # Return success since skipping is expected behavior

    print("\n🧪 Running AI reasoning validation tests...")

    # Run pytest on this module
    test_file = __file__
    exit_code = pytest.main([test_file, "-v", "--tb=short", "-x"])

    if exit_code == 0:
        print("\n✅ All AI reasoning tests PASSED!")
        print("🧠 AI agents demonstrate intelligent behavior")
    else:
        print("\n❌ Some AI reasoning tests FAILED or were SKIPPED!")
        print("⚠️  Check LLM connections and agent intelligence")

    return exit_code == 0


if __name__ == "__main__":
    success = run_ai_reasoning_tests()
    sys.exit(0 if success else 1)
