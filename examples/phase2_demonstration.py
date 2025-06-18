"""
Phase 2 Demonstration: Dynamic Context Enhancement + Multimodal Integration

Demonstrates the enhanced biochemical context injection and question-driven
reasoning frameworks implemented in Phase 2 of the Intelligence Enhancement.
"""

import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import with proper module paths
try:
    from src.reasoning.context_enhancer import get_context_enhancer, get_context_memory
    from src.reasoning.enhanced_prompt_provider import get_enhanced_prompt_provider
    from src.reasoning.frameworks.biochemical_reasoning import ReasoningDepth
except ImportError:
    # Alternative import approach
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.reasoning.context_enhancer import get_context_enhancer, get_context_memory
    from src.reasoning.enhanced_prompt_provider import get_enhanced_prompt_provider
    from src.reasoning.frameworks.biochemical_reasoning import ReasoningDepth

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def demonstrate_context_enhancement():
    """Demonstrate automatic biochemical context enhancement"""
    print("=" * 60)
    print("PHASE 2 DEMONSTRATION: Context Enhancement")
    print("=" * 60)

    # Get context enhancer
    enhancer = get_context_enhancer()

    # Example FBA result with raw biochemical IDs
    raw_fba_result = {
        "growth_rate": 0.8739,
        "fluxes": {
            "PFK": 7.477,
            "EX_glc__D_e": -10.0,
            "BIOMASS_Ecoli_core_w_GAM": 0.8739,
            "TPI": 7.477,
            "GAPD": 7.477,
        },
        "exchange_fluxes": {"EX_glc__D_e": -10.0, "EX_o2_e": -21.8, "EX_co2_e": 22.8},
    }

    print("\n1. Raw FBA Result:")
    print(json.dumps(raw_fba_result, indent=2))

    # Enhance with biochemical context
    enhanced_result = enhancer.enhance_tool_result("run_metabolic_fba", raw_fba_result)

    print("\n2. Enhanced Result with Biochemical Context:")
    print(json.dumps(enhanced_result, indent=2, default=str))

    return enhanced_result


def demonstrate_reasoning_frameworks():
    """Demonstrate question-driven reasoning frameworks"""
    print("\n" + "=" * 60)
    print("REASONING FRAMEWORKS DEMONSTRATION")
    print("=" * 60)

    # Import frameworks
    try:
        from src.reasoning.frameworks import (
            BiochemicalReasoningFramework,
            GrowthAnalysisFramework,
            PathwayAnalysisFramework,
        )
    except ImportError:
        from reasoning.frameworks import (
            BiochemicalReasoningFramework,
            GrowthAnalysisFramework,
            PathwayAnalysisFramework,
        )

    # Example growth data
    growth_data = {
        "growth_rate": 0.43,
        "biomass_yield": 0.35,
        "exchange_fluxes": {"EX_glc__D_e": -15.2, "EX_o2_e": -18.5},
    }

    # Growth analysis framework
    growth_framework = GrowthAnalysisFramework()
    growth_analysis = growth_framework.analyze_growth_context(growth_data)

    print("\n1. Growth Analysis Framework Output:")
    print(f"Growth Characterization: {growth_analysis['growth_characterization']}")
    print(f"Limiting Factors: {growth_analysis['limiting_factors']}")
    print("Reasoning Questions:")
    for i, question in enumerate(growth_analysis["reasoning_questions"][:3], 1):
        print(f"   {i}. {question}")

    # Pathway analysis framework
    pathway_framework = PathwayAnalysisFramework()

    flux_data = {
        "pathway_analysis": {
            "Glycolysis": {
                "active_reactions": 8,
                "total_flux": 45.2,
                "reactions": [
                    {"id": "PFK", "flux": 7.5, "name": "Phosphofructokinase"},
                    {
                        "id": "GAPD",
                        "flux": 7.5,
                        "name": "Glyceraldehyde-3-phosphate dehydrogenase",
                    },
                ],
            },
            "TCA cycle": {
                "active_reactions": 6,
                "total_flux": 12.8,
                "reactions": [{"id": "CS", "flux": 2.1, "name": "Citrate synthase"}],
            },
        }
    }

    pathway_analysis = pathway_framework.analyze_pathway_activity(flux_data)

    print("\n2. Pathway Analysis Framework Output:")
    print(f"Active Pathways: {list(pathway_analysis['active_pathways'].keys())}")
    print(
        f"Metabolic Strategy: {pathway_analysis['metabolic_strategy']['primary_objective']}"
    )
    print("Reasoning Questions:")
    for i, question in enumerate(pathway_analysis["reasoning_questions"][:3], 1):
        print(f"   {i}. {question}")

    return growth_analysis, pathway_analysis


def demonstrate_enhanced_prompt_generation():
    """Demonstrate enhanced prompt generation with context and reasoning"""
    print("\n" + "=" * 60)
    print("ENHANCED PROMPT GENERATION DEMONSTRATION")
    print("=" * 60)

    # Get enhanced prompt provider
    prompt_provider = get_enhanced_prompt_provider()

    # Example tool result
    tool_result = {
        "tool_name": "run_metabolic_fba",
        "growth_rate": 0.67,
        "fluxes": {"PFK": 8.2, "EX_glc__D_e": -12.0, "BIOMASS_Ecoli_core_w_GAM": 0.67},
        "biomass_yield": 0.42,
    }

    # Generate enhanced prompt
    enhanced_prompt = prompt_provider.get_enhanced_prompt(
        prompt_id="enhanced_analysis",
        context_data=tool_result,
        tool_name="run_metabolic_fba",
        reasoning_depth=ReasoningDepth.INTERMEDIATE,
    )

    print("\n1. Enhanced Prompt with Biochemical Context:")
    print(enhanced_prompt)

    # Generate tool-specific prompt
    tool_specific_prompt = prompt_provider.get_tool_specific_prompt(
        tool_name="run_metabolic_fba",
        tool_result=tool_result,
        analysis_goal="growth_optimization",
    )

    print("\n2. Tool-Specific Growth Analysis Prompt:")
    print(
        tool_specific_prompt[:800] + "..."
        if len(tool_specific_prompt) > 800
        else tool_specific_prompt
    )

    return enhanced_prompt


def demonstrate_cross_tool_synthesis():
    """Demonstrate cross-tool synthesis with context integration"""
    print("\n" + "=" * 60)
    print("CROSS-TOOL SYNTHESIS DEMONSTRATION")
    print("=" * 60)

    prompt_provider = get_enhanced_prompt_provider()

    # Multiple tool results
    tool_results = [
        {
            "tool_name": "run_metabolic_fba",
            "growth_rate": 0.8739,
            "fluxes": {"PFK": 7.5, "EX_glc__D_e": -10.0},
        },
        {
            "tool_name": "run_flux_variability_analysis",
            "variability": {
                "PFK": {"min": 5.2, "max": 9.8, "flexibility": 4.6},
                "EX_glc__D_e": {"min": -12.0, "max": -8.0, "flexibility": 4.0},
            },
        },
        {
            "tool_name": "find_minimal_media",
            "media_components": ["cpd00027", "cpd00001", "cpd00007"],
            "concentrations": {"cpd00027": 10.0},
        },
    ]

    # Generate synthesis prompt
    synthesis_prompt = prompt_provider.get_synthesis_prompt(
        tool_results=tool_results, analysis_focus="metabolic_efficiency"
    )

    print("\n1. Cross-Tool Synthesis Prompt:")
    print(
        synthesis_prompt[:1000] + "..."
        if len(synthesis_prompt) > 1000
        else synthesis_prompt
    )

    return synthesis_prompt


def demonstrate_context_memory():
    """Demonstrate context memory across analysis session"""
    print("\n" + "=" * 60)
    print("CONTEXT MEMORY DEMONSTRATION")
    print("=" * 60)

    context_memory = get_context_memory()

    # Simulate building context across multiple analyses
    try:
        from src.reasoning.context_enhancer import BiochemicalContext
    except ImportError:
        from reasoning.context_enhancer import BiochemicalContext

    # Add some important compounds
    glucose_context = BiochemicalContext("cpd00027", "compound")
    glucose_context.name = "D-Glucose"
    glucose_context.formula = "C6H12O6"

    atp_context = BiochemicalContext("cpd00002", "compound")
    atp_context.name = "ATP"
    atp_context.formula = "C10H16N5O13P3"

    context_memory.remember_entities(
        {"cpd00027": glucose_context, "cpd00002": atp_context}, importance=0.9
    )

    # Get session context
    session_context = context_memory.get_session_context()

    print("\n1. Session Context Summary:")
    print(json.dumps(session_context, indent=2, default=str))

    # Get context for reasoning
    reasoning_context = context_memory.get_context_for_reasoning()

    print("\n2. Context for AI Reasoning:")
    print(reasoning_context)

    return session_context


def run_comprehensive_demonstration():
    """Run comprehensive Phase 2 demonstration"""
    print("🧬 ModelSEEDagent Phase 2 Intelligence Enhancement Demonstration")
    print("Dynamic Context Enhancement + Multimodal Integration")
    print("=" * 80)

    try:
        # 1. Context Enhancement
        demonstrate_context_enhancement()

        # 2. Reasoning Frameworks
        growth_analysis, pathway_analysis = demonstrate_reasoning_frameworks()

        # 3. Enhanced Prompt Generation
        demonstrate_enhanced_prompt_generation()

        # 4. Cross-Tool Synthesis
        demonstrate_cross_tool_synthesis()

        # 5. Context Memory
        demonstrate_context_memory()

        print("\n" + "=" * 80)
        print("✅ PHASE 2 DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)

        # Summary
        print("\n📊 PHASE 2 CAPABILITIES DEMONSTRATED:")
        print(f"   ✓ Automatic biochemical context enhancement")
        print(f"   ✓ Question-driven reasoning frameworks")
        print(f"   ✓ Enhanced prompt generation with context")
        print(f"   ✓ Cross-tool synthesis capabilities")
        print(f"   ✓ Session context memory")

        print(f"\n🎯 KEY IMPROVEMENTS:")
        print(f"   • Raw biochemical IDs → Human-readable context")
        print(f"   • Generic analysis → Framework-guided reasoning")
        print(f"   • Single-tool focus → Cross-tool integration")
        print(f"   • Static prompts → Dynamic context-aware prompts")
        print(f"   • No memory → Progressive context building")

        return True

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_demonstration()

    if success:
        print(
            f"\n🚀 Phase 2 implementation ready for Phase 3: Reasoning Quality Validation"
        )
    else:
        print(f"\n❌ Phase 2 demonstration encountered errors - check implementation")
        sys.exit(1)
