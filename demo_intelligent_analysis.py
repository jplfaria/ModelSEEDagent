#!/usr/bin/env python3
"""
ModelSEEDagent Intelligent Analysis Demo
Showcases AI-driven tool selection and comprehensive reasoning
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.tools import ToolRegistry


class IntelligentMetabolicAgent:
    """AI Agent that intelligently selects tools and provides reasoning"""

    def __init__(self):
        self.model_path = str(project_root / "data" / "examples" / "e_coli_core.xml")
        self.analysis_results = {}
        self.reasoning_log = []

    def log_reasoning(self, step, decision, rationale):
        """Log agent reasoning for each decision"""
        self.reasoning_log.append(
            {
                "step": step,
                "decision": decision,
                "rationale": rationale,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def analyze_query_intent(self, query):
        """Analyze user query and determine appropriate analysis strategy"""
        print(f"🧠 **AI Agent Decision Making Process**")
        print(f"📝 User Query: '{query}'")
        print()

        # Simulate intelligent query analysis
        query_lower = query.lower()

        if any(
            word in query_lower
            for word in ["comprehensive", "complete", "full", "thorough"]
        ):
            strategy = "comprehensive"
            tools = [
                "run_metabolic_fba",
                "analyze_essentiality",
                "find_minimal_media",
                "run_flux_variability_analysis",
                "identify_auxotrophies",
                "search_biochem",
            ]
            reasoning = "Query requests comprehensive analysis - selecting full metabolic characterization toolkit"

        elif any(
            word in query_lower for word in ["growth", "fba", "flux", "production"]
        ):
            strategy = "growth_focused"
            tools = [
                "run_metabolic_fba",
                "run_flux_variability_analysis",
                "find_minimal_media",
            ]
            reasoning = (
                "Query focuses on growth and flux - selecting FBA-centered analysis"
            )

        elif any(
            word in query_lower
            for word in ["essential", "gene", "knockout", "deletion"]
        ):
            strategy = "essentiality_focused"
            tools = [
                "run_metabolic_fba",
                "analyze_essentiality",
                "run_gene_deletion_analysis",
            ]
            reasoning = (
                "Query targets gene essentiality - selecting gene-focused analysis"
            )

        elif any(
            word in query_lower
            for word in ["media", "nutrition", "requirements", "auxotroph"]
        ):
            strategy = "media_focused"
            tools = ["find_minimal_media", "identify_auxotrophies", "run_metabolic_fba"]
            reasoning = (
                "Query about nutritional requirements - selecting media analysis"
            )

        else:
            strategy = "general"
            tools = ["run_metabolic_fba", "analyze_essentiality", "find_minimal_media"]
            reasoning = "General query - selecting core metabolic analysis toolkit"

        self.log_reasoning("Query Analysis", f"Strategy: {strategy}", reasoning)

        print(f"🎯 **Agent Strategy Decision**: {strategy.title()}")
        print(f"💭 **Reasoning**: {reasoning}")
        print(f"🔧 **Selected Tools**: {', '.join(tools)}")
        print()

        return strategy, tools

    def execute_analysis_workflow(self, tools):
        """Execute analysis workflow with intelligent tool ordering"""
        print(f"⚡ **Executing Intelligent Analysis Workflow**")
        print("=" * 60)

        # Intelligent tool ordering based on dependencies
        ordered_tools = self._optimize_tool_order(tools)

        for i, tool_name in enumerate(ordered_tools, 1):
            print(f"\n🔬 **Step {i}: {self._get_tool_description(tool_name)}**")

            # Explain why this tool is being used now
            rationale = self._explain_tool_selection(tool_name, i, len(ordered_tools))
            print(f"💭 Agent reasoning: {rationale}")

            self.log_reasoning(f"Tool Execution {i}", tool_name, rationale)

            # Execute tool
            success = self._execute_tool(tool_name)

            if success:
                print(f"✅ Analysis completed successfully")
            else:
                print(f"❌ Analysis failed - adjusting strategy")
                self.log_reasoning(
                    f"Tool Failure {i}",
                    tool_name,
                    "Tool failed, continuing with remaining analysis",
                )

            # Small delay for demo effect
            time.sleep(0.5)

    def _optimize_tool_order(self, tools):
        """Intelligently order tools based on dependencies and logic"""
        # Define tool priorities and dependencies
        priority_order = {
            "run_metabolic_fba": 1,  # Always start with basic growth
            "find_minimal_media": 2,  # Media requirements next
            "analyze_essentiality": 3,  # Then essentiality
            "run_flux_variability_analysis": 4,  # Flux analysis
            "identify_auxotrophies": 5,  # Auxotrophy analysis
            "run_gene_deletion_analysis": 6,  # Gene analysis
            "search_biochem": 7,  # Database queries last
            "resolve_biochem_entity": 8,
        }

        # Sort tools by priority
        ordered = sorted(tools, key=lambda x: priority_order.get(x, 99))

        reasoning = f"Optimized tool order: FBA first (baseline), then media requirements, followed by gene analysis"
        self.log_reasoning("Workflow Planning", "Tool Ordering", reasoning)

        return ordered

    def _get_tool_description(self, tool_name):
        """Get human-readable description for tool"""
        descriptions = {
            "run_metabolic_fba": "Flux Balance Analysis (Growth Prediction)",
            "analyze_essentiality": "Essential Gene Identification",
            "find_minimal_media": "Minimal Media Requirements",
            "run_flux_variability_analysis": "Flux Variability Analysis",
            "identify_auxotrophies": "Auxotrophy Detection",
            "run_gene_deletion_analysis": "Gene Deletion Impact",
            "search_biochem": "Biochemistry Database Search",
            "resolve_biochem_entity": "Compound ID Resolution",
        }
        return descriptions.get(tool_name, tool_name)

    def _explain_tool_selection(self, tool_name, step, total_steps):
        """Explain why this tool is being used at this step"""
        explanations = {
            "run_metabolic_fba": "Starting with FBA to establish baseline growth capabilities and metabolic flux distribution",
            "find_minimal_media": "Determining minimal nutritional requirements to understand organism's dependencies",
            "analyze_essentiality": "Identifying essential genes and reactions critical for survival",
            "run_flux_variability_analysis": "Analyzing flux ranges to understand metabolic flexibility and constraints",
            "identify_auxotrophies": "Detecting biosynthetic deficiencies that require external supplementation",
            "run_gene_deletion_analysis": "Systematic gene knockout analysis to quantify gene importance",
            "search_biochem": "Querying biochemistry database for additional metabolite information",
        }
        return explanations.get(
            tool_name, f"Executing {tool_name} as part of comprehensive analysis"
        )

    def _execute_tool(self, tool_name):
        """Execute a single tool and store results"""
        try:
            tool = ToolRegistry.create_tool(tool_name, {})

            # Prepare inputs based on tool type
            if tool_name in ["search_biochem"]:
                inputs = {"query": "ATP"}
            elif tool_name in ["resolve_biochem_entity"]:
                inputs = {"entity_id": "cpd00027"}  # Fixed parameter name
            else:
                inputs = {"model_path": self.model_path}

            result = tool._run_tool(inputs)

            if result.success:
                self.analysis_results[tool_name] = {
                    "success": True,
                    "message": result.message,
                    "data": (
                        result.data
                        if isinstance(result.data, (dict, list, str, int, float))
                        else str(result.data)
                    ),
                }
                return True
            else:
                self.analysis_results[tool_name] = {
                    "success": False,
                    "error": result.error,
                }
                return False

        except Exception as e:
            self.analysis_results[tool_name] = {"success": False, "error": str(e)}
            return False

    def generate_comprehensive_reasoning(self, query):
        """Generate AI reasoning based on all analysis results"""
        print(f"\n🧬 **AI Agent Comprehensive Analysis & Reasoning**")
        print("=" * 80)

        # Analyze results
        successful_analyses = [
            k for k, v in self.analysis_results.items() if v["success"]
        ]
        failed_analyses = [
            k for k, v in self.analysis_results.items() if not v["success"]
        ]

        print(f"📊 **Analysis Summary**:")
        print(f"   • Query: {query}")
        print(f"   • Tools executed: {len(self.analysis_results)}")
        print(f"   • Successful: {len(successful_analyses)}")
        print(f"   • Failed: {len(failed_analyses)}")
        print()

        # Extract key findings
        key_findings = {}

        if "run_metabolic_fba" in successful_analyses:
            fba_data = self.analysis_results["run_metabolic_fba"]["data"]
            if isinstance(fba_data, dict) and "objective_value" in fba_data:
                key_findings["growth_rate"] = fba_data["objective_value"]

        if "analyze_essentiality" in successful_analyses:
            ess_data = self.analysis_results["analyze_essentiality"]["data"]
            if isinstance(ess_data, dict):
                if "essential_genes" in ess_data:
                    key_findings["essential_genes"] = len(ess_data["essential_genes"])
                if "essential_reactions" in ess_data:
                    key_findings["essential_reactions"] = len(
                        ess_data["essential_reactions"]
                    )

        if "find_minimal_media" in successful_analyses:
            media_data = self.analysis_results["find_minimal_media"]["data"]
            if isinstance(media_data, dict) and "minimal_media" in media_data:
                key_findings["minimal_media_components"] = len(
                    media_data["minimal_media"]
                )

        # Generate intelligent reasoning
        print(f"🎯 **Key Metabolic Insights**:")

        if "growth_rate" in key_findings:
            growth = key_findings["growth_rate"]
            print(f"   • **Growth Capacity**: {growth:.3f} h⁻¹")
            if growth > 0.5:
                print(
                    f"     💡 Agent insight: High growth rate indicates robust metabolic network"
                )
            elif growth > 0.1:
                print(
                    f"     💡 Agent insight: Moderate growth suggests some metabolic constraints"
                )
            else:
                print(
                    f"     💡 Agent insight: Low growth indicates significant metabolic limitations"
                )

        if "essential_genes" in key_findings:
            ess_genes = key_findings["essential_genes"]
            print(f"   • **Essential Genes**: {ess_genes} critical genes identified")
            print(
                f"     💡 Agent insight: {ess_genes/137*100:.1f}% of genes are essential for survival"
            )

        if "essential_reactions" in key_findings:
            ess_rxns = key_findings["essential_reactions"]
            print(f"   • **Essential Reactions**: {ess_rxns} critical reactions")
            print(
                f"     💡 Agent insight: These reactions represent metabolic bottlenecks"
            )

        if "minimal_media_components" in key_findings:
            media_comp = key_findings["minimal_media_components"]
            print(
                f"   • **Nutritional Requirements**: {media_comp} minimal media components"
            )
            if media_comp < 10:
                print(f"     💡 Agent insight: Organism is nutritionally versatile")
            elif media_comp < 20:
                print(f"     💡 Agent insight: Moderate nutritional requirements")
            else:
                print(f"     💡 Agent insight: Complex nutritional needs")

        print(f"\n🧠 **AI Agent Conclusions**:")

        # Generate contextual conclusions
        if key_findings:
            print(f"   Based on the comprehensive analysis, E. coli demonstrates:")

            if "growth_rate" in key_findings and key_findings["growth_rate"] > 0.5:
                print(
                    f"   • 🚀 **Metabolic Robustness**: High growth potential indicates efficient metabolism"
                )

            if "essential_genes" in key_findings:
                ess_percent = key_findings["essential_genes"] / 137 * 100
                if ess_percent < 10:
                    print(
                        f"   • 🧬 **Genetic Resilience**: Low essentiality suggests genetic redundancy"
                    )
                else:
                    print(
                        f"   • 🧬 **Genetic Constraints**: {ess_percent:.1f}% gene essentiality indicates network sensitivity"
                    )

            if "minimal_media_components" in key_findings:
                if key_findings["minimal_media_components"] < 15:
                    print(
                        f"   • 🍽️ **Nutritional Efficiency**: Minimal media requirements suggest biosynthetic capability"
                    )

        # Strategy recommendations
        print(f"\n🎯 **AI Recommendations for Further Analysis**:")

        if "run_flux_variability_analysis" not in successful_analyses:
            print(
                f"   • Consider flux variability analysis to understand metabolic flexibility"
            )

        if "identify_auxotrophies" not in successful_analyses:
            print(f"   • Perform auxotrophy analysis to identify biosynthetic gaps")

        if failed_analyses:
            print(f"   • Investigate failed analyses: {', '.join(failed_analyses)}")

        print(
            f"\n📈 **Metabolic Model Assessment**: E. coli core model represents a {self._assess_model_quality(key_findings)}"
        )

        # Save reasoning report
        self._save_reasoning_report(query, key_findings)

    def _assess_model_quality(self, findings):
        """Assess overall model quality based on findings"""
        if not findings:
            return "basic metabolic network requiring further validation"

        quality_score = 0

        if "growth_rate" in findings and findings["growth_rate"] > 0.5:
            quality_score += 2
        elif "growth_rate" in findings and findings["growth_rate"] > 0.1:
            quality_score += 1

        if "essential_genes" in findings:
            ess_percent = findings["essential_genes"] / 137 * 100
            if 5 <= ess_percent <= 15:  # Reasonable essentiality range
                quality_score += 2
            elif ess_percent < 20:
                quality_score += 1

        if "minimal_media_components" in findings:
            if 10 <= findings["minimal_media_components"] <= 25:
                quality_score += 1

        if quality_score >= 4:
            return "well-curated, biologically realistic metabolic network"
        elif quality_score >= 2:
            return "functional metabolic model with good predictive capability"
        else:
            return "basic metabolic framework suitable for initial studies"

    def _save_reasoning_report(self, query, findings):
        """Save comprehensive reasoning report"""
        report = {
            "query": query,
            "analysis_timestamp": datetime.now().isoformat(),
            "key_findings": findings,
            "reasoning_log": self.reasoning_log,
            "analysis_results": self.analysis_results,
            "model_assessment": self._assess_model_quality(findings),
        }

        output_file = project_root / "demo_analysis_report.json"
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n💾 **Complete analysis report saved**: {output_file}")


def run_demo():
    """Run the intelligent analysis demo"""
    print("🧬 **ModelSEEDagent Intelligent Analysis Demo**")
    print("🤖 AI Agent with Decision-Making and Reasoning")
    print("=" * 70)
    print()

    # Demo query (you can change this)
    demo_query = "Perform a comprehensive metabolic analysis of E. coli to understand its growth capabilities, essential genes, and nutritional requirements"

    # Alternative demo queries you can try:
    # demo_query = "What are the growth limitations and essential genes in this E. coli model?"
    # demo_query = "Analyze the minimal media requirements and auxotrophies for E. coli"
    # demo_query = "Do a complete flux analysis to understand E. coli metabolism"

    agent = IntelligentMetabolicAgent()

    # Step 1: Intelligent query analysis and tool selection
    strategy, tools = agent.analyze_query_intent(demo_query)

    # Step 2: Execute workflow with reasoning
    agent.execute_analysis_workflow(tools)

    # Step 3: Generate comprehensive AI reasoning
    agent.generate_comprehensive_reasoning(demo_query)

    print(
        f"\n🎉 **Demo Complete!** AI Agent successfully analyzed E. coli metabolism with intelligent reasoning"
    )


if __name__ == "__main__":
    run_demo()
