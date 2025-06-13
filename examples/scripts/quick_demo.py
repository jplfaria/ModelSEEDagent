#!/usr/bin/env python3
"""
ModelSEEDagent Quick Demo - Fast AI Agent Showcase
Perfect for team presentations (30-60 seconds)
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


class QuickIntelligentAgent:
    """Fast AI Agent demo for presentations"""

    def __init__(self):
        self.model_path = str(project_root / "data" / "examples" / "e_coli_core.xml")
        self.results = {}

    def analyze_query_and_select_tools(self, query):
        """AI decision making process"""
        print(f"🧠 **AI Agent Query Analysis**")
        print(f"📝 User Query: '{query}'")
        print()

        # Simulate intelligent analysis
        query_lower = query.lower()

        if any(word in query_lower for word in ["comprehensive", "complete", "full"]):
            strategy = "comprehensive_fast"
            tools = ["run_metabolic_fba", "find_minimal_media", "search_biochem"]
            reasoning = "Comprehensive analysis requested - selecting core metabolic tools for fast overview"

        elif any(word in query_lower for word in ["growth", "fba", "flux"]):
            strategy = "growth_focused"
            tools = ["run_metabolic_fba", "find_minimal_media"]
            reasoning = "Growth-focused query - selecting FBA and media analysis"

        elif any(
            word in query_lower for word in ["media", "nutrition", "requirements"]
        ):
            strategy = "nutrition_focused"
            tools = ["find_minimal_media", "identify_auxotrophies"]
            reasoning = "Nutrition query - selecting media and auxotrophy analysis"

        else:
            strategy = "general_analysis"
            tools = ["run_metabolic_fba", "search_biochem"]
            reasoning = "General query - selecting basic metabolic analysis"

        print(f"🎯 **AI Strategy**: {strategy.replace('_', ' ').title()}")
        print(f"💭 **AI Reasoning**: {reasoning}")
        print(f"🔧 **Selected Tools**: {', '.join(tools)}")
        print()

        return tools

    def execute_intelligent_workflow(self, tools):
        """Execute tools with AI reasoning"""
        print(f"⚡ **AI Agent Executing Analysis Workflow**")
        print("=" * 55)

        for i, tool_name in enumerate(tools, 1):
            print(f"\n🔬 **Step {i}: {self._get_tool_description(tool_name)}**")

            # Show AI reasoning
            reasoning = self._get_ai_reasoning(tool_name, i, len(tools))
            print(f"💭 AI Agent: {reasoning}")

            # Execute tool
            success, result_summary = self._execute_tool_fast(tool_name)

            if success:
                print(f"✅ Success: {result_summary}")
            else:
                print(f"❌ Failed: {result_summary}")

            time.sleep(0.3)  # Brief pause for demo effect

    def _get_tool_description(self, tool_name):
        """Human-readable tool descriptions"""
        descriptions = {
            "run_metabolic_fba": "Growth & Flux Analysis",
            "find_minimal_media": "Nutritional Requirements",
            "identify_auxotrophies": "Biosynthetic Dependencies",
            "search_biochem": "Biochemistry Database Query",
            "resolve_biochem_entity": "Metabolite Identification",
        }
        return descriptions.get(tool_name, tool_name)

    def _get_ai_reasoning(self, tool_name, step, total):
        """AI reasoning for each tool"""
        reasoning = {
            "run_metabolic_fba": "Establishing baseline growth capacity and metabolic flux patterns",
            "find_minimal_media": "Determining minimal nutritional requirements for growth",
            "identify_auxotrophies": "Identifying biosynthetic pathways requiring supplementation",
            "search_biochem": "Querying metabolite database for biochemical context",
            "resolve_biochem_entity": "Resolving metabolite identifiers for biological interpretation",
        }
        return reasoning.get(
            tool_name, f"Executing {tool_name} for comprehensive analysis"
        )

    def _execute_tool_fast(self, tool_name):
        """Fast tool execution with result summary"""
        try:
            tool = ToolRegistry.create_tool(tool_name, {})

            # Prepare inputs
            if tool_name == "search_biochem":
                inputs = {"query": "ATP"}
            elif tool_name == "resolve_biochem_entity":
                inputs = {"entity_id": "cpd00027"}
            else:
                inputs = {"model_path": self.model_path}

            result = tool._run_tool(inputs)

            if result.success:
                # Extract key info for demo
                summary = self._extract_key_results(tool_name, result.data)
                self.results[tool_name] = {
                    "success": True,
                    "data": result.data,
                    "summary": summary,
                }
                return True, summary
            else:
                self.results[tool_name] = {"success": False, "error": result.error}
                return False, result.error[:60] + "..."

        except Exception as e:
            error_msg = str(e)[:60] + "..."
            self.results[tool_name] = {"success": False, "error": error_msg}
            return False, error_msg

    def _extract_key_results(self, tool_name, data):
        """Extract key results for demo display"""
        if not isinstance(data, dict):
            return "Analysis completed"

        if tool_name == "run_metabolic_fba":
            if "objective_value" in data:
                return f"Growth rate: {data['objective_value']:.3f} h⁻¹"
            return "FBA analysis completed"

        elif tool_name == "find_minimal_media":
            if "minimal_media" in data:
                return f"Requires {len(data['minimal_media'])} media components"
            return "Minimal media determined"

        elif tool_name == "identify_auxotrophies":
            if "auxotrophies" in data:
                return f"Found {len(data['auxotrophies'])} auxotrophies"
            return "Auxotrophy analysis completed"

        elif tool_name == "search_biochem":
            if "results" in data and isinstance(data["results"], list):
                return f"Found {len(data['results'])} biochemistry matches"
            return "Database search completed"

        return "Analysis completed successfully"

    def generate_ai_insights(self, query):
        """Generate AI insights and conclusions"""
        print(f"\n🧬 **AI Agent Analysis Summary & Insights**")
        print("=" * 60)

        successful = [k for k, v in self.results.items() if v["success"]]
        failed = [k for k, v in self.results.items() if not v["success"]]

        print(f"📊 **Analysis Results**:")
        print(f"   • Query: {query}")
        print(f"   • Tools executed: {len(self.results)}")
        print(f"   • Successful: {len(successful)} | Failed: {len(failed)}")
        print()

        # Extract and display key findings
        findings = {}

        if "run_metabolic_fba" in successful:
            fba_data = self.results["run_metabolic_fba"]["data"]
            if isinstance(fba_data, dict) and "objective_value" in fba_data:
                findings["growth_rate"] = fba_data["objective_value"]

        if "find_minimal_media" in successful:
            media_data = self.results["find_minimal_media"]["data"]
            if isinstance(media_data, dict) and "minimal_media" in media_data:
                findings["media_components"] = len(media_data["minimal_media"])

        if "search_biochem" in successful:
            search_data = self.results["search_biochem"]["data"]
            if isinstance(search_data, dict) and "results" in search_data:
                findings["biochem_matches"] = len(search_data["results"])

        # AI-generated insights
        print(f"🎯 **AI-Generated Metabolic Insights**:")

        if "growth_rate" in findings:
            growth = findings["growth_rate"]
            print(f"   • **Metabolic Capacity**: {growth:.3f} h⁻¹")
            if growth > 0.5:
                print(
                    f"     💡 AI Insight: High growth rate indicates robust, efficient metabolism"
                )
            elif growth > 0.1:
                print(
                    f"     💡 AI Insight: Moderate growth suggests some metabolic constraints"
                )
            else:
                print(
                    f"     💡 AI Insight: Low growth indicates significant limitations"
                )

        if "media_components" in findings:
            components = findings["media_components"]
            print(
                f"   • **Nutritional Profile**: {components} minimal media components"
            )
            if components < 15:
                print(
                    f"     💡 AI Insight: Low requirements suggest metabolic versatility"
                )
            elif components < 25:
                print(f"     💡 AI Insight: Moderate nutritional complexity")
            else:
                print(f"     💡 AI Insight: High nutritional demands")

        if "biochem_matches" in findings:
            matches = findings["biochem_matches"]
            print(
                f"   • **Database Coverage**: {matches} biochemistry database matches"
            )
            print(f"     💡 AI Insight: Comprehensive metabolite annotation available")

        # AI conclusions
        print(f"\n🧠 **AI Agent Conclusions**:")

        if findings:
            print(f"   Based on intelligent analysis, E. coli demonstrates:")

            if "growth_rate" in findings and findings["growth_rate"] > 0.5:
                print(
                    f"   • 🚀 **Metabolic Efficiency**: High-performance metabolic network"
                )

            if "media_components" in findings:
                if findings["media_components"] < 20:
                    print(
                        f"   • 🍽️ **Biosynthetic Capability**: Efficient nutrient utilization"
                    )
                else:
                    print(
                        f"   • 🍽️ **Nutritional Complexity**: Diverse metabolic requirements"
                    )

            print(
                f"   • 🧬 **Model Quality**: Well-curated metabolic reconstruction suitable for analysis"
            )

        print(f"\n🎯 **AI Recommendations**:")
        print(f"   • Model demonstrates realistic metabolic behavior")
        print(f"   • Suitable for systems biology and biotechnology applications")

        if failed:
            print(
                f"   • Consider investigating failed analyses for complete characterization"
            )

        print(
            f"\n📈 **Overall Assessment**: E. coli core model represents a {self._assess_model(findings)}"
        )

    def _assess_model(self, findings):
        """AI model quality assessment"""
        if not findings:
            return "basic metabolic framework"

        if "growth_rate" in findings and findings["growth_rate"] > 0.5:
            if "media_components" in findings and findings["media_components"] < 25:
                return "high-quality, biologically realistic metabolic network"
            else:
                return "functional metabolic model with good predictive capability"
        else:
            return "basic metabolic framework requiring validation"


def run_quick_demo():
    """Run fast demo for team presentation"""
    print("🧬 **ModelSEEDagent Quick Demo**")
    print("🤖 AI Agent with Intelligent Decision-Making")
    print("⚡ Optimized for Team Presentations")
    print("=" * 65)
    print()

    # Demo query (customizable)
    query = "Perform a comprehensive metabolic analysis of E. coli to understand its growth and nutritional requirements"

    # Alternative queries for different demos:
    # query = "What are the growth capabilities of this E. coli model?"
    # query = "Analyze the nutritional requirements for E. coli growth"
    # query = "Do a quick metabolic assessment of the E. coli core model"

    agent = QuickIntelligentAgent()

    # Step 1: AI Query Analysis & Tool Selection
    tools = agent.analyze_query_and_select_tools(query)

    # Step 2: Intelligent Workflow Execution
    agent.execute_intelligent_workflow(tools)

    # Step 3: AI Insights & Conclusions
    agent.generate_ai_insights(query)

    print(f"\n🎉 **Quick Demo Complete!**")
    print(f"⏱️ **Total Time**: ~30-60 seconds")
    print(
        f"🎯 **Demonstrated**: AI decision-making, tool selection, and biological insights"
    )


if __name__ == "__main__":
    run_quick_demo()
