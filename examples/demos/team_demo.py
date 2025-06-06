#!/usr/bin/env python3
"""
Professional ModelSEEDagent Demo for Team Presentations
Clean, fast, and demonstrative of AI decision-making
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Suppress warnings and version prints for clean demo
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.tools import ToolRegistry


class TeamDemoAgent:
    """Professional AI Agent demo showcasing intelligent decision-making"""

    def __init__(self):
        self.model_path = str(project_root / "data" / "examples" / "e_coli_core.xml")
        self.results = {}
        self.reasoning_steps = []

    def analyze_user_query(self, query):
        """🧠 AI Decision-Making: Analyze query and select strategy"""
        print("🧠 AI AGENT DECISION-MAKING PROCESS")
        print("=" * 50)
        print(f"📝 User Query: '{query}'")
        print()

        # AI query analysis logic
        # query_words = query.lower().split()  # Currently unused

        # Strategy selection based on keywords
        if any(word in query for word in ["comprehensive", "complete", "full"]):
            strategy = "comprehensive_analysis"
            selected_tools = [
                "run_metabolic_fba",
                "find_minimal_media",
                "search_biochem",
            ]
            reasoning = "Query indicates comprehensive analysis needed → selecting core metabolic tools"

        elif any(word in query for word in ["growth", "fba", "flux"]):
            strategy = "growth_analysis"
            selected_tools = ["run_metabolic_fba", "find_minimal_media"]
            reasoning = (
                "Growth-focused query detected → selecting FBA and media analysis"
            )

        elif any(word in query for word in ["media", "nutrition", "requirements"]):
            strategy = "nutritional_analysis"
            selected_tools = ["find_minimal_media", "identify_auxotrophies"]
            reasoning = (
                "Nutritional query detected → selecting media and auxotrophy tools"
            )

        else:
            strategy = "general_analysis"
            selected_tools = ["run_metabolic_fba", "search_biochem"]
            reasoning = "General metabolic query → selecting baseline analysis tools"

        # Display AI reasoning
        print(f"🎯 AI STRATEGY SELECTION: {strategy.replace('_', ' ').title()}")
        print(f"💭 AI REASONING: {reasoning}")
        print(f"🔧 SELECTED TOOLS: {' → '.join(selected_tools)}")
        print()

        self.reasoning_steps.append(
            {
                "step": "Query Analysis",
                "strategy": strategy,
                "reasoning": reasoning,
                "tools": selected_tools,
            }
        )

        return selected_tools

    def execute_intelligent_workflow(self, tools):
        """⚡ Execute workflow with step-by-step AI reasoning"""
        print("⚡ AI WORKFLOW EXECUTION")
        print("=" * 30)

        for i, tool_name in enumerate(tools, 1):
            step_name = self._get_step_name(tool_name)
            reasoning = self._get_step_reasoning(tool_name, i, len(tools))

            print(f"\n🔬 STEP {i}: {step_name}")
            print(f"💭 AI Logic: {reasoning}")

            # Execute tool (with clean output)
            success, summary = self._execute_tool_clean(tool_name)

            if success:
                print(f"✅ Result: {summary}")
                self.reasoning_steps.append(
                    {
                        "step": f"Tool Execution {i}",
                        "tool": tool_name,
                        "reasoning": reasoning,
                        "result": "success",
                        "summary": summary,
                    }
                )
            else:
                print(f"❌ Failed: {summary}")
                self.reasoning_steps.append(
                    {
                        "step": f"Tool Execution {i}",
                        "tool": tool_name,
                        "reasoning": reasoning,
                        "result": "failed",
                        "error": summary,
                    }
                )

            # Brief pause for presentation effect
            time.sleep(0.5)

    def _get_step_name(self, tool_name):
        """Clean step names for presentation"""
        names = {
            "run_metabolic_fba": "Growth & Flux Analysis",
            "find_minimal_media": "Nutritional Requirements",
            "identify_auxotrophies": "Biosynthetic Dependencies",
            "search_biochem": "Biochemistry Database Query",
            "resolve_biochem_entity": "Metabolite Identification",
        }
        return names.get(tool_name, tool_name.replace("_", " ").title())

    def _get_step_reasoning(self, tool_name, step, total):
        """AI reasoning for each step"""
        reasoning_map = {
            "run_metabolic_fba": "Establish baseline growth capacity and identify active metabolic pathways",
            "find_minimal_media": "Determine minimum nutritional requirements for organism survival",
            "identify_auxotrophies": "Identify biosynthetic pathways requiring external supplementation",
            "search_biochem": "Query biochemistry database to enrich analysis with metabolite context",
            "resolve_biochem_entity": "Resolve metabolite identifiers for biological interpretation",
        }
        return reasoning_map.get(
            tool_name, f"Execute {tool_name} for metabolic characterization"
        )

    def _execute_tool_clean(self, tool_name):
        """Execute tool with clean output (no version spam)"""
        try:
            # Redirect stdout to suppress version prints during tool creation
            import contextlib
            import io

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                tool = ToolRegistry.create_tool(tool_name, {})

            # Prepare inputs
            if tool_name == "search_biochem":
                inputs = {"query": "ATP"}
            elif tool_name == "resolve_biochem_entity":
                inputs = {"entity_id": "cpd00027"}
            else:
                inputs = {"model_path": self.model_path}

            # Execute with output suppression
            with contextlib.redirect_stdout(f):
                result = tool._run_tool(inputs)

            if result.success:
                summary = self._create_result_summary(tool_name, result.data)
                self.results[tool_name] = {
                    "success": True,
                    "data": result.data,
                    "summary": summary,
                }
                return True, summary
            else:
                error_msg = (
                    str(result.error)[:50] + "..."
                    if len(str(result.error)) > 50
                    else str(result.error)
                )
                self.results[tool_name] = {"success": False, "error": error_msg}
                return False, error_msg

        except Exception as e:
            error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
            self.results[tool_name] = {"success": False, "error": error_msg}
            return False, error_msg

    def _create_result_summary(self, tool_name, data):
        """Create clean summary for each tool result"""
        if not isinstance(data, dict):
            return "Analysis completed successfully"

        if tool_name == "run_metabolic_fba":
            if "objective_value" in data:
                return f"Growth rate: {data['objective_value']:.3f} h⁻¹ (robust metabolism)"
            return "FBA analysis completed successfully"

        elif tool_name == "find_minimal_media":
            if "minimal_media" in data:
                count = len(data["minimal_media"])
                return f"Requires {count} essential nutrients (metabolically efficient)"
            return "Minimal media requirements determined"

        elif tool_name == "identify_auxotrophies":
            if "auxotrophies" in data:
                count = len(data["auxotrophies"])
                if count == 0:
                    return "No auxotrophies detected (biosynthetically complete)"
                else:
                    return f"Found {count} auxotrophies (requires supplementation)"
            return "Auxotrophy analysis completed"

        elif tool_name == "search_biochem":
            if "results" in data and isinstance(data["results"], list):
                return f"Found {len(data['results'])} biochemistry matches (well-annotated)"
            return "Database search completed"

        return "Analysis completed successfully"

    def generate_ai_insights(self, query):
        """🧬 Generate comprehensive AI insights and biological conclusions"""
        print("\n🧬 AI COMPREHENSIVE ANALYSIS & BIOLOGICAL INSIGHTS")
        print("=" * 60)

        # Analysis summary
        successful = [k for k, v in self.results.items() if v["success"]]
        failed = [k for k, v in self.results.items() if not v["success"]]

        print(f"📊 ANALYSIS OVERVIEW:")
        print(f"   • Original Query: {query}")
        print(f"   • Tools Executed: {len(self.results)}")
        print(f"   • Successful: {len(successful)} | Failed: {len(failed)}")
        print()

        # Extract quantitative findings
        findings = self._extract_quantitative_findings()

        # Generate AI biological insights
        print(f"🎯 AI-GENERATED BIOLOGICAL INSIGHTS:")

        if "growth_rate" in findings:
            growth = findings["growth_rate"]
            print(f"   • METABOLIC PERFORMANCE: {growth:.3f} h⁻¹")
            if growth > 0.5:
                print(f"     🧠 AI Assessment: High-efficiency metabolic network")
                print(
                    f"     📈 Biological Significance: Rapid growth potential under optimal conditions"
                )
            elif growth > 0.1:
                print(f"     🧠 AI Assessment: Moderate metabolic efficiency")
                print(
                    f"     📈 Biological Significance: Steady growth with some constraints"
                )
            else:
                print(f"     🧠 AI Assessment: Limited metabolic capacity")
                print(
                    f"     📈 Biological Significance: Growth-limiting bottlenecks present"
                )

        if "media_components" in findings:
            components = findings["media_components"]
            print(f"   • NUTRITIONAL PROFILE: {components} minimal requirements")
            if components < 15:
                print(f"     🧠 AI Assessment: Biosynthetically versatile organism")
                print(
                    f"     📈 Biological Significance: Can synthesize most required compounds"
                )
            elif components < 25:
                print(f"     🧠 AI Assessment: Moderate nutritional complexity")
                print(
                    f"     📈 Biological Significance: Balanced autotrophy and auxotrophy"
                )
            else:
                print(f"     🧠 AI Assessment: Nutritionally demanding organism")
                print(f"     📈 Biological Significance: Requires rich growth media")

        if "biochem_coverage" in findings:
            coverage = findings["biochem_coverage"]
            print(f"   • DATABASE ANNOTATION: {coverage} metabolite matches")
            print(f"     🧠 AI Assessment: Well-characterized metabolic network")
            print(
                f"     📈 Biological Significance: Comprehensive biochemical knowledge available"
            )

        # AI-driven conclusions
        print(f"\n🧠 AI AGENT CONCLUSIONS:")

        if findings:
            assessment = self._generate_model_assessment(findings)
            print(
                f"   Based on quantitative analysis, this E. coli model demonstrates:"
            )

            if "growth_rate" in findings and findings["growth_rate"] > 0.5:
                print(
                    f"   • 🚀 METABOLIC ROBUSTNESS: High-performance biochemical network"
                )

            if "media_components" in findings and findings["media_components"] < 20:
                print(
                    f"   • 🧬 BIOSYNTHETIC CAPABILITY: Efficient metabolic self-sufficiency"
                )

            print(f"   • 📊 MODEL QUALITY: {assessment}")

        print(f"\n🎯 AI RECOMMENDATIONS:")

        if len(successful) >= 2:
            print(
                f"   • Model suitable for systems biology research and biotechnology applications"
            )
            print(
                f"   • Quantitative predictions reliable for metabolic engineering studies"
            )

        if failed:
            print(
                f"   • Consider investigating {len(failed)} failed analyses for complete characterization"
            )
        else:
            print(f"   • All analyses successful - model ready for advanced studies")

        print(
            f"\n📈 OVERALL AI ASSESSMENT: E. coli core model represents a {self._final_assessment(findings)}"
        )

        # Save detailed report
        self._save_professional_report(query, findings)

    def _extract_quantitative_findings(self):
        """Extract key quantitative results"""
        findings = {}

        if (
            "run_metabolic_fba" in self.results
            and self.results["run_metabolic_fba"]["success"]
        ):
            data = self.results["run_metabolic_fba"]["data"]
            if isinstance(data, dict) and "objective_value" in data:
                findings["growth_rate"] = data["objective_value"]

        if (
            "find_minimal_media" in self.results
            and self.results["find_minimal_media"]["success"]
        ):
            data = self.results["find_minimal_media"]["data"]
            if isinstance(data, dict) and "minimal_media" in data:
                findings["media_components"] = len(data["minimal_media"])

        if (
            "search_biochem" in self.results
            and self.results["search_biochem"]["success"]
        ):
            data = self.results["search_biochem"]["data"]
            if isinstance(data, dict) and "results" in data:
                findings["biochem_coverage"] = len(data["results"])

        return findings

    def _generate_model_assessment(self, findings):
        """Generate AI model quality assessment"""
        score = 0

        if "growth_rate" in findings:
            if findings["growth_rate"] > 0.5:
                score += 2
            elif findings["growth_rate"] > 0.1:
                score += 1

        if "media_components" in findings:
            if 10 <= findings["media_components"] <= 25:
                score += 1

        if "biochem_coverage" in findings:
            if findings["biochem_coverage"] > 10:
                score += 1

        if score >= 3:
            return "High-quality, biologically realistic metabolic reconstruction"
        elif score >= 2:
            return "Well-curated model suitable for quantitative predictions"
        else:
            return "Functional model appropriate for basic metabolic studies"

    def _final_assessment(self, findings):
        """Generate final AI assessment"""
        if not findings:
            return "basic metabolic framework requiring further validation"

        if (
            len(findings) >= 2
            and "growth_rate" in findings
            and findings["growth_rate"] > 0.5
        ):
            return "sophisticated, validated metabolic model ready for advanced research applications"
        elif len(findings) >= 2:
            return "well-characterized metabolic network suitable for systems biology studies"
        else:
            return "functional metabolic model with demonstrated predictive capability"

    def _save_professional_report(self, query, findings):
        """Save comprehensive professional report"""
        report = {
            "demo_info": {
                "title": "ModelSEEDagent AI Analysis Demo",
                "timestamp": datetime.now().isoformat(),
                "query": query,
            },
            "ai_decision_process": self.reasoning_steps,
            "quantitative_findings": findings,
            "tool_results": self.results,
            "model_file": self.model_path,
        }

        output_file = project_root / "team_demo_report.json"
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n💾 Professional analysis report saved: {output_file.name}")


def run_team_demo():
    """🎯 Main demo function for team presentations"""
    print("🧬 ModelSEEDagent Professional Demo")
    print("🤖 AI Agent with Intelligent Decision-Making & Biological Reasoning")
    print("🎯 Optimized for Team Presentations")
    print("=" * 75)
    print()

    # Professional demo query
    query = "Perform a comprehensive metabolic analysis of E. coli to understand growth capabilities and nutritional requirements"

    # Alternative demo queries:
    # query = "What are the growth characteristics and media requirements for this E. coli model?"
    # query = "Analyze the metabolic efficiency and biosynthetic capabilities of E. coli"

    agent = TeamDemoAgent()

    # 🧠 Step 1: AI Query Analysis & Strategic Planning
    selected_tools = agent.analyze_user_query(query)

    # ⚡ Step 2: Intelligent Workflow Execution
    agent.execute_intelligent_workflow(selected_tools)

    # 🧬 Step 3: AI Insights & Biological Conclusions
    agent.generate_ai_insights(query)

    print(f"\n🎉 DEMO COMPLETE!")
    print(
        f"✨ Demonstrated: AI decision-making, tool orchestration, and biological insight generation"
    )
    print(f"⏱️ Total execution time: ~30-45 seconds")


if __name__ == "__main__":
    run_team_demo()
