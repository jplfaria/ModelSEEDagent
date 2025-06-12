#!/usr/bin/env python3
"""
Split Comprehensive Testbed Results with Biological Validation
============================================================

This script splits the comprehensive testbed results into individual files
with enhanced biological validation analysis and cross-tool comparisons.

Output structure:
- testbed_results/comprehensive/
  - individual/
    - e_coli_core/
      - FBA_results.json (with biological validation)
      - MediaSelector_results.json
      - BiochemEntityResolver_results.json
      - etc.
    - iML1515/
    - EcoliMG1655/
    - B_aphidicola/
  - analysis/
    - biological_validation_summary.json
    - cross_model_comparison.json
    - tool_category_analysis.json
"""

import json
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class ComprehensiveResultsAnalyzer:
    """Analyzes comprehensive testbed results with biological validation"""

    def __init__(self):
        self.tool_categories = {
            "COBRA": [
                "FBA",
                "ModelAnalysis",
                "FluxVariability",
                "GeneDeletion",
                "Essentiality",
                "FluxSampling",
                "ProductionEnvelope",
                "Auxotrophy",
                "MinimalMedia",
                "MissingMedia",
                "ReactionExpression",
            ],
            "AI_Media": [
                "MediaSelector",
                "MediaManipulator",
                "MediaCompatibility",
                "MediaComparator",
                "MediaOptimization",
                "AuxotrophyPrediction",
            ],
            "Biochemistry": ["BiochemEntityResolver", "BiochemSearch"],
        }

        self.model_types = {
            "BiGG": ["e_coli_core", "iML1515"],
            "ModelSEED": ["EcoliMG1655", "B_aphidicola"],
        }

    def split_comprehensive_results(self, input_file_path: str):
        """Split comprehensive results with enhanced analysis"""

        # Load comprehensive results
        with open(input_file_path, "r") as f:
            data = json.load(f)

        # Create enhanced output directory structure
        base_output_dir = Path("testbed_results/comprehensive")
        individual_dir = base_output_dir / "individual"
        analysis_dir = base_output_dir / "analysis"

        base_output_dir.mkdir(exist_ok=True)
        individual_dir.mkdir(exist_ok=True)
        analysis_dir.mkdir(exist_ok=True)

        metadata = data["metadata"]
        models_tested = metadata["models_tested"]

        print(f"🔄 Splitting comprehensive testbed results")
        print(
            f"📊 Processing {len(models_tested)} models × {metadata['tools_tested']['total_count']} tools"
        )
        print(f"🔬 Including biological validation analysis")
        print("=" * 70)

        # Split individual results
        individual_files = self._split_individual_results(
            data, individual_dir, models_tested
        )

        # Generate enhanced analyses
        analyses = self._generate_comprehensive_analyses(data, analysis_dir)

        # Create comprehensive summary
        summary = self._create_comprehensive_summary(data, individual_files, analyses)

        summary_file = base_output_dir / "comprehensive_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n📁 Results split into: {base_output_dir}/")
        print(f"📄 Comprehensive summary: {summary_file}")
        print(f"🔬 Enhanced analyses in: {analysis_dir}/")

        return individual_files, analyses, summary_file

    def _split_individual_results(
        self, data: Dict, individual_dir: Path, models_tested: List[str]
    ) -> List[str]:
        """Split individual tool results with enhanced format"""

        files_created = []

        for model_name in models_tested:
            model_dir = individual_dir / model_name
            model_dir.mkdir(exist_ok=True)

            print(f"\n🧬 Processing {model_name.upper()} model:")

            if model_name not in data["results"]:
                print(f"  ⚠️  No results found for {model_name}")
                continue

            model_results = data["results"][model_name]

            # Process each tool for this model
            for tool_name, tool_result in model_results.items():
                output_filename = f"{tool_name}_results.json"
                output_path = model_dir / output_filename

                # Determine tool category
                tool_category = "Unknown"
                for category, tools in self.tool_categories.items():
                    if tool_name in tools:
                        tool_category = category
                        break

                # Create enhanced individual result
                individual_result = {
                    "metadata": {
                        "original_timestamp": data["metadata"]["timestamp"],
                        "model_name": model_name,
                        "tool_name": tool_name,
                        "tool_category": tool_category,
                        "split_timestamp": datetime.now().isoformat(),
                        "success": tool_result.get("success", False),
                        "execution_time": tool_result.get("execution_time", 0),
                        "has_biological_validation": "biological_validation"
                        in tool_result,
                    },
                    "test_parameters": {
                        "tool": tool_result.get("tool", tool_name),
                        "model": tool_result.get("model", model_name),
                        "model_path": tool_result.get("model_path", ""),
                    },
                    "execution_details": {
                        "success": tool_result.get("success", False),
                        "execution_time": tool_result.get("execution_time", 0),
                        "error": tool_result.get("error"),
                        "warnings": tool_result.get("warnings", []),
                    },
                    "tool_output": tool_result.get("output"),
                    "key_metrics": tool_result.get("key_metrics", {}),
                    "biological_validation": tool_result.get(
                        "biological_validation", {}
                    ),
                    "debug_info": tool_result.get("debug_info", {}),
                }

                # Add category-specific analysis
                if tool_category == "COBRA":
                    individual_result["cobra_analysis"] = self._analyze_cobra_result(
                        tool_result
                    )
                elif tool_category == "AI_Media":
                    individual_result["media_analysis"] = self._analyze_media_result(
                        tool_result
                    )
                elif tool_category == "Biochemistry":
                    individual_result["biochem_analysis"] = (
                        self._analyze_biochem_result(tool_result)
                    )

                # Write enhanced individual file
                with open(output_path, "w") as f:
                    json.dump(individual_result, f, indent=2, default=str)

                files_created.append(str(output_path))

                # Progress reporting
                status = "✅" if tool_result.get("success", False) else "❌"
                time_str = f"{tool_result.get('execution_time', 0):.2f}s"
                file_size = os.path.getsize(output_path) / 1024

                # Biological validation indicator
                bio_indicator = ""
                if "biological_validation" in tool_result:
                    validation = tool_result["biological_validation"]
                    scores = validation.get("scores", {})
                    if scores:
                        avg_score = sum(scores.values()) / len(scores)
                        bio_indicator = f"Bio:{avg_score:.2f}"
                    else:
                        bio_indicator = "Bio:✓"

                print(
                    f"  {status} {tool_name:<22} {time_str:>8} {bio_indicator:>10} → {output_filename} ({file_size:.1f} KB)"
                )

        return files_created

    def _analyze_cobra_result(self, tool_result: Dict) -> Dict[str, Any]:
        """Analyze COBRA tool specific results"""
        analysis = {"category": "COBRA", "metrics": {}}

        if not tool_result.get("success", False):
            return analysis

        output = tool_result.get("output", {})
        if isinstance(output, dict):
            data = output.get("data", {})

            # Extract COBRA-specific metrics (check data is not None)
            if data and "objective_value" in data:
                analysis["metrics"]["growth_rate"] = data["objective_value"]
            if data and "significant_fluxes" in data:
                analysis["metrics"]["active_reactions"] = len(
                    data["significant_fluxes"]
                )
            if data and "essential_genes" in data:
                analysis["metrics"]["essential_genes"] = len(data["essential_genes"])

        return analysis

    def _analyze_media_result(self, tool_result: Dict) -> Dict[str, Any]:
        """Analyze AI Media tool specific results"""
        analysis = {"category": "AI_Media", "metrics": {}}

        if not tool_result.get("success", False):
            return analysis

        output = tool_result.get("output", {})
        if isinstance(output, dict):
            data = output.get("data", {})

            # Extract media-specific metrics (check data is not None)
            if data and "compatibility_score" in data:
                analysis["metrics"]["media_compatibility"] = data["compatibility_score"]
            if data and "growth_rate" in data:
                analysis["metrics"]["growth_rate"] = data["growth_rate"]
            if data and "best_media" in data:
                analysis["metrics"]["optimal_media"] = data["best_media"]

        return analysis

    def _analyze_biochem_result(self, tool_result: Dict) -> Dict[str, Any]:
        """Analyze Biochemistry tool specific results"""
        analysis = {"category": "Biochemistry", "metrics": {}}

        if not tool_result.get("success", False):
            return analysis

        output = tool_result.get("output", {})
        if isinstance(output, dict):
            data = output.get("data", {})

            # Extract biochemistry-specific metrics (check data is not None)
            if data and "resolved" in data:
                analysis["metrics"]["resolution_success"] = data["resolved"]
            if data and "total_results" in data:
                analysis["metrics"]["search_results"] = data["total_results"]
            if data and "aliases" in data:
                analysis["metrics"]["cross_database_aliases"] = len(data["aliases"])

        return analysis

    def _generate_comprehensive_analyses(
        self, data: Dict, analysis_dir: Path
    ) -> Dict[str, str]:
        """Generate comprehensive cross-tool and cross-model analyses"""

        analyses = {}

        # 1. Biological Validation Summary
        bio_validation_file = analysis_dir / "biological_validation_summary.json"
        bio_validation_analysis = self._analyze_biological_validation(data)
        with open(bio_validation_file, "w") as f:
            json.dump(bio_validation_analysis, f, indent=2)
        analyses["biological_validation"] = str(bio_validation_file)

        # 2. Cross-Model Comparison
        cross_model_file = analysis_dir / "cross_model_comparison.json"
        cross_model_analysis = self._analyze_cross_model_performance(data)
        with open(cross_model_file, "w") as f:
            json.dump(cross_model_analysis, f, indent=2)
        analyses["cross_model"] = str(cross_model_file)

        # 3. Tool Category Analysis
        tool_category_file = analysis_dir / "tool_category_analysis.json"
        tool_category_analysis = self._analyze_tool_categories(data)
        with open(tool_category_file, "w") as f:
            json.dump(tool_category_analysis, f, indent=2)
        analyses["tool_categories"] = str(tool_category_file)

        # 4. Model Format Compatibility
        format_compat_file = analysis_dir / "model_format_compatibility.json"
        format_compat_analysis = self._analyze_model_format_compatibility(data)
        with open(format_compat_file, "w") as f:
            json.dump(format_compat_analysis, f, indent=2)
        analyses["format_compatibility"] = str(format_compat_file)

        print(f"\n📊 Generated {len(analyses)} comprehensive analyses:")
        for analysis_name, file_path in analyses.items():
            file_size = os.path.getsize(file_path) / 1024
            print(
                f"  📈 {analysis_name:<25} → {os.path.basename(file_path)} ({file_size:.1f} KB)"
            )

        return analyses

    def _analyze_biological_validation(self, data: Dict) -> Dict[str, Any]:
        """Analyze biological validation results across all tests"""

        analysis = {
            "summary": {
                "total_tests_with_validation": 0,
                "tests_with_bio_scores": 0,
                "avg_bio_score": 0,
                "validation_categories": {},
            },
            "by_model": {},
            "by_tool": {},
            "insights_frequency": {},
            "warnings_frequency": {},
        }

        all_bio_scores = []
        all_insights = []
        all_warnings = []

        for model_name, model_results in data["results"].items():
            model_bio_scores = []

            for tool_name, tool_result in model_results.items():
                if "biological_validation" in tool_result:
                    analysis["summary"]["total_tests_with_validation"] += 1
                    validation = tool_result["biological_validation"]

                    # Extract scores
                    scores = validation.get("scores", {})
                    if scores:
                        analysis["summary"]["tests_with_bio_scores"] += 1
                        avg_score = sum(scores.values()) / len(scores)
                        all_bio_scores.append(avg_score)
                        model_bio_scores.append(avg_score)

                        # Tool-specific bio scores
                        if tool_name not in analysis["by_tool"]:
                            analysis["by_tool"][tool_name] = []
                        analysis["by_tool"][tool_name].append(avg_score)

                    # Collect insights and warnings
                    insights = validation.get("biological_insights", [])
                    warnings = validation.get("warnings", [])

                    all_insights.extend(insights)
                    all_warnings.extend(warnings)

            # Model-specific analysis
            analysis["by_model"][model_name] = {
                "avg_bio_score": (
                    statistics.mean(model_bio_scores) if model_bio_scores else 0
                ),
                "bio_score_count": len(model_bio_scores),
                "score_range": (
                    [min(model_bio_scores), max(model_bio_scores)]
                    if model_bio_scores
                    else [0, 0]
                ),
            }

        # Overall statistics
        if all_bio_scores:
            analysis["summary"]["avg_bio_score"] = statistics.mean(all_bio_scores)
            analysis["summary"]["bio_score_std"] = (
                statistics.stdev(all_bio_scores) if len(all_bio_scores) > 1 else 0
            )
            analysis["summary"]["bio_score_range"] = [
                min(all_bio_scores),
                max(all_bio_scores),
            ]

        # Tool category averages
        for tool_name, scores in analysis["by_tool"].items():
            analysis["by_tool"][tool_name] = {
                "avg_score": statistics.mean(scores),
                "score_count": len(scores),
                "score_range": [min(scores), max(scores)] if scores else [0, 0],
            }

        # Frequency analysis
        from collections import Counter

        insights_counter = Counter(all_insights)
        analysis["insights_frequency"] = dict(insights_counter.most_common(10))

        warnings_counter = Counter(all_warnings)
        analysis["warnings_frequency"] = dict(warnings_counter.most_common(10))

        return analysis

    def _analyze_cross_model_performance(self, data: Dict) -> Dict[str, Any]:
        """Compare tool performance across different models"""

        analysis = {
            "model_comparison": {},
            "tool_consistency": {},
            "format_specific_performance": {
                "BiGG": {"models": [], "success_rates": {}},
                "ModelSEED": {"models": [], "success_rates": {}},
            },
        }

        # Classify models by format
        for model_name in data["results"].keys():
            if model_name in self.model_types["BiGG"]:
                analysis["format_specific_performance"]["BiGG"]["models"].append(
                    model_name
                )
            elif model_name in self.model_types["ModelSEED"]:
                analysis["format_specific_performance"]["ModelSEED"]["models"].append(
                    model_name
                )

        # Analyze tool consistency across models
        all_tools = set()
        for model_results in data["results"].values():
            all_tools.update(model_results.keys())

        for tool_name in all_tools:
            tool_results = {}
            for model_name, model_results in data["results"].items():
                if tool_name in model_results:
                    tool_results[model_name] = model_results[tool_name].get(
                        "success", False
                    )

            analysis["tool_consistency"][tool_name] = {
                "tested_on_models": list(tool_results.keys()),
                "success_by_model": tool_results,
                "overall_success_rate": (
                    sum(tool_results.values()) / len(tool_results)
                    if tool_results
                    else 0
                ),
                "consistency_score": len(set(tool_results.values()))
                == 1,  # All same result
            }

        return analysis

    def _analyze_tool_categories(self, data: Dict) -> Dict[str, Any]:
        """Analyze performance by tool category"""

        analysis = {
            "category_performance": {},
            "category_comparison": {},
            "execution_time_analysis": {},
        }

        for category, tool_names in self.tool_categories.items():
            category_results = {
                "total_tests": 0,
                "successful_tests": 0,
                "avg_execution_time": 0,
                "execution_times": [],
                "bio_scores": [],
            }

            for model_results in data["results"].values():
                for tool_name in tool_names:
                    if tool_name in model_results:
                        result = model_results[tool_name]
                        category_results["total_tests"] += 1

                        if result.get("success", False):
                            category_results["successful_tests"] += 1

                        exec_time = result.get("execution_time", 0)
                        category_results["execution_times"].append(exec_time)

                        # Biological scores
                        validation = result.get("biological_validation", {})
                        scores = validation.get("scores", {})
                        if scores:
                            avg_score = sum(scores.values()) / len(scores)
                            category_results["bio_scores"].append(avg_score)

            # Calculate statistics
            if category_results["execution_times"]:
                category_results["avg_execution_time"] = statistics.mean(
                    category_results["execution_times"]
                )
                category_results["execution_time_std"] = (
                    statistics.stdev(category_results["execution_times"])
                    if len(category_results["execution_times"]) > 1
                    else 0
                )

            if category_results["bio_scores"]:
                category_results["avg_bio_score"] = statistics.mean(
                    category_results["bio_scores"]
                )

            category_results["success_rate"] = (
                category_results["successful_tests"] / category_results["total_tests"]
                if category_results["total_tests"] > 0
                else 0
            )

            analysis["category_performance"][category] = category_results

        return analysis

    def _analyze_model_format_compatibility(self, data: Dict) -> Dict[str, Any]:
        """Analyze compatibility between ModelSEED and BiGG models"""

        analysis = {
            "format_summary": {
                "BiGG_models": [],
                "ModelSEED_models": [],
                "format_specific_tools": {},
            },
            "media_tool_compatibility": {},
            "biochem_tool_effectiveness": {},
        }

        # Classify models
        for model_name in data["results"].keys():
            if model_name in self.model_types["BiGG"]:
                analysis["format_summary"]["BiGG_models"].append(model_name)
            elif model_name in self.model_types["ModelSEED"]:
                analysis["format_summary"]["ModelSEED_models"].append(model_name)

        # Analyze media tool compatibility across formats
        media_tools = self.tool_categories["AI_Media"]
        for tool_name in media_tools:
            format_performance = {"BiGG": [], "ModelSEED": []}

            for model_name, model_results in data["results"].items():
                if tool_name in model_results:
                    result = model_results[tool_name]
                    success = result.get("success", False)

                    # Get compatibility score if available
                    compatibility_score = None
                    key_metrics = result.get("key_metrics", {})
                    if "media_compatibility" in key_metrics:
                        compatibility_score = key_metrics["media_compatibility"]

                    model_format = (
                        "BiGG"
                        if model_name in self.model_types["BiGG"]
                        else "ModelSEED"
                    )
                    format_performance[model_format].append(
                        {
                            "model": model_name,
                            "success": success,
                            "compatibility_score": compatibility_score,
                        }
                    )

            analysis["media_tool_compatibility"][tool_name] = format_performance

        return analysis

    def _create_comprehensive_summary(
        self, data: Dict, individual_files: List[str], analyses: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create comprehensive summary of the split results"""

        metadata = data["metadata"]

        # Calculate summary statistics
        total_tests = metadata["total_tests"]
        successful_tests = metadata["successful_tests"]
        success_rate = metadata["success_rate"]

        # Model-specific summaries
        model_summaries = {}
        for model_name in metadata["models_tested"]:
            if model_name in data["results"]:
                model_results = data["results"][model_name]
                model_success = sum(
                    1 for r in model_results.values() if r.get("success", False)
                )
                model_total = len(model_results)
                model_summaries[model_name] = {
                    "success_rate": (
                        model_success / model_total if model_total > 0 else 0
                    ),
                    "successful_tests": model_success,
                    "total_tests": model_total,
                    "tool_count": model_total,
                }

        summary = {
            "split_metadata": {
                "original_timestamp": metadata["timestamp"],
                "split_timestamp": datetime.now().isoformat(),
                "testbed_version": metadata.get("testbed_version", "comprehensive_v1"),
                "total_files_created": len(individual_files),
                "analyses_generated": len(analyses),
            },
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "overall_success_rate": success_rate,
                "models_tested": metadata["models_tested"],
                "tools_by_category": metadata["tools_tested"],
                "biological_validation_summary": metadata.get(
                    "biological_validation_summary", {}
                ),
            },
            "model_summaries": model_summaries,
            "file_structure": {
                "individual_results": {},
                "comprehensive_analyses": analyses,
            },
            "quality_metrics": {
                "avg_execution_time": self._calculate_avg_execution_time(data),
                "biological_validation_coverage": self._calculate_bio_validation_coverage(
                    data
                ),
                "tool_category_coverage": self._calculate_category_coverage(data),
            },
        }

        # Organize individual files by model
        for model_name in metadata["models_tested"]:
            model_files = [f for f in individual_files if f"/{model_name}/" in f]
            summary["file_structure"]["individual_results"][model_name] = {
                "directory": f"testbed_results/comprehensive/individual/{model_name}/",
                "files": [os.path.basename(f) for f in model_files],
                "file_count": len(model_files),
            }

        return summary

    def _calculate_avg_execution_time(self, data: Dict) -> float:
        """Calculate average execution time across all tests"""
        all_times = []
        for model_results in data["results"].values():
            for result in model_results.values():
                exec_time = result.get("execution_time", 0)
                if exec_time > 0:
                    all_times.append(exec_time)
        return statistics.mean(all_times) if all_times else 0

    def _calculate_bio_validation_coverage(self, data: Dict) -> float:
        """Calculate percentage of tests with biological validation"""
        total_tests = 0
        validated_tests = 0

        for model_results in data["results"].values():
            for result in model_results.values():
                total_tests += 1
                if "biological_validation" in result:
                    validated_tests += 1

        return validated_tests / total_tests if total_tests > 0 else 0

    def _calculate_category_coverage(self, data: Dict) -> Dict[str, float]:
        """Calculate success rate by tool category"""
        category_coverage = {}

        for category, tools in self.tool_categories.items():
            category_success = 0
            category_total = 0

            for model_results in data["results"].values():
                for tool_name in tools:
                    if tool_name in model_results:
                        category_total += 1
                        if model_results[tool_name].get("success", False):
                            category_success += 1

            category_coverage[category] = (
                category_success / category_total if category_total > 0 else 0
            )

        return category_coverage


def main():
    """Main execution function"""

    print("🔄 Comprehensive Testbed Results Splitter with Biological Analysis")
    print("=" * 70)

    # Find the latest comprehensive testbed results file
    testbed_dir = Path("testbed_results")
    result_files = list(testbed_dir.glob("*_comprehensive_testbed_results.json"))

    if not result_files:
        print("❌ No comprehensive testbed results files found!")
        print("   Run 'python scripts/comprehensive_tool_testbed.py' first")
        return

    # Use the latest file
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)

    print(f"📁 Input file: {latest_file}")
    print(f"📏 File size: {latest_file.stat().st_size / 1024:.1f} KB")

    try:
        analyzer = ComprehensiveResultsAnalyzer()
        individual_files, analyses, summary_file = analyzer.split_comprehensive_results(
            str(latest_file)
        )

        print(f"\n✅ Successfully split comprehensive results:")
        print(f"📄 {len(individual_files)} individual tool result files")
        print(f"📊 {len(analyses)} comprehensive analyses")
        print(f"📋 Comprehensive summary: {summary_file}")
        print(f"🔬 Including biological validation and cross-tool comparisons")

    except Exception as e:
        print(f"❌ Error splitting comprehensive results: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
