"""GeneDeletion Summarizer

Priority implementation for Smart Summarization Framework.
Reduces GeneDeletion output while preserving essential gene insights.
Focus on critical genes for growth and metabolism.
"""

from typing import Any, Dict, List, Optional, Union
import json

from ..base import ToolResult
from ..smart_summarization import BaseSummarizer, summarizer_registry


class GeneDeletionSummarizer(BaseSummarizer):
    """Smart summarizer for GeneDeletion tool
    
    Transforms detailed gene deletion results into three-tier summarization:
    - key_findings: Critical insights about essential genes (≤2KB)
    - summary_dict: Statistical summary and gene categories (≤5KB) 
    - full_data_path: Complete deletion results on disk
    
    Focus on essential genes and growth-critical pathways.
    """
    
    def get_tool_name(self) -> str:
        return "run_gene_deletion_analysis"
    
    def summarize(self, raw_output: Any, artifact_path: str, model_stats: Optional[Dict[str, Union[str, int]]] = None) -> ToolResult:
        """Summarize GeneDeletion output preserving essential gene insights
        
        Args:
            raw_output: Original gene deletion tool output with results/analysis structure
            artifact_path: Path where full deletion data is stored
            model_stats: Model metadata (reactions, genes, etc.)
            
        Returns:
            ToolResult with three-tier gene deletion summarization
        """
        # Extract gene deletion data from tool output
        if isinstance(raw_output, dict):
            deletion_results = raw_output.get('deletion_results', {})
            analysis_data = raw_output.get('analysis', {})
            wild_type_growth = raw_output.get('wild_type_growth', 0)
        else:
            # Fallback for unexpected format
            deletion_results = raw_output
            analysis_data = {}
            wild_type_growth = 0
        
        # Extract key metadata
        model_id = model_stats.get('model_id', 'unknown') if model_stats else 'unknown'
        genes_tested = len(deletion_results) if deletion_results else 0
        
        # Generate key findings (≤2KB) - Focus on essential genes for LLM
        key_findings = self._generate_key_findings(
            genes_tested, analysis_data, wild_type_growth, model_id
        )
        
        # Generate summary dict (≤5KB) - Structured data for analysis
        summary_dict = self._generate_summary_dict(
            genes_tested, analysis_data, deletion_results, wild_type_growth, model_stats
        )
        
        # Validate size limits
        self.validate_size_limits(key_findings, summary_dict)
        
        return ToolResult(
            success=True,
            message=f"Gene deletion analysis summarized: {genes_tested} genes tested",
            full_data_path=artifact_path,
            summary_dict=summary_dict,
            key_findings=key_findings,
            tool_name=self.get_tool_name(),
            model_stats=model_stats,
            schema_version="1.0"
        )
    
    def _generate_key_findings(self, genes_tested: int, analysis: Dict[str, Any], 
                              wild_type_growth: float, model_id: str) -> List[str]:
        """Generate critical insights for LLM consumption (≤2KB)"""
        key_findings = [
            f"Gene deletion analysis of {model_id}: {genes_tested} genes tested",
            f"Wild-type growth rate: {wild_type_growth:.3f}"
        ]
        
        # Extract critical gene categories from analysis
        summary = analysis.get('summary', {})
        if summary:
            essential_count = summary.get('essential_count', 0)
            impaired_count = summary.get('impaired_count', 0)
            no_effect_count = summary.get('no_effect_count', 0)
            improved_count = summary.get('improved_count', 0)
            essentiality_rate = summary.get('essentiality_rate', 0)
            
            # Calculate percentages
            essential_pct = (essential_count / genes_tested * 100) if genes_tested > 0 else 0
            impaired_pct = (impaired_count / genes_tested * 100) if genes_tested > 0 else 0
            no_effect_pct = (no_effect_count / genes_tested * 100) if genes_tested > 0 else 0
            
            key_findings.extend([
                f"Essential genes: {essential_count} ({essential_pct:.1f}%) - lethal when deleted",
                f"Growth-impaired genes: {impaired_count} ({impaired_pct:.1f}%) - reduce growth",
                f"Non-essential genes: {no_effect_count} ({no_effect_pct:.1f}%) - minimal impact"
            ])
            
            # Critical insights about essentiality
            if essentiality_rate > 0.15:
                key_findings.append("⚠️  High essentiality rate - model may be over-constrained")
            elif essentiality_rate < 0.05:
                key_findings.append("⚠️  Low essentiality rate - check deletion thresholds")
            else:
                key_findings.append("✓ Normal essentiality rate for metabolic model")
            
            # Beneficial mutations
            if improved_count > 0:
                improved_pct = (improved_count / genes_tested * 100) if genes_tested > 0 else 0
                key_findings.append(f"Growth-improving deletions: {improved_count} ({improved_pct:.1f}%)")
        
        # Extract examples of essential genes (critical for understanding)
        essential_genes = analysis.get('essential_genes', [])
        if essential_genes:
            # Show top essential genes (up to 5)
            top_essential = essential_genes[:5]
            key_findings.append(f"Key essential genes: {', '.join(top_essential)}")
        
        # Severity breakdown for impaired genes
        severely_impaired = analysis.get('severely_impaired', [])
        moderately_impaired = analysis.get('moderately_impaired', [])
        mildly_impaired = analysis.get('mildly_impaired', [])
        
        if severely_impaired:
            key_findings.append(f"Severely impaired genes: {len(severely_impaired)} (growth 1-10%)")
        if moderately_impaired:
            key_findings.append(f"Moderately impaired genes: {len(moderately_impaired)} (growth 10-50%)")
        if mildly_impaired:
            key_findings.append(f"Mildly impaired genes: {len(mildly_impaired)} (growth 50-90%)")
        
        return key_findings
    
    def _generate_summary_dict(self, genes_tested: int, analysis: Dict[str, Any],
                              deletion_results: Dict[str, Any], wild_type_growth: float,
                              model_stats: Optional[Dict[str, Union[str, int]]]) -> Dict[str, Any]:
        """Generate structured summary for analysis (≤5KB)"""
        summary_dict = {
            "deletion_statistics": {
                "total_genes_tested": genes_tested,
                "wild_type_growth": round(wild_type_growth, 4),
                "model_coverage": genes_tested / model_stats.get('num_genes', genes_tested) if model_stats else 1.0
            },
            "gene_categories": {},
            "essentiality_analysis": {},
            "growth_impact_distribution": {},
            "critical_genes": {},
            "model_context": model_stats or {},
            "analysis_metadata": {
                "deletion_method": "systematic_gene_knockout",
                "framework_version": "1.0",
                "growth_thresholds": {
                    "essential": "< 1% wild-type",
                    "severe": "1-10% wild-type", 
                    "moderate": "10-50% wild-type",
                    "mild": "50-90% wild-type"
                }
            }
        }
        
        # Gene categories summary
        if analysis:
            summary = analysis.get('summary', {})
            essential_genes = analysis.get('essential_genes', [])
            severely_impaired = analysis.get('severely_impaired', [])
            moderately_impaired = analysis.get('moderately_impaired', [])
            mildly_impaired = analysis.get('mildly_impaired', [])
            no_effect = analysis.get('no_effect', [])
            improved_growth = analysis.get('improved_growth', [])
            
            summary_dict["gene_categories"] = {
                "essential": {
                    "count": len(essential_genes),
                    "percentage": (len(essential_genes) / genes_tested * 100) if genes_tested > 0 else 0,
                    "examples": essential_genes[:10]  # Top 10 essential genes
                },
                "severely_impaired": {
                    "count": len(severely_impaired),
                    "percentage": (len(severely_impaired) / genes_tested * 100) if genes_tested > 0 else 0,
                    "examples": severely_impaired[:5]  # Top 5 severely impaired
                },
                "moderately_impaired": {
                    "count": len(moderately_impaired),
                    "percentage": (len(moderately_impaired) / genes_tested * 100) if genes_tested > 0 else 0
                },
                "mildly_impaired": {
                    "count": len(mildly_impaired),
                    "percentage": (len(mildly_impaired) / genes_tested * 100) if genes_tested > 0 else 0
                },
                "no_effect": {
                    "count": len(no_effect),
                    "percentage": (len(no_effect) / genes_tested * 100) if genes_tested > 0 else 0
                },
                "improved_growth": {
                    "count": len(improved_growth),
                    "percentage": (len(improved_growth) / genes_tested * 100) if genes_tested > 0 else 0,
                    "examples": improved_growth[:5] if improved_growth else []
                }
            }
            
            # Essentiality analysis
            essentiality_rate = summary.get('essentiality_rate', 0)
            summary_dict["essentiality_analysis"] = {
                "essentiality_rate": round(essentiality_rate, 4),
                "essentiality_category": self._categorize_essentiality_rate(essentiality_rate),
                "total_critical_genes": len(essential_genes) + len(severely_impaired),
                "robustness_score": (len(no_effect) / genes_tested) if genes_tested > 0 else 0
            }
        
        # Growth impact distribution
        if deletion_results:
            growth_values = [result.get('growth_ratio', 0) for result in deletion_results.values()]
            if growth_values:
                summary_dict["growth_impact_distribution"] = {
                    "mean_growth_retention": round(sum(growth_values) / len(growth_values), 4),
                    "min_growth_retention": round(min(growth_values), 4),
                    "max_growth_retention": round(max(growth_values), 4),
                    "lethal_deletions": sum(1 for g in growth_values if g < 0.01),
                    "beneficial_deletions": sum(1 for g in growth_values if g > 1.0)
                }
        
        # Critical genes (essential + severely impaired) with growth data
        if analysis and deletion_results:
            essential_genes = analysis.get('essential_genes', [])
            severely_impaired = analysis.get('severely_impaired', [])
            critical_gene_ids = essential_genes + severely_impaired
            
            critical_genes_data = []
            for gene_id in critical_gene_ids[:15]:  # Limit to top 15 critical genes
                if gene_id in deletion_results:
                    gene_data = deletion_results[gene_id]
                    critical_genes_data.append({
                        "gene_id": gene_id,
                        "growth_ratio": round(gene_data.get('growth_ratio', 0), 4),
                        "category": "essential" if gene_id in essential_genes else "severely_impaired"
                    })
            
            summary_dict["critical_genes"] = {
                "total_critical": len(critical_gene_ids),
                "critical_gene_details": critical_genes_data
            }
        
        return summary_dict
    
    def _categorize_essentiality_rate(self, rate: float) -> str:
        """Categorize essentiality rate for interpretation"""
        if rate > 0.15:
            return "high"
        elif rate > 0.05:
            return "normal"
        else:
            return "low"


# Register the summarizer
gene_deletion_summarizer = GeneDeletionSummarizer()
summarizer_registry.register(gene_deletion_summarizer)