"""
Phase 8 Advanced Agentic Capabilities Extension for RealTimeMetabolicAgent

This file contains the Phase 8 methods to add to the RealTimeMetabolicAgent class.
"""


def add_phase8_methods(cls):
    """Add Phase 8 methods to RealTimeMetabolicAgent class"""

    def _should_use_reasoning_chains(self, query: str) -> bool:
        """Determine if query benefits from multi-step reasoning chains"""
        complex_indicators = [
            "comprehensive",
            "complete",
            "analyze",
            "investigate",
            "what",
            "why",
            "how",
            "compare",
            "optimize",
        ]
        return any(indicator in query.lower() for indicator in complex_indicators)

    def _should_use_hypothesis_driven(self, query: str) -> bool:
        """Determine if query benefits from hypothesis-driven analysis"""
        hypothesis_indicators = [
            "why",
            "cause",
            "reason",
            "limitation",
            "problem",
            "slow",
            "fast",
            "high",
            "low",
            "unusual",
            "unexpected",
        ]
        return any(indicator in query.lower() for indicator in hypothesis_indicators)

    def _analyze_model_characteristics(self, query: str):
        """Analyze model characteristics for learning system"""
        return {
            "model_path": self.default_model_path,
            "query_type": self._classify_query_type(query),
            "complexity": "high" if len(query.split()) > 10 else "medium",
        }

    def _classify_query_type(self, query: str) -> str:
        """Classify query type for pattern matching"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["comprehensive", "complete", "full"]):
            return "comprehensive_analysis"
        elif any(word in query_lower for word in ["growth", "biomass", "rate"]):
            return "growth_analysis"
        elif any(word in query_lower for word in ["essential", "gene", "knockout"]):
            return "essentiality_analysis"
        elif any(word in query_lower for word in ["media", "nutrient", "auxotroph"]):
            return "media_analysis"
        elif any(word in query_lower for word in ["flux", "pathway", "variability"]):
            return "flux_analysis"
        else:
            return "general_analysis"

    # Add all methods to the class
    cls._should_use_reasoning_chains = _should_use_reasoning_chains
    cls._should_use_hypothesis_driven = _should_use_hypothesis_driven
    cls._analyze_model_characteristics = _analyze_model_characteristics
    cls._classify_query_type = _classify_query_type

    return cls


# This enables importing the extension
__all__ = ["add_phase8_methods"]
