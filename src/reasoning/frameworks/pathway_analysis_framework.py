"""
Pathway Analysis Framework for ModelSEEDagent

Specialized reasoning framework for metabolic pathway analysis,
providing pathway-specific context and mechanistic insights.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from .biochemical_reasoning import BiochemicalReasoningFramework, ReasoningDepth

logger = logging.getLogger(__name__)


class PathwayAnalysisFramework(BiochemicalReasoningFramework):
    """
    Specialized framework for pathway analysis reasoning

    Focuses on metabolic pathway activity, regulation, and
    cross-pathway interactions.
    """

    def __init__(self):
        super().__init__()
        self.pathway_patterns = self._initialize_pathway_patterns()
        self.pathway_interactions = self._initialize_pathway_interactions()

    def analyze_pathway_activity(
        self,
        flux_data: Dict[str, Any],
        pathway_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze metabolic pathway activity patterns

        Args:
            flux_data: Flux analysis results
            pathway_context: Additional pathway context information

        Returns:
            Comprehensive pathway analysis with reasoning guidance
        """
        analysis = {
            "active_pathways": self._identify_active_pathways(flux_data),
            "pathway_coordination": self._analyze_pathway_coordination(flux_data),
            "regulatory_patterns": self._infer_regulatory_patterns(flux_data),
            "metabolic_strategy": self._infer_metabolic_strategy(flux_data),
            "reasoning_questions": self._generate_pathway_questions(flux_data),
            "mechanistic_insights": self._generate_pathway_insights(flux_data),
        }

        return analysis

    def get_pathway_reasoning_prompts(
        self, pathway_data: Dict[str, Any], focus_pathway: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Get pathway-specific reasoning prompts

        Args:
            pathway_data: Pathway analysis results
            focus_pathway: Specific pathway to focus on

        Returns:
            Dictionary of reasoning prompts
        """
        prompts = {}

        if focus_pathway:
            prompts.update(
                self._get_focused_pathway_prompts(pathway_data, focus_pathway)
            )
        else:
            prompts.update(self._get_general_pathway_prompts(pathway_data))

        prompts.update(self._get_integration_prompts(pathway_data))

        return prompts

    def _initialize_pathway_patterns(self) -> Dict[str, Any]:
        """Initialize pathway-specific analysis patterns"""
        return {
            "central_carbon_metabolism": {
                "pathways": ["Glycolysis", "TCA cycle", "Pentose phosphate pathway"],
                "key_questions": [
                    "How is glucose being metabolized?",
                    "What is the balance between glycolysis and PPP?",
                    "Is the TCA cycle operating efficiently?",
                ],
                "regulatory_focus": [
                    "allosteric_regulation",
                    "redox_balance",
                    "energy_charge",
                ],
            },
            "amino_acid_metabolism": {
                "pathways": ["Amino acid biosynthesis", "Amino acid degradation"],
                "key_questions": [
                    "Which amino acids are being synthesized vs imported?",
                    "How does nitrogen availability affect amino acid metabolism?",
                    "Are there amino acid regulatory networks active?",
                ],
                "regulatory_focus": [
                    "nitrogen_regulation",
                    "amino_acid_availability",
                    "growth_phase_regulation",
                ],
            },
            "nucleotide_metabolism": {
                "pathways": ["Purine metabolism", "Pyrimidine metabolism"],
                "key_questions": [
                    "How are nucleotides being synthesized for DNA/RNA?",
                    "Is there balanced purine/pyrimidine production?",
                    "How does energy availability affect nucleotide synthesis?",
                ],
                "regulatory_focus": [
                    "energy_availability",
                    "growth_rate_coupling",
                    "salvage_vs_de_novo",
                ],
            },
            "lipid_metabolism": {
                "pathways": ["Fatty acid biosynthesis", "Fatty acid degradation"],
                "key_questions": [
                    "How are membrane lipids being synthesized?",
                    "Is there fatty acid beta-oxidation activity?",
                    "How does membrane composition relate to growth conditions?",
                ],
                "regulatory_focus": [
                    "membrane_homeostasis",
                    "carbon_storage",
                    "stress_adaptation",
                ],
            },
            "energy_metabolism": {
                "pathways": ["Respiratory chain", "Fermentation", "Photosynthesis"],
                "key_questions": [
                    "How is ATP being generated?",
                    "What is the respiratory efficiency?",
                    "Are there alternative energy generation modes?",
                ],
                "regulatory_focus": [
                    "oxygen_availability",
                    "carbon_source_quality",
                    "energy_demand",
                ],
            },
        }

    def _initialize_pathway_interactions(self) -> Dict[str, List[str]]:
        """Initialize known pathway interaction patterns"""
        return {
            "Glycolysis": [
                "TCA cycle",
                "Pentose phosphate pathway",
                "Fatty acid biosynthesis",
                "Amino acid biosynthesis",
            ],
            "TCA cycle": [
                "Glycolysis",
                "Amino acid biosynthesis",
                "Nucleotide biosynthesis",
                "Fatty acid biosynthesis",
            ],
            "Pentose phosphate pathway": [
                "Glycolysis",
                "Nucleotide biosynthesis",
                "Amino acid biosynthesis",
            ],
            "Fatty acid biosynthesis": [
                "Glycolysis",
                "TCA cycle",
                "Membrane biogenesis",
            ],
            "Amino acid biosynthesis": [
                "Glycolysis",
                "TCA cycle",
                "Nitrogen assimilation",
            ],
        }

    def _identify_active_pathways(self, flux_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify active metabolic pathways from flux data"""
        active_pathways = {}

        # Extract pathway activity from flux data
        if "pathway_analysis" in flux_data:
            pathway_data = flux_data["pathway_analysis"]

            for pathway_name, pathway_info in pathway_data.items():
                activity_score = self._calculate_pathway_activity(pathway_info)

                active_pathways[pathway_name] = {
                    "activity_score": activity_score,
                    "active_reactions": pathway_info.get("active_reactions", 0),
                    "total_flux": pathway_info.get("total_flux", 0.0),
                    "key_reactions": pathway_info.get("reactions", [])[:3],  # Top 3
                    "activity_level": self._classify_activity_level(activity_score),
                }

        return active_pathways

    def _analyze_pathway_coordination(
        self, flux_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze coordination between metabolic pathways"""
        coordination_analysis = {
            "coupled_pathways": [],
            "competing_pathways": [],
            "sequential_pathways": [],
            "coordination_score": 0.0,
        }

        active_pathways = self._identify_active_pathways(flux_data)

        # Check for pathway interactions
        for pathway1 in active_pathways:
            for pathway2 in active_pathways:
                if pathway1 != pathway2:
                    interaction = self._assess_pathway_interaction(
                        pathway1, pathway2, active_pathways
                    )

                    if interaction["type"] == "coupled":
                        coordination_analysis["coupled_pathways"].append(interaction)
                    elif interaction["type"] == "competing":
                        coordination_analysis["competing_pathways"].append(interaction)
                    elif interaction["type"] == "sequential":
                        coordination_analysis["sequential_pathways"].append(interaction)

        # Calculate overall coordination score
        coordination_analysis["coordination_score"] = (
            self._calculate_coordination_score(coordination_analysis)
        )

        return coordination_analysis

    def _infer_regulatory_patterns(self, flux_data: Dict[str, Any]) -> Dict[str, Any]:
        """Infer regulatory patterns from pathway activity"""
        regulatory_patterns = {
            "catabolite_repression": self._detect_catabolite_repression(flux_data),
            "nitrogen_regulation": self._detect_nitrogen_regulation(flux_data),
            "energy_regulation": self._detect_energy_regulation(flux_data),
            "growth_phase_regulation": self._detect_growth_phase_regulation(flux_data),
            "stress_response": self._detect_stress_response(flux_data),
        }

        return regulatory_patterns

    def _infer_metabolic_strategy(self, flux_data: Dict[str, Any]) -> Dict[str, Any]:
        """Infer overall metabolic strategy from pathway patterns"""
        strategy = {
            "primary_objective": "unknown",
            "carbon_utilization_strategy": "unknown",
            "energy_generation_strategy": "unknown",
            "biosynthetic_strategy": "unknown",
            "metabolic_flexibility": 0.0,
        }

        active_pathways = self._identify_active_pathways(flux_data)

        # Infer primary objective
        if self._is_growth_optimized(active_pathways):
            strategy["primary_objective"] = "growth_maximization"
        elif self._is_efficiency_optimized(active_pathways):
            strategy["primary_objective"] = "efficiency_maximization"
        elif self._is_stress_adapted(active_pathways):
            strategy["primary_objective"] = "stress_survival"

        # Infer carbon strategy
        strategy["carbon_utilization_strategy"] = self._infer_carbon_strategy(
            active_pathways
        )

        # Infer energy strategy
        strategy["energy_generation_strategy"] = self._infer_energy_strategy(
            active_pathways
        )

        # Calculate metabolic flexibility
        strategy["metabolic_flexibility"] = self._calculate_metabolic_flexibility(
            flux_data
        )

        return strategy

    def _generate_pathway_questions(self, flux_data: Dict[str, Any]) -> List[str]:
        """Generate pathway-specific reasoning questions"""
        questions = []

        active_pathways = self._identify_active_pathways(flux_data)

        # Add questions based on active pathways
        for pathway_name, pathway_info in active_pathways.items():
            if pathway_info["activity_level"] == "high":
                questions.append(f"Why is {pathway_name} highly active?")
                questions.append(f"What drives the high flux through {pathway_name}?")
            elif pathway_info["activity_level"] == "low":
                questions.append(f"Why is {pathway_name} showing low activity?")

        # Add coordination questions
        coordination = self._analyze_pathway_coordination(flux_data)
        if coordination["coupled_pathways"]:
            questions.append(
                "How are the coupled pathways coordinating their activities?"
            )

        if coordination["competing_pathways"]:
            questions.append("What resolves competition between alternative pathways?")

        # Add strategy questions
        questions.extend(
            [
                "What overall metabolic strategy does this pathway pattern represent?",
                "How do pathway activities reflect environmental adaptation?",
                "What regulatory mechanisms coordinate pathway activities?",
            ]
        )

        return questions

    def _generate_pathway_insights(
        self, flux_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate mechanistic insights about pathway behavior"""
        insights = []

        active_pathways = self._identify_active_pathways(flux_data)
        metabolic_strategy = self._infer_metabolic_strategy(flux_data)

        # Strategy insights
        insights.append(
            {
                "type": "metabolic_strategy",
                "insight": f"The organism appears to be following a {metabolic_strategy['primary_objective']} strategy",
                "evidence": f"Pathway activity pattern shows {metabolic_strategy['carbon_utilization_strategy']} carbon utilization",
                "implications": "This suggests specific environmental adaptation and regulatory priorities",
            }
        )

        # Pathway-specific insights
        for pathway_name, pathway_info in active_pathways.items():
            if pathway_info["activity_score"] > 0.7:
                insights.append(
                    {
                        "type": "high_pathway_activity",
                        "insight": f"{pathway_name} shows exceptionally high activity",
                        "evidence": f"Activity score: {pathway_info['activity_score']:.2f}",
                        "implications": "This pathway may be rate-limiting or under strong positive regulation",
                    }
                )

        return insights

    def _calculate_pathway_activity(self, pathway_info: Dict[str, Any]) -> float:
        """Calculate pathway activity score"""
        active_reactions = pathway_info.get("active_reactions", 0)
        total_flux = pathway_info.get("total_flux", 0.0)

        # Normalize and combine metrics
        activity_score = min(1.0, (active_reactions / 10.0) + (total_flux / 100.0))
        return activity_score

    def _classify_activity_level(self, activity_score: float) -> str:
        """Classify pathway activity level"""
        if activity_score > 0.7:
            return "high"
        elif activity_score > 0.4:
            return "moderate"
        elif activity_score > 0.1:
            return "low"
        else:
            return "inactive"

    def _assess_pathway_interaction(
        self, pathway1: str, pathway2: str, active_pathways: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess interaction between two pathways"""
        activity1 = active_pathways[pathway1]["activity_score"]
        activity2 = active_pathways[pathway2]["activity_score"]

        # Check for known interactions
        known_interactions = self.pathway_interactions.get(pathway1, [])

        interaction = {
            "pathway1": pathway1,
            "pathway2": pathway2,
            "strength": abs(activity1 - activity2),
            "type": "independent",
        }

        if pathway2 in known_interactions:
            if abs(activity1 - activity2) < 0.2:  # Similar activities
                interaction["type"] = "coupled"
            elif activity1 > 0.5 and activity2 < 0.3:  # One high, one low
                interaction["type"] = "competing"
            else:
                interaction["type"] = "sequential"

        return interaction

    def _calculate_coordination_score(
        self, coordination_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall pathway coordination score"""
        coupled_count = len(coordination_analysis["coupled_pathways"])
        competing_count = len(coordination_analysis["competing_pathways"])

        # Higher coordination score for more coupled, fewer competing pathways
        coordination_score = coupled_count * 0.3 - competing_count * 0.1
        return max(0.0, min(1.0, coordination_score))

    def _detect_catabolite_repression(
        self, flux_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect catabolite repression patterns"""
        # Look for glucose preference and alternative carbon source repression
        glucose_activity = self._get_pathway_activity(flux_data, "Glycolysis")
        alternative_activity = self._get_pathway_activity(
            flux_data, "Alternative carbon metabolism"
        )

        return {
            "detected": glucose_activity > 0.7 and alternative_activity < 0.3,
            "primary_carbon_source": "glucose" if glucose_activity > 0.7 else "unknown",
            "repressed_pathways": (
                ["Alternative carbon metabolism"] if alternative_activity < 0.3 else []
            ),
        }

    def _detect_nitrogen_regulation(self, flux_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect nitrogen regulation patterns"""
        amino_acid_biosynthesis = self._get_pathway_activity(
            flux_data, "Amino acid biosynthesis"
        )
        nitrogen_assimilation = self._get_pathway_activity(
            flux_data, "Nitrogen assimilation"
        )

        return {
            "detected": amino_acid_biosynthesis > 0.5,
            "nitrogen_limitation": nitrogen_assimilation > 0.8,
            "biosynthetic_activity": amino_acid_biosynthesis,
        }

    def _detect_energy_regulation(self, flux_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect energy-related regulation patterns"""
        respiratory_activity = self._get_pathway_activity(
            flux_data, "Respiratory chain"
        )
        fermentation_activity = self._get_pathway_activity(flux_data, "Fermentation")

        return {
            "detected": respiratory_activity > 0.5 or fermentation_activity > 0.5,
            "energy_mode": (
                "respiratory"
                if respiratory_activity > fermentation_activity
                else "fermentative"
            ),
            "energy_efficiency": respiratory_activity
            / (respiratory_activity + fermentation_activity + 0.01),
        }

    def _detect_growth_phase_regulation(
        self, flux_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect growth phase-specific regulation"""
        biosynthetic_activity = self._get_average_biosynthetic_activity(flux_data)
        catabolic_activity = self._get_average_catabolic_activity(flux_data)

        return {
            "detected": True,
            "inferred_phase": (
                "exponential" if biosynthetic_activity > 0.6 else "stationary"
            ),
            "biosynthetic_vs_catabolic_ratio": biosynthetic_activity
            / (catabolic_activity + 0.01),
        }

    def _detect_stress_response(self, flux_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect stress response patterns"""
        stress_pathways = ["Stress response", "DNA repair", "Protein folding"]
        stress_activity = sum(
            self._get_pathway_activity(flux_data, pathway)
            for pathway in stress_pathways
        ) / len(stress_pathways)

        return {
            "detected": stress_activity > 0.4,
            "stress_level": (
                "high"
                if stress_activity > 0.7
                else "moderate" if stress_activity > 0.4 else "low"
            ),
            "active_stress_pathways": [
                p
                for p in stress_pathways
                if self._get_pathway_activity(flux_data, p) > 0.5
            ],
        }

    def _get_pathway_activity(
        self, flux_data: Dict[str, Any], pathway_name: str
    ) -> float:
        """Get activity score for a specific pathway"""
        pathway_analysis = flux_data.get("pathway_analysis", {})
        pathway_info = pathway_analysis.get(pathway_name, {})
        return self._calculate_pathway_activity(pathway_info)

    def _get_average_biosynthetic_activity(self, flux_data: Dict[str, Any]) -> float:
        """Get average activity of biosynthetic pathways"""
        biosynthetic_pathways = [
            "Amino acid biosynthesis",
            "Nucleotide biosynthesis",
            "Fatty acid biosynthesis",
            "Cofactor biosynthesis",
        ]

        activities = [
            self._get_pathway_activity(flux_data, pathway)
            for pathway in biosynthetic_pathways
        ]
        return sum(activities) / len(activities) if activities else 0.0

    def _get_average_catabolic_activity(self, flux_data: Dict[str, Any]) -> float:
        """Get average activity of catabolic pathways"""
        catabolic_pathways = [
            "Glycolysis",
            "TCA cycle",
            "Fatty acid degradation",
            "Amino acid degradation",
        ]

        activities = [
            self._get_pathway_activity(flux_data, pathway)
            for pathway in catabolic_pathways
        ]
        return sum(activities) / len(activities) if activities else 0.0

    def _is_growth_optimized(self, active_pathways: Dict[str, Any]) -> bool:
        """Check if pathway pattern indicates growth optimization"""
        biosynthetic_activity = sum(
            info["activity_score"]
            for name, info in active_pathways.items()
            if "biosynthesis" in name.lower()
        ) / max(1, len(active_pathways))
        return biosynthetic_activity > 0.6

    def _is_efficiency_optimized(self, active_pathways: Dict[str, Any]) -> bool:
        """Check if pathway pattern indicates efficiency optimization"""
        respiratory_activity = active_pathways.get("Respiratory chain", {}).get(
            "activity_score", 0
        )
        return respiratory_activity > 0.7

    def _is_stress_adapted(self, active_pathways: Dict[str, Any]) -> bool:
        """Check if pathway pattern indicates stress adaptation"""
        stress_activity = active_pathways.get("Stress response", {}).get(
            "activity_score", 0
        )
        return stress_activity > 0.5

    def _infer_carbon_strategy(self, active_pathways: Dict[str, Any]) -> str:
        """Infer carbon utilization strategy"""
        glycolysis_activity = active_pathways.get("Glycolysis", {}).get(
            "activity_score", 0
        )
        ppp_activity = active_pathways.get("Pentose phosphate pathway", {}).get(
            "activity_score", 0
        )

        if glycolysis_activity > 0.7:
            return "glucose_specialist"
        elif ppp_activity > glycolysis_activity:
            return "biosynthetic_precursor_focused"
        else:
            return "mixed_carbon_utilization"

    def _infer_energy_strategy(self, active_pathways: Dict[str, Any]) -> str:
        """Infer energy generation strategy"""
        respiratory_activity = active_pathways.get("Respiratory chain", {}).get(
            "activity_score", 0
        )
        fermentation_activity = active_pathways.get("Fermentation", {}).get(
            "activity_score", 0
        )

        if respiratory_activity > fermentation_activity:
            return "respiratory_efficiency"
        elif fermentation_activity > 0.5:
            return "fermentative_rapid_growth"
        else:
            return "mixed_energy_generation"

    def _calculate_metabolic_flexibility(self, flux_data: Dict[str, Any]) -> float:
        """Calculate metabolic flexibility score"""
        # Based on number of active pathways and flux variability
        active_pathways = self._identify_active_pathways(flux_data)
        active_count = len(
            [p for p in active_pathways.values() if p["activity_score"] > 0.3]
        )

        flexibility_score = min(1.0, active_count / 10.0)
        return flexibility_score

    def _get_focused_pathway_prompts(
        self, pathway_data: Dict[str, Any], focus_pathway: str
    ) -> Dict[str, str]:
        """Get prompts focused on a specific pathway"""
        return {
            f"{focus_pathway}_analysis": f"""
            Focus specifically on {focus_pathway}:
            - What is the activity level and why?
            - Which reactions in this pathway are most active?
            - How does this pathway interact with others?
            - What regulates this pathway's activity?
            """,
            f"{focus_pathway}_mechanisms": f"""
            Explain the biochemical mechanisms in {focus_pathway}:
            - What are the key enzymatic steps?
            - How do allosteric regulations control flux?
            - What cofactors and metabolites are critical?
            """,
        }

    def _get_general_pathway_prompts(
        self, pathway_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Get general pathway analysis prompts"""
        return {
            "pathway_overview": """
            Analyze the overall pathway activity pattern:
            - Which pathways dominate the metabolic landscape?
            - How do pathways coordinate their activities?
            - What does this pattern reveal about metabolic strategy?
            """,
            "pathway_regulation": """
            Examine pathway regulation:
            - What regulatory mechanisms coordinate pathway activities?
            - How do environmental conditions influence pathway choice?
            - What metabolic switches or decision points are evident?
            """,
        }

    def _get_integration_prompts(self, pathway_data: Dict[str, Any]) -> Dict[str, str]:
        """Get prompts for pathway integration analysis"""
        return {
            "systems_integration": """
            Analyze pathway integration at the systems level:
            - How do different metabolic modules connect and communicate?
            - What emergent properties arise from pathway interactions?
            - How does the organism balance competing metabolic demands?
            """,
            "evolutionary_perspective": """
            Consider the evolutionary perspective:
            - How might this pathway organization reflect evolutionary optimization?
            - What environmental pressures shaped this metabolic architecture?
            - How do pathway redundancies and specializations contribute to fitness?
            """,
        }
