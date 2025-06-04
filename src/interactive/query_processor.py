"""
Intelligent Query Processor for Interactive Analysis

Processes natural language queries about metabolic modeling,
provides context-aware responses, and generates smart suggestions.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class QueryType(Enum):
    """Types of metabolic modeling queries"""

    STRUCTURAL_ANALYSIS = "structural_analysis"
    GROWTH_ANALYSIS = "growth_analysis"
    PATHWAY_ANALYSIS = "pathway_analysis"
    FLUX_ANALYSIS = "flux_analysis"
    OPTIMIZATION = "optimization"
    COMPARISON = "comparison"
    VISUALIZATION = "visualization"
    MODEL_MODIFICATION = "model_modification"
    GENERAL_QUESTION = "general_question"
    HELP_REQUEST = "help_request"


class QueryComplexity(Enum):
    """Query complexity levels"""

    SIMPLE = "simple"  # Single tool, basic analysis
    MODERATE = "moderate"  # Multiple tools, some dependencies
    COMPLEX = "complex"  # Multiple tools, complex dependencies
    EXPERT = "expert"  # Advanced analysis, domain expertise


class QueryIntent(Enum):
    """User intent behind the query"""

    EXPLORE = "explore"  # Exploratory analysis
    VALIDATE = "validate"  # Validation/verification
    OPTIMIZE = "optimize"  # Optimization/improvement
    COMPARE = "compare"  # Comparison analysis
    DIAGNOSE = "diagnose"  # Problem diagnosis
    LEARN = "learn"  # Educational/learning
    REPORT = "report"  # Generate reports


@dataclass
class QueryAnalysis:
    """Comprehensive analysis of a user query"""

    original_query: str
    query_type: QueryType
    complexity: QueryComplexity
    intent: QueryIntent
    confidence: float

    # Extracted entities
    model_references: List[str] = field(default_factory=list)
    pathway_references: List[str] = field(default_factory=list)
    metabolite_references: List[str] = field(default_factory=list)
    gene_references: List[str] = field(default_factory=list)
    numeric_values: List[float] = field(default_factory=list)

    # Analysis details
    suggested_tools: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    context_dependencies: List[str] = field(default_factory=list)

    # Suggestions and improvements
    clarification_questions: List[str] = field(default_factory=list)
    alternative_phrasings: List[str] = field(default_factory=list)
    follow_up_suggestions: List[str] = field(default_factory=list)

    # Metadata
    processing_time: float = 0.0
    language_confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "original_query": self.original_query,
            "query_type": self.query_type.value,
            "complexity": self.complexity.value,
            "intent": self.intent.value,
            "confidence": self.confidence,
            "model_references": self.model_references,
            "pathway_references": self.pathway_references,
            "metabolite_references": self.metabolite_references,
            "gene_references": self.gene_references,
            "numeric_values": self.numeric_values,
            "suggested_tools": self.suggested_tools,
            "required_inputs": self.required_inputs,
            "expected_outputs": self.expected_outputs,
            "context_dependencies": self.context_dependencies,
            "clarification_questions": self.clarification_questions,
            "alternative_phrasings": self.alternative_phrasings,
            "follow_up_suggestions": self.follow_up_suggestions,
            "processing_time": self.processing_time,
            "language_confidence": self.language_confidence,
        }


class QueryProcessor:
    """Intelligent processor for metabolic modeling queries"""

    def __init__(self):
        self.query_patterns = self._initialize_patterns()
        self.metabolic_vocabulary = self._initialize_vocabulary()
        self.tool_mappings = self._initialize_tool_mappings()
        self.context_memory: Dict[str, Any] = {}

    def _initialize_patterns(self) -> Dict[QueryType, List[str]]:
        """Initialize regex patterns for query type detection"""
        return {
            QueryType.STRUCTURAL_ANALYSIS: [
                r"\b(structure|components?|reactions?|metabolites?|genes?)\b",
                r"\b(analyze|examine|inspect|describe)\b.*\b(model|network)\b",
                r"\b(how many|count|number of)\b.*\b(reactions?|metabolites?|genes?)\b",
                r"\b(stoichiometry|connectivity|topology)\b",
            ],
            QueryType.GROWTH_ANALYSIS: [
                r"\b(growth|biomass|proliferation)\b",
                r"\b(growth rate|doubling time|generation time)\b",
                r"\b(can.*grow|grow.*on|growth.*condition)\b",
                r"\b(media|medium|nutrient|substrate)\b.*\b(growth|biomass)\b",
            ],
            QueryType.PATHWAY_ANALYSIS: [
                r"\b(pathway|route|path)\b",
                r"\b(glycolysis|TCA|citric acid|pentose phosphate|fatty acid)\b",
                r"\b(metabolism|metabolic.*pathway)\b",
                r"\b(carbon|nitrogen|sulfur).*\b(metabolism|pathway)\b",
            ],
            QueryType.FLUX_ANALYSIS: [
                r"\b(flux|flow|rate)\b",
                r"\b(FBA|flux.*balance|flux.*analysis)\b",
                r"\b(objective|maximize|minimize)\b.*\b(flux|growth)\b",
                r"\b(variability|FVA|flux.*variability)\b",
            ],
            QueryType.OPTIMIZATION: [
                r"\b(optimize|optimization|improve|enhance)\b",
                r"\b(maximize|minimize|optimal)\b",
                r"\b(yield|production|efficiency)\b",
                r"\b(design|engineer|modify)\b.*\b(strain|organism)\b",
            ],
            QueryType.COMPARISON: [
                r"\b(compare|comparison|versus|vs|against)\b",
                r"\b(difference|different|similar|similarity)\b",
                r"\b(better|worse|faster|slower)\b.*\b(than|compared)\b",
            ],
            QueryType.VISUALIZATION: [
                r"\b(plot|graph|chart|visualize|show|display)\b",
                r"\b(heatmap|network|diagram|figure)\b",
                r"\b(export|save|generate).*\b(image|figure|plot)\b",
            ],
            QueryType.MODEL_MODIFICATION: [
                r"\b(add|remove|delete|modify|edit)\b.*\b(reaction|gene|metabolite)\b",
                r"\b(knockout|knockin|overexpression)\b",
                r"\b(constrain|bound|limit)\b.*\b(flux|reaction)\b",
            ],
            QueryType.HELP_REQUEST: [
                r"\b(help|how.*do|how.*can|explain|tutorial)\b",
                r"\b(what.*is|what.*does|how.*work)\b",
                r"\b(example|demonstrate|show.*me)\b",
            ],
        }

    def _initialize_vocabulary(self) -> Dict[str, List[str]]:
        """Initialize metabolic modeling vocabulary"""
        return {
            "models": [
                "ecoli",
                "e.coli",
                "iAF1260",
                "iML1515",
                "yeast",
                "saccharomyces",
                "recon",
                "human",
                "core",
                "genome-scale",
                "constraint-based",
            ],
            "pathways": [
                "glycolysis",
                "gluconeogenesis",
                "TCA",
                "citric acid cycle",
                "krebs cycle",
                "pentose phosphate",
                "fatty acid synthesis",
                "fatty acid oxidation",
                "amino acid metabolism",
                "nucleotide metabolism",
                "central carbon metabolism",
            ],
            "metabolites": [
                "glucose",
                "pyruvate",
                "acetyl-coa",
                "ATP",
                "NADH",
                "FADH2",
                "succinate",
                "fumarate",
                "malate",
                "oxaloacetate",
                "citrate",
                "biomass",
                "CO2",
                "oxygen",
                "lactate",
                "ethanol",
            ],
            "methods": [
                "FBA",
                "flux balance analysis",
                "FVA",
                "flux variability analysis",
                "MOMA",
                "minimization of metabolic adjustment",
                "pFBA",
                "parsimonious FBA",
                "essentiality",
                "gene knockout",
                "growth rate",
                "yield",
            ],
            "conditions": [
                "aerobic",
                "anaerobic",
                "minimal media",
                "rich media",
                "glucose",
                "acetate",
                "ethanol",
                "glycerol",
                "lactate",
                "succinate",
            ],
        }

    def _initialize_tool_mappings(self) -> Dict[QueryType, List[str]]:
        """Initialize tool mappings for each query type"""
        return {
            QueryType.STRUCTURAL_ANALYSIS: [
                "analyze_metabolic_model",
                "model_statistics",
            ],
            QueryType.GROWTH_ANALYSIS: ["run_metabolic_fba", "growth_analysis"],
            QueryType.PATHWAY_ANALYSIS: ["analyze_pathway", "pathway_visualization"],
            QueryType.FLUX_ANALYSIS: ["run_metabolic_fba", "flux_variability_analysis"],
            QueryType.OPTIMIZATION: ["optimization_analysis", "strain_design"],
            QueryType.COMPARISON: ["compare_models", "comparative_analysis"],
            QueryType.VISUALIZATION: ["generate_visualization", "network_plot"],
            QueryType.MODEL_MODIFICATION: ["modify_model", "gene_knockout"],
            QueryType.GENERAL_QUESTION: ["analyze_metabolic_model"],
            QueryType.HELP_REQUEST: ["help_system"],
        }

    def analyze_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> QueryAnalysis:
        """Perform comprehensive analysis of a user query"""
        start_time = datetime.now()

        # Clean and preprocess query
        cleaned_query = self._preprocess_query(query)

        # Detect query type
        query_type, type_confidence = self._detect_query_type(cleaned_query)

        # Analyze complexity
        complexity = self._analyze_complexity(cleaned_query, query_type)

        # Detect intent
        intent = self._detect_intent(cleaned_query, query_type)

        # Extract entities
        entities = self._extract_entities(cleaned_query)

        # Get tool suggestions
        suggested_tools = self._suggest_tools(query_type, complexity, entities)

        # Generate analysis metadata
        required_inputs = self._analyze_required_inputs(query_type, entities)
        expected_outputs = self._analyze_expected_outputs(query_type, entities)
        context_deps = self._analyze_context_dependencies(query_type, context)

        # Generate suggestions
        clarifications = self._generate_clarification_questions(query_type, entities)
        alternatives = self._generate_alternative_phrasings(query)
        follow_ups = self._generate_follow_up_suggestions(query_type, entities)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            complexity=complexity,
            intent=intent,
            confidence=type_confidence,
            model_references=entities.get("models", []),
            pathway_references=entities.get("pathways", []),
            metabolite_references=entities.get("metabolites", []),
            gene_references=entities.get("genes", []),
            numeric_values=entities.get("numbers", []),
            suggested_tools=suggested_tools,
            required_inputs=required_inputs,
            expected_outputs=expected_outputs,
            context_dependencies=context_deps,
            clarification_questions=clarifications,
            alternative_phrasings=alternatives,
            follow_up_suggestions=follow_ups,
            processing_time=processing_time,
            language_confidence=self._assess_language_confidence(query),
        )

    def _preprocess_query(self, query: str) -> str:
        """Clean and preprocess the query text"""
        # Convert to lowercase
        query = query.lower().strip()

        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query)

        # Normalize common abbreviations
        abbreviations = {
            r"\bfba\b": "flux balance analysis",
            r"\bfva\b": "flux variability analysis",
            r"\btca\b": "citric acid cycle",
            r"\bppp\b": "pentose phosphate pathway",
            r"\batp\b": "ATP",
            r"\bnadh\b": "NADH",
        }

        for abbrev, expansion in abbreviations.items():
            query = re.sub(abbrev, expansion, query)

        return query

    def _detect_query_type(self, query: str) -> Tuple[QueryType, float]:
        """Detect the type of query with confidence score"""
        type_scores = {}

        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches

            if score > 0:
                type_scores[query_type] = score

        if not type_scores:
            return QueryType.GENERAL_QUESTION, 0.5

        # Get the query type with highest score
        best_type = max(type_scores, key=type_scores.get)
        max_score = type_scores[best_type]
        total_score = sum(type_scores.values())

        confidence = max_score / max(total_score, 1)
        return best_type, min(confidence, 1.0)

    def _analyze_complexity(self, query: str, query_type: QueryType) -> QueryComplexity:
        """Analyze query complexity based on content and type"""
        complexity_indicators = {
            "simple": [
                r"\b(what|how many|count|list|show)\b",
                r"\b(basic|simple|quick)\b",
            ],
            "moderate": [
                r"\b(compare|analyze|calculate)\b",
                r"\b(growth.*rate|flux.*analysis)\b",
            ],
            "complex": [
                r"\b(optimize|design|predict|simulate)\b",
                r"\b(multiple|several|various|different)\b.*\b(condition|scenario)\b",
            ],
            "expert": [
                r"\b(algorithm|parameter|constraint|mathematical)\b",
                r"\b(custom|advanced|sophisticated)\b",
            ],
        }

        scores = {}
        for level, patterns in complexity_indicators.items():
            score = sum(
                len(re.findall(pattern, query, re.IGNORECASE)) for pattern in patterns
            )
            scores[level] = score

        # Default complexity based on query type
        type_complexity = {
            QueryType.STRUCTURAL_ANALYSIS: QueryComplexity.SIMPLE,
            QueryType.GROWTH_ANALYSIS: QueryComplexity.MODERATE,
            QueryType.PATHWAY_ANALYSIS: QueryComplexity.MODERATE,
            QueryType.FLUX_ANALYSIS: QueryComplexity.COMPLEX,
            QueryType.OPTIMIZATION: QueryComplexity.EXPERT,
            QueryType.COMPARISON: QueryComplexity.COMPLEX,
            QueryType.VISUALIZATION: QueryComplexity.SIMPLE,
            QueryType.MODEL_MODIFICATION: QueryComplexity.EXPERT,
            QueryType.GENERAL_QUESTION: QueryComplexity.SIMPLE,
            QueryType.HELP_REQUEST: QueryComplexity.SIMPLE,
        }

        # Use pattern-based complexity if strong indicators found
        if scores.get("expert", 0) > 0:
            return QueryComplexity.EXPERT
        elif scores.get("complex", 0) > 0:
            return QueryComplexity.COMPLEX
        elif scores.get("moderate", 0) > 0:
            return QueryComplexity.MODERATE
        elif scores.get("simple", 0) > 0:
            return QueryComplexity.SIMPLE

        return type_complexity.get(query_type, QueryComplexity.MODERATE)

    def _detect_intent(self, query: str, query_type: QueryType) -> QueryIntent:
        """Detect user intent behind the query"""
        intent_patterns = {
            QueryIntent.EXPLORE: [r"\b(explore|investigate|examine|what.*happen)\b"],
            QueryIntent.VALIDATE: [r"\b(validate|verify|check|confirm|test)\b"],
            QueryIntent.OPTIMIZE: [r"\b(optimize|improve|maximize|minimize|enhance)\b"],
            QueryIntent.COMPARE: [r"\b(compare|versus|difference|better|worse)\b"],
            QueryIntent.DIAGNOSE: [r"\b(why|problem|issue|wrong|error|debug)\b"],
            QueryIntent.LEARN: [r"\b(learn|understand|explain|teach|how.*work)\b"],
            QueryIntent.REPORT: [r"\b(report|document|export|save|generate)\b"],
        }

        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(
                len(re.findall(pattern, query, re.IGNORECASE)) for pattern in patterns
            )
            intent_scores[intent] = score

        if intent_scores:
            return max(intent_scores, key=intent_scores.get)

        # Default intent based on query type
        type_intent_mapping = {
            QueryType.STRUCTURAL_ANALYSIS: QueryIntent.EXPLORE,
            QueryType.GROWTH_ANALYSIS: QueryIntent.VALIDATE,
            QueryType.PATHWAY_ANALYSIS: QueryIntent.EXPLORE,
            QueryType.FLUX_ANALYSIS: QueryIntent.VALIDATE,
            QueryType.OPTIMIZATION: QueryIntent.OPTIMIZE,
            QueryType.COMPARISON: QueryIntent.COMPARE,
            QueryType.VISUALIZATION: QueryIntent.REPORT,
            QueryType.MODEL_MODIFICATION: QueryIntent.OPTIMIZE,
            QueryType.GENERAL_QUESTION: QueryIntent.LEARN,
            QueryType.HELP_REQUEST: QueryIntent.LEARN,
        }

        return type_intent_mapping.get(query_type, QueryIntent.EXPLORE)

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract relevant entities from the query"""
        entities = {
            "models": [],
            "pathways": [],
            "metabolites": [],
            "genes": [],
            "numbers": [],
        }

        # Extract model references
        for model in self.metabolic_vocabulary["models"]:
            if model.lower() in query:
                entities["models"].append(model)

        # Extract pathway references
        for pathway in self.metabolic_vocabulary["pathways"]:
            if pathway.lower() in query:
                entities["pathways"].append(pathway)

        # Extract metabolite references
        for metabolite in self.metabolic_vocabulary["metabolites"]:
            if metabolite.lower() in query:
                entities["metabolites"].append(metabolite)

        # Extract numeric values
        numbers = re.findall(r"\b\d+\.?\d*\b", query)
        entities["numbers"] = [float(num) for num in numbers]

        # Extract gene references (pattern-based)
        gene_patterns = [r"\b[a-z]{2,4}[A-Z]\b", r"\bb\d{4}\b"]
        for pattern in gene_patterns:
            genes = re.findall(pattern, query)
            entities["genes"].extend(genes)

        return entities

    def _suggest_tools(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
        entities: Dict[str, List[str]],
    ) -> List[str]:
        """Suggest appropriate tools for the query"""
        base_tools = self.tool_mappings.get(query_type, [])

        # Add complexity-based tools
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            if query_type == QueryType.FLUX_ANALYSIS:
                base_tools.append("advanced_flux_analysis")
            elif query_type == QueryType.OPTIMIZATION:
                base_tools.append("multi_objective_optimization")

        # Add entity-based tools
        if entities.get("pathways"):
            base_tools.append("pathway_analysis")
        if entities.get("models") and len(entities["models"]) > 1:
            base_tools.append("model_comparison")

        return list(set(base_tools))  # Remove duplicates

    def _analyze_required_inputs(
        self, query_type: QueryType, entities: Dict[str, List[str]]
    ) -> List[str]:
        """Analyze what inputs are required for this query"""
        base_inputs = {
            QueryType.STRUCTURAL_ANALYSIS: ["model_file"],
            QueryType.GROWTH_ANALYSIS: ["model_file", "media_conditions"],
            QueryType.PATHWAY_ANALYSIS: ["model_file", "pathway_id"],
            QueryType.FLUX_ANALYSIS: ["model_file", "objective_function"],
            QueryType.OPTIMIZATION: ["model_file", "optimization_target"],
            QueryType.COMPARISON: ["model_files", "comparison_criteria"],
            QueryType.VISUALIZATION: ["analysis_results"],
            QueryType.MODEL_MODIFICATION: ["model_file", "modification_specs"],
        }

        inputs = base_inputs.get(query_type, ["model_file"])

        # Add entity-specific inputs
        if entities.get("metabolites"):
            inputs.append("metabolite_list")
        if entities.get("pathways"):
            inputs.append("pathway_definitions")
        if entities.get("numbers"):
            inputs.append("parameter_values")

        return inputs

    def _analyze_expected_outputs(
        self, query_type: QueryType, entities: Dict[str, List[str]]
    ) -> List[str]:
        """Analyze what outputs are expected from this query"""
        base_outputs = {
            QueryType.STRUCTURAL_ANALYSIS: [
                "model_statistics",
                "reaction_list",
                "metabolite_list",
            ],
            QueryType.GROWTH_ANALYSIS: ["growth_rate", "biomass_composition"],
            QueryType.PATHWAY_ANALYSIS: ["pathway_fluxes", "pathway_diagram"],
            QueryType.FLUX_ANALYSIS: ["flux_distribution", "optimal_value"],
            QueryType.OPTIMIZATION: ["optimal_solution", "optimization_report"],
            QueryType.COMPARISON: ["comparison_table", "difference_analysis"],
            QueryType.VISUALIZATION: ["plot_image", "interactive_visualization"],
            QueryType.MODEL_MODIFICATION: ["modified_model", "modification_report"],
        }

        return base_outputs.get(query_type, ["analysis_report"])

    def _analyze_context_dependencies(
        self, query_type: QueryType, context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Analyze what context dependencies exist"""
        dependencies = []

        if query_type == QueryType.COMPARISON and context:
            if not context.get("previous_models"):
                dependencies.append("previous_model_analysis")

        if query_type == QueryType.OPTIMIZATION and context:
            if not context.get("baseline_flux"):
                dependencies.append("baseline_flux_analysis")

        return dependencies

    def _generate_clarification_questions(
        self, query_type: QueryType, entities: Dict[str, List[str]]
    ) -> List[str]:
        """Generate helpful clarification questions"""
        questions = []

        if query_type == QueryType.GROWTH_ANALYSIS and not entities.get("metabolites"):
            questions.append("What growth medium or carbon source should I use?")

        if query_type == QueryType.OPTIMIZATION and not entities.get("numbers"):
            questions.append("What is your target value or optimization goal?")

        if query_type == QueryType.COMPARISON and len(entities.get("models", [])) < 2:
            questions.append("Which models or conditions would you like to compare?")

        if query_type == QueryType.PATHWAY_ANALYSIS and not entities.get("pathways"):
            questions.append("Which specific pathway are you interested in?")

        return questions

    def _generate_alternative_phrasings(self, query: str) -> List[str]:
        """Generate alternative ways to phrase the query"""
        alternatives = []

        # Simple pattern-based alternatives
        if "how" in query.lower():
            alternatives.append(
                query.replace("How", "What is the method to").replace(
                    "how", "what is the method to"
                )
            )

        if "analyze" in query.lower():
            alternatives.append(query.replace("analyze", "examine"))
            alternatives.append(query.replace("analyze", "investigate"))

        if "calculate" in query.lower():
            alternatives.append(query.replace("calculate", "compute"))
            alternatives.append(query.replace("calculate", "determine"))

        return alternatives[:3]  # Limit to 3 alternatives

    def _generate_follow_up_suggestions(
        self, query_type: QueryType, entities: Dict[str, List[str]]
    ) -> List[str]:
        """Generate relevant follow-up suggestions"""
        suggestions = []

        if query_type == QueryType.STRUCTURAL_ANALYSIS:
            suggestions.extend(
                [
                    "Would you like to analyze growth capabilities next?",
                    "Should I examine specific pathways in this model?",
                    "Would you like to visualize the metabolic network?",
                ]
            )

        elif query_type == QueryType.GROWTH_ANALYSIS:
            suggestions.extend(
                [
                    "Would you like to perform flux variability analysis?",
                    "Should I analyze gene essentiality?",
                    "Would you like to test different media conditions?",
                ]
            )

        elif query_type == QueryType.PATHWAY_ANALYSIS:
            suggestions.extend(
                [
                    "Would you like to optimize this pathway?",
                    "Should I compare with other pathways?",
                    "Would you like to visualize pathway fluxes?",
                ]
            )

        return suggestions[:3]  # Limit to 3 suggestions

    def _assess_language_confidence(self, query: str) -> float:
        """Assess confidence in language understanding"""
        # Simple heuristics for language confidence
        confidence = 1.0

        # Reduce confidence for very short queries
        if len(query.split()) < 3:
            confidence *= 0.7

        # Reduce confidence for queries with many typos/non-words
        words = query.split()
        potential_typos = sum(
            1
            for word in words
            if len(word) > 3
            and not any(
                vocab_word in word.lower()
                for vocab_list in self.metabolic_vocabulary.values()
                for vocab_word in vocab_list
            )
        )

        if potential_typos > len(words) * 0.3:
            confidence *= 0.8

        return confidence

    def display_analysis(self, analysis: QueryAnalysis) -> None:
        """Display query analysis in a beautiful format"""
        # Main analysis panel
        analysis_table = Table(show_header=False, box=box.SIMPLE)
        analysis_table.add_column("Aspect", style="bold cyan")
        analysis_table.add_column("Value", style="bold white")

        analysis_table.add_row(
            "Query Type", analysis.query_type.value.replace("_", " ").title()
        )
        analysis_table.add_row("Complexity", analysis.complexity.value.title())
        analysis_table.add_row("Intent", analysis.intent.value.title())
        analysis_table.add_row("Confidence", f"{analysis.confidence:.1%}")
        analysis_table.add_row("Processing Time", f"{analysis.processing_time:.3f}s")

        console.print(
            Panel(
                analysis_table,
                title="[bold blue]🧠 Query Analysis[/bold blue]",
                border_style="blue",
            )
        )

        # Extracted entities
        if any(
            [
                analysis.model_references,
                analysis.pathway_references,
                analysis.metabolite_references,
                analysis.gene_references,
            ]
        ):
            entities_table = Table(show_header=False, box=box.SIMPLE)
            entities_table.add_column("Type", style="bold yellow")
            entities_table.add_column("Found", style="white")

            if analysis.model_references:
                entities_table.add_row("Models", ", ".join(analysis.model_references))
            if analysis.pathway_references:
                entities_table.add_row(
                    "Pathways", ", ".join(analysis.pathway_references)
                )
            if analysis.metabolite_references:
                entities_table.add_row(
                    "Metabolites", ", ".join(analysis.metabolite_references)
                )
            if analysis.gene_references:
                entities_table.add_row("Genes", ", ".join(analysis.gene_references))
            if analysis.numeric_values:
                entities_table.add_row(
                    "Numbers", ", ".join(map(str, analysis.numeric_values))
                )

            console.print(
                Panel(
                    entities_table,
                    title="[bold green]🔍 Extracted Entities[/bold green]",
                    border_style="green",
                )
            )

        # Suggestions
        if analysis.suggested_tools:
            console.print(
                f"\n[bold yellow]🔧 Suggested Tools:[/bold yellow] {', '.join(analysis.suggested_tools)}"
            )

        if analysis.clarification_questions:
            console.print(f"\n[bold cyan]❓ Clarification Questions:[/bold cyan]")
            for i, question in enumerate(analysis.clarification_questions, 1):
                console.print(f"  {i}. {question}")

        if analysis.follow_up_suggestions:
            console.print(f"\n[bold green]💡 Follow-up Suggestions:[/bold green]")
            for i, suggestion in enumerate(analysis.follow_up_suggestions, 1):
                console.print(f"  {i}. {suggestion}")

    def update_context(self, key: str, value: Any) -> None:
        """Update context memory for future queries"""
        self.context_memory[key] = value
