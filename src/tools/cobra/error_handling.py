"""
Enhanced Error Handling and Debugging Framework for COBRA Tools

This module provides comprehensive error handling, validation, and debugging
utilities to improve user experience and problem resolution.
"""

import difflib
import glob
import importlib
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cobra
from cobra.exceptions import OptimizationError

from ..base import ToolResult

# Set up logging
logger = logging.getLogger(__name__)


def get_available_solvers() -> Dict[str, Any]:
    """
    Get available optimization solvers compatible with current COBRApy version

    Returns:
        Dictionary mapping solver names to their interface modules
    """
    solver_interfaces = [
        "glpk_interface",
        "cplex_interface",
        "gurobi_interface",
        "scipy_interface",
    ]
    available = {}

    for solver_name in solver_interfaces:
        try:
            module = importlib.import_module(f"optlang.{solver_name}")
            if hasattr(module, "Model"):
                try:
                    # Test if solver actually works by creating a simple model
                    _ = module.Model()
                    solver_key = solver_name.replace("_interface", "")
                    available[solver_key] = module
                except Exception:
                    # Solver module exists but not functional
                    pass
        except ImportError:
            # Solver module not installed
            pass

    return available


class ModelValidationError(Exception):
    """Custom exception for model validation issues"""

    def __init__(self, message: str, suggestions: List[str] = None):
        super().__init__(message)
        self.suggestions = suggestions or []


class ParameterValidationError(Exception):
    """Custom exception for parameter validation issues"""

    def __init__(self, message: str, suggestions: List[str] = None):
        super().__init__(message)
        self.suggestions = suggestions or []


def create_error_result(
    message: str,
    error: str,
    suggestions: List[str] = None,
    context: Dict[str, Any] = None,
) -> ToolResult:
    """
    Create a standardized error result with helpful information

    Args:
        message: User-friendly error message
        error: Technical error details
        suggestions: List of actionable suggestions for the user
        context: Additional context information

    Returns:
        ToolResult with comprehensive error information
    """
    return ToolResult(
        success=False,
        message=message,
        error=error,
        data={
            "suggestions": suggestions or [],
            "context": context or {},
            "error_type": (
                type(error).__name__ if hasattr(error, "__class__") else "Unknown"
            ),
        },
    )


def validate_model_path(model_path: Any) -> str:
    """
    Comprehensive model path validation with helpful suggestions

    Args:
        model_path: Path to validate

    Returns:
        Validated model path string

    Raises:
        ParameterValidationError: With specific guidance for fixing issues
    """
    # Type validation
    if not isinstance(model_path, str):
        raise ParameterValidationError(
            f"Model path must be a string, got {type(model_path).__name__}",
            suggestions=[
                "Provide path as string: '/path/to/model.xml'",
                "Ensure path is not None or other data type",
            ],
        )

    # Format validation
    valid_extensions = (".xml", ".sbml", ".json")
    if not model_path.endswith(valid_extensions):
        raise ParameterValidationError(
            f"Model file must have extension {valid_extensions}, got: {model_path}",
            suggestions=[
                "Use SBML format (.xml) for best compatibility",
                "Convert other formats to SBML using appropriate tools",
                "Check file extension spelling and case",
            ],
        )

    # Existence validation with suggestions
    if not os.path.exists(model_path):
        suggestions = []

        # Try to find similar files
        dir_path = os.path.dirname(model_path) or "."
        filename = os.path.basename(model_path)

        if os.path.exists(dir_path):
            # Look for similar filenames
            available_files = [
                f for f in os.listdir(dir_path) if f.endswith(valid_extensions)
            ]
            similar_files = difflib.get_close_matches(
                filename, available_files, n=3, cutoff=0.6
            )

            if similar_files:
                suggestions.append(f"Did you mean: {similar_files[0]}?")
                if len(similar_files) > 1:
                    suggestions.append(f"Other similar files: {similar_files[1:]}")

            if available_files:
                suggestions.append(
                    f"Available model files in {dir_path}: {available_files[:5]}"
                )
            else:
                suggestions.append(f"No model files found in {dir_path}")
        else:
            suggestions.append(f"Directory does not exist: {dir_path}")
            suggestions.append("Check path spelling and ensure directory exists")

        # Add general suggestions
        suggestions.extend(
            [
                "Verify the complete file path is correct",
                "Check file permissions and accessibility",
                "Ensure you're running from the correct working directory",
            ]
        )

        raise ParameterValidationError(
            f"Model file not found: {model_path}", suggestions=suggestions
        )

    # File size check
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        raise ParameterValidationError(
            f"Model file is empty: {model_path}",
            suggestions=[
                "Check if file download completed successfully",
                "Verify file wasn't corrupted during transfer",
                "Try re-downloading or regenerating the model file",
            ],
        )

    # Warn about very large files
    if file_size > 100 * 1024 * 1024:  # 100MB
        logger.warning(
            f"Large model file detected ({file_size / 1024 / 1024:.1f} MB). "
            f"Analysis may take longer than usual."
        )

    return model_path


def validate_numerical_parameters(
    params: Dict[str, Any], tool_name: str
) -> Dict[str, Any]:
    """
    Validate numerical parameters with tool-specific guidance

    Args:
        params: Dictionary of parameters to validate
        tool_name: Name of the tool for context-specific validation

    Returns:
        Validated parameters dictionary

    Raises:
        ParameterValidationError: With specific parameter guidance
    """
    validated = {}

    # General threshold validation
    for param_name, value in params.items():
        if not isinstance(value, (int, float, type(None))):
            continue

        if value is None:
            continue

        # Threshold parameters
        if "threshold" in param_name.lower():
            if value < 0:
                raise ParameterValidationError(
                    f"Parameter '{param_name}' must be non-negative, got {value}",
                    suggestions=[
                        "Use positive values for thresholds",
                        "Typical flux thresholds: 1e-6 to 1e-3",
                        "Typical growth thresholds: 1e-6 to 1e-3",
                        "Set to 0 to include all values",
                    ],
                )

            if value > 1.0 and "growth" in param_name.lower():
                logger.warning(
                    f"Growth threshold {value} > 1.0 is unusual. "
                    f"Typical values are 1e-6 to 1e-3 for absolute thresholds."
                )

        # Fraction parameters
        if "fraction" in param_name.lower():
            if not 0 <= value <= 1:
                raise ParameterValidationError(
                    f"Parameter '{param_name}' must be between 0 and 1, got {value}",
                    suggestions=[
                        "Fractions represent percentages: 0.01 = 1%, 0.1 = 10%",
                        "For essentiality: 0.01 (1% of wild-type) is typical",
                        "For flux variability: 0.9 (90% of optimal) is common",
                    ],
                )

        validated[param_name] = value

    # Tool-specific validation
    if tool_name == "essentiality_analysis":
        if "essentiality_threshold" in params:
            threshold = params["essentiality_threshold"]
            if threshold is not None and (threshold < 0.001 or threshold > 0.2):
                logger.warning(
                    f"Essentiality threshold {threshold} is outside typical range [0.001, 0.2]. "
                    f"Recommended: 0.01 (strict), 0.05 (moderate), 0.1 (lenient)"
                )

    elif tool_name == "flux_variability_analysis":
        if "fraction_of_optimum" in params:
            fraction = params["fraction_of_optimum"]
            if fraction is not None and fraction < 0.5:
                logger.warning(
                    f"Very low fraction_of_optimum ({fraction}) may result in very wide flux ranges. "
                    f"Consider values ≥ 0.9 for realistic biological constraints."
                )

    elif tool_name == "flux_sampling":
        if "n_samples" in params:
            n_samples = params["n_samples"]
            if n_samples is not None:
                if n_samples < 100:
                    logger.warning(
                        f"Low sample count ({n_samples}) may give unreliable statistics. "
                        f"Recommended: ≥ 1000 samples for robust analysis."
                    )
                elif n_samples > 10000:
                    logger.warning(
                        f"High sample count ({n_samples}) will increase computation time significantly."
                    )

    return validated


def diagnose_optimization_failure(
    model: cobra.Model, operation: str
) -> Tuple[str, List[str]]:
    """
    Diagnose common causes of optimization failure

    Args:
        model: COBRApy model that failed optimization
        operation: Description of the operation that failed

    Returns:
        Tuple of (diagnosis message, list of suggestions)
    """
    issues = []
    suggestions = []

    # Check exchange reactions
    exchanges = [rxn for rxn in model.reactions if rxn.id.startswith("EX_")]
    uptake_allowed = [rxn for rxn in exchanges if rxn.lower_bound < 0]

    if not exchanges:
        issues.append("No exchange reactions found")
        suggestions.extend(
            [
                "Model may be missing exchange reactions for nutrients",
                "Check if model uses different exchange reaction naming convention",
                "Ensure model includes proper boundary conditions",
            ]
        )
    elif not uptake_allowed:
        issues.append("No uptake reactions allowed (all exchange bounds ≥ 0)")
        suggestions.extend(
            [
                "Open essential nutrient uptake: model.reactions.get_by_id('EX_glc__D_e').lower_bound = -1000",
                "Check medium composition and exchange reaction bounds",
                "Use model.medium to set up growth medium",
            ]
        )

    # Check for essential nutrients
    essential_nutrients = ["EX_glc__D_e", "EX_o2_e", "EX_nh4_e", "EX_pi_e", "EX_so4_e"]
    blocked_nutrients = []

    for nutrient in essential_nutrients:
        if nutrient in model.reactions:
            rxn = model.reactions.get_by_id(nutrient)
            if rxn.lower_bound >= 0:
                blocked_nutrients.append(nutrient)

    if blocked_nutrients:
        issues.append(f"Essential nutrients blocked: {blocked_nutrients}")
        suggestions.extend(
            [
                f"Allow uptake: model.reactions.get_by_id('{blocked_nutrients[0]}').lower_bound = -1000",
                "Set up complete growth medium with all essential nutrients",
                "Use model.medium = model.medium to reset to default medium",
            ]
        )

    # Check objective function
    objective_reactions = [
        rxn for rxn in model.reactions if rxn.objective_coefficient != 0
    ]
    if not objective_reactions:
        issues.append("No objective function defined")
        suggestions.extend(
            [
                "Set biomass objective: model.objective = 'biomass_reaction_id'",
                "Find biomass reaction: [r.id for r in model.reactions if 'biomass' in r.id.lower()]",
                "Or set any reaction as objective: model.reactions.get_by_id('reaction_id').objective_coefficient = 1",
            ]
        )
    elif len(objective_reactions) == 1:
        obj_rxn = objective_reactions[0]
        if all(coeff == 0 for coeff in obj_rxn.bounds):
            issues.append(f"Objective reaction '{obj_rxn.id}' has zero bounds")
            suggestions.append(
                f"Fix objective bounds: model.reactions.get_by_id('{obj_rxn.id}').bounds = (0, 1000)"
            )

    # Check for obvious constraint conflicts
    infeasible_reactions = []
    for rxn in model.reactions:
        if rxn.lower_bound > rxn.upper_bound:
            infeasible_reactions.append(rxn.id)

    if infeasible_reactions:
        issues.append(f"Reactions with impossible bounds: {infeasible_reactions[:3]}")
        suggestions.append("Fix reaction bounds: lower_bound must be ≤ upper_bound")

    # Generate diagnosis
    if issues:
        diagnosis = f"{operation} failed. Detected issues: " + "; ".join(issues)
    else:
        diagnosis = f"{operation} failed due to model infeasibility (cause unclear)"
        suggestions.extend(
            [
                "Try different solver: model.solver = 'cplex' or 'gurobi'",
                "Check for subtle constraint conflicts",
                "Validate model with: cobra.flux_analysis.find_blocked_reactions(model)",
            ]
        )

    return diagnosis, suggestions


def safe_optimize(
    model: cobra.Model, operation: str = "Optimization"
) -> cobra.Solution:
    """
    Safely optimize a model with comprehensive error handling

    Args:
        model: COBRApy model to optimize
        operation: Description of the operation for error messages

    Returns:
        Optimization solution

    Raises:
        OptimizationError: With detailed diagnosis and suggestions
    """
    try:
        solution = model.optimize()

        if solution.status == "optimal":
            # Check for near-zero growth
            if solution.objective_value < 1e-10:
                logger.warning(
                    f"{operation}: Growth is very low ({solution.objective_value:.2e}). "
                    f"Check medium composition and model constraints."
                )
            return solution

        elif solution.status == "infeasible":
            diagnosis, suggestions = diagnose_optimization_failure(model, operation)
            raise OptimizationError(f"{diagnosis}. Suggestions: {suggestions[:3]}")

        elif solution.status == "unbounded":
            raise OptimizationError(
                f"{operation} failed: Model is unbounded. "
                f"Missing constraints on exchange reactions. "
                f"Set reasonable bounds: model.reactions.get_by_id('EX_rxn').bounds = (-1000, 1000)"
            )

        else:
            available_solvers = list(get_available_solvers().keys())
            raise OptimizationError(
                f"{operation} failed with status: {solution.status}. "
                f"Try different solver from: {available_solvers}"
            )

    except Exception as e:
        if "solver" in str(e).lower() or "glpk" in str(e).lower():
            available_solvers = list(get_available_solvers().keys())
            raise OptimizationError(
                f"{operation} failed due to solver issue: {str(e)}. "
                f"Available solvers: {available_solvers}. "
                f"Install solvers: pip install python-glpk-cffi"
            )
        raise


def validate_solver_availability(solver: str) -> str:
    """
    Validate solver availability and suggest alternatives

    Args:
        solver: Name of the solver to validate

    Returns:
        Available solver name (may be different from input if fallback used)

    Raises:
        RuntimeError: If no solvers are available
    """
    available_solvers = list(get_available_solvers().keys())

    if not available_solvers:
        raise RuntimeError(
            "No optimization solvers available. "
            "Install at least one solver: "
            "pip install python-glpk-cffi (for GLPK) or "
            "pip install cplex (for CPLEX) or "
            "pip install gurobipy (for Gurobi)"
        )

    if solver not in available_solvers:
        fallback = available_solvers[0]
        logger.warning(
            f"Requested solver '{solver}' not available. "
            f"Using '{fallback}' instead. Available: {available_solvers}"
        )
        return fallback

    return solver


def create_progress_logger(total_items: int, operation: str) -> callable:
    """
    Create a progress logging function for long operations

    Args:
        total_items: Total number of items to process
        operation: Description of the operation

    Returns:
        Function to call with current progress
    """

    def log_progress(current: int, extra_info: str = ""):
        if total_items <= 1:
            return

        percentage = (current / total_items) * 100

        # Log at specific intervals
        if current == 0:
            logger.info(f"Starting {operation} for {total_items} items")
        elif current % max(1, total_items // 20) == 0 or current == total_items:
            logger.info(
                f"{operation} progress: {current}/{total_items} ({percentage:.1f}%) {extra_info}"
            )

    return log_progress
