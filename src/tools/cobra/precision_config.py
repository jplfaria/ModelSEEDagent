"""
Unified Numerical Precision Configuration for COBRA Tools

This module provides consistent numerical precision settings across all
metabolic modeling tools to ensure reliable and reproducible results.
"""

from pydantic import BaseModel, Field


class PrecisionConfig(BaseModel):
    """
    Unified precision configuration for all COBRA tools

    This ensures consistent numerical handling across different analysis tools
    and prevents precision-related errors in metabolic modeling calculations.
    """

    model_config = {"protected_namespaces": ()}

    # Solver numerical tolerance (very small - for solver accuracy)
    model_tolerance: float = Field(
        default=1e-9,
        description="Numerical tolerance for optimization solver",
        ge=1e-12,
        le=1e-6,
    )

    # Analysis thresholds (larger - for biological significance)
    flux_threshold: float = Field(
        default=1e-6,
        description="Minimum significant flux magnitude for reporting",
        ge=1e-9,
        le=1e-3,
    )

    growth_threshold: float = Field(
        default=1e-6,
        description="Minimum significant growth rate for viability",
        ge=1e-9,
        le=1e-3,
    )

    # Specialized thresholds
    essentiality_growth_fraction: float = Field(
        default=0.01,
        description="Growth fraction threshold for essentiality (1% of wild-type)",
        ge=0.001,
        le=0.1,
    )

    correlation_threshold: float = Field(
        default=0.7,
        description="Statistical correlation threshold for flux relationships",
        ge=0.5,
        le=0.95,
    )

    # Numerical comparison tolerance (smallest - for floating point comparisons)
    numerical_tolerance: float = Field(
        default=1e-12,
        description="Tolerance for numerical floating-point comparisons",
        ge=1e-15,
        le=1e-9,
    )

    # Gene deletion effect thresholds (as fractions of wild-type growth)
    essential_growth_fraction: float = Field(
        default=0.01,
        description="Growth fraction below which gene is considered essential",
        ge=0.001,
        le=0.05,
    )

    severe_effect_fraction: float = Field(
        default=0.10,
        description="Growth fraction threshold for severe deletion effect",
        ge=0.05,
        le=0.25,
    )

    moderate_effect_fraction: float = Field(
        default=0.50,
        description="Growth fraction threshold for moderate deletion effect",
        ge=0.25,
        le=0.75,
    )

    mild_effect_fraction: float = Field(
        default=0.90,
        description="Growth fraction threshold for mild deletion effect",
        ge=0.75,
        le=0.99,
    )


def get_default_precision() -> PrecisionConfig:
    """Get default precision configuration"""
    return PrecisionConfig()


def create_precision_config(**kwargs) -> PrecisionConfig:
    """Create precision configuration with custom values"""
    return PrecisionConfig(**kwargs)


def safe_divide(
    numerator: float, denominator: float, tolerance: float = 1e-12
) -> float:
    """
    Perform safe division with numerical tolerance

    Args:
        numerator: Dividend
        denominator: Divisor
        tolerance: Minimum absolute value for denominator

    Returns:
        Division result or float('inf') if denominator too small
    """
    if abs(denominator) > tolerance:
        return numerator / denominator
    else:
        if abs(numerator) > tolerance:
            return float("inf") if numerator * denominator >= 0 else float("-inf")
        else:
            return 0.0  # 0/0 case


def is_significant_flux(flux: float, threshold: float = 1e-6) -> bool:
    """Check if flux magnitude is above significance threshold"""
    return abs(flux) > threshold


def is_significant_growth(growth: float, threshold: float = 1e-6) -> bool:
    """Check if growth rate is above significance threshold"""
    return growth > threshold


def calculate_growth_fraction(
    knockout_growth: float, wild_type_growth: float, tolerance: float = 1e-12
) -> float:
    """
    Calculate growth fraction with numerical safety

    Args:
        knockout_growth: Growth rate after gene deletion
        wild_type_growth: Wild-type growth rate
        tolerance: Numerical tolerance for denominator check

    Returns:
        Growth fraction (knockout/wild-type) with safe handling
    """
    return safe_divide(knockout_growth, wild_type_growth, tolerance)
