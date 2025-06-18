"""
Centralized Prompt Registry for ModelSEEDagent

This module provides a centralized system for managing all AI prompts with
version control, A/B testing capabilities, and usage tracking.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PromptCategory(Enum):
    """Categories for organizing prompts by function"""

    TOOL_SELECTION = "tool_selection"
    RESULT_ANALYSIS = "result_analysis"
    WORKFLOW_PLANNING = "workflow_planning"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    SYNTHESIS = "synthesis"
    QUALITY_ASSESSMENT = "quality_assessment"
    SYSTEM_CONFIGURATION = "system_configuration"
    INTERACTION = "interaction"


@dataclass
class PromptTemplate:
    """Structure for prompt templates with metadata"""

    prompt_id: str
    category: PromptCategory
    version: str
    description: str
    template: str
    variables: List[str]
    validation_rules: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    usage_count: int = 0
    success_rate: float = 0.0
    avg_response_time: float = 0.0


@dataclass
class PromptUsage:
    """Track individual prompt usage for analytics"""

    prompt_id: str
    version: str
    timestamp: datetime
    variables: Dict[str, Any]
    response_time: float
    success: bool
    context: Dict[str, Any]
    reasoning_quality: Optional[float] = None


@dataclass
class ABTestConfiguration:
    """Configuration for A/B testing prompts"""

    test_id: str
    prompt_a_id: str
    prompt_b_id: str
    traffic_split: float  # 0.0-1.0, percentage going to A
    start_date: datetime
    end_date: Optional[datetime]
    metrics: List[str]
    active: bool = True


class PromptRegistry:
    """Centralized prompt management system"""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent / "config"
        self.config_dir.mkdir(exist_ok=True)

        # Core storage
        self.prompts: Dict[str, PromptTemplate] = {}
        self.usage_history: List[PromptUsage] = []
        self.ab_tests: Dict[str, ABTestConfiguration] = {}

        # Performance tracking
        self.performance_cache: Dict[str, Dict[str, float]] = {}

        # Load existing configuration
        self._load_configuration()

    def register_prompt(
        self,
        prompt_id: str,
        template: str,
        category: PromptCategory,
        description: str,
        variables: List[str],
        validation_rules: Optional[Dict[str, Any]] = None,
        version: str = "1.0",
    ) -> bool:
        """Register a new prompt template"""
        try:
            # Validate inputs
            if not prompt_id or not template:
                raise ValueError("prompt_id and template are required")

            if prompt_id in self.prompts:
                logger.warning(f"Prompt {prompt_id} already exists, updating version")
                # Auto-increment version if prompt exists
                existing_version = self.prompts[prompt_id].version
                major, minor = map(int, existing_version.split("."))
                version = f"{major}.{minor + 1}"

            # Create prompt template
            prompt = PromptTemplate(
                prompt_id=prompt_id,
                category=category,
                version=version,
                description=description,
                template=template,
                variables=variables,
                validation_rules=validation_rules or {},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            # Validate template
            if not self._validate_template(prompt):
                raise ValueError(f"Template validation failed for {prompt_id}")

            self.prompts[prompt_id] = prompt
            self._save_configuration()

            logger.info(
                f"Registered prompt {prompt_id} v{version} in category {category.value}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register prompt {prompt_id}: {e}")
            return False

    def get_prompt(
        self,
        prompt_id: str,
        variables: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """
        Retrieve and format a prompt with usage tracking

        Returns:
            Tuple of (formatted_prompt, version_used)
        """
        start_time = time.time()

        try:
            # Check for A/B test
            active_prompt_id = self._get_ab_test_prompt(prompt_id)

            if active_prompt_id not in self.prompts:
                raise KeyError(f"Prompt {active_prompt_id} not found")

            prompt = self.prompts[active_prompt_id]
            variables = variables or {}

            # Validate required variables
            missing_vars = set(prompt.variables) - set(variables.keys())
            if missing_vars:
                raise ValueError(f"Missing required variables: {missing_vars}")

            # Format template
            formatted_prompt = prompt.template.format(**variables)

            # Track usage
            usage = PromptUsage(
                prompt_id=active_prompt_id,
                version=prompt.version,
                timestamp=datetime.now(),
                variables=variables,
                response_time=time.time() - start_time,
                success=True,
                context=context or {},
            )

            self.usage_history.append(usage)
            prompt.usage_count += 1
            prompt.updated_at = datetime.now()

            # Update performance metrics
            self._update_performance_metrics(active_prompt_id, usage)

            return formatted_prompt, prompt.version

        except Exception as e:
            # Track failed usage
            usage = PromptUsage(
                prompt_id=prompt_id,
                version="unknown",
                timestamp=datetime.now(),
                variables=variables or {},
                response_time=time.time() - start_time,
                success=False,
                context=context or {},
            )
            self.usage_history.append(usage)

            logger.error(f"Failed to get prompt {prompt_id}: {e}")
            raise

    def track_prompt_outcome(
        self,
        prompt_id: str,
        success: bool,
        reasoning_quality: Optional[float] = None,
        response_time: Optional[float] = None,
    ) -> None:
        """Track the outcome of a prompt usage for quality metrics"""
        try:
            if prompt_id in self.prompts:
                prompt = self.prompts[prompt_id]

                # Update success rate
                total_uses = prompt.usage_count
                if total_uses > 0:
                    current_success_rate = prompt.success_rate
                    new_success_rate = (
                        current_success_rate * (total_uses - 1)
                        + (1.0 if success else 0.0)
                    ) / total_uses
                    prompt.success_rate = new_success_rate

                # Update response time
                if response_time is not None:
                    if prompt.avg_response_time == 0.0:
                        prompt.avg_response_time = response_time
                    else:
                        prompt.avg_response_time = (prompt.avg_response_time * 0.9) + (
                            response_time * 0.1
                        )

                # Update latest usage with quality score
                if self.usage_history and reasoning_quality is not None:
                    latest_usage = self.usage_history[-1]
                    if latest_usage.prompt_id == prompt_id:
                        latest_usage.reasoning_quality = reasoning_quality

                self._save_configuration()

        except Exception as e:
            logger.error(f"Failed to track outcome for prompt {prompt_id}: {e}")

    def setup_ab_test(
        self,
        test_id: str,
        prompt_a_id: str,
        prompt_b_id: str,
        traffic_split: float = 0.5,
        duration_days: Optional[int] = None,
        metrics: Optional[List[str]] = None,
    ) -> bool:
        """Set up A/B testing between two prompts"""
        try:
            if prompt_a_id not in self.prompts or prompt_b_id not in self.prompts:
                raise ValueError("Both prompts must be registered before A/B testing")

            if not 0.0 <= traffic_split <= 1.0:
                raise ValueError("Traffic split must be between 0.0 and 1.0")

            end_date = None
            if duration_days:
                from datetime import timedelta

                end_date = datetime.now() + timedelta(days=duration_days)

            ab_test = ABTestConfiguration(
                test_id=test_id,
                prompt_a_id=prompt_a_id,
                prompt_b_id=prompt_b_id,
                traffic_split=traffic_split,
                start_date=datetime.now(),
                end_date=end_date,
                metrics=metrics or ["success_rate", "reasoning_quality"],
                active=True,
            )

            self.ab_tests[test_id] = ab_test
            self._save_configuration()

            logger.info(f"Started A/B test {test_id}: {prompt_a_id} vs {prompt_b_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to setup A/B test {test_id}: {e}")
            return False

    def get_prompt_analytics(self, prompt_id: str) -> Dict[str, Any]:
        """Get analytics for a specific prompt"""
        if prompt_id not in self.prompts:
            return {}

        prompt = self.prompts[prompt_id]

        # Calculate usage statistics
        recent_usage = [u for u in self.usage_history if u.prompt_id == prompt_id]

        return {
            "prompt_id": prompt_id,
            "version": prompt.version,
            "category": prompt.category.value,
            "total_usage": prompt.usage_count,
            "success_rate": prompt.success_rate,
            "avg_response_time": prompt.avg_response_time,
            "recent_usage_count": len(recent_usage),
            "avg_reasoning_quality": self._calculate_avg_reasoning_quality(
                recent_usage
            ),
            "performance_trend": self._calculate_performance_trend(prompt_id),
            "created_at": prompt.created_at.isoformat(),
            "last_used": prompt.updated_at.isoformat(),
        }

    def get_category_analytics(self, category: PromptCategory) -> Dict[str, Any]:
        """Get analytics for all prompts in a category"""
        category_prompts = [p for p in self.prompts.values() if p.category == category]

        if not category_prompts:
            return {"category": category.value, "prompt_count": 0}

        total_usage = sum(p.usage_count for p in category_prompts)
        avg_success_rate = sum(p.success_rate for p in category_prompts) / len(
            category_prompts
        )
        avg_response_time = sum(p.avg_response_time for p in category_prompts) / len(
            category_prompts
        )

        return {
            "category": category.value,
            "prompt_count": len(category_prompts),
            "total_usage": total_usage,
            "avg_success_rate": avg_success_rate,
            "avg_response_time": avg_response_time,
            "prompts": [p.prompt_id for p in category_prompts],
        }

    def export_prompts(
        self, category: Optional[PromptCategory] = None
    ) -> Dict[str, Any]:
        """Export prompts for backup or migration"""
        prompts_to_export = self.prompts.values()

        if category:
            prompts_to_export = [p for p in prompts_to_export if p.category == category]

        return {
            "export_timestamp": datetime.now().isoformat(),
            "prompt_count": len(prompts_to_export),
            "prompts": [asdict(p) for p in prompts_to_export],
            "ab_tests": [asdict(test) for test in self.ab_tests.values()],
            "usage_summary": {
                "total_usage_records": len(self.usage_history),
                "categories": [cat.value for cat in PromptCategory],
            },
        }

    def _validate_template(self, prompt: PromptTemplate) -> bool:
        """Validate a prompt template"""
        try:
            # Check for required variables in template
            import re

            template_vars = re.findall(r"\{(\w+)\}", prompt.template)

            # Verify all declared variables are used
            unused_vars = set(prompt.variables) - set(template_vars)
            if unused_vars:
                logger.warning(f"Unused variables in {prompt.prompt_id}: {unused_vars}")

            # Test formatting with dummy variables
            dummy_vars = {var: "test" for var in prompt.variables}
            formatted = prompt.template.format(**dummy_vars)

            # Check validation rules
            for rule, value in prompt.validation_rules.items():
                if rule == "min_length" and len(formatted) < value:
                    return False
                elif rule == "max_length" and len(formatted) > value:
                    return False
                elif rule == "must_contain" and value not in formatted:
                    return False

            return True

        except Exception as e:
            logger.error(f"Template validation failed: {e}")
            return False

    def _get_ab_test_prompt(self, prompt_id: str) -> str:
        """Determine which prompt to use based on active A/B tests"""
        for test in self.ab_tests.values():
            if test.active and (
                test.prompt_a_id == prompt_id or test.prompt_b_id == prompt_id
            ):
                # Check if test is still active
                if test.end_date and datetime.now() > test.end_date:
                    test.active = False
                    continue

                # Simple hash-based traffic splitting
                import hashlib

                hash_input = f"{prompt_id}{datetime.now().strftime('%Y%m%d%H')}"
                hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

                if (hash_value % 100) / 100.0 < test.traffic_split:
                    return test.prompt_a_id
                else:
                    return test.prompt_b_id

        return prompt_id

    def _update_performance_metrics(self, prompt_id: str, usage: PromptUsage) -> None:
        """Update performance metrics for a prompt"""
        if prompt_id not in self.performance_cache:
            self.performance_cache[prompt_id] = {
                "response_times": [],
                "success_count": 0,
                "total_count": 0,
            }

        cache = self.performance_cache[prompt_id]
        cache["response_times"].append(usage.response_time)
        cache["total_count"] += 1

        if usage.success:
            cache["success_count"] += 1

        # Keep only recent data (last 100 uses)
        if len(cache["response_times"]) > 100:
            cache["response_times"] = cache["response_times"][-100:]

    def _calculate_avg_reasoning_quality(
        self, usage_records: List[PromptUsage]
    ) -> Optional[float]:
        """Calculate average reasoning quality from usage records"""
        quality_scores = [
            u.reasoning_quality
            for u in usage_records
            if u.reasoning_quality is not None
        ]

        if not quality_scores:
            return None

        return sum(quality_scores) / len(quality_scores)

    def _calculate_performance_trend(self, prompt_id: str) -> str:
        """Calculate performance trend for a prompt"""
        if prompt_id not in self.performance_cache:
            return "insufficient_data"

        cache = self.performance_cache[prompt_id]
        response_times = cache["response_times"]

        if len(response_times) < 10:
            return "insufficient_data"

        # Simple trend calculation
        recent_avg = sum(response_times[-10:]) / 10
        overall_avg = sum(response_times) / len(response_times)

        if recent_avg < overall_avg * 0.9:
            return "improving"
        elif recent_avg > overall_avg * 1.1:
            return "degrading"
        else:
            return "stable"

    def _load_configuration(self) -> None:
        """Load configuration from disk"""
        try:
            config_file = self.config_dir / "prompt_registry.json"
            if config_file.exists():
                with open(config_file, "r") as f:
                    data = json.load(f)

                # Load prompts
                for prompt_data in data.get("prompts", []):
                    # Convert category string back to enum
                    prompt_data["category"] = PromptCategory(prompt_data["category"])
                    # Convert datetime strings back to datetime objects
                    prompt_data["created_at"] = datetime.fromisoformat(
                        prompt_data["created_at"]
                    )
                    prompt_data["updated_at"] = datetime.fromisoformat(
                        prompt_data["updated_at"]
                    )

                    prompt = PromptTemplate(**prompt_data)
                    self.prompts[prompt.prompt_id] = prompt

                # Load A/B tests
                for test_data in data.get("ab_tests", []):
                    test_data["start_date"] = datetime.fromisoformat(
                        test_data["start_date"]
                    )
                    if test_data["end_date"]:
                        test_data["end_date"] = datetime.fromisoformat(
                            test_data["end_date"]
                        )

                    test = ABTestConfiguration(**test_data)
                    self.ab_tests[test.test_id] = test

                logger.info(
                    f"Loaded {len(self.prompts)} prompts and {len(self.ab_tests)} A/B tests"
                )

        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")
            # Start with empty configuration

    def _save_configuration(self) -> None:
        """Save configuration to disk"""
        try:
            config_file = self.config_dir / "prompt_registry.json"

            # Prepare data for serialization
            prompts_data = []
            for prompt in self.prompts.values():
                prompt_dict = asdict(prompt)
                prompt_dict["category"] = prompt.category.value
                prompt_dict["created_at"] = prompt.created_at.isoformat()
                prompt_dict["updated_at"] = prompt.updated_at.isoformat()
                prompts_data.append(prompt_dict)

            ab_tests_data = []
            for test in self.ab_tests.values():
                test_dict = asdict(test)
                test_dict["start_date"] = test.start_date.isoformat()
                if test.end_date:
                    test_dict["end_date"] = test.end_date.isoformat()
                ab_tests_data.append(test_dict)

            data = {
                "prompts": prompts_data,
                "ab_tests": ab_tests_data,
                "last_updated": datetime.now().isoformat(),
            }

            with open(config_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")


# Global registry instance
_global_registry: Optional[PromptRegistry] = None


def get_prompt_registry() -> PromptRegistry:
    """Get the global prompt registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = PromptRegistry()
    return _global_registry


def reset_prompt_registry() -> None:
    """Reset the global registry (mainly for testing)"""
    global _global_registry
    _global_registry = None
