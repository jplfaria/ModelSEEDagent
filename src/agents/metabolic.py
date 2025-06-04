import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Additional imports for vector store functionality
import numpy as np
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config.prompts import load_prompt_template, load_prompts
from ..llm.base import BaseLLM
from ..tools.base import BaseTool, ToolResult
from .base import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger(__name__)


# -------------------------------
# Simulation Results Store
# -------------------------------
class SimulationResultsStore:
    """
    A store for simulation results from COBRApy simulations (FBA, pFBA, geometric FBA, FVA, etc.)
    It saves results in a JSON format and exports fluxes as CSV files.
    """

    def __init__(self):
        self.results = {}

    def save_results(
        self,
        tool_name: str,
        model_id: str,
        solution,
        additional_metadata: Dict[str, Any] = None,
    ) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        result_id = f"{tool_name}_{model_id}_{timestamp}"
        # Assume solution.fluxes is a pandas Series; convert to dict
        result_data = {
            "objective_value": float(solution.objective_value),
            "fluxes": solution.fluxes.to_dict(),
            "status": solution.status,
            "timestamp": timestamp,
            "model_id": model_id,
            "tool": tool_name,
            "metadata": additional_metadata or {},
        }
        self.results[result_id] = result_data
        logger.info(
            f"SimulationResultsStore: Saved results for {tool_name} with ID {result_id}."
        )
        return result_id

    def export_results(self, result_id: str, output_dir: str) -> Dict[str, str]:
        """Exports the simulation result as a JSON file and fluxes as a CSV file."""
        if result_id not in self.results:
            logger.error(
                f"SimulationResultsStore: No results found for ID {result_id}."
            )
            return {}
        os.makedirs(output_dir, exist_ok=True)
        result = self.results[result_id]
        json_filename = f"{result_id}.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        # Export fluxes as CSV
        try:
            import pandas as pd

            flux_df = pd.DataFrame.from_dict(
                result["fluxes"], orient="index", columns=["flux"]
            )
            csv_filename = f"{result_id}_fluxes.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            flux_df.to_csv(csv_path)
        except Exception as e:
            logger.error(f"SimulationResultsStore: Error exporting fluxes CSV: {e}")
            csv_path = ""
        logger.info(
            f"SimulationResultsStore: Exported results to {json_path} and {csv_path}."
        )
        return {"json_file": json_path, "csv_file": csv_path}


# -------------------------------
# End SimulationResultsStore
# -------------------------------


# -------------------------------
# Simple Vector Store for Memory
# -------------------------------
class SimpleVectorStore:
    """
    A simple vector store using TF-IDF embeddings for retrieval.
    """

    def __init__(self):
        self.documents: List[str] = []
        self.vectorizer = TfidfVectorizer()
        self.embeddings = None

    def add_document(self, text: str) -> None:
        self.documents.append(text)
        self.embeddings = self.vectorizer.fit_transform(self.documents)
        logger.info(
            f"VectorStore: Added document (len={len(text)}). Total documents: {len(self.documents)}."
        )

    def query(self, query_text: str, top_n: int = 1) -> List[str]:
        if not self.documents or self.embeddings is None:
            logger.info("VectorStore: No documents available for query.")
            return []
        query_vec = self.vectorizer.transform([query_text])
        sim = cosine_similarity(query_vec, self.embeddings)
        top_indices = np.argsort(sim[0])[-top_n:][::-1]
        logger.info(f"VectorStore: Query returned {len(top_indices)} results.")
        return [self.documents[i] for i in top_indices]


# -------------------------------
# End SimpleVectorStore
# -------------------------------


class CustomReActOutputParser(ReActSingleInputOutputParser):
    """Custom parser for combined Action/Final Answer responses."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "For troubleshooting" in text:
            text = text.split("For troubleshooting")[0].strip()
        sequences = re.split(r"\nThought:", text)
        if not sequences:
            return AgentFinish({"output": "No valid response generated"}, "")
        last_sequence = sequences[-1].strip()
        if "Final Answer:" in last_sequence:
            final_answer = last_sequence.split("Final Answer:")[-1].strip()
            return AgentFinish({"output": final_answer}, text)
        if "Action:" in last_sequence and "Action Input:" in last_sequence:
            action_match = re.search(r"Action: (.*?)(?:\n|$)", last_sequence)
            input_match = re.search(r"Action Input: (.*?)(?:\n|$)", last_sequence)
            if action_match and input_match:
                action = action_match.group(1).strip()
                action_input = input_match.group(1).strip()
                return AgentAction(action, action_input, text)
        return AgentFinish({"output": last_sequence}, text)


class MetabolicAgent(BaseAgent):
    """Agent for metabolic model analysis with enhanced memory, token management, simulation result export, and logging."""

    def __init__(
        self, llm: BaseLLM, tools: List[BaseTool], config: Dict[str, Any] | AgentConfig
    ):
        # Initialize variables needed before super().__init__
        self._current_tool_results = "No previous results"
        self._tools_dict = {t.tool_name: t for t in tools}
        self._tool_output_counters = {}

        # Set up logging directory structure BEFORE super().__init__
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_base = Path(__file__).parent.parent.parent / "logs"
        self.run_dir = self.log_base / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create directories for the new structure
        self.results_dir = self.run_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.observations_dir = self.run_dir / "observations"
        self.observations_dir.mkdir(exist_ok=True)
        self.steps_dir = self.run_dir / "steps"
        self.steps_dir.mkdir(exist_ok=True)

        # Call super().__init__ which will trigger _setup_agent()
        super().__init__(llm, tools, config)

        # Create the execution log
        self.exec_log = self.run_dir / "execution.json"
        with open(self.exec_log, "w") as f:
            json.dump([], f)

        self.current_iteration = 0
        self._last_query = ""
        self.vector_store = SimpleVectorStore()
        self.simulation_store = SimulationResultsStore()

        logger.info(f"Created run directory with improved structure: {self.run_dir}")
        print(f"DEBUG: Run directory created at {self.run_dir}")

    def _normalize_tool_name(self, tool_name: str) -> str:
        return tool_name.lower().replace(" ", "_")

    def _create_prompt(self) -> PromptTemplate:
        try:
            template = load_prompt_template("metabolic")
            if template:
                logger.info("Using prompt template from config")
                if "tool_names" not in template.input_variables:
                    template.input_variables.append("tool_names")
                if "tools" not in template.input_variables:
                    template.input_variables.append("tools")
                return template
        except Exception as e:
            logger.warning(
                f"Could not load prompt template from config: {e}. Using default."
            )
        return PromptTemplate(
            template=(
                "You are a metabolic modeling expert. Analyze metabolic models using the available tools.\n"
                "IMPORTANT: Follow these rules exactly:\n"
                "1. Only provide ONE response type at a time - either an Action or a Final Answer, never both\n"
                "2. Use Action when you need to call a tool\n"
                "3. Use Final Answer only when you have all necessary information and are done\n\n"
                "Previous Results:\n"
                "{tool_results}\n\n"
                "Available Tools:\n"
                "- run_metabolic_fba: Run FBA on a model to calculate growth rates and reaction fluxes.\n"
                "- analyze_metabolic_model: Analyze model structure and network properties.\n"
                "- check_missing_media: Check for missing media components that might be preventing growth.\n"
                "- find_minimal_media: Determine the minimal set of media components required for growth.\n"
                "- analyze_reaction_expression: Analyze reaction fluxes under provided media.\n"
                "- identify_auxotrophies: Identify auxotrophies by testing nutrient removal.\n\n"
                "Use this EXACT format - do not deviate from it:\n\n"
                "When using a tool:\n"
                "  Thought: [your reasoning]\n"
                "  Action: tool_name  # Choose one from the list above\n"
                "  Action Input: [input for the tool]\n"
                "  Observation: [result from the tool]\n"
                "... (repeat as needed)\n\n"
                "When providing final answer:\n"
                "  Thought: [summarize findings]\n"
                "  Final Answer: [final summary]\n\n"
                "Question: {input}\n"
                "{agent_scratchpad}"
            ),
            input_variables=[
                "input",
                "agent_scratchpad",
                "tool_results",
                "tools",
                "tool_names",
            ],
        )

    def _summarize_tool_results(self, results: str) -> str:
        summary_prompt = f"Please summarize the following simulation output concisely:\n\n{results}\n\nSummary:"
        summary = self.llm.predict(summary_prompt, max_tokens=150)
        logger.info("Tool results summarization completed")
        return summary.strip()

    def _log_execution(self, step: str, content: Any) -> None:
        try:
            with open(self.exec_log, "r") as f:
                log = json.load(f)
        except Exception:
            log = []
        large_content_threshold = (
            self.config.get("summarization_threshold", 1000)
            if isinstance(self.config, dict)
            else 1000
        )
        log_content = content
        if isinstance(content, str) and len(content) > large_content_threshold:
            file_name = f"large_output_{datetime.now().strftime('%H%M%S_%f')}.txt"
            output_file = self.run_dir / file_name
            try:
                with open(output_file, "w") as f_out:
                    f_out.write(content)
                log_content = f"Large output stored in file: {str(output_file)}"
            except Exception as e:
                logger.error(f"Failed to write large output to file {output_file}: {e}")
                log_content = f"Failed to store large output; length: {len(content)}"
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "iteration": self.current_iteration,
            "token_usage": getattr(self.llm, "_tokens", None),
            "content": log_content,
        }
        log.append(log_entry)
        try:
            with open(self.exec_log, "w") as f:
                json.dump(log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update log file {self.exec_log}: {e}")

    def _format_tool_results(self, results: Dict[str, Any]) -> str:
        output = []
        for tool_name, data in results.items():
            if not isinstance(data, dict):
                try:
                    data = data.dict()
                except Exception:
                    data = {}
            if tool_name == "analyze_metabolic_model":
                output.append(
                    f"Model contains {data.get('num_reactions', 'N/A')} reactions, {data.get('num_metabolites', 'N/A')} metabolites."
                )
            elif tool_name == "run_metabolic_fba":
                growth_rate = data.get("objective_value", "Unknown")
                # If simulation results include a result file, include its name.
                result_file = data.get("result_file", "N/A")
                output.append(
                    f"FBA Analysis: Growth rate is {growth_rate}. (Results file: {result_file})"
                )
        return "\n".join(output) if output else "No tool results available."

    def _get_tools_used(self, steps: List[Dict[str, Any]]) -> Dict[str, int]:
        tool_usage = {}
        for step in steps:
            tool_name = self._normalize_tool_name(step.get("action", "unknown"))
            if tool_name in self._tools_dict:
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        return tool_usage

    def _create_agent(self) -> AgentExecutor:
        prompt = self._create_prompt()
        tools_renderer = lambda tools_list: "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in tools_list]
        )
        prompt = prompt.partial(
            tools=tools_renderer(self.tools),
            tool_names=", ".join([t.name for t in self.tools]),
        )
        from langchain.agents.react.agent import create_react_agent

        runnable = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
            output_parser=CustomReActOutputParser(),
        )

        # Create the standard agent executor (without modifying tools)
        agent_executor = AgentExecutor(
            agent=runnable,
            tools=self.tools,
            verbose=self.config.verbose,
            max_iterations=self.config.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        return agent_executor

    def _format_result(self, result: Dict[str, Any]) -> AgentResult:
        try:
            steps = result.get("intermediate_steps", [])
            output = result.get("output", "")
            formatted_steps = []
            tool_outputs = {}
            all_files = {}

            # Process each step and save files
            for step_idx, step in enumerate(steps):
                if isinstance(step, tuple) and len(step) == 2:
                    action, observation = step
                    tool_name = self._normalize_tool_name(
                        action.tool if hasattr(action, "tool") else str(action)
                    )

                    # Get action input
                    action_input = getattr(action, "tool_input", str(action))

                    # Process valid tool steps
                    if tool_name in [
                        "run_metabolic_fba",
                        "analyze_metabolic_model",
                        "check_missing_media",
                        "find_minimal_media",
                        "analyze_reaction_expression",
                        "identify_auxotrophies",
                    ]:

                        # Save observation to file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        # Save observation
                        observation_file = (
                            self.observations_dir
                            / f"{timestamp}_{tool_name}_observation.json"
                        )
                        observation_data = {
                            "tool": tool_name,
                            "input": str(action_input),
                            "observation": str(observation),
                            "timestamp": timestamp,
                            "step": step_idx,
                        }

                        try:
                            with open(observation_file, "w") as f:
                                json.dump(observation_data, f, indent=2)
                            all_files[f"observation_{tool_name}_{step_idx}"] = str(
                                observation_file
                            )
                        except Exception as e:
                            logger.error(f"Error saving observation: {e}")

                        # Save tool results if available
                        if hasattr(observation, "data") and observation.data:
                            try:
                                # Save results JSON
                                result_file = (
                                    self.results_dir
                                    / f"{timestamp}_{tool_name}_result.json"
                                )
                                with open(result_file, "w") as f:
                                    json.dump(observation.data, f, indent=2)
                                all_files[f"result_{tool_name}_{step_idx}"] = str(
                                    result_file
                                )

                                # For FBA tools with fluxes, save CSV
                                if (
                                    tool_name == "run_metabolic_fba"
                                    and "significant_fluxes" in observation.data
                                ):
                                    import pandas as pd

                                    flux_df = pd.DataFrame.from_dict(
                                        observation.data.get("significant_fluxes", {}),
                                        orient="index",
                                        columns=["flux"],
                                    )
                                    flux_file = (
                                        self.results_dir
                                        / f"{timestamp}_{tool_name}_fluxes.csv"
                                    )
                                    flux_df.to_csv(flux_file)
                                    all_files[f"fluxes_{tool_name}_{step_idx}"] = str(
                                        flux_file
                                    )
                            except Exception as e:
                                logger.error(f"Error saving result files: {e}")

                        # Save step info
                        step_file = (
                            self.steps_dir / f"{timestamp}_{tool_name}_step.json"
                        )
                        step_data = {
                            "step": step_idx,
                            "tool": tool_name,
                            "input": str(action_input),
                            "timestamp": timestamp,
                            "files": all_files,
                        }
                        try:
                            with open(step_file, "w") as f:
                                json.dump(step_data, f, indent=2)
                            all_files[f"step_{tool_name}_{step_idx}"] = str(step_file)
                        except Exception as e:
                            logger.error(f"Error saving step data: {e}")

                        # Add to tool outputs and formatted steps
                        tool_outputs[tool_name] = observation
                        formatted_steps.append(
                            {
                                "action": tool_name,
                                "action_input": action_input,
                                "observation": observation,
                                "files": {
                                    k: v
                                    for k, v in all_files.items()
                                    if f"_{tool_name}_" in k
                                },
                            }
                        )

            # Process tool results for summary
            tool_results_str = self._format_tool_results(tool_outputs)
            summarization_threshold = (
                self.config.get("summarization_threshold", 500)
                if isinstance(self.config, dict)
                else 500
            )

            try:
                # Try to get token count - handle mock objects in tests
                token_count = self.llm.estimate_tokens(tool_results_str)
                if hasattr(token_count, "__gt__"):  # Check if it's a real number
                    should_summarize = token_count > summarization_threshold
                else:
                    # Fallback for mock objects or non-numeric returns
                    should_summarize = (
                        len(tool_results_str) > summarization_threshold * 4
                    )  # Rough estimate: 4 chars per token
            except (TypeError, AttributeError):
                # Fallback for any comparison issues (e.g., with mock objects)
                should_summarize = (
                    len(tool_results_str) > summarization_threshold * 4
                )  # Rough estimate: 4 chars per token

            if should_summarize:
                summarized = self._summarize_tool_results(tool_results_str)
                self._current_tool_results = summarized
            else:
                self._current_tool_results = tool_results_str

            # Add to vector store for context
            document_to_store = str(tool_results_str)
            if self.vector_store:
                self.vector_store.add_document(document_to_store)
                retrieved_context = self.vector_store.query(
                    str(self._last_query), top_n=1
                )
            else:
                retrieved_context = []

            # Clean output
            if isinstance(output, str):
                output = output.split("For troubleshooting")[0].strip()
                output = re.sub(
                    r"Thought:|Action:|Action Input:|Observation:", "", output
                ).strip()

            # Get latest log entry
            try:
                with open(self.exec_log, "r") as f:
                    log_entries = json.load(f)
                last_log = log_entries[-1] if log_entries else {}
            except Exception:
                last_log = {}

            # Save final report
            if output:
                final_report_file = (
                    self.run_dir
                    / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                )
                try:
                    with open(final_report_file, "w") as f:
                        f.write(output)
                    all_files["final_report"] = str(final_report_file)
                except Exception as e:
                    logger.error(f"Failed to save final report: {e}")

            return AgentResult(
                success=True,
                message="Analysis completed successfully",
                data={
                    "final_answer": output or "Analysis completed",
                    "tool_results": formatted_steps,
                    "summary": self._current_tool_results,
                    "retrieved_context": retrieved_context,
                    "log_summary": f"Enhanced logging enabled. Run directory: {str(self.run_dir)}. Last log entry: {last_log}",
                    "files": all_files,
                },
                intermediate_steps=formatted_steps,
                metadata={
                    "iterations": len(formatted_steps),
                    "tools_used": self._get_tools_used(formatted_steps),
                    "last_tool": (
                        formatted_steps[-1]["action"] if formatted_steps else None
                    ),
                    "run_dir": str(self.run_dir),
                },
            )
        except Exception as e:
            logger.exception("Error formatting agent result")
            return AgentResult(
                success=False, message="Error formatting agent result", error=str(e)
            )

    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            query = str(input_data.get("input", ""))
            self._last_query = query
            input_data.update(
                {"input": query, "tool_results": self._current_tool_results}
            )
            self._log_execution("start", {"query": query})
            result = self._agent_executor.invoke(input_data)
            formatted_result = self._format_result(result)
            self._log_execution("complete", formatted_result.model_dump())
            return formatted_result
        except Exception as e:
            self._log_execution("error", {"error": str(e)})
            logger.error(f"Agent execution failed: {e}")
            return AgentResult(success=False, message="Execution failed", error=str(e))

    def analyze_model(
        self, query: str, analysis_type: Optional[str] = None
    ) -> AgentResult:
        """Analyze a metabolic model with the given query"""
        # Incorporate analysis_type into the query if provided
        if analysis_type:
            query = f"Perform {analysis_type} analysis: {query}"

        self._last_query = query
        input_data = {"input": query}
        return self.run(input_data)

    def suggest_improvements(self, model_path: str) -> AgentResult:
        """Suggest improvements for a metabolic model"""
        query = f"Analyze the metabolic model at {model_path} and suggest specific improvements to enhance its performance, accuracy, or biological relevance. Consider aspects like missing reactions, media optimization, and growth predictions."
        self._last_query = query
        input_data = {"input": query}
        return self.run(input_data)

    def compare_models(self, model_paths: List[str]) -> AgentResult:
        """Compare multiple metabolic models"""
        models_str = ", ".join(model_paths)
        query = f"Compare the metabolic models at the following paths: {models_str}. Analyze their differences in terms of reactions, metabolites, growth capabilities, and biological coverage. Provide a comprehensive comparison report."
        self._last_query = query
        input_data = {"input": query}
        return self.run(input_data)

    async def arun(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            if "tool_results" not in input_data:
                input_data.update(
                    {
                        "tool_results": self._current_tool_results,
                        "tools": "\n".join(
                            f"- {t.name}: {t.description}" for t in self.tools
                        ),
                    }
                )
            logger.info(f"Starting async analysis with input: {input_data}")
            result = await self._agent_executor.ainvoke(input_data)
            formatted_result = self._format_result(result)
            self._log_execution("complete", formatted_result.model_dump())
            return formatted_result
        except Exception as e:
            error_msg = f"Async agent execution failed: {str(e)}"
            self._log_execution("error", {"error": error_msg})
            logger.error(error_msg)
            return AgentResult(success=False, message=error_msg, error=str(e))
