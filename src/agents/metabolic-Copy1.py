from typing import Dict, Any, List, Optional, Union
import json
import logging
import re
import os
from pathlib import Path
from datetime import datetime

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from .base import BaseAgent, AgentResult, AgentConfig
from ..tools.base import BaseTool, ToolResult
from ..llm.base import BaseLLM
from ..config.prompts import load_prompts, load_prompt_template

# Additional imports for vector store functionality
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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

    def save_results(self, tool_name: str, model_id: str, solution, additional_metadata: Dict[str, Any] = None) -> str:
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
            "metadata": additional_metadata or {}
        }
        self.results[result_id] = result_data
        logger.info(f"SimulationResultsStore: Saved results for {tool_name} with ID {result_id}.")
        return result_id

    def export_results(self, result_id: str, output_dir: str) -> Dict[str, str]:
        """Exports the simulation result as a JSON file and fluxes as a CSV file."""
        if result_id not in self.results:
            logger.error(f"SimulationResultsStore: No results found for ID {result_id}.")
            return {}
        os.makedirs(output_dir, exist_ok=True)
        result = self.results[result_id]
        json_filename = f"{result_id}.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        # Export fluxes as CSV
        try:
            import pandas as pd
            flux_df = pd.DataFrame.from_dict(result["fluxes"], orient='index', columns=['flux'])
            csv_filename = f"{result_id}_fluxes.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            flux_df.to_csv(csv_path)
        except Exception as e:
            logger.error(f"SimulationResultsStore: Error exporting fluxes CSV: {e}")
            csv_path = ""
        logger.info(f"SimulationResultsStore: Exported results to {json_path} and {csv_path}.")
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
        logger.info(f"VectorStore: Added document (len={len(text)}). Total documents: {len(self.documents)}.")
        print(f"DEBUG: VectorStore added document; count = {len(self.documents)}.")

    def query(self, query_text: str, top_n: int = 1) -> List[str]:
        if not self.documents or self.embeddings is None:
            logger.info("VectorStore: No documents available for query.")
            print("DEBUG: VectorStore empty on query.")
            return []
        query_vec = self.vectorizer.transform([query_text])
        sim = cosine_similarity(query_vec, self.embeddings)
        top_indices = np.argsort(sim[0])[-top_n:][::-1]
        logger.info(f"VectorStore: Query '{query_text}' returned indices: {top_indices}.")
        print(f"DEBUG: VectorStore query indices: {top_indices}.")
        return [self.documents[i] for i in top_indices]
# -------------------------------
# End SimpleVectorStore
# -------------------------------

class CustomReActOutputParser(ReActSingleInputOutputParser):
    """Custom parser for combined Action/Final Answer responses."""
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "For troubleshooting" in text:
            text = text.split("For troubleshooting")[0].strip()
        sequences = re.split(r'\nThought:', text)
        if not sequences:
            return AgentFinish({"output": "No valid response generated"}, "")
        last_sequence = sequences[-1].strip()
        if "Final Answer:" in last_sequence:
            final_answer = last_sequence.split("Final Answer:")[-1].strip()
            return AgentFinish({"output": final_answer}, text)
        if "Action:" in last_sequence and "Action Input:" in last_sequence:
            action_match = re.search(r'Action: (.*?)(?:\n|$)', last_sequence)
            input_match = re.search(r'Action Input: (.*?)(?:\n|$)', last_sequence)
            if action_match and input_match:
                action = action_match.group(1).strip()
                action_input = input_match.group(1).strip()
                return AgentAction(action, action_input, text)
        return AgentFinish({"output": last_sequence}, text)

class MetabolicAgent(BaseAgent):
    """Agent for metabolic model analysis with enhanced memory, token management, simulation result export, and logging."""
    
    def __init__(self, llm: BaseLLM, tools: List[BaseTool], config: Dict[str, Any] | AgentConfig):
        self._current_tool_results = "No previous results"
        self._tools_dict = {t.tool_name: t for t in tools}
        self._tool_output_counters = {}
        super().__init__(llm, tools, config)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_base = Path(__file__).parent.parent.parent / "logs"
        self.run_dir = self.log_base / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tool_dirs = {}
        for tool in tools:
            tool_dir = self.run_dir / tool.tool_name
            tool_dir.mkdir(parents=True, exist_ok=True)
            self.tool_dirs[tool.tool_name] = tool_dir
        self.exec_log = self.run_dir / "execution.json"
        with open(self.exec_log, 'w') as f:
            json.dump([], f)
        self.current_iteration = 0
        self._last_query = ""
        self.vector_store = SimpleVectorStore()
        self.simulation_store = SimulationResultsStore()  # New simulation results store
        logger.info(f"Created run directory: {self.run_dir}")
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
            logger.warning(f"Could not load prompt template from config: {e}. Using default.")
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
            input_variables=["input", "agent_scratchpad", "tool_results", "tools", "tool_names"]
        )
    
    def _summarize_tool_results(self, results: str) -> str:
        summary_prompt = f"Please summarize the following simulation output concisely:\n\n{results}\n\nSummary:"
        summary = self.llm.predict(summary_prompt, max_tokens=150)
        print("DEBUG: Summarization prompt sent, summary obtained.")
        return summary.strip()
    
    def _log_tool_output(self, tool_name: str, content: Dict[str, Any]) -> None:
        print(f"DEBUG: Logging output for {tool_name}: {content}")
        try:
            tool_dir = self.tool_dirs.get(tool_name)
            if not tool_dir:
                logger.error(f"Log directory for {tool_name} not found.")
                print(f"DEBUG: Log directory for {tool_name} not found.")
                return
            counter = self._tool_output_counters.get(tool_name, 0) + 1
            self._tool_output_counters[tool_name] = counter
            timestamp = datetime.now().strftime("%H%M%S_%f")
            filename = f"{tool_name}_output_{counter}_{timestamp}.json"
            log_file = tool_dir / filename
            with open(log_file, 'w') as f:
                json.dump(content, f, indent=2)
            logger.debug(f"Logged {tool_name} output to {log_file}")
            print(f"DEBUG: Logged output for {tool_name} to {log_file}")
        except Exception as e:
            logger.error(f"Failed to log {tool_name} output: {e}")
            print(f"DEBUG: Failed to log output for {tool_name}: {e}")
    
    def _log_execution(self, step: str, content: Any) -> None:
        try:
            with open(self.exec_log, 'r') as f:
                log = json.load(f)
        except Exception:
            log = []
        large_content_threshold = self.config.get("summarization_threshold", 1000) if isinstance(self.config, dict) else 1000
        log_content = content
        if isinstance(content, str) and len(content) > large_content_threshold:
            file_name = f"large_output_{datetime.now().strftime('%H%M%S_%f')}.txt"
            output_file = self.run_dir / file_name
            try:
                with open(output_file, 'w') as f_out:
                    f_out.write(content)
                log_content = f"Large output stored in file: {str(output_file)}"
            except Exception as e:
                logger.error(f"Failed to write large output to file {output_file}: {e}")
                log_content = f"Failed to store large output; length: {len(content)}"
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "iteration": self.current_iteration,
            "token_usage": getattr(self.llm, '_tokens', None),
            "content": log_content
        }
        log.append(log_entry)
        try:
            with open(self.exec_log, 'w') as f:
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
                output.append(f"Model contains {data.get('num_reactions', 'N/A')} reactions, {data.get('num_metabolites', 'N/A')} metabolites.")
            elif tool_name == "run_metabolic_fba":
                growth_rate = data.get("objective_value", "Unknown")
                # If simulation results include a result file, include its name.
                result_file = data.get("result_file", "N/A")
                output.append(f"FBA Analysis: Growth rate is {growth_rate}. (Results file: {result_file})")
        return "\n".join(output) if output else "No tool results available."
    
    def _get_tools_used(self, steps: List[Dict[str, Any]]) -> Dict[str, int]:
        tool_usage = {}
        for step in steps:
            tool_name = self._normalize_tool_name(step.get("action", "unknown"))
            if tool_name in self._tools_dict:
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        return tool_usage
    
    def _format_result(self, result: Dict[str, Any]) -> AgentResult:
        try:
            steps = result.get("intermediate_steps", [])
            output = result.get("output", "")
            formatted_steps = []
            tool_outputs = {}
            for step in steps:
                if isinstance(step, tuple) and len(step) == 2:
                    action, observation = step
                    tool_name = self._normalize_tool_name(
                        action.tool if hasattr(action, "tool") else str(action)
                    )
                    if tool_name in ["run_metabolic_fba", "analyze_metabolic_model", "check_missing_media", "find_minimal_media", "analyze_reaction_expression", "identify_auxotrophies"]:
                        tool_outputs[tool_name] = observation
                        step_data = {
                            "action": tool_name,
                            "action_input": getattr(action, "tool_input", str(action)),
                            "observation": observation
                        }
                        formatted_steps.append(step_data)
                        self._log_tool_output(tool_name, step_data)
            tool_results_str = self._format_tool_results(tool_outputs)
            summarization_threshold = self.config.get("summarization_threshold", 500) if isinstance(self.config, dict) else 500
            if self.llm.estimate_tokens(tool_results_str) > summarization_threshold:
                summarized = self._summarize_tool_results(tool_results_str)
                self._current_tool_results = summarized
            else:
                self._current_tool_results = tool_results_str
            # Add tool results to vector store for memory (store as string)
            document_to_store = str(tool_results_str)
            if self.vector_store:
                self.vector_store.add_document(document_to_store)
                retrieved_context = self.vector_store.query(str(self._last_query), top_n=1)
            else:
                retrieved_context = []
            if isinstance(output, str):
                output = output.split("For troubleshooting")[0].strip()
                output = re.sub(r'Thought:|Action:|Action Input:|Observation:', '', output).strip()
            try:
                with open(self.exec_log, 'r') as f:
                    log_entries = json.load(f)
                last_log = log_entries[-1] if log_entries else {}
            except Exception:
                last_log = {}
            return AgentResult(
                success=True,
                message="Analysis completed successfully",
                data={
                    "final_answer": output or "Analysis completed",
                    "tool_results": formatted_steps,
                    "summary": self._current_tool_results,
                    "retrieved_context": retrieved_context,
                    "log_summary": f"Enhanced logging enabled. Run directory: {str(self.run_dir)}. Last log entry: {last_log}"
                },
                intermediate_steps=formatted_steps,
                metadata={
                    "iterations": len(formatted_steps),
                    "tools_used": self._get_tools_used(formatted_steps),
                    "last_tool": formatted_steps[-1]["action"] if formatted_steps else None
                }
            )
        except Exception as e:
            logger.exception("Error formatting agent result")
            return AgentResult(success=False, message="Error formatting agent result", error=str(e))
    
    def _create_agent(self) -> AgentExecutor:
        prompt = self._create_prompt()
        tools_renderer = lambda tools_list: "\n".join([f"- {tool.name}: {tool.description}" for tool in tools_list])
        prompt = prompt.partial(tools=tools_renderer(self.tools), tool_names=", ".join([t.name for t in self.tools]))
        from langchain.agents.react.agent import create_react_agent
        runnable = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt, output_parser=CustomReActOutputParser())
        
        # Create a dictionary that maps tool names to their instances
        tool_map = {tool.name: tool for tool in self.tools}
        
        # Create wrapped tools that inject output directory
        wrapped_tools = []
        for tool in self.tools:
            # Create a wrapped version of the tool's run method
            original_run = tool.run
            
            def wrapped_run(input_data, _tool=tool, _run_dir=self.run_dir):
                # If input is a string, convert to dict with model_path and output_dir
                if isinstance(input_data, str):
                    model_path = input_data
                    tool_dir = _run_dir / _tool.name
                    input_data = {
                        "model_path": model_path,
                        "output_dir": str(tool_dir)
                    }
                # If input is a dict but missing output_dir, add it
                elif isinstance(input_data, dict) and "model_path" in input_data and "output_dir" not in input_data:
                    tool_dir = _run_dir / _tool.name
                    input_data["output_dir"] = str(tool_dir)
                
                # Call the original run method with the enhanced input
                return original_run(input_data)
            
            # Replace the tool's run method with our wrapped version
            tool.run = wrapped_run.__get__(tool, type(tool))
            wrapped_tools.append(tool)
        
        # Create and return the agent executor with wrapped tools
        return AgentExecutor(
            agent=runnable, 
            tools=wrapped_tools, 
            verbose=self.config.verbose, 
            max_iterations=self.config.max_iterations, 
            handle_parsing_errors=True, 
            return_intermediate_steps=True
        )
    
    def analyze_model(self, query: str) -> AgentResult:
        try:
            logger.info(f"Starting model analysis with query: {query}")
            return self.run({"input": query, "tool_results": self._current_tool_results, "tools": "\n".join(f"- {t.name}: {t.description}" for t in self.tools)})
        except Exception as e:
            error_msg = f"Error in analyze_model: {str(e)}"
            logger.exception(error_msg)
            return AgentResult(success=False, message=error_msg, error=str(e))
    
    async def arun(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            if "tool_results" not in input_data:
                input_data.update({"tool_results": self._current_tool_results, "tools": "\n".join(f"- {t.name}: {t.description}" for t in self.tools)})
            logger.info(f"Starting async analysis with input: {input_data}")
            result = await self._agent_executor.ainvoke(input_data)
            formatted_result = self._format_result(result)
            self._log_execution("complete", formatted_result.dict())
            return formatted_result
        except Exception as e:
            error_msg = f"Async agent execution failed: {str(e)}"
            self._log_execution("error", {"error": error_msg})
            logger.error(error_msg)
            return AgentResult(success=False, message=error_msg, error=str(e))