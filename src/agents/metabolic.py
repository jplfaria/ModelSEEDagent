from typing import Dict, Any, List, Optional, Union
import json
import logging
import re
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

logger = logging.getLogger(__name__)

class CustomReActOutputParser(ReActSingleInputOutputParser):
    """Custom parser that handles combined Action/Final Answer responses"""
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # Clean up troubleshooting messages
        if "For troubleshooting" in text:
            text = text.split("For troubleshooting")[0].strip()
        
        # Get the last complete sequence
        sequences = re.split(r'\nThought:', text)
        if not sequences:
            return AgentFinish({"output": "No valid response generated"}, "")
            
        last_sequence = sequences[-1].strip()
        
        # Check for final answer
        if "Final Answer:" in last_sequence:
            final_answer = last_sequence.split("Final Answer:")[-1].strip()
            return AgentFinish({"output": final_answer}, text)
            
        # Look for action
        if "Action:" in last_sequence and "Action Input:" in last_sequence:
            action_match = re.search(r'Action: (.*?)(?:\n|$)', last_sequence)
            input_match = re.search(r'Action Input: (.*?)(?:\n|$)', last_sequence)
            
            if action_match and input_match:
                action = action_match.group(1).strip()
                action_input = input_match.group(1).strip()
                return AgentAction(action, action_input, text)
        
        # If no clear pattern, return as final answer
        return AgentFinish({"output": last_sequence}, text)

class MetabolicAgent(BaseAgent):
    """Agent for metabolic model analysis with improved error handling"""
    
    def __init__(
        self,
        llm: BaseLLM,
        tools: List[BaseTool],
        config: Dict[str, Any] | AgentConfig
    ):
        # Store initial tool results before base init
        self._current_tool_results = "No previous results"
        self._tools_dict = {t.tool_name: t for t in tools}
        
        # Initialize base agent
        super().__init__(llm, tools, config)
        
        # Setup logging directory structure
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_base = Path(__file__).parent.parent.parent / "logs"
        self.run_dir = self.log_base / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tool-specific log directories
        self.tool_dirs = {}
        for tool in tools:
            tool_dir = self.run_dir / tool.tool_name
            tool_dir.mkdir(parents=True, exist_ok=True)
            self.tool_dirs[tool.tool_name] = tool_dir
        
        # Main execution log
        self.exec_log = self.run_dir / "execution.json"
        with open(self.exec_log, 'w') as f:
            json.dump([], f)
            
        logger.info(f"Created run directory: {self.run_dir}")
    
    def _create_prompt(self) -> PromptTemplate:
        """Create prompt template with strict formatting requirements"""
        try:
            # Try to load from config first
            template = load_prompt_template("metabolic")
            if template:
                logger.info("Using prompt template from config")
                # Add tool_names to template variables
                template.input_variables = list(set(template.input_variables + ["tool_names"]))
                return template
        except Exception as e:
            logger.warning(f"Could not load prompt template from config: {e}. Using default.")
        
        # Fall back to default template
        return PromptTemplate(
            template=(
                MetabolicAgentConfig.PREFIX +
                MetabolicAgentConfig.FORMAT_INSTRUCTIONS +
                MetabolicAgentConfig.SUFFIX
            ),
            input_variables=["input", "agent_scratchpad", "tool_results", "tools", "tool_names"]
        )

    def _create_agent(self) -> AgentExecutor:
        """Create agent with custom output parser"""
        prompt = self._create_prompt()
        
        # Create the agent with custom parser
        runnable = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
            output_parser=CustomReActOutputParser()
        )
        
        return AgentExecutor(
            agent=runnable,
            tools=self.tools,
            verbose=self.config.verbose,
            max_iterations=self.config.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

    def _log_tool_output(self, tool_name: str, content: Dict[str, Any]) -> None:
        """Log tool output to tool-specific directory"""
        try:
            tool_dir = self.tool_dirs[tool_name]
            timestamp = datetime.now().strftime("%H%M%S")
            log_file = tool_dir / f"output_{timestamp}.json"
            
            with open(log_file, 'w') as f:
                json.dump(content, f, indent=2)
            logger.debug(f"Logged {tool_name} output to {log_file}")
        except Exception as e:
            logger.error(f"Failed to log {tool_name} output: {e}")

    def _log_execution(self, step: str, content: Any) -> None:
        """Log execution step"""
        try:
            with open(self.exec_log, 'r') as f:
                log = json.load(f)
            
            log.append({
                "timestamp": datetime.now().isoformat(),
                "step": step,
                "content": content
            })
            
            with open(self.exec_log, 'w') as f:
                json.dump(log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log execution step: {e}")

    def _clean_reactions_data(self, reactions_data: Dict[str, float]) -> List[Dict[str, Any]]:
            """Clean and sort reaction data by absolute flux value"""
            reactions = []
            for rxn, flux in reactions_data.items():
                if isinstance(flux, (int, float)) and abs(flux) > 1e-6:
                    reactions.append({
                        "id": rxn,
                        "flux": flux,
                        "abs_flux": abs(flux)
                    })
            return sorted(reactions, key=lambda x: x["abs_flux"], reverse=True)
    
    def _format_tool_results(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Format results for LLM consumption with improved pathway context"""
        output = []
        
        for tool_name, data in results.items():
            if tool_name == "analyze_metabolic_model":
                try:
                    # Parse observation if it's a string
                    if isinstance(data, str):
                        lines = [line.strip() for line in data.split('\n') if line.strip()]
                        parsed_data = {}
                        for line in lines:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                parsed_data[key.strip('- ')] = value.strip()
                        data = parsed_data

                    model_info = (
                        f"Model Structure Analysis:\n"
                        f"- Network Size: {data.get('Number of reactions', 'N/A')} reactions, "
                        f"{data.get('Number of metabolites', 'N/A')} metabolites, "
                        f"{data.get('Number of genes', 'N/A')} genes\n"
                        f"- Compartments: {data.get('Number of compartments', 'N/A')} "
                        f"({data.get('Compartments', 'not specified')})\n"
                        f"- Model Objective: {data.get('Model objective') or data.get('Objective function', 'N/A')}\n"
                        f"- Model Type: {data.get('Model type', 'metabolic')}"
                    )
                    output.append(model_info)

                except Exception as e:
                    logger.warning(f"Error formatting model analysis: {e}")
                    output.append(str(data))

            elif tool_name == "run_metabolic_fba":
                try:
                    if isinstance(data, str):
                        lines = [line.strip() for line in data.split('\n') if line.strip()]
                        parsed_data = {}
                        for line in lines:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                key = key.strip('- ')
                                value = value.strip()
                                try:
                                    if value.replace('.', '').replace('-', '').isdigit():
                                        value = float(value)
                                except:
                                    pass
                                parsed_data[key] = value
                        data = parsed_data

                    # Find growth rate
                    growth_rate = None
                    for key, value in data.items():
                        if 'Biomass' in key:
                            growth_rate = value
                            break

                    # Process reaction fluxes
                    reactions_data = {}
                    exchange_fluxes = {}
                    for key, value in data.items():
                        if isinstance(value, (int, float)):
                            if key.startswith('EX_') or key.endswith('_ex'):
                                exchange_fluxes[key] = value
                            else:
                                reactions_data[key] = value

                    # Format central metabolism reactions
                    central_reactions = self._clean_reactions_data(reactions_data)[:15]  # Top 15 reactions

                    fba_results = [
                        f"FBA Results:",
                        f"- Growth Rate: {growth_rate} h⁻¹",
                        "\nKey Central Metabolism Fluxes:"
                    ]
                    
                    for rxn in central_reactions:
                        fba_results.append(
                            f"- {rxn['id']}: {rxn['flux']:0.3f}"
                        )

                    if exchange_fluxes:
                        fba_results.extend([
                            "\nKey Exchange Fluxes:",
                            *[f"- {rxn}: {flux:0.3f}" 
                              for rxn, flux in sorted(exchange_fluxes.items(), 
                                                    key=lambda x: abs(x[1]), 
                                                    reverse=True)[:5]
                            ]
                        ])

                    output.append("\n".join(fba_results))

                except Exception as e:
                    logger.warning(f"Error formatting FBA results: {e}")
                    output.append(str(data))

        formatted_output = "\n\n".join(output) if output else "No previous results available"
        return formatted_output
    
    def _get_tools_used(self, steps: List[Dict[str, Any]]) -> Dict[str, int]:
        """Track which tools were used and how many times, with improved accuracy"""
        tool_usage = {}
        for step in steps:
            tool_name = self._normalize_tool_name(step.get("action", "unknown"))
            if tool_name in ["run_metabolic_fba", "analyze_metabolic_model"]:
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        return tool_usage

    def _format_result(self, result: Dict[str, Any]) -> AgentResult:
            """Format the raw agent result with improved error handling"""
            try:
                # Extract steps and final answer
                steps = result.get("intermediate_steps", [])
                output = result.get("output", "")
                formatted_steps = []
                tool_outputs = {}
    
                # Process intermediate steps
                for step in steps:
                    if isinstance(step, tuple) and len(step) == 2:
                        action, observation = step
                        tool_name = self._normalize_tool_name(
                            action.tool if hasattr(action, "tool") else str(action)
                        )
                        
                        if tool_name in ["run_metabolic_fba", "analyze_metabolic_model"]:
                            # Parse observation to dictionary if it's a string
                            if isinstance(observation, str):
                                try:
                                    # Convert observation to structured data
                                    obs_lines = observation.split('\n')
                                    obs_dict = {}
                                    current_section = "main"
                                    for line in obs_lines:
                                        line = line.strip()
                                        if line.startswith('- '):
                                            key, value = line[2:].split(':', 1)
                                            obs_dict[key.strip()] = value.strip()
                                    observation = obs_dict
                                except:
                                    # Keep original if parsing fails
                                    pass
                            
                            tool_outputs[tool_name] = observation
                            formatted_steps.append({
                                "action": tool_name,
                                "action_input": getattr(action, "tool_input", str(action)),
                                "observation": observation
                            })
    
                # Update current tool results for next iteration
                if tool_outputs:
                    self._current_tool_results = self._format_tool_results(tool_outputs)
    
                # Clean any troubleshooting messages from output
                if isinstance(output, str):
                    output = output.split("For troubleshooting")[0].strip()
                    # Clean up any additional format markers
                    output = re.sub(r'Thought:|Action:|Action Input:|Observation:', '', output)
                    output = output.strip()
    
                return AgentResult(
                    success=True,
                    message="Analysis completed successfully",
                    data={
                        "final_answer": output or "Analysis completed",
                        "tool_results": formatted_steps,
                        "summary": self._current_tool_results  # Add formatted summary
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
                return AgentResult(
                    success=False,
                    message="Error formatting agent result",
                    error=str(e)
                )

    def _get_tools_used(self, steps: List[Dict[str, Any]]) -> Dict[str, int]:
        """Track which tools were used and how many times"""
        tool_usage = {}
        for step in steps:
            tool_name = self._normalize_tool_name(step.get("action", "unknown"))
            if tool_name in self._tools_dict:
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        return tool_usage

    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """Run agent with improved error handling"""
        try:
            # Add formatting reminder to input
            query = input_data.get("input", "")
            if not query.endswith("Remember to provide only one response type at a time."):
                query += " Remember to provide only one response type at a time."
            
            # Ensure required variables are present
            input_data.update({
                "input": query,
                "tool_results": self._current_tool_results,
                "tools": "\n".join(
                    f"- {t.name}: {t.description}" 
                    for t in self.tools
                )
            })
            
            # Log start
            if "Model path:" in query:
                model_path = query.split("Model path:")[-1].strip()
                self._log_execution("start", {
                    "query": query,
                    "model_path": model_path
                })
            
            # Run agent
            result = self._agent_executor.invoke(input_data)
            
            # Format result
            formatted_result = self._format_result(result)
            self._log_execution("complete", formatted_result.dict())
            
            # Log locations
            logger.info(f"\nRun logs available at: {self.run_dir}")
            for tool_name, tool_dir in self.tool_dirs.items():
                logger.info(f"{tool_name} logs: {tool_dir}")
            
            return formatted_result
            
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            self._log_execution("error", {"error": error_msg})
            logger.error(error_msg)
            return AgentResult(
                success=False,
                message=error_msg,
                error=str(e)
            )

    def analyze_model(self, query: str) -> AgentResult:
        """Analyze model with enhanced error handling"""
        try:
            logger.info(f"Starting model analysis with query: {query}")
            return self.run({
                "input": query,
                "tool_results": self._current_tool_results,
                "tools": "\n".join(
                    f"- {t.name}: {t.description}" 
                    for t in self.tools
                )
            })
        except Exception as e:
            error_msg = f"Error in analyze_model: {str(e)}"
            logger.exception(error_msg)
            return AgentResult(
                success=False,
                message=error_msg,
                error=str(e)
            )

    async def arun(self, input_data: Dict[str, Any]) -> AgentResult:
        """Run agent asynchronously with logging"""
        try:
            if "tool_results" not in input_data:
                input_data.update({
                    "tool_results": self._current_tool_results,
                    "tools": "\n".join(
                        f"- {t.name}: {t.description}" 
                        for t in self.tools
                    )
                })
                
            logger.info(f"Starting async analysis with input: {input_data}")
            result = await self._agent_executor.ainvoke(input_data)
            formatted_result = self._format_result(result)
            self._log_execution("complete", formatted_result.dict())
            return formatted_result
        except Exception as e:
            error_msg = f"Async agent execution failed: {str(e)}"
            self._log_execution("error", {"error": error_msg})
            logger.error(error_msg)
            return AgentResult(
                success=False,
                message=error_msg,
                error=str(e)
            )