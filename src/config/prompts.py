# src/config/prompts.py
from typing import Dict, Any, Optional, Union
import yaml
from pathlib import Path
import logging
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate as LangChainPromptTemplate

logger = logging.getLogger(__name__)

class PromptTemplate(BaseModel):
    """Model for prompt templates"""
    name: str
    description: str
    template: str
    input_variables: list[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentPrompts(BaseModel):
    """Collection of prompts for an agent type"""
    prefix: str
    suffix: str
    format_instructions: str
    examples: Optional[list[Dict[str, str]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ToolPrompts(BaseModel):
    """Collection of prompts for tools"""
    description: str
    usage: str
    examples: Optional[list[Dict[str, str]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Prompts(BaseModel):
    """Main prompts configuration"""
    agents: Dict[str, AgentPrompts]
    tools: Dict[str, ToolPrompts]
    templates: Dict[str, PromptTemplate]

class PromptManager:
    """Manager for handling prompt templates"""
    
    _instance = None
    _prompts: Optional[Prompts] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def load_prompts(cls, prompts_dir: Optional[Union[str, Path]] = None) -> Prompts:
        """
        Load prompts from YAML files in the prompts directory.
        
        Args:
            prompts_dir: Directory containing prompt YAML files
            
        Returns:
            Loaded prompts configuration
            
        Raises:
            FileNotFoundError: If prompts directory is not found
            ValueError: If prompts are invalid
        """
        if prompts_dir is None:
            prompts_dir = cls._find_prompts_dir()
        else:
            prompts_dir = Path(prompts_dir)
        
        try:
            # Load agent prompts
            agents_file = prompts_dir / "metabolic.yaml"  # Changed from agents.yaml
            with open(agents_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract sections
            agent_config = config.get('agent', {})
            tools_config = config.get('tools', {})
            templates_config = {}  # Will populate from agent config
            
            # Create template from agent prompts
            if 'prefix' in agent_config and 'format_instructions' in agent_config and 'suffix' in agent_config:
                templates_config['metabolic_agent'] = {
                    'name': 'metabolic_agent',
                    'description': config.get('description', ''),
                    'template': agent_config['prefix'] + '\n\n' + agent_config['format_instructions'] + '\n\n' + agent_config['suffix'],
                    'input_variables': ['input', 'agent_scratchpad', 'tool_results', 'tools']
                }
            
            prompts = Prompts(
                agents={
                    'metabolic': AgentPrompts(
                        prefix=agent_config.get('prefix', ''),
                        suffix=agent_config.get('suffix', ''),
                        format_instructions=agent_config.get('format_instructions', ''),
                        metadata=agent_config.get('metadata', {})
                    )
                },
                tools={k: ToolPrompts(**v) for k, v in tools_config.items()},
                templates={k: PromptTemplate(**v) for k, v in templates_config.items()}
            )
            
            cls._prompts = prompts
            logger.info(f"Successfully loaded prompts from {prompts_dir}")
            return prompts
            
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise
    
    @classmethod
    def _find_prompts_dir(cls) -> Path:
        """Find prompts directory in default locations"""
        search_paths = [
            Path.cwd() / 'config' / 'prompts',
            Path.home() / '.config' / 'metabolic_agent' / 'prompts',
            Path(__file__).parent.parent.parent / 'config' / 'prompts'
        ]
        
        for path in search_paths:
            if path.exists() and path.is_dir():
                return path
        
        raise FileNotFoundError(
            "Prompts directory not found in default locations: " +
            ", ".join(str(p) for p in search_paths)
        )
    
    @classmethod
    def get_prompts(cls) -> Prompts:
        """Get the current prompts configuration"""
        if cls._prompts is None:
            cls.load_prompts()
        return cls._prompts
    
    @classmethod
    def get_agent_prompt(cls, agent_type: str) -> AgentPrompts:
        """Get prompts for a specific agent type"""
        prompts = cls.get_prompts()
        if agent_type not in prompts.agents:
            raise ValueError(f"No prompts found for agent type: {agent_type}")
        return prompts.agents[agent_type]

def load_prompt_template(template_name: str) -> Optional[LangChainPromptTemplate]:
    """Load a prompt template and convert to LangChain format"""
    try:
        prompts = PromptManager.get_prompts()
        if template_name in prompts.templates:
            template = prompts.templates[template_name]
            return LangChainPromptTemplate(
                template=template.template,
                input_variables=template.input_variables
            )
        elif template_name == "metabolic":
            # Convert agent prompts to template
            agent_prompts = prompts.agents.get("metabolic")
            if agent_prompts:
                template = (
                    agent_prompts.prefix +
                    "\n\n" +
                    agent_prompts.format_instructions +
                    "\n\n" +
                    agent_prompts.suffix
                )
                return LangChainPromptTemplate(
                    template=template,
                    input_variables=["input", "agent_scratchpad", "tool_results", "tools"]
                )
        return None
    except Exception as e:
        logger.error(f"Error loading prompt template {template_name}: {e}")
        return None

# Convenience functions
def load_prompts(prompts_dir: Optional[Union[str, Path]] = None) -> Prompts:
    return PromptManager.load_prompts(prompts_dir)

def get_prompt(agent_type: str = None, tool_name: str = None, template_name: str = None) -> Union[AgentPrompts, ToolPrompts, PromptTemplate]:
    if agent_type:
        return PromptManager.get_agent_prompt(agent_type)
    elif template_name:
        prompts = PromptManager.get_prompts()
        return prompts.templates[template_name]
    else:
        raise ValueError("Must specify either agent_type or template_name")