# src/config/settings.py
import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LLMConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    llm_backend: str = "argo"
    safety_settings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "max_api_calls": 100,
            "max_tokens": 50000
        }
    )

class ArgoConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    user: str
    system_content: str
    models: Dict[str, Dict[str, str]]
    default_model: str = "gpt4"

class OpenAIConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    api_key: str
    api_name: str = "gpt-3.5-turbo"  # Changed from model_name
    system_content: str

class LocalModelConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_path: str
    device: str = "mps"
    max_tokens: int = 500
    temperature: float = 0.7

class LocalLLMConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_path: str
    system_content: str
    device: str = "mps"
    max_tokens: int = 500
    temperature: float = 0.7
    models: Dict[str, LocalModelConfig] = Field(default_factory=dict)

class ToolConfigs(BaseModel):
    model_config = {"protected_namespaces": ()}
    fba_config: Dict[str, Any] = Field(default_factory=dict)
    analysis_config: Dict[str, Any] = Field(default_factory=dict)

class ToolConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    configs: ToolConfigs

class AgentConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    max_iterations: int = 5
    verbose: bool = False
    handle_parsing_errors: bool = True
    additional_config: Dict[str, Any] = Field(default_factory=dict)

class Config(BaseModel):
    model_config = {"protected_namespaces": ()}
    llm: LLMConfig
    argo: Optional[ArgoConfig] = None
    openai: Optional[OpenAIConfig] = None
    local: Optional[LocalLLMConfig] = None
    tools: ToolConfig
    agent: AgentConfig

class ConfigManager:
    """Manager for handling configuration loading and access"""
    
    _instance = None
    _config: Optional[Config] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def load_config(cls, config_path: Optional[Union[str, Path]] = None) -> Config:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file. If None, looks in default locations.
            
        Returns:
            Loaded configuration
            
        Raises:
            FileNotFoundError: If configuration file is not found
            ValueError: If configuration is invalid
        """
        if config_path is None:
            config_path = cls._find_config()
        
        try:
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            # Process environment variables in config
            processed_config = cls._process_env_vars(raw_config)
            
            # Validate and create config object
            config = Config(**processed_config)
            cls._config = config
            
            logger.info(f"Successfully loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    @classmethod
    def _find_config(cls) -> Path:
        """Find configuration file in default locations"""
        search_paths = [
            Path.cwd() / 'config' / 'config.yaml',
            Path.home() / '.config' / 'metabolic_agent' / 'config.yaml',
            Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(
            "Configuration file not found in default locations: " +
            ", ".join(str(p) for p in search_paths)
        )
    
    @staticmethod
    def _process_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """Process environment variables in configuration"""
        def process_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.environ.get(env_var, value)
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(v) for v in value]
            return value
        
        return process_value(config)
    
    @classmethod
    def get_config(cls) -> Config:
        """Get the current configuration"""
        if cls._config is None:
            cls.load_config()
        return cls._config
    
    @classmethod
    def save_config(cls, config: Config, path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file"""
        if path is None:
            path = cls._find_config()
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Convert config to dict and save
            config_dict = config.model_dump()
            with open(path, 'w') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False)
            
            logger.info(f"Successfully saved configuration to {path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    @classmethod
    def update_config(cls, updates: Dict[str, Any]) -> Config:
        """Update current configuration with new values"""
        current_config = cls.get_config()
        config_dict = current_config.model_dump()
        
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        updated_dict = deep_update(config_dict, updates)
        cls._config = Config(**updated_dict)
        return cls._config

# Convenience functions
def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    return ConfigManager.load_config(config_path)

def save_config(config: Config, path: Optional[Union[str, Path]] = None) -> None:
    ConfigManager.save_config(config, path)

def get_config() -> Config:
    return ConfigManager.get_config()

def update_config(updates: Dict[str, Any]) -> Config:
    return ConfigManager.update_config(updates)