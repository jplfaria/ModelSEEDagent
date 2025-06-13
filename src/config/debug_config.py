"""
Debug Configuration Management
=============================

Centralized debug configuration system with environment variable control.
Supports different levels of debug verbosity for different components.

Environment Variables:
- MODELSEED_DEBUG_LEVEL: overall debug level (QUIET, NORMAL, VERBOSE, TRACE)
- MODELSEED_DEBUG_COBRAKBASE: enable cobrakbase debug messages (true/false)
- MODELSEED_DEBUG_LANGGRAPH: enable LangGraph initialization debug (true/false)
- MODELSEED_DEBUG_HTTP: enable HTTP/SSL debug messages (true/false)
- MODELSEED_DEBUG_TOOLS: enable tool execution debug (true/false)
- MODELSEED_DEBUG_LLM: enable LLM interaction debug (true/false)
- MODELSEED_LOG_LLM_INPUTS: enable complete LLM input logging (true/false)
"""

import logging
import os
from enum import Enum
from typing import Dict, Any


class DebugLevel(Enum):
    """Debug verbosity levels"""
    QUIET = "quiet"      # Minimal output, errors only
    NORMAL = "normal"    # Standard info messages
    VERBOSE = "verbose"  # Detailed debugging
    TRACE = "trace"      # Maximum verbosity


class DebugFlags:
    """Debug flag configuration"""
    
    def __init__(self):
        # Overall debug level
        self.debug_level = self._get_debug_level()
        
        # Component-specific flags
        self.cobrakbase_debug = self._get_bool_env("MODELSEED_DEBUG_COBRAKBASE", False)
        self.langgraph_debug = self._get_bool_env("MODELSEED_DEBUG_LANGGRAPH", False)
        self.http_debug = self._get_bool_env("MODELSEED_DEBUG_HTTP", False)
        self.tools_debug = self._get_bool_env("MODELSEED_DEBUG_TOOLS", False)
        self.llm_debug = self._get_bool_env("MODELSEED_DEBUG_LLM", False)
        
        # Special logging flags
        self.log_llm_inputs = self._get_bool_env("MODELSEED_LOG_LLM_INPUTS", False)
        
        # Auto-configure based on debug level
        self._auto_configure()
        
    def _get_debug_level(self) -> DebugLevel:
        """Get debug level from environment"""
        level_str = os.getenv("MODELSEED_DEBUG_LEVEL", "normal").lower()
        try:
            return DebugLevel(level_str)
        except ValueError:
            return DebugLevel.NORMAL
            
    def _get_bool_env(self, var_name: str, default: bool) -> bool:
        """Get boolean environment variable"""
        value = os.getenv(var_name, str(default)).lower()
        return value in ("true", "1", "yes", "on")
        
    def _auto_configure(self):
        """Auto-configure flags based on debug level"""
        if self.debug_level == DebugLevel.QUIET:
            # Override all flags to False for quiet mode
            self.cobrakbase_debug = False
            self.langgraph_debug = False
            self.http_debug = False
            self.tools_debug = False
            self.llm_debug = False
            
        elif self.debug_level == DebugLevel.TRACE:
            # Enable all debugging for trace mode (unless explicitly disabled)
            if not os.getenv("MODELSEED_DEBUG_COBRAKBASE"):
                self.cobrakbase_debug = True
            if not os.getenv("MODELSEED_DEBUG_LANGGRAPH"):
                self.langgraph_debug = True
            if not os.getenv("MODELSEED_DEBUG_HTTP"):
                self.http_debug = True
            if not os.getenv("MODELSEED_DEBUG_TOOLS"):
                self.tools_debug = True
            if not os.getenv("MODELSEED_DEBUG_LLM"):
                self.llm_debug = True
                
    def get_logging_level(self) -> int:
        """Get Python logging level based on debug level"""
        if self.debug_level == DebugLevel.QUIET:
            return logging.WARNING
        elif self.debug_level == DebugLevel.NORMAL:
            return logging.INFO
        elif self.debug_level == DebugLevel.VERBOSE:
            return logging.DEBUG
        else:  # TRACE
            return 5  # Very detailed
            
    def should_suppress_cobrakbase(self) -> bool:
        """Should cobrakbase messages be suppressed?"""
        return not self.cobrakbase_debug
        
    def should_suppress_langgraph(self) -> bool:
        """Should LangGraph initialization messages be suppressed?"""
        return not self.langgraph_debug
        
    def should_suppress_http(self) -> bool:
        """Should HTTP/SSL messages be suppressed?"""
        return not self.http_debug
        
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for display"""
        return {
            "debug_level": self.debug_level.value,
            "component_flags": {
                "cobrakbase": self.cobrakbase_debug,
                "langgraph": self.langgraph_debug,
                "http": self.http_debug,
                "tools": self.tools_debug,
                "llm": self.llm_debug,
            },
            "special_flags": {
                "log_llm_inputs": self.log_llm_inputs,
            }
        }
        
    def apply_logging_config(self):
        """Apply logging configuration to Python logging system"""
        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(self.get_logging_level())
        
        # Configure specific loggers based on flags
        if self.should_suppress_http():
            # Suppress HTTP/SSL debug noise
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            
        if self.should_suppress_langgraph():
            # Suppress LangGraph agent initialization spam
            logging.getLogger("src.agents.langgraph_metabolic").setLevel(logging.WARNING)
            logging.getLogger("src.agents").setLevel(logging.WARNING)
            
        # Tool execution logging
        tools_level = logging.DEBUG if self.tools_debug else logging.INFO
        logging.getLogger("src.tools").setLevel(tools_level)
        
        # LLM interaction logging
        llm_level = logging.DEBUG if self.llm_debug else logging.INFO
        logging.getLogger("src.llm").setLevel(llm_level)


# Global debug configuration instance
debug_config = DebugFlags()


def get_debug_config() -> DebugFlags:
    """Get the global debug configuration"""
    return debug_config


def configure_logging():
    """Configure logging based on debug flags"""
    debug_config.apply_logging_config()
    

def print_debug_status():
    """Print current debug configuration status"""
    config = debug_config.get_config_summary()
    
    print(f"🔍 Debug Configuration:")
    print(f"   Overall Level: {config['debug_level'].upper()}")
    print(f"   Component Flags:")
    for component, enabled in config['component_flags'].items():
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        print(f"     {component:12}: {status}")
    print(f"   Special Flags:")
    for flag, enabled in config['special_flags'].items():
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        print(f"     {flag:12}: {status}")


def get_logging_level_for_component(component: str) -> int:
    """Get logging level for specific component"""
    component_flags = {
        "cobrakbase": debug_config.cobrakbase_debug,
        "langgraph": debug_config.langgraph_debug,
        "http": debug_config.http_debug,
        "tools": debug_config.tools_debug,
        "llm": debug_config.llm_debug,
    }
    
    if component in component_flags and component_flags[component]:
        return logging.DEBUG
    else:
        return debug_config.get_logging_level()