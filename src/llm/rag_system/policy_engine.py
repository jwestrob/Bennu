#!/usr/bin/env python3
"""
Simple policy engine for user-configurable preferences.
Provides cost, latency, and behavior controls for genomic analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for user preferences and policies."""
    
    # Performance and cost policies
    max_tokens_per_query: int = 30000   # Match o3's 30K context limit for proper chunking decisions
    max_latency_seconds: int = 600      # 10 minutes for complex o3 reasoning
    allow_expensive_tools: bool = True
    max_refinement_depth: int = 10      # Higher limit to allow proper recursive analysis when needed
    
    # Execution preferences
    prefer_traditional_mode: bool = True
    enable_tool_integration: bool = True
    enable_literature_search: bool = True
    enable_code_interpreter: bool = True
    
    # Quality and reliability
    min_confidence_threshold: float = 0.7
    max_result_count: int = 1000
    enable_context_compression: bool = True
    compression_threshold: int = 30000
    
    # Tool-specific settings
    literature_search_max_results: int = 5
    code_interpreter_timeout: int = 30
    
    # Debug and logging
    enable_debug_logging: bool = False
    save_execution_logs: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_tokens_per_query < 1000:
            raise ValueError("max_tokens_per_query must be at least 1000")
        if self.max_latency_seconds < 10:
            raise ValueError("max_latency_seconds must be at least 10")
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise ValueError("min_confidence_threshold must be between 0.0 and 1.0")
        if self.max_refinement_depth < 1:
            raise ValueError("max_refinement_depth must be at least 1")


class PolicyEngine:
    """
    Manages user policies and applies them to query execution.
    
    Provides centralized policy enforcement and configuration management.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize policy engine.
        
        Args:
            config_file: Path to configuration file (JSON format)
        """
        self.config_file = config_file
        self.policies = PolicyConfig()
        
        # Load configuration if file provided
        if config_file:
            self.load_config(config_file)
        
        logger.info(f"🎯 Policy engine initialized with {self._get_policy_summary()}")
    
    def load_config(self, config_file: str):
        """Load policy configuration from JSON file."""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update policy configuration
                for key, value in config_data.items():
                    if hasattr(self.policies, key):
                        setattr(self.policies, key, value)
                    else:
                        logger.warning(f"Unknown policy configuration key: {key}")
                
                logger.info(f"✅ Loaded policy configuration from {config_file}")
            else:
                logger.warning(f"Configuration file not found: {config_file}")
        except Exception as e:
            logger.error(f"Failed to load policy configuration: {e}")
    
    def save_config(self, config_file: str):
        """Save current policy configuration to JSON file."""
        try:
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert dataclass to dictionary
            config_data = {
                key: getattr(self.policies, key)
                for key in self.policies.__dataclass_fields__.keys()
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"💾 Saved policy configuration to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save policy configuration: {e}")
    
    def should_use_tool(self, tool_name: str) -> bool:
        """Check if a tool should be used based on policies."""
        if not self.policies.enable_tool_integration:
            return False
        
        if tool_name == "literature_search":
            return self.policies.enable_literature_search
        elif tool_name == "code_interpreter":
            return self.policies.enable_code_interpreter and self.policies.allow_expensive_tools
        
        return True
    
    def should_use_agentic_mode(self, complexity: str, estimated_tokens: int) -> bool:
        """Check if agentic mode should be used based on policies."""
        # Honor user preference for traditional mode
        if self.policies.prefer_traditional_mode and complexity != "complex":
            return False
        
        # Check token budget
        if estimated_tokens > self.policies.max_tokens_per_query:
            logger.warning(f"Query exceeds token budget ({estimated_tokens} > {self.policies.max_tokens_per_query})")
            return False
        
        return True
    
    def get_timeout_for_operation(self, operation: str) -> int:
        """Get timeout for specific operations."""
        if operation == "code_interpreter":
            return self.policies.code_interpreter_timeout
        elif operation == "literature_search":
            return 30  # Default timeout for literature search
        else:
            return self.policies.max_latency_seconds
    
    def should_compress_context(self, context_size: int) -> bool:
        """Check if context should be compressed."""
        return (self.policies.enable_context_compression and 
                context_size > self.policies.compression_threshold)
    
    def get_max_results(self, query_type: str) -> int:
        """Get maximum results for query type."""
        if query_type == "literature_search":
            return self.policies.literature_search_max_results
        else:
            return self.policies.max_result_count
    
    def is_confidence_acceptable(self, confidence: float) -> bool:
        """Check if confidence level meets threshold."""
        return confidence >= self.policies.min_confidence_threshold
    
    def update_policy(self, key: str, value: Any):
        """Update a single policy setting."""
        if hasattr(self.policies, key):
            old_value = getattr(self.policies, key)
            setattr(self.policies, key, value)
            logger.info(f"🔧 Updated policy {key}: {old_value} → {value}")
        else:
            logger.warning(f"Unknown policy key: {key}")
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of current policies."""
        return {
            "execution_mode": "traditional" if self.policies.prefer_traditional_mode else "adaptive",
            "tool_integration": self.policies.enable_tool_integration,
            "expensive_tools": self.policies.allow_expensive_tools,
            "max_tokens": self.policies.max_tokens_per_query,
            "max_latency": self.policies.max_latency_seconds,
            "compression_enabled": self.policies.enable_context_compression,
            "min_confidence": self.policies.min_confidence_threshold
        }
    
    def _get_policy_summary(self) -> str:
        """Get a brief summary of current policies."""
        mode = "traditional" if self.policies.prefer_traditional_mode else "adaptive"
        tools = "enabled" if self.policies.enable_tool_integration else "disabled"
        return f"mode={mode}, tools={tools}, max_tokens={self.policies.max_tokens_per_query}"


# Global policy engine instance
policy_engine = PolicyEngine()


def get_policy_engine() -> PolicyEngine:
    """Get the global policy engine instance."""
    return policy_engine


def load_user_config(config_file: str):
    """Load user configuration from file."""
    global policy_engine
    policy_engine.load_config(config_file)


def save_user_config(config_file: str):
    """Save current configuration to file."""
    global policy_engine
    policy_engine.save_config(config_file)


def update_policy(key: str, value: Any):
    """Update a single policy setting."""
    global policy_engine
    policy_engine.update_policy(key, value)


def get_current_policies() -> Dict[str, Any]:
    """Get current policy configuration."""
    global policy_engine
    return policy_engine.get_policy_summary()