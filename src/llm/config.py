#!/usr/bin/env python3
"""
Configuration management for LLM integration.
Designed for containerized deployment with environment-based configuration.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import json
from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="your_new_password", description="Neo4j password")
    lancedb_path: str = Field(default="data/stage06_esm2/lancedb", description="LanceDB database path")


class LLMConfig(BaseModel):
    """LLM integration configuration."""
    
    # Database connections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    # LLM provider settings
    llm_provider: str = Field(default="openai", description="LLM provider (openai, anthropic, local)")
    llm_model: str = Field(default="o3", description="LLM model name")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    
    # Embedding settings
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model for queries")
    embedding_dim: int = Field(default=320, description="ESM2 embedding dimension")
    
    # RAG settings
    max_context_length: int = Field(default=8000, description="Maximum context length for LLM")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold for protein search")
    max_results_per_query: int = Field(default=10, description="Maximum results per database query")
    
    # Performance settings
    timeout_seconds: int = Field(default=30, description="Query timeout in seconds")
    cache_enabled: bool = Field(default=True, description="Enable query result caching")
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create configuration from environment variables."""
        config_data = {}
        
        # Database settings
        db_config = {}
        if os.getenv('NEO4J_URI'):
            db_config['neo4j_uri'] = os.getenv('NEO4J_URI')
        if os.getenv('NEO4J_USER'):
            db_config['neo4j_user'] = os.getenv('NEO4J_USER')
        if os.getenv('NEO4J_PASSWORD'):
            db_config['neo4j_password'] = os.getenv('NEO4J_PASSWORD')
        if os.getenv('LANCEDB_PATH'):
            db_config['lancedb_path'] = os.getenv('LANCEDB_PATH')
        
        if db_config:
            config_data['database'] = db_config
        
        # LLM settings
        if os.getenv('LLM_PROVIDER'):
            config_data['llm_provider'] = os.getenv('LLM_PROVIDER')
        if os.getenv('LLM_MODEL'):
            config_data['llm_model'] = os.getenv('LLM_MODEL')
        if os.getenv('OPENAI_API_KEY'):
            config_data['openai_api_key'] = os.getenv('OPENAI_API_KEY')
        if os.getenv('ANTHROPIC_API_KEY'):
            config_data['anthropic_api_key'] = os.getenv('ANTHROPIC_API_KEY')
        
        # Performance settings
        if os.getenv('MAX_CONTEXT_LENGTH'):
            config_data['max_context_length'] = int(os.getenv('MAX_CONTEXT_LENGTH'))
        if os.getenv('SIMILARITY_THRESHOLD'):
            config_data['similarity_threshold'] = float(os.getenv('SIMILARITY_THRESHOLD'))
        if os.getenv('TIMEOUT_SECONDS'):
            config_data['timeout_seconds'] = int(os.getenv('TIMEOUT_SECONDS'))
        
        return cls(**config_data)
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'LLMConfig':
        """Load configuration from JSON file."""
        with open(config_path) as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.dict(), f, indent=2)
    
    def get_api_key(self) -> Optional[str]:
        """Get the appropriate API key based on provider."""
        if self.llm_provider == "openai":
            return self.openai_api_key or os.getenv('OPENAI_API_KEY')
        elif self.llm_provider == "anthropic":
            return self.anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        return None
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate configuration and return status of components."""
        status = {}
        
        # Check database connections
        status['neo4j_configured'] = bool(
            self.database.neo4j_uri and 
            self.database.neo4j_user and 
            self.database.neo4j_password
        )
        
        status['lancedb_configured'] = bool(
            self.database.lancedb_path and 
            Path(self.database.lancedb_path).exists()
        )
        
        # Check LLM provider
        api_key = self.get_api_key()
        status['llm_configured'] = bool(api_key)
        
        # Check paths exist
        status['all_paths_exist'] = status['lancedb_configured']
        
        return status


# Default configuration for containerized deployment
DEFAULT_CONTAINER_CONFIG = LLMConfig(
    database=DatabaseConfig(
        neo4j_uri="bolt://neo4j:7687",  # Docker service name
        neo4j_user="neo4j",
        neo4j_password="genomics2024",
        lancedb_path="/data/lancedb"  # Container mount point
    ),
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    max_context_length=8000,
    similarity_threshold=0.7,
    timeout_seconds=30
)
