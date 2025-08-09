"""
Shared test fixtures and configuration for agent tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_neo4j_query_results():
    """Mock Neo4j query results for testing."""
    return [
        {"name": "test_protein", "id": "protein_001", "description": "Test protein"},
        {"name": "another_protein", "id": "protein_002", "description": "Another test protein"}
    ]


@pytest.fixture  
def empty_neo4j_results():
    """Empty Neo4j results for absence testing."""
    return []


@pytest.fixture
def mock_vector_search_results():
    """Mock vector search results."""
    return [
        {"protein_id": "protein_001", "similarity": 0.85},
        {"protein_id": "protein_002", "similarity": 0.72}
    ]


@pytest.fixture
def low_similarity_vector_results():
    """Low similarity vector results for absence testing."""
    return [
        {"protein_id": "protein_003", "similarity": 0.45},
        {"protein_id": "protein_004", "similarity": 0.38}
    ]