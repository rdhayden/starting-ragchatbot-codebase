"""Shared fixtures and mocks for RAG chatbot tests"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


@pytest.fixture
def sample_search_results():
    """Create sample SearchResults for testing"""
    return SearchResults(
        documents=[
            "Claude models support tool use for external integrations.",
            "The API allows you to define custom tools with schemas."
        ],
        metadata=[
            {"course_title": "Building Towards Computer Use", "lesson_number": 3},
            {"course_title": "Building Towards Computer Use", "lesson_number": 4}
        ],
        distances=[0.15, 0.25]
    )


@pytest.fixture
def empty_search_results():
    """Create empty SearchResults for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    """Create SearchResults with error for testing"""
    return SearchResults.empty("No course found matching 'NonExistent Course'")


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore for testing"""
    mock = Mock()
    mock.max_results = 5
    return mock


@pytest.fixture
def sample_course():
    """Create a sample Course for testing"""
    return Course(
        title="Introduction to MCP Servers",
        course_link="https://example.com/mcp-course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Building Servers", lesson_link="https://example.com/lesson2"),
            Lesson(lesson_number=3, title="Advanced Topics", lesson_link="https://example.com/lesson3")
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample CourseChunks for testing"""
    return [
        CourseChunk(
            content="MCP servers allow Claude to interact with external systems.",
            course_title="Introduction to MCP Servers",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="You can build custom tools using the MCP protocol.",
            course_title="Introduction to MCP Servers",
            lesson_number=2,
            chunk_index=1
        )
    ]


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response without tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_content = Mock()
    mock_content.text = "This is a direct response about course content."
    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture
def mock_anthropic_tool_use_response():
    """Create a mock Anthropic API response with tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    # Create tool use block
    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "search_course_content"
    tool_use_block.id = "tool_123"
    tool_use_block.input = {
        "query": "tool use",
        "course_name": "Building Towards Computer Use"
    }

    mock_response.content = [tool_use_block]
    return mock_response


@pytest.fixture
def mock_anthropic_final_response():
    """Create a mock final response after tool execution"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_content = Mock()
    mock_content.text = "Based on the course materials, tool use allows Claude to call external functions."
    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture
def sample_course_outline():
    """Sample course outline data"""
    return {
        "course_title": "Building Towards Computer Use",
        "course_link": "https://example.com/course",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "https://example.com/l1"},
            {"lesson_number": 2, "lesson_title": "Tool Use Basics", "lesson_link": "https://example.com/l2"},
            {"lesson_number": 3, "lesson_title": "Advanced Tools", "lesson_link": "https://example.com/l3"}
        ]
    }


class MockConfig:
    """Mock configuration for testing"""
    ANTHROPIC_API_KEY = "test-api-key"
    ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 5
    MAX_HISTORY = 2
    CHROMA_PATH = "./test_chroma_db"


@pytest.fixture
def mock_config():
    """Return mock configuration"""
    return MockConfig()
