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

# API testing imports
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel


@pytest.fixture
def sample_search_results():
    """Create sample SearchResults for testing"""
    return SearchResults(
        documents=[
            "Claude models support tool use for external integrations.",
            "The API allows you to define custom tools with schemas.",
        ],
        metadata=[
            {"course_title": "Building Towards Computer Use", "lesson_number": 3},
            {"course_title": "Building Towards Computer Use", "lesson_number": 4},
        ],
        distances=[0.15, 0.25],
    )


@pytest.fixture
def empty_search_results():
    """Create empty SearchResults for testing"""
    return SearchResults(documents=[], metadata=[], distances=[])


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
            Lesson(
                lesson_number=1,
                title="Getting Started",
                lesson_link="https://example.com/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Building Servers",
                lesson_link="https://example.com/lesson2",
            ),
            Lesson(
                lesson_number=3,
                title="Advanced Topics",
                lesson_link="https://example.com/lesson3",
            ),
        ],
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample CourseChunks for testing"""
    return [
        CourseChunk(
            content="MCP servers allow Claude to interact with external systems.",
            course_title="Introduction to MCP Servers",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="You can build custom tools using the MCP protocol.",
            course_title="Introduction to MCP Servers",
            lesson_number=2,
            chunk_index=1,
        ),
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
        "course_name": "Building Towards Computer Use",
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
            {
                "lesson_number": 1,
                "lesson_title": "Introduction",
                "lesson_link": "https://example.com/l1",
            },
            {
                "lesson_number": 2,
                "lesson_title": "Tool Use Basics",
                "lesson_link": "https://example.com/l2",
            },
            {
                "lesson_number": 3,
                "lesson_title": "Advanced Tools",
                "lesson_link": "https://example.com/l3",
            },
        ],
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


# =============================================================================
# API Testing Fixtures
# =============================================================================

@pytest.fixture
def mock_rag_system():
    """Create a mock RAGSystem for API testing"""
    mock = Mock()

    # Mock session_manager
    mock.session_manager = Mock()
    mock.session_manager.create_session.return_value = "test-session-123"

    # Mock query method - returns (answer, sources)
    mock.query.return_value = (
        "This is a test answer about course content.",
        [
            {"text": "Building Towards Computer Use - Lesson 3", "link": "https://example.com/lesson3"},
            {"text": "Building Towards Computer Use - Lesson 4", "link": "https://example.com/lesson4"}
        ]
    )

    # Mock get_course_analytics
    mock.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": [
            "Building Towards Computer Use",
            "Introduction to MCP Servers",
            "Advanced Claude Techniques"
        ]
    }

    return mock


@pytest.fixture
def mock_rag_system_error():
    """Create a mock RAGSystem that raises errors"""
    mock = Mock()
    mock.session_manager = Mock()
    mock.session_manager.create_session.return_value = "test-session-123"
    mock.query.side_effect = Exception("Database connection failed")
    mock.get_course_analytics.side_effect = Exception("Failed to fetch analytics")
    return mock


@pytest.fixture
def mock_rag_system_empty():
    """Create a mock RAGSystem with empty results"""
    mock = Mock()
    mock.session_manager = Mock()
    mock.session_manager.create_session.return_value = "test-session-456"
    mock.query.return_value = (
        "I couldn't find any relevant information about that topic.",
        []
    )
    mock.get_course_analytics.return_value = {
        "total_courses": 0,
        "course_titles": []
    }
    return mock


def create_test_app(mock_rag_system):
    """
    Create a test FastAPI app without static file mounting.
    This avoids the '../frontend' directory dependency.
    """
    from pydantic import BaseModel
    from typing import List, Optional
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Course Materials RAG System - Test")

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            source_items = []
            for s in sources:
                if isinstance(s, dict):
                    source_items.append(SourceItem(text=s["text"], link=s.get("link")))
                else:
                    source_items.append(SourceItem(text=s, link=None))

            return QueryResponse(
                answer=answer,
                sources=source_items,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API"}

    return app


@pytest.fixture
def test_client(mock_rag_system):
    """Create a TestClient with mocked RAGSystem"""
    app = create_test_app(mock_rag_system)
    return TestClient(app)


@pytest.fixture
def test_client_error(mock_rag_system_error):
    """Create a TestClient that returns errors"""
    app = create_test_app(mock_rag_system_error)
    return TestClient(app)


@pytest.fixture
def test_client_empty(mock_rag_system_empty):
    """Create a TestClient with empty results"""
    app = create_test_app(mock_rag_system_empty)
    return TestClient(app)


@pytest.fixture
def sample_query_request():
    """Sample query request data"""
    return {
        "query": "What is tool use in Claude?",
        "session_id": None
    }


@pytest.fixture
def sample_query_request_with_session():
    """Sample query request with existing session"""
    return {
        "query": "Tell me more about the previous topic",
        "session_id": "existing-session-789"
    }
