"""Tests for RAGSystem handling content-query related questions"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method"""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_orchestrates_components_correctly(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test that query() orchestrates all components in correct order"""
        # Setup mocks
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Generated response about courses"
        mock_ai_class.return_value = mock_ai

        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store_class.return_value = mock_store

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_class.return_value = mock_session

        mock_doc_class.return_value = Mock()

        # Create RAG system
        rag = RAGSystem(mock_config)

        # Execute query
        response, sources = rag.query(
            "What is prompt caching?", session_id="test_session"
        )

        # Verify AI generator was called
        mock_ai.generate_response.assert_called_once()

        # Verify session history was retrieved
        mock_session.get_conversation_history.assert_called_with("test_session")

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_passes_tools_to_ai_generator(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test that query() passes tool definitions to AI generator"""
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai

        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store_class.return_value = mock_store

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_class.return_value = mock_session

        mock_doc_class.return_value = Mock()

        rag = RAGSystem(mock_config)

        rag.query("Test question")

        # Check that tools were passed
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None

        # Check tool manager was passed
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is not None

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_includes_conversation_history(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test that query() includes conversation history when session exists"""
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai

        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store_class.return_value = mock_store

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = (
            "User: Previous\nAssistant: Previous response"
        )
        mock_session_class.return_value = mock_session

        mock_doc_class.return_value = Mock()

        rag = RAGSystem(mock_config)

        rag.query("Follow up question", session_id="session_123")

        call_kwargs = mock_ai.generate_response.call_args[1]
        assert (
            call_kwargs["conversation_history"]
            == "User: Previous\nAssistant: Previous response"
        )

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_updates_session_history(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test that query() adds exchange to session history"""
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "AI response"
        mock_ai_class.return_value = mock_ai

        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store_class.return_value = mock_store

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_class.return_value = mock_session

        mock_doc_class.return_value = Mock()

        rag = RAGSystem(mock_config)

        rag.query("User question", session_id="session_456")

        mock_session.add_exchange.assert_called_once_with(
            "session_456", "User question", "AI response"
        )

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_does_not_update_session_without_id(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test that query() does not update session when no session_id provided"""
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai

        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store_class.return_value = mock_store

        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_doc_class.return_value = Mock()

        rag = RAGSystem(mock_config)

        rag.query("Question without session")

        mock_session.add_exchange.assert_not_called()

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_returns_sources_from_tool_manager(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test that query() returns sources captured by tool manager"""
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai

        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store_class.return_value = mock_store

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_class.return_value = mock_session

        mock_doc_class.return_value = Mock()

        rag = RAGSystem(mock_config)

        # Manually add sources to the search tool
        rag.search_tool.last_sources = [
            {"text": "Course A - Lesson 1", "link": "https://example.com/a1"},
            {"text": "Course B - Lesson 2", "link": "https://example.com/b2"},
        ]

        response, sources = rag.query("Test query")

        assert len(sources) == 2
        assert sources[0]["text"] == "Course A - Lesson 1"
        assert sources[1]["link"] == "https://example.com/b2"

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_resets_sources_after_retrieval(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test that sources are reset after being retrieved"""
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai

        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store_class.return_value = mock_store

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_class.return_value = mock_session

        mock_doc_class.return_value = Mock()

        rag = RAGSystem(mock_config)

        # Add sources
        rag.search_tool.last_sources = [
            {"text": "Source", "link": "http://example.com"}
        ]

        # First query
        response1, sources1 = rag.query("First query")

        # Sources should be cleared
        assert rag.search_tool.last_sources == []

        # Second query should have empty sources if no search happens
        response2, sources2 = rag.query("Second query")
        assert sources2 == []


class TestRAGSystemToolRegistration:
    """Tests for RAGSystem tool registration"""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_registers_search_tool(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test that CourseSearchTool is registered"""
        mock_ai_class.return_value = Mock()
        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store_class.return_value = mock_store
        mock_session_class.return_value = Mock()
        mock_doc_class.return_value = Mock()

        rag = RAGSystem(mock_config)

        tool_defs = rag.tool_manager.get_tool_definitions()
        tool_names = [t["name"] for t in tool_defs]

        assert "search_course_content" in tool_names

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_registers_outline_tool(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test that CourseOutlineTool is registered"""
        mock_ai_class.return_value = Mock()
        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store_class.return_value = mock_store
        mock_session_class.return_value = Mock()
        mock_doc_class.return_value = Mock()

        rag = RAGSystem(mock_config)

        tool_defs = rag.tool_manager.get_tool_definitions()
        tool_names = [t["name"] for t in tool_defs]

        assert "get_course_outline" in tool_names


class TestRAGSystemPromptFormatting:
    """Tests for how RAGSystem formats prompts"""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_wraps_user_question_in_prompt(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test that user query is wrapped in instructional prompt"""
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai

        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store_class.return_value = mock_store

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_class.return_value = mock_session

        mock_doc_class.return_value = Mock()

        rag = RAGSystem(mock_config)

        rag.query("What is Claude?")

        call_kwargs = mock_ai.generate_response.call_args[1]
        query_sent = call_kwargs["query"]

        assert "What is Claude?" in query_sent
        assert "course materials" in query_sent.lower()


class TestRAGSystemAddCourseDocument:
    """Tests for RAGSystem.add_course_document()"""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_add_course_document_processes_and_stores(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test that add_course_document processes doc and stores in vector store"""
        mock_ai_class.return_value = Mock()

        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store_class.return_value = mock_store

        mock_session_class.return_value = Mock()

        # Setup document processor
        mock_doc = Mock()
        sample_course = Course(
            title="Test Course", lessons=[Lesson(lesson_number=1, title="Intro")]
        )
        sample_chunks = [
            CourseChunk(
                content="Chunk 1",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
            )
        ]
        mock_doc.process_course_document.return_value = (sample_course, sample_chunks)
        mock_doc_class.return_value = mock_doc

        rag = RAGSystem(mock_config)

        course, chunk_count = rag.add_course_document("/path/to/course.txt")

        # Verify document was processed
        mock_doc.process_course_document.assert_called_once_with("/path/to/course.txt")

        # Verify course metadata was added
        mock_store.add_course_metadata.assert_called_once_with(sample_course)

        # Verify chunks were added
        mock_store.add_course_content.assert_called_once_with(sample_chunks)

        # Verify return values
        assert course.title == "Test Course"
        assert chunk_count == 1

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_add_course_document_handles_errors(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test that add_course_document handles processing errors gracefully"""
        mock_ai_class.return_value = Mock()

        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store_class.return_value = mock_store

        mock_session_class.return_value = Mock()

        mock_doc = Mock()
        mock_doc.process_course_document.side_effect = Exception("Parse error")
        mock_doc_class.return_value = mock_doc

        rag = RAGSystem(mock_config)

        course, chunk_count = rag.add_course_document("/invalid/path.txt")

        assert course is None
        assert chunk_count == 0


class TestRAGSystemCourseAnalytics:
    """Tests for RAGSystem.get_course_analytics()"""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_get_course_analytics_returns_stats(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test that get_course_analytics returns correct statistics"""
        mock_ai_class.return_value = Mock()

        mock_store = Mock()
        mock_store.get_course_count.return_value = 5
        mock_store.get_existing_course_titles.return_value = [
            "Course A",
            "Course B",
            "Course C",
            "Course D",
            "Course E",
        ]
        mock_store_class.return_value = mock_store

        mock_session_class.return_value = Mock()
        mock_doc_class.return_value = Mock()

        rag = RAGSystem(mock_config)

        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course A" in analytics["course_titles"]


class TestRAGSystemContentQueryHandling:
    """End-to-end tests for content query handling"""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_content_query_flow_with_search_results(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test complete flow when search returns results"""
        # Setup AI generator to simulate tool use
        mock_ai = Mock()
        mock_ai.generate_response.return_value = (
            "Prompt caching reduces API costs by reusing processed prompts."
        )
        mock_ai_class.return_value = mock_ai

        # Setup vector store with search results
        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store.search.return_value = SearchResults(
            documents=["Prompt caching content here..."],
            metadata=[{"course_title": "Computer Use Course", "lesson_number": 5}],
            distances=[0.1],
        )
        mock_store.get_lesson_link.return_value = "https://example.com/lesson5"
        mock_store_class.return_value = mock_store

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_class.return_value = mock_session

        mock_doc_class.return_value = Mock()

        rag = RAGSystem(mock_config)

        # Simulate search tool execution by setting sources
        rag.search_tool.last_sources = [
            {
                "text": "Computer Use Course - Lesson 5",
                "link": "https://example.com/lesson5",
            }
        ]

        response, sources = rag.query("What is prompt caching?")

        assert "Prompt caching" in response
        assert len(sources) == 1
        assert "Computer Use Course" in sources[0]["text"]

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_content_query_flow_with_no_results(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test flow when search returns no results"""
        mock_ai = Mock()
        mock_ai.generate_response.return_value = (
            "I couldn't find information about that topic in the course materials."
        )
        mock_ai_class.return_value = mock_ai

        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        mock_store_class.return_value = mock_store

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_class.return_value = mock_session

        mock_doc_class.return_value = Mock()

        rag = RAGSystem(mock_config)

        response, sources = rag.query("Tell me about quantum computing")

        assert "couldn't find" in response.lower() or sources == []

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_content_query_with_conversation_context(
        self,
        mock_session_class,
        mock_doc_class,
        mock_store_class,
        mock_ai_class,
        mock_config,
    ):
        """Test that conversation context is used for follow-up questions"""
        mock_ai = Mock()
        mock_ai.generate_response.return_value = (
            "As I mentioned, prompt caching is very useful."
        )
        mock_ai_class.return_value = mock_ai

        mock_store = Mock()
        mock_store.get_existing_course_titles.return_value = []
        mock_store_class.return_value = mock_store

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = (
            "User: What is prompt caching?\n"
            "Assistant: Prompt caching is a feature that saves API costs."
        )
        mock_session_class.return_value = mock_session

        mock_doc_class.return_value = Mock()

        rag = RAGSystem(mock_config)

        response, sources = rag.query("Can you tell me more?", session_id="session_789")

        # Verify conversation history was passed
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert "prompt caching" in call_kwargs["conversation_history"].lower()
