"""Integration tests using real components (except external API calls)"""
import pytest
import tempfile
import os
import shutil
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from document_processor import DocumentProcessor
from models import Course, Lesson, CourseChunk
from rag_system import RAGSystem


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def real_vector_store(temp_chroma_dir):
    """Create a real VectorStore with temporary storage"""
    return VectorStore(
        chroma_path=temp_chroma_dir,
        embedding_model="all-MiniLM-L6-v2",
        max_results=5
    )


@pytest.fixture
def sample_course_file(tmp_path):
    """Create a sample course file for testing"""
    content = """Course Title: Introduction to AI Tools
Course Link: https://example.com/ai-tools-course
Course Instructor: Dr. Test

Lesson 0: Introduction
Lesson Link: https://example.com/ai-tools-course/lesson0
Welcome to the Introduction to AI Tools course. In this course,
you will learn about various AI-powered tools and how to use them
effectively in your projects.

Lesson 1: Understanding Large Language Models
Lesson Link: https://example.com/ai-tools-course/lesson1
Large Language Models (LLMs) are deep learning models trained on
massive amounts of text data. They can generate human-like text,
answer questions, and perform various NLP tasks.

Lesson 2: Tool Use and Function Calling
Lesson Link: https://example.com/ai-tools-course/lesson2
Tool use allows AI models to interact with external systems.
Function calling enables structured output and API integration.
Claude can use tools defined with JSON schemas.

Lesson 3: Prompt Engineering Basics
Lesson Link: https://example.com/ai-tools-course/lesson3
Prompt engineering is the practice of designing effective prompts.
Good prompts are clear, specific, and provide context.
Chain of thought prompting improves reasoning capabilities.
"""
    file_path = tmp_path / "test_course.txt"
    file_path.write_text(content)
    return str(file_path)


class TestVectorStoreIntegration:
    """Integration tests for VectorStore with real ChromaDB"""

    def test_add_and_search_course_content(self, real_vector_store):
        """Test adding course content and searching for it"""
        # Add sample course
        course = Course(
            title="Test AI Course",
            course_link="https://example.com/test",
            instructor="Test Instructor",
            lessons=[
                Lesson(lesson_number=1, title="Intro", lesson_link="https://example.com/l1"),
                Lesson(lesson_number=2, title="Advanced", lesson_link="https://example.com/l2")
            ]
        )
        real_vector_store.add_course_metadata(course)

        # Add content chunks
        chunks = [
            CourseChunk(
                content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                course_title="Test AI Course",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Neural networks are computational models inspired by biological neurons.",
                course_title="Test AI Course",
                lesson_number=2,
                chunk_index=1
            )
        ]
        real_vector_store.add_course_content(chunks)

        # Search for content
        results = real_vector_store.search("machine learning")

        assert not results.is_empty()
        assert "machine learning" in results.documents[0].lower()

    def test_search_with_course_filter(self, real_vector_store):
        """Test that course filter correctly limits search results"""
        # Add two different courses
        course1 = Course(title="Python Basics", lessons=[])
        course2 = Course(title="JavaScript Fundamentals", lessons=[])

        real_vector_store.add_course_metadata(course1)
        real_vector_store.add_course_metadata(course2)

        # Add content for both
        chunks = [
            CourseChunk(content="Variables store data values", course_title="Python Basics", lesson_number=1, chunk_index=0),
            CourseChunk(content="Variables store data values in JavaScript", course_title="JavaScript Fundamentals", lesson_number=1, chunk_index=1)
        ]
        real_vector_store.add_course_content(chunks)

        # Search with course filter
        results = real_vector_store.search("variables", course_name="Python")

        # Should only get Python results
        for meta in results.metadata:
            assert meta["course_title"] == "Python Basics"

    def test_search_returns_empty_for_nonexistent_course(self, real_vector_store):
        """Test that searching in nonexistent course returns appropriate error"""
        results = real_vector_store.search("test query", course_name="NonExistent Course")

        assert results.error is not None
        assert "No course found" in results.error

    def test_get_course_outline(self, real_vector_store):
        """Test retrieving course outline with lessons"""
        course = Course(
            title="Complete Web Development",
            course_link="https://example.com/webdev",
            instructor="Web Expert",
            lessons=[
                Lesson(lesson_number=1, title="HTML Basics", lesson_link="https://example.com/l1"),
                Lesson(lesson_number=2, title="CSS Styling", lesson_link="https://example.com/l2"),
                Lesson(lesson_number=3, title="JavaScript", lesson_link="https://example.com/l3")
            ]
        )
        real_vector_store.add_course_metadata(course)

        outline = real_vector_store.get_course_outline("Web Development")

        assert outline is not None
        assert outline["course_title"] == "Complete Web Development"
        assert outline["course_link"] == "https://example.com/webdev"
        assert len(outline["lessons"]) == 3
        assert outline["lessons"][0]["lesson_title"] == "HTML Basics"

    def test_get_lesson_link(self, real_vector_store):
        """Test retrieving specific lesson link"""
        course = Course(
            title="Data Science 101",
            lessons=[
                Lesson(lesson_number=1, title="Pandas", lesson_link="https://example.com/pandas"),
                Lesson(lesson_number=2, title="NumPy", lesson_link="https://example.com/numpy")
            ]
        )
        real_vector_store.add_course_metadata(course)

        link = real_vector_store.get_lesson_link("Data Science 101", 2)

        assert link == "https://example.com/numpy"


class TestSearchToolIntegration:
    """Integration tests for CourseSearchTool with real VectorStore"""

    def test_search_tool_with_real_vector_store(self, real_vector_store):
        """Test CourseSearchTool with real VectorStore"""
        # Setup course data
        course = Course(
            title="API Development Course",
            lessons=[Lesson(lesson_number=1, title="REST APIs", lesson_link="https://example.com/api1")]
        )
        real_vector_store.add_course_metadata(course)

        chunks = [
            CourseChunk(
                content="REST APIs use HTTP methods like GET, POST, PUT, and DELETE.",
                course_title="API Development Course",
                lesson_number=1,
                chunk_index=0
            )
        ]
        real_vector_store.add_course_content(chunks)

        # Use the tool
        tool = CourseSearchTool(real_vector_store)
        result = tool.execute(query="HTTP methods")

        assert "[API Development Course - Lesson 1]" in result
        assert "HTTP methods" in result or "GET" in result

    def test_search_tool_tracks_sources(self, real_vector_store):
        """Test that search tool correctly tracks sources for UI"""
        course = Course(
            title="Testing Course",
            lessons=[Lesson(lesson_number=5, title="Unit Tests", lesson_link="https://example.com/unit-tests")]
        )
        real_vector_store.add_course_metadata(course)

        chunks = [
            CourseChunk(
                content="Unit tests verify individual components work correctly.",
                course_title="Testing Course",
                lesson_number=5,
                chunk_index=0
            )
        ]
        real_vector_store.add_course_content(chunks)

        tool = CourseSearchTool(real_vector_store)
        tool.execute(query="unit testing")

        assert len(tool.last_sources) > 0
        assert "Testing Course" in tool.last_sources[0]["text"]
        assert tool.last_sources[0]["link"] == "https://example.com/unit-tests"


class TestDocumentProcessorIntegration:
    """Integration tests for DocumentProcessor"""

    def test_process_course_document(self, sample_course_file):
        """Test processing a real course document"""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)

        course, chunks = processor.process_course_document(sample_course_file)

        assert course.title == "Introduction to AI Tools"
        assert course.course_link == "https://example.com/ai-tools-course"
        assert course.instructor == "Dr. Test"
        assert len(course.lessons) == 4  # Lesson 0, 1, 2, 3

        # Verify lessons
        assert course.lessons[0].lesson_number == 0
        assert course.lessons[0].title == "Introduction"
        assert course.lessons[1].title == "Understanding Large Language Models"

        # Verify chunks were created
        assert len(chunks) > 0
        assert all(isinstance(c, CourseChunk) for c in chunks)

    def test_chunks_have_correct_metadata(self, sample_course_file):
        """Test that chunks have correct course and lesson metadata"""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)

        course, chunks = processor.process_course_document(sample_course_file)

        for chunk in chunks:
            assert chunk.course_title == "Introduction to AI Tools"
            assert chunk.lesson_number is not None
            assert chunk.chunk_index >= 0


class TestToolManagerIntegration:
    """Integration tests for ToolManager with real tools"""

    def test_tool_manager_with_real_tools(self, real_vector_store):
        """Test ToolManager with real search and outline tools"""
        # Add course data
        course = Course(
            title="Integration Test Course",
            course_link="https://example.com/integration",
            lessons=[Lesson(lesson_number=1, title="Testing", lesson_link="https://example.com/l1")]
        )
        real_vector_store.add_course_metadata(course)

        chunks = [
            CourseChunk(
                content="Integration testing verifies multiple components work together.",
                course_title="Integration Test Course",
                lesson_number=1,
                chunk_index=0
            )
        ]
        real_vector_store.add_course_content(chunks)

        # Setup tool manager
        manager = ToolManager()
        search_tool = CourseSearchTool(real_vector_store)
        outline_tool = CourseOutlineTool(real_vector_store)
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        # Test search execution
        search_result = manager.execute_tool("search_course_content", query="integration testing")
        assert "Integration Test Course" in search_result

        # Test outline execution
        outline_result = manager.execute_tool("get_course_outline", course_name="Integration")
        assert "Integration Test Course" in outline_result
        assert "Testing" in outline_result

        # Test source retrieval
        sources = manager.get_last_sources()
        assert len(sources) > 0


class TestRAGSystemIntegration:
    """Integration tests for the complete RAG system"""

    @patch('rag_system.AIGenerator')
    def test_rag_system_with_real_document_processing(
        self, mock_ai_class, temp_chroma_dir, sample_course_file
    ):
        """Test RAG system processing a real document"""
        # Mock AI generator to avoid API calls
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "This course covers AI tools."
        mock_ai_class.return_value = mock_ai

        # Create config for testing
        class TestConfig:
            ANTHROPIC_API_KEY = "test-key"
            ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
            EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            CHUNK_SIZE = 500
            CHUNK_OVERLAP = 50
            MAX_RESULTS = 5
            MAX_HISTORY = 2
            CHROMA_PATH = temp_chroma_dir

        rag = RAGSystem(TestConfig())

        # Add the sample course document
        course, chunk_count = rag.add_course_document(sample_course_file)

        assert course is not None
        assert course.title == "Introduction to AI Tools"
        assert chunk_count > 0

        # Test that course was added to vector store
        analytics = rag.get_course_analytics()
        assert analytics["total_courses"] == 1
        assert "Introduction to AI Tools" in analytics["course_titles"]

    @patch('rag_system.AIGenerator')
    def test_rag_system_query_uses_real_search(
        self, mock_ai_class, temp_chroma_dir, sample_course_file
    ):
        """Test that RAG system query actually searches the vector store"""
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Tool use enables external interactions."
        mock_ai_class.return_value = mock_ai

        class TestConfig:
            ANTHROPIC_API_KEY = "test-key"
            ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
            EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            CHUNK_SIZE = 500
            CHUNK_OVERLAP = 50
            MAX_RESULTS = 5
            MAX_HISTORY = 2
            CHROMA_PATH = temp_chroma_dir

        rag = RAGSystem(TestConfig())

        # Add course document
        rag.add_course_document(sample_course_file)

        # Execute search manually through tool manager
        search_result = rag.tool_manager.execute_tool(
            "search_course_content",
            query="tool use function calling"
        )

        # Verify search found relevant content
        assert "Introduction to AI Tools" in search_result or "Tool use" in search_result


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_search_results(self, real_vector_store):
        """Test behavior when search returns no results"""
        tool = CourseSearchTool(real_vector_store)
        result = tool.execute(query="completely random gibberish xyz123")

        assert "No relevant content found" in result

    def test_search_with_special_characters(self, real_vector_store):
        """Test search handles special characters gracefully"""
        course = Course(title="Test Course", lessons=[])
        real_vector_store.add_course_metadata(course)

        chunks = [
            CourseChunk(
                content="Regular content here",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            )
        ]
        real_vector_store.add_course_content(chunks)

        tool = CourseSearchTool(real_vector_store)

        # These should not crash
        result1 = tool.execute(query="test <script>alert('xss')</script>")
        result2 = tool.execute(query="test'; DROP TABLE users;--")
        result3 = tool.execute(query="test\nwith\nnewlines")

        # Just verify they return some result without crashing
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert isinstance(result3, str)

    def test_very_long_query(self, real_vector_store):
        """Test handling of very long query strings"""
        course = Course(title="Test", lessons=[])
        real_vector_store.add_course_metadata(course)

        chunks = [
            CourseChunk(content="Content", course_title="Test", lesson_number=1, chunk_index=0)
        ]
        real_vector_store.add_course_content(chunks)

        tool = CourseSearchTool(real_vector_store)

        long_query = "test " * 1000  # Very long query
        result = tool.execute(query=long_query)

        assert isinstance(result, str)
