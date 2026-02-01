"""Tests for CourseSearchTool.execute() method in search_tools.py"""

import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager, Tool
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() method"""

    def test_execute_basic_query_returns_formatted_results(
        self, mock_vector_store, sample_search_results
    ):
        """Test that execute() returns properly formatted results for basic query"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="tool use")

        assert "[Building Towards Computer Use - Lesson 3]" in result
        assert "Claude models support tool use" in result
        mock_vector_store.search.assert_called_once_with(
            query="tool use", course_name=None, lesson_number=None
        )

    def test_execute_with_course_filter(self, mock_vector_store, sample_search_results):
        """Test that execute() passes course_name filter to vector store"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="prompting", course_name="MCP")

        mock_vector_store.search.assert_called_once_with(
            query="prompting", course_name="MCP", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test that execute() passes lesson_number filter to vector store"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="introduction", lesson_number=1)

        mock_vector_store.search.assert_called_once_with(
            query="introduction", course_name=None, lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test that execute() passes both course and lesson filters"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(
            query="computer use",
            course_name="Building Towards Computer Use",
            lesson_number=2,
        )

        mock_vector_store.search.assert_called_once_with(
            query="computer use",
            course_name="Building Towards Computer Use",
            lesson_number=2,
        )

    def test_execute_returns_error_when_search_fails(
        self, mock_vector_store, error_search_results
    ):
        """Test that execute() returns error message when search has error"""
        mock_vector_store.search.return_value = error_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="something", course_name="NonExistent Course")

        assert "No course found matching 'NonExistent Course'" in result

    def test_execute_returns_no_results_message(
        self, mock_vector_store, empty_search_results
    ):
        """Test that execute() returns appropriate message when no results found"""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="very specific obscure topic")

        assert "No relevant content found" in result

    def test_execute_no_results_includes_filter_info(
        self, mock_vector_store, empty_search_results
    ):
        """Test that empty results message includes filter context"""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="topic", course_name="MCP", lesson_number=5)

        assert "in course 'MCP'" in result
        assert "in lesson 5" in result

    def test_execute_tracks_sources(self, mock_vector_store, sample_search_results):
        """Test that execute() populates last_sources for UI"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson3"

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="tool use")

        assert len(tool.last_sources) == 2
        assert (
            tool.last_sources[0]["text"] == "Building Towards Computer Use - Lesson 3"
        )
        assert tool.last_sources[0]["link"] == "https://example.com/lesson3"

    def test_execute_handles_missing_lesson_number_in_metadata(self, mock_vector_store):
        """Test that execute() handles metadata without lesson_number gracefully"""
        results = SearchResults(
            documents=["Some content without lesson info"],
            metadata=[{"course_title": "Test Course"}],  # No lesson_number
            distances=[0.1],
        )
        mock_vector_store.search.return_value = results
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")

        # Should format without lesson number
        assert "[Test Course]" in result
        assert "Lesson" not in result.split("]")[0]  # No lesson in header

    def test_execute_formats_multiple_results(
        self, mock_vector_store, sample_search_results
    ):
        """Test that multiple results are separated properly"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="tools")

        # Results should be separated by double newlines
        parts = result.split("\n\n")
        assert len(parts) == 2


class TestCourseSearchToolDefinition:
    """Tests for CourseSearchTool.get_tool_definition()"""

    def test_tool_definition_has_required_fields(self, mock_vector_store):
        """Test that tool definition has all required Anthropic fields"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["name"] == "search_course_content"

    def test_tool_definition_schema_has_query_required(self, mock_vector_store):
        """Test that query is marked as required in schema"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        schema = definition["input_schema"]
        assert "query" in schema["properties"]
        assert "query" in schema["required"]

    def test_tool_definition_has_optional_filters(self, mock_vector_store):
        """Test that course_name and lesson_number are optional"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        schema = definition["input_schema"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        # These should NOT be in required
        assert "course_name" not in schema["required"]
        assert "lesson_number" not in schema["required"]


class TestCourseOutlineToolExecute:
    """Tests for CourseOutlineTool.execute() method"""

    def test_execute_returns_formatted_outline(
        self, mock_vector_store, sample_course_outline
    ):
        """Test that execute() returns properly formatted course outline"""
        mock_vector_store.get_course_outline.return_value = sample_course_outline

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="Computer Use")

        assert "Course Title: Building Towards Computer Use" in result
        assert "Course Link:" in result
        assert "Lesson 1: Introduction" in result
        assert "Lesson 2: Tool Use Basics" in result

    def test_execute_returns_error_for_unknown_course(self, mock_vector_store):
        """Test that execute() returns error for non-existent course"""
        mock_vector_store.get_course_outline.return_value = None

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="NonExistent")

        assert "No course found matching 'NonExistent'" in result

    def test_execute_tracks_source(self, mock_vector_store, sample_course_outline):
        """Test that execute() populates last_sources"""
        mock_vector_store.get_course_outline.return_value = sample_course_outline

        tool = CourseOutlineTool(mock_vector_store)
        tool.execute(course_name="Computer Use")

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Building Towards Computer Use"
        assert tool.last_sources[0]["link"] == "https://example.com/course"


class TestToolManager:
    """Tests for ToolManager class"""

    def test_register_tool_adds_tool(self, mock_vector_store):
        """Test that tools can be registered"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_register_tool_rejects_tool_without_name(self):
        """Test that tools without name in definition are rejected"""
        manager = ToolManager()

        # Create a mock tool with no name
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {"description": "No name"}

        with pytest.raises(ValueError, match="must have a 'name'"):
            manager.register_tool(mock_tool)

    def test_get_tool_definitions_returns_all_tools(self, mock_vector_store):
        """Test that get_tool_definitions returns definitions for all registered tools"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "search_course_content" in names
        assert "get_course_outline" in names

    def test_execute_tool_calls_correct_tool(
        self, mock_vector_store, sample_search_results
    ):
        """Test that execute_tool routes to the correct tool"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")

        assert "[Building Towards Computer Use" in result

    def test_execute_tool_returns_error_for_unknown_tool(self):
        """Test that execute_tool returns error for non-existent tool"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", param="value")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources_returns_sources_from_tools(
        self, mock_vector_store, sample_search_results
    ):
        """Test that get_last_sources aggregates sources from tools"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()

        assert len(sources) == 2

    def test_reset_sources_clears_all_tool_sources(
        self, mock_vector_store, sample_search_results
    ):
        """Test that reset_sources clears sources from all tools"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        manager.execute_tool("search_course_content", query="test")
        manager.reset_sources()

        assert manager.get_last_sources() == []
        assert tool.last_sources == []
