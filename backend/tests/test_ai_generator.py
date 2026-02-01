"""Tests for AIGenerator correctly calling CourseSearchTool"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestAIGeneratorToolCalling:
    """Tests for AIGenerator's tool calling behavior"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_passes_tools_to_api(self, mock_anthropic_class):
        """Test that tools are passed to the Anthropic API call"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Setup response without tool use
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Direct response"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content", "description": "Search courses"}]

        generator.generate_response(
            query="What is tool use?", tools=tools, tool_manager=Mock()
        )

        # Verify tools were passed to API
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test that API is called without tools parameter when none provided"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Response without tools"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        result = generator.generate_response(query="Hello")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_executes_tool_when_requested(self, mock_anthropic_class):
        """Test that tool is executed when Claude requests tool use"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use request
        tool_use_response = Mock()
        tool_use_response.stop_reason = "tool_use"
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_123"
        tool_use_block.input = {
            "query": "prompt caching",
            "course_name": "Computer Use",
        }
        tool_use_response.content = [tool_use_block]

        # Second response: final answer
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Prompt caching saves cost and latency."
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Setup tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Search results about prompt caching..."
        )

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content", "description": "Search courses"}]

        result = generator.generate_response(
            query="Tell me about prompt caching",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Verify tool was executed with correct parameters
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="prompt caching", course_name="Computer Use"
        )

        # Verify final response is returned
        assert result == "Prompt caching saves cost and latency."

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_handles_multiple_tool_calls(self, mock_anthropic_class):
        """Test that multiple tool calls in one response are all executed"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Response with multiple tool use blocks
        tool_use_response = Mock()
        tool_use_response.stop_reason = "tool_use"

        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.id = "tool_1"
        tool_block1.input = {"query": "caching"}

        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "get_course_outline"
        tool_block2.id = "tool_2"
        tool_block2.input = {"course_name": "MCP"}

        tool_use_response.content = [tool_block1, tool_block2]

        # Final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Combined answer from both tools"
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        result = generator.generate_response(
            query="Test query", tools=[{"name": "test"}], tool_manager=mock_tool_manager
        )

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_sends_tool_results_to_api(self, mock_anthropic_class):
        """Test that tool results are sent back to the API correctly"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Setup tool use response
        tool_use_response = Mock()
        tool_use_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_abc123"
        tool_block.input = {"query": "test"}
        tool_use_response.content = [tool_block]

        # Setup final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Final answer"
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        generator.generate_response(
            query="Test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Check the second API call includes tool results
        second_call = mock_client.messages.create.call_args_list[1]
        messages = second_call[1]["messages"]

        # Should have: original user message, assistant tool use, user tool result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Check tool result format
        tool_result = messages[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_abc123"
        assert tool_result["content"] == "Tool execution result"

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_includes_conversation_history(
        self, mock_anthropic_class
    ):
        """Test that conversation history is included in system prompt"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Response with context"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        history = "User: Previous question\nAssistant: Previous answer"

        generator.generate_response(
            query="Follow-up question", conversation_history=history
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        system_content = call_kwargs["system"]

        assert "Previous conversation:" in system_content
        assert "Previous question" in system_content
        assert "Previous answer" in system_content

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_includes_tools_on_follow_up_call(
        self, mock_anthropic_class
    ):
        """Test that tools ARE included in follow-up API calls (supporting sequential tool use)"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Tool use response
        tool_use_response = Mock()
        tool_use_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_id"
        tool_block.input = {"query": "test"}
        tool_use_response.content = [tool_block]

        # Final response (text, no more tool use)
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Final"
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        generator.generate_response(
            query="Test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Check second call DOES have tools (supporting sequential tool use)
        second_call = mock_client.messages.create.call_args_list[1]
        assert "tools" in second_call[1]
        assert second_call[1]["tools"] == [{"name": "search_course_content"}]


class TestAIGeneratorConfiguration:
    """Tests for AIGenerator configuration and initialization"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_uses_configured_model(self, mock_anthropic_class):
        """Test that the configured model is used in API calls"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Response"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="claude-opus-4-20250514")

        generator.generate_response(query="Test")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-20250514"

    @patch("ai_generator.anthropic.Anthropic")
    def test_uses_zero_temperature(self, mock_anthropic_class):
        """Test that temperature is set to 0 for deterministic responses"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Response"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        generator.generate_response(query="Test")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0

    @patch("ai_generator.anthropic.Anthropic")
    def test_system_prompt_contains_tool_instructions(self, mock_anthropic_class):
        """Test that system prompt includes instructions for tool usage"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Response"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        generator.generate_response(query="Test")

        call_kwargs = mock_client.messages.create.call_args[1]
        system_prompt = call_kwargs["system"]

        assert "search_course_content" in system_prompt
        assert "get_course_outline" in system_prompt


class TestAIGeneratorIntegrationWithToolManager:
    """Integration tests for AIGenerator with real ToolManager"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_end_to_end_tool_execution(
        self, mock_anthropic_class, mock_vector_store, sample_search_results
    ):
        """Test complete flow: API call -> tool execution -> final response"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Setup vector store
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        # Tool use response from Claude
        tool_use_response = Mock()
        tool_use_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_xyz"
        tool_block.input = {"query": "tool use basics"}
        tool_use_response.content = [tool_block]

        # Final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Tool use allows Claude to call external functions."
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Setup real ToolManager with CourseSearchTool
        tool_manager = ToolManager()
        from search_tools import CourseSearchTool

        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        result = generator.generate_response(
            query="What is tool use?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify the flow completed correctly
        assert result == "Tool use allows Claude to call external functions."

        # Verify search was called
        mock_vector_store.search.assert_called_once_with(
            query="tool use basics", course_name=None, lesson_number=None
        )

        # Verify sources were tracked
        sources = tool_manager.get_last_sources()
        assert len(sources) == 2


class TestSequentialToolCalling:
    """Tests for sequential tool calling behavior (up to 2 rounds)"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_sequential_tool_calls(self, mock_anthropic_class):
        """Test that Claude can make 2 tool calls in sequence"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        tool_use_response_1 = Mock()
        tool_use_response_1.stop_reason = "tool_use"
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "MCP basics"}
        tool_use_response_1.content = [tool_block_1]

        # Second response: another tool use
        tool_use_response_2 = Mock()
        tool_use_response_2.stop_reason = "tool_use"
        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "get_course_outline"
        tool_block_2.id = "tool_2"
        tool_block_2.input = {"course_name": "MCP"}
        tool_use_response_2.content = [tool_block_2]

        # Third response: final text answer
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Based on both searches, here is the answer."
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [
            tool_use_response_1,
            tool_use_response_2,
            final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        result = generator.generate_response(
            query="Tell me about MCP and list all lessons",
            tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
            tool_manager=mock_tool_manager,
        )

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify 3 API calls were made (2 with tools, 1 final)
        assert mock_client.messages.create.call_count == 3

        # Verify final response
        assert result == "Based on both searches, here is the answer."

    @patch("ai_generator.anthropic.Anthropic")
    def test_stops_after_max_rounds(self, mock_anthropic_class):
        """Verify that max 2 rounds of tool use is enforced"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create a tool use response generator
        def create_tool_response(tool_id):
            response = Mock()
            response.stop_reason = "tool_use"
            tool_block = Mock()
            tool_block.type = "tool_use"
            tool_block.name = "search_course_content"
            tool_block.id = tool_id
            tool_block.input = {"query": "test"}
            response.content = [tool_block]
            return response

        # Final response (no more tools)
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Final answer after max rounds"
        final_response.content = [final_content]

        # 3 tool use responses + final (but we should stop at 2 rounds + final)
        mock_client.messages.create.side_effect = [
            create_tool_response("tool_1"),
            create_tool_response("tool_2"),
            final_response,  # This should be the final call WITHOUT tools
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        result = generator.generate_response(
            query="Test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Verify exactly 2 tool executions (max rounds)
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify exactly 3 API calls (2 with tools + 1 final without tools)
        assert mock_client.messages.create.call_count == 3

        # Verify the final call does NOT have tools
        final_call = mock_client.messages.create.call_args_list[2]
        assert "tools" not in final_call[1]

    @patch("ai_generator.anthropic.Anthropic")
    def test_stops_when_no_tool_use(self, mock_anthropic_class):
        """Test that loop stops early if Claude gives text response instead of tool use"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: direct text (no tool use)
        text_response = Mock()
        text_response.stop_reason = "end_turn"
        text_content = Mock()
        text_content.text = "I can answer without tools"
        text_response.content = [text_content]

        mock_client.messages.create.return_value = text_response

        mock_tool_manager = Mock()

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        result = generator.generate_response(
            query="Hello",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Verify no tools were executed
        mock_tool_manager.execute_tool.assert_not_called()

        # Verify only 1 API call
        assert mock_client.messages.create.call_count == 1

        # Verify response
        assert result == "I can answer without tools"

    @patch("ai_generator.anthropic.Anthropic")
    def test_preserves_context_between_rounds(self, mock_anthropic_class):
        """Test that messages accumulate correctly between rounds"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Track message counts at each call
        message_counts = []

        def capture_messages(**kwargs):
            message_counts.append(len(kwargs["messages"]))
            return responses.pop(0)

        # First tool use response
        tool_use_response_1 = Mock()
        tool_use_response_1.stop_reason = "tool_use"
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "first search"}
        tool_use_response_1.content = [tool_block_1]

        # Second tool use response
        tool_use_response_2 = Mock()
        tool_use_response_2.stop_reason = "tool_use"
        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "search_course_content"
        tool_block_2.id = "tool_2"
        tool_block_2.input = {"query": "second search"}
        tool_use_response_2.content = [tool_block_2]

        # Final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Final"
        final_response.content = [final_content]

        responses = [tool_use_response_1, tool_use_response_2, final_response]
        mock_client.messages.create.side_effect = capture_messages

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["First result", "Second result"]

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        generator.generate_response(
            query="Compare two topics",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Verify message accumulation:
        # Call 1: 1 message (user query)
        # Call 2: 3 messages (user query, assistant tool use 1, tool result 1)
        # Call 3: 5 messages (user query, assistant 1, result 1, assistant 2, result 2)
        assert message_counts == [1, 3, 5]

        # Verify 3 API calls were made
        assert mock_client.messages.create.call_count == 3

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_error_triggers_final_call(self, mock_anthropic_class):
        """Test that tool execution error terminates loop gracefully"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Tool use response
        tool_use_response = Mock()
        tool_use_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_1"
        tool_block.input = {"query": "test"}
        tool_use_response.content = [tool_block]

        # Final response after error
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Sorry, there was an issue with the search."
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Tool manager that raises an exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception(
            "Database connection failed"
        )

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        result = generator.generate_response(
            query="Search for something",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Should still return a result (graceful degradation)
        assert result == "Sorry, there was an issue with the search."

        # Verify only 2 API calls (initial + final after error)
        assert mock_client.messages.create.call_count == 2

        # Verify the final call does NOT have tools
        final_call = mock_client.messages.create.call_args_list[1]
        assert "tools" not in final_call[1]

    @patch("ai_generator.anthropic.Anthropic")
    def test_second_round_includes_tools(self, mock_anthropic_class):
        """Test that tools are available on follow-up calls within the loop"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First tool use
        tool_use_response_1 = Mock()
        tool_use_response_1.stop_reason = "tool_use"
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "first"}
        tool_use_response_1.content = [tool_block_1]

        # Final text response (second round)
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Final answer"
        final_response.content = [final_content]

        mock_client.messages.create.side_effect = [tool_use_response_1, final_response]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content"}, {"name": "get_course_outline"}]

        generator.generate_response(
            query="Test", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify second call still has tools (enabling sequential tool use)
        second_call = mock_client.messages.create.call_args_list[1]
        assert "tools" in second_call[1]
        assert second_call[1]["tools"] == tools
