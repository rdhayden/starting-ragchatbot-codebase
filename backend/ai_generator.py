import anthropic
from typing import List, Optional, Dict, Any, Tuple


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content**: Search course materials for specific content or educational materials
2. **get_course_outline**: Get course structure including course title, course link, and complete lesson list (lesson number and title for each lesson)

Tool Selection:
- **Course outline questions** (syllabus, structure, lesson list, what's covered): Use `get_course_outline`
- **Content questions** (specific topics, concepts, details): Use `search_course_content`
- **General knowledge questions**: Answer using existing knowledge without tools
- **Up to 2 tool calls per query** - use multiple calls when needed for comparison or multi-part questions
- After reviewing results, you may call another tool if more information would help

Response Protocol for Outline Queries:
- When returning course outlines, include: course title, course link, and all lessons with their numbers and titles
- Present lesson lists in a clear, readable format

Response Protocol for All Queries:
- Provide direct answers only — no reasoning process, search explanations, or question-type analysis
- Do not mention "based on the search results" or "based on the outline"
- If a tool yields no results, state this clearly without offering alternatives

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 sequential tool calling rounds.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize messages and round counter
        messages = [{"role": "user", "content": query}]
        round_count = 0
        max_rounds = 2

        while round_count < max_rounds:
            # Prepare API call parameters with tools
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # Exit if no tool use requested or no tool manager
            if response.stop_reason != "tool_use" or not tool_manager:
                return self._extract_text(response)

            # Execute tools
            tool_results, had_error = self._execute_tools(response, tool_manager)

            # If error, break to final call without tools
            if had_error:
                break

            # Append assistant response + tool results to messages
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            round_count += 1

        # After loop: final call WITHOUT tools
        return self._make_final_call(messages, system_content)

    def _execute_tools(self, response, tool_manager) -> Tuple[List[Dict], bool]:
        """
        Execute all tool calls from a response.

        Args:
            response: The API response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (tool_results list, had_error boolean)
        """
        tool_results = []
        had_error = False

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )
                except Exception as e:
                    tool_result = f"Error executing tool: {str(e)}"
                    had_error = True

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        return tool_results, had_error

    def _make_final_call(self, messages: List[Dict], system_content: str) -> str:
        """
        Make final API call without tools.

        Args:
            messages: Conversation messages
            system_content: System prompt content

        Returns:
            Response text
        """
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        final_response = self.client.messages.create(**final_params)
        return self._extract_text(final_response)

    def _extract_text(self, response) -> str:
        """
        Extract text from response content blocks.

        Args:
            response: API response object

        Returns:
            Text content from the response
        """
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""
