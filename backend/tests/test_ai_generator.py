"""
Tests for AIGenerator.

Tests validate:
1. Tool calling behavior (two-phase API calls)
2. Response generation without tools
3. Tool execution handling
4. System prompt construction
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_generator import AIGenerator


class TestAIGeneratorInitialization:
    """Test AIGenerator initialization"""

    def test_initialization_with_valid_params(self):
        """Test AIGenerator initializes correctly"""
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        assert generator.model == "claude-sonnet-4-20250514"
        assert generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    def test_system_prompt_exists(self):
        """Test that SYSTEM_PROMPT is defined"""
        assert hasattr(AIGenerator, 'SYSTEM_PROMPT')
        assert len(AIGenerator.SYSTEM_PROMPT) > 0

    def test_system_prompt_mentions_tools(self):
        """Test that system prompt describes the tools"""
        prompt = AIGenerator.SYSTEM_PROMPT

        # Should mention both tools
        assert "search_course_content" in prompt or "course content" in prompt.lower()
        assert "get_course_outline" in prompt or "course outline" in prompt.lower()


class TestResponseGenerationWithoutTools:
    """Test response generation without tool use"""

    @patch('anthropic.Anthropic')
    def test_generate_simple_response(self, mock_anthropic_class):
        """Test generating a response without tool use"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="This is a simple answer.")]
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response

        # Test
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.client = mock_client

        result = generator.generate_response("What is AI?")

        assert result == "This is a simple answer."
        mock_client.messages.create.assert_called_once()

    @patch('anthropic.Anthropic')
    def test_generate_response_with_history(self, mock_anthropic_class):
        """Test that conversation history is included in system prompt"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Answer with context.")]
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response

        # Test
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.client = mock_client

        history = "User: Previous question\nAssistant: Previous answer"
        result = generator.generate_response("Follow-up question", conversation_history=history)

        # Check that system prompt includes history
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous question" in system_content


class TestToolCallingBehavior:
    """Test AIGenerator's tool calling functionality"""

    @patch('anthropic.Anthropic')
    def test_tool_use_triggers_execution(self, mock_anthropic_class):
        """Test that tool_use stop_reason triggers tool execution"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_123"
        tool_use_block.input = {"query": "What is RAG?"}

        first_response = Mock()
        first_response.content = [tool_use_block]
        first_response.stop_reason = "tool_use"

        # Second response: final answer
        second_response = Mock()
        second_response.content = [Mock(text="RAG is Retrieval-Augmented Generation.")]
        second_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [first_response, second_response]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results about RAG..."

        # Test
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.client = mock_client

        tools = [{"name": "search_course_content", "description": "Search tool"}]
        result = generator.generate_response(
            "What is RAG?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Should make two API calls
        assert mock_client.messages.create.call_count == 2

        # Tool should be executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="What is RAG?"
        )

        # Should return final response
        assert result == "RAG is Retrieval-Augmented Generation."

    @patch('anthropic.Anthropic')
    def test_tools_passed_to_api(self, mock_anthropic_class):
        """Test that tools are passed to the API correctly"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Answer without tools.")]
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response

        # Test
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.client = mock_client

        tools = [
            {
                "name": "search_course_content",
                "description": "Search tool",
                "input_schema": {"type": "object"}
            }
        ]

        result = generator.generate_response("Test query", tools=tools)

        # Check that tools were passed
        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tools"] == tools
        assert "tool_choice" in call_args[1]
        assert call_args[1]["tool_choice"]["type"] == "auto"

    @patch('anthropic.Anthropic')
    def test_no_tools_when_not_provided(self, mock_anthropic_class):
        """Test that tools are not passed if not provided"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Answer.")]
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response

        # Test
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.client = mock_client

        result = generator.generate_response("Test query")

        # Check that tools were NOT passed
        call_args = mock_client.messages.create.call_args
        assert "tools" not in call_args[1]


class TestToolExecutionFlow:
    """Test the tool execution flow in detail"""

    @patch('anthropic.Anthropic')
    def test_tool_results_passed_correctly(self, mock_anthropic_class):
        """Test that tool results are formatted and passed correctly"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_abc"
        tool_use_block.input = {"query": "test"}

        first_response = Mock()
        first_response.content = [tool_use_block]
        first_response.stop_reason = "tool_use"

        # Second response
        second_response = Mock()
        second_response.content = [Mock(text="Final answer")]
        second_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [first_response, second_response]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"

        # Test
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.client = mock_client

        tools = [{"name": "search_course_content"}]
        result = generator.generate_response(
            "Test",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Check second API call has tool results
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]

        # Should have 3 messages: user, assistant (tool use), user (tool result)
        assert len(messages) == 3

        # Last message should contain tool result
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        assert isinstance(tool_result_msg["content"], list)
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "tool_abc"
        assert tool_result_msg["content"][0]["content"] == "Tool execution result"

    @patch('anthropic.Anthropic')
    def test_multiple_tool_calls(self, mock_anthropic_class):
        """Test handling multiple tool calls in one response"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: multiple tool uses
        tool_use_1 = Mock()
        tool_use_1.type = "tool_use"
        tool_use_1.name = "search_course_content"
        tool_use_1.id = "tool_1"
        tool_use_1.input = {"query": "test1"}

        tool_use_2 = Mock()
        tool_use_2.type = "tool_use"
        tool_use_2.name = "get_course_outline"
        tool_use_2.id = "tool_2"
        tool_use_2.input = {"course_title": "Course A"}

        first_response = Mock()
        first_response.content = [tool_use_1, tool_use_2]
        first_response.stop_reason = "tool_use"

        # Second response
        second_response = Mock()
        second_response.content = [Mock(text="Final answer")]
        second_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [first_response, second_response]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        # Test
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.client = mock_client

        tools = [{"name": "search_course_content"}, {"name": "get_course_outline"}]
        result = generator.generate_response(
            "Test",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Both tools should be executed
        assert mock_tool_manager.execute_tool.call_count == 2


class TestErrorHandling:
    """Test error handling in AIGenerator"""

    @patch('anthropic.Anthropic')
    def test_handles_api_errors_gracefully(self, mock_anthropic_class):
        """Test that API errors are propagated appropriately"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_client.messages.create.side_effect = Exception("API Error")

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.client = mock_client

        # Should raise the exception
        with pytest.raises(Exception) as exc_info:
            generator.generate_response("Test query")

        assert "API Error" in str(exc_info.value)
