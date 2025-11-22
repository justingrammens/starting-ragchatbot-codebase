"""
Tests for CourseSearchTool and ToolManager.

Tests validate:
1. CourseSearchTool.execute() with various search scenarios
2. Tool definition correctness
3. Source tracking functionality
4. Error handling
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolDefinition:
    """Test tool definition structure"""

    def test_tool_definition_structure(self, mock_vector_store):
        """Test that tool definition has correct structure"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert "name" in definition
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

    def test_tool_definition_parameters(self, mock_vector_store):
        """Test that tool definition has correct parameters"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema

        props = schema["properties"]
        assert "query" in props
        assert "course_name" in props
        assert "lesson_number" in props

        # Query is required
        assert "required" in schema
        assert "query" in schema["required"]


class TestCourseSearchToolExecution:
    """Test CourseSearchTool.execute() method"""

    def test_execute_with_valid_results(self, mock_vector_store, mock_search_results):
        """Test execute returns formatted results when search succeeds"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results

        result = tool.execute(query="What is RAG?")

        # Should return formatted text
        assert isinstance(result, str)
        assert len(result) > 0

        # Should contain course context
        assert "Introduction to RAG Systems" in result

        # Should contain the actual content
        assert "RAG" in result or "retrieval" in result.lower()

    def test_execute_with_empty_results(self, mock_vector_store, mock_empty_search_results):
        """Test execute handles empty results gracefully"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_empty_search_results

        result = tool.execute(query="nonexistent topic")

        # Should return "no content found" message
        assert isinstance(result, str)
        assert "no relevant content found" in result.lower()

    def test_execute_with_error_results(self, mock_vector_store, mock_error_search_results):
        """Test execute handles errors from vector store"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_error_search_results

        result = tool.execute(query="test query")

        # Should return the error message
        assert isinstance(result, str)
        assert "error" in result.lower()

    def test_execute_with_course_filter(self, mock_vector_store, mock_search_results):
        """Test execute passes course_name filter to vector store"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results

        result = tool.execute(query="What is RAG?", course_name="Introduction to RAG")

        # Verify vector store was called with course filter
        mock_vector_store.search.assert_called_once()
        call_args = mock_vector_store.search.call_args
        assert call_args[1]["course_name"] == "Introduction to RAG"

    def test_execute_with_lesson_filter(self, mock_vector_store, mock_search_results):
        """Test execute passes lesson_number filter to vector store"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results

        result = tool.execute(query="What is RAG?", lesson_number=1)

        # Verify vector store was called with lesson filter
        mock_vector_store.search.assert_called_once()
        call_args = mock_vector_store.search.call_args
        assert call_args[1]["lesson_number"] == 1

    def test_execute_with_both_filters(self, mock_vector_store, mock_search_results):
        """Test execute passes both course and lesson filters"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results

        result = tool.execute(
            query="What is RAG?",
            course_name="Introduction to RAG",
            lesson_number=1
        )

        # Verify both filters were passed
        call_args = mock_vector_store.search.call_args
        assert call_args[1]["course_name"] == "Introduction to RAG"
        assert call_args[1]["lesson_number"] == 1


class TestCourseSearchToolSourceTracking:
    """Test source tracking functionality"""

    def test_sources_tracked_after_search(self, mock_vector_store, mock_search_results):
        """Test that sources are tracked after successful search"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results

        result = tool.execute(query="What is RAG?")

        # Sources should be populated
        assert hasattr(tool, 'last_sources')
        assert len(tool.last_sources) > 0

    def test_sources_contain_correct_info(self, mock_vector_store, mock_search_results):
        """Test that tracked sources have text and URL"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results

        result = tool.execute(query="What is RAG?")

        # Each source should have text and url fields
        for source in tool.last_sources:
            assert "text" in source
            assert "url" in source

    def test_sources_empty_for_no_results(self, mock_vector_store, mock_empty_search_results):
        """Test that sources are empty when no results found"""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_empty_search_results

        result = tool.execute(query="nonexistent")

        # Sources should be empty
        assert len(tool.last_sources) == 0


class TestCourseOutlineTool:
    """Test CourseOutlineTool"""

    def test_outline_tool_definition(self, mock_vector_store):
        """Test outline tool has correct definition"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "course_title" in definition["input_schema"]["properties"]

    def test_outline_execute_success(self, mock_vector_store):
        """Test outline tool returns formatted outline"""
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_title="Introduction to RAG")

        assert isinstance(result, str)
        assert "Course:" in result
        assert "Lessons" in result

    def test_outline_execute_not_found(self, mock_vector_store):
        """Test outline tool handles course not found"""
        tool = CourseOutlineTool(mock_vector_store)
        mock_vector_store.get_course_outline.return_value = None

        result = tool.execute(course_title="Nonexistent Course")

        assert "no course found" in result.lower()


class TestToolManager:
    """Test ToolManager functionality"""

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        # Tool should be registered
        assert "search_course_content" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        assert any(d["name"] == "search_course_content" for d in definitions)
        assert any(d["name"] == "get_course_outline" for d in definitions)

    def test_execute_tool(self, mock_vector_store, mock_search_results):
        """Test executing a tool by name"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        mock_vector_store.search.return_value = mock_search_results

        result = manager.execute_tool("search_course_content", query="What is RAG?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_nonexistent_tool(self, mock_vector_store):
        """Test executing a tool that doesn't exist"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result.lower()

    def test_get_last_sources(self, mock_vector_store, mock_search_results):
        """Test retrieving sources from last search"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        mock_vector_store.search.return_value = mock_search_results

        # Execute search
        manager.execute_tool("search_course_content", query="What is RAG?")

        # Get sources
        sources = manager.get_last_sources()
        assert len(sources) > 0

    def test_reset_sources(self, mock_vector_store, mock_search_results):
        """Test resetting sources"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        mock_vector_store.search.return_value = mock_search_results

        # Execute search and reset
        manager.execute_tool("search_course_content", query="What is RAG?")
        manager.reset_sources()

        # Sources should be empty
        sources = manager.get_last_sources()
        assert len(sources) == 0
