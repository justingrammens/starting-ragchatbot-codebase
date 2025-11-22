"""
Shared fixtures and test utilities for RAG system tests.
"""
import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


@pytest.fixture
def mock_config():
    """Mock configuration with safe defaults"""
    config = Mock()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5  # Should be > 0
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Introduction to RAG Systems",
        course_link="https://example.com/rag-course",
        instructor="Dr. Test",
        lessons=[
            Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"),
            Lesson(lesson_number=1, title="Vector Databases", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="LLM Integration", lesson_link="https://example.com/lesson2"),
        ]
    )


@pytest.fixture
def sample_chunks(sample_course):
    """Create sample course chunks"""
    return [
        CourseChunk(
            content="This is an introduction to RAG systems. RAG combines retrieval and generation.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Vector databases store embeddings. They enable semantic search.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="LLMs can generate responses based on retrieved context.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=2
        ),
    ]


@pytest.fixture
def mock_search_results():
    """Create mock search results from VectorStore"""
    return SearchResults(
        documents=[
            "This is an introduction to RAG systems. RAG combines retrieval and generation.",
            "Vector databases store embeddings. They enable semantic search."
        ],
        metadata=[
            {"course_title": "Introduction to RAG Systems", "lesson_number": 0, "chunk_index": 0},
            {"course_title": "Introduction to RAG Systems", "lesson_number": 1, "chunk_index": 1}
        ],
        distances=[0.15, 0.23],
        error=None
    )


@pytest.fixture
def mock_empty_search_results():
    """Create empty search results"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )


@pytest.fixture
def mock_error_search_results():
    """Create error search results"""
    return SearchResults.empty("Search error: Database connection failed")


@pytest.fixture
def mock_vector_store(mock_search_results):
    """Mock VectorStore with predefined search behavior"""
    store = Mock()
    store.search = Mock(return_value=mock_search_results)
    store.get_lesson_link = Mock(return_value="https://example.com/lesson0")
    store.get_course_outline = Mock(return_value={
        'title': 'Introduction to RAG Systems',
        'course_link': 'https://example.com/rag-course',
        'instructor': 'Dr. Test',
        'lessons': [
            {'lesson_number': 0, 'lesson_title': 'Introduction', 'lesson_link': 'https://example.com/lesson0'},
            {'lesson_number': 1, 'lesson_title': 'Vector Databases', 'lesson_link': 'https://example.com/lesson1'}
        ],
        'lesson_count': 2
    })
    return store


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic API client"""
    client = Mock()

    # Mock response without tool use
    text_response = Mock()
    text_response.content = [Mock(text="This is a sample response from Claude.")]
    text_response.stop_reason = "end_turn"

    # Mock response with tool use
    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "search_course_content"
    tool_use_block.id = "tool_12345"
    tool_use_block.input = {"query": "What is RAG?"}

    tool_response = Mock()
    tool_response.content = [tool_use_block]
    tool_response.stop_reason = "tool_use"

    # Mock final response after tool execution
    final_response = Mock()
    final_response.content = [Mock(text="RAG stands for Retrieval-Augmented Generation.")]
    final_response.stop_reason = "end_turn"

    client.messages.create = Mock(side_effect=[tool_response, final_response])

    return client


@pytest.fixture
def tool_definitions():
    """Sample tool definitions for testing"""
    return [
        {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    ]
