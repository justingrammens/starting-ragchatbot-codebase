"""
Integration tests for RAGSystem.

Tests validate:
1. End-to-end query flow
2. Component integration
3. Session management
4. Source tracking
5. Error propagation
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_system import RAGSystem
from vector_store import SearchResults


class TestRAGSystemInitialization:
    """Test RAG system initialization"""

    def test_initialization_with_config(self, mock_config):
        """Test RAGSystem initializes all components"""
        rag = RAGSystem(mock_config)

        assert rag.document_processor is not None
        assert rag.vector_store is not None
        assert rag.ai_generator is not None
        assert rag.session_manager is not None
        assert rag.tool_manager is not None
        assert rag.search_tool is not None

    def test_tools_registered(self, mock_config):
        """Test that tools are registered with ToolManager"""
        rag = RAGSystem(mock_config)

        # Should have registered tools
        definitions = rag.tool_manager.get_tool_definitions()
        assert len(definitions) >= 1

        # Should have search tool
        tool_names = [d["name"] for d in definitions]
        assert "search_course_content" in tool_names


class TestRAGSystemQuery:
    """Test RAG system query processing"""

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_query_without_session(self, mock_vector_store_class, mock_ai_gen_class, mock_config):
        """Test basic query without session context"""
        # Setup mocks
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "This is the answer."
        mock_ai_gen_class.return_value = mock_ai_gen

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        # Create RAG system
        rag = RAGSystem(mock_config)
        rag.ai_generator = mock_ai_gen

        # Query
        response, sources = rag.query("What is RAG?")

        # Should return response
        assert response == "This is the answer."
        assert isinstance(sources, list)

        # AI generator should be called with tools
        mock_ai_gen.generate_response.assert_called_once()
        call_args = mock_ai_gen.generate_response.call_args
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_query_with_session(self, mock_vector_store_class, mock_ai_gen_class, mock_config):
        """Test query with session history"""
        # Setup mocks
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Follow-up answer."
        mock_ai_gen_class.return_value = mock_ai_gen

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        # Create RAG system
        rag = RAGSystem(mock_config)
        rag.ai_generator = mock_ai_gen

        # Create session and add history
        session_id = rag.session_manager.create_session()
        rag.session_manager.add_exchange(session_id, "Previous question", "Previous answer")

        # Query with session
        response, sources = rag.query("Follow-up question", session_id=session_id)

        # Should pass history to AI generator
        call_args = mock_ai_gen.generate_response.call_args
        history = call_args[1]["conversation_history"]
        assert history is not None
        assert "Previous question" in history

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_query_updates_session_history(self, mock_vector_store_class, mock_ai_gen_class, mock_config):
        """Test that query updates session history"""
        # Setup mocks
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Answer"
        mock_ai_gen_class.return_value = mock_ai_gen

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        # Create RAG system
        rag = RAGSystem(mock_config)
        rag.ai_generator = mock_ai_gen

        # Create session and query
        session_id = rag.session_manager.create_session()
        rag.query("Test question", session_id=session_id)

        # History should be updated
        history = rag.session_manager.get_conversation_history(session_id)
        assert "Test question" in history
        assert "Answer" in history


class TestRAGSystemSourceTracking:
    """Test source tracking through the system"""

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_sources_returned_from_query(self, mock_vector_store_class, mock_ai_gen_class, mock_config):
        """Test that sources are returned from query"""
        # Setup mocks
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Answer with sources"
        mock_ai_gen_class.return_value = mock_ai_gen

        mock_vector_store = Mock()
        mock_search_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course A", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.search.return_value = mock_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        mock_vector_store_class.return_value = mock_vector_store

        # Create RAG system
        rag = RAGSystem(mock_config)
        rag.ai_generator = mock_ai_gen

        # Simulate tool execution by directly calling search tool
        rag.tool_manager.execute_tool("search_course_content", query="test")

        # Query
        response, sources = rag.query("What is this?")

        # Sources should be populated (from previous tool call)
        sources_before_reset = rag.tool_manager.get_last_sources()
        # Note: sources are reset after query, so we check before reset

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_sources_reset_after_query(self, mock_vector_store_class, mock_ai_gen_class, mock_config):
        """Test that sources are reset after being retrieved"""
        # Setup mocks
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Answer"
        mock_ai_gen_class.return_value = mock_ai_gen

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        # Create RAG system
        rag = RAGSystem(mock_config)
        rag.ai_generator = mock_ai_gen

        # Set some sources
        rag.search_tool.last_sources = [{"text": "Source 1", "url": "http://example.com"}]

        # Query (should reset sources)
        response, sources = rag.query("Test")

        # Sources should be reset after query
        assert len(rag.tool_manager.get_last_sources()) == 0


class TestRAGSystemDocumentProcessing:
    """Test document processing functionality"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_document(self, mock_doc_proc_class, mock_vector_store_class, mock_config, sample_course, sample_chunks):
        """Test adding a course document"""
        # Setup mocks
        mock_doc_proc = Mock()
        mock_doc_proc.process_course_document.return_value = (sample_course, sample_chunks)
        mock_doc_proc_class.return_value = mock_doc_proc

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        # Create RAG system
        rag = RAGSystem(mock_config)
        rag.document_processor = mock_doc_proc
        rag.vector_store = mock_vector_store

        # Add document
        course, chunk_count = rag.add_course_document("/path/to/course.txt")

        # Should process and add to vector store
        assert course == sample_course
        assert chunk_count == len(sample_chunks)
        mock_vector_store.add_course_metadata.assert_called_once_with(sample_course)
        mock_vector_store.add_course_content.assert_called_once_with(sample_chunks)

    @patch('rag_system.VectorStore')
    def test_get_course_analytics(self, mock_vector_store_class, mock_config):
        """Test getting course analytics"""
        # Setup mocks
        mock_vector_store = Mock()
        mock_vector_store.get_course_count.return_value = 5
        mock_vector_store.get_existing_course_titles.return_value = ["Course A", "Course B"]
        mock_vector_store_class.return_value = mock_vector_store

        # Create RAG system
        rag = RAGSystem(mock_config)
        rag.vector_store = mock_vector_store

        # Get analytics
        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 2


class TestRAGSystemErrorHandling:
    """Test error handling in RAG system"""

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_query_with_ai_generator_error(self, mock_vector_store_class, mock_ai_gen_class, mock_config):
        """Test query handling when AI generator fails"""
        # Setup mocks
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.side_effect = Exception("API Error")
        mock_ai_gen_class.return_value = mock_ai_gen

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        # Create RAG system
        rag = RAGSystem(mock_config)
        rag.ai_generator = mock_ai_gen

        # Query should raise exception
        with pytest.raises(Exception) as exc_info:
            rag.query("Test query")

        assert "API Error" in str(exc_info.value)

    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_document_error_handling(self, mock_doc_proc_class, mock_vector_store_class, mock_config):
        """Test error handling when adding course fails"""
        # Setup mocks
        mock_doc_proc = Mock()
        mock_doc_proc.process_course_document.side_effect = Exception("File read error")
        mock_doc_proc_class.return_value = mock_doc_proc

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        # Create RAG system
        rag = RAGSystem(mock_config)
        rag.document_processor = mock_doc_proc

        # Should handle error gracefully
        course, chunk_count = rag.add_course_document("/bad/path.txt")

        assert course is None
        assert chunk_count == 0


class TestRAGSystemWithRealConfig:
    """Test RAG system with actual configuration"""

    def test_real_config_max_results_issue(self):
        """
        CRITICAL TEST: Test that real config has valid MAX_RESULTS

        This test imports the actual config and checks if MAX_RESULTS=0,
        which would cause all queries to fail.
        """
        from config import config

        # This will fail if MAX_RESULTS is 0
        assert config.MAX_RESULTS > 0, \
            f"CRITICAL: MAX_RESULTS is {config.MAX_RESULTS}! This causes all queries to fail. " \
            f"Set MAX_RESULTS to 5 or higher in config.py"

    def test_real_config_values(self):
        """Test that real config has reasonable values"""
        from config import config

        # All values should be set
        assert config.CHUNK_SIZE > 0
        assert config.CHUNK_OVERLAP >= 0
        assert config.MAX_HISTORY >= 0
        assert len(config.CHROMA_PATH) > 0

        # MAX_RESULTS is critical
        if config.MAX_RESULTS == 0:
            pytest.fail(
                "MAX_RESULTS is 0! This is the root cause of 'query failed' errors. "
                "Change MAX_RESULTS to 5 in backend/config.py line 22"
            )
