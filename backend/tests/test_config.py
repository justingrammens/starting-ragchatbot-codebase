"""
Tests for configuration validation.

This test module validates that all configuration settings are properly set
and have valid values that won't cause system failures.
"""
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import config


class TestConfigValidation:
    """Test suite for configuration validation"""

    def test_anthropic_api_key_exists(self):
        """Test that ANTHROPIC_API_KEY is set"""
        assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY must be set"
        assert len(config.ANTHROPIC_API_KEY) > 0, "ANTHROPIC_API_KEY cannot be empty"

    def test_anthropic_model_is_valid(self):
        """Test that ANTHROPIC_MODEL is set to a valid model"""
        assert config.ANTHROPIC_MODEL, "ANTHROPIC_MODEL must be set"
        assert "claude" in config.ANTHROPIC_MODEL.lower(), "ANTHROPIC_MODEL should be a Claude model"

    def test_max_results_is_positive(self):
        """
        CRITICAL TEST: Validates that MAX_RESULTS > 0

        This test will FAIL if MAX_RESULTS is set to 0, which causes
        all search queries to return 0 results and makes the system unusable.
        """
        assert config.MAX_RESULTS > 0, \
            f"MAX_RESULTS must be greater than 0 (current value: {config.MAX_RESULTS}). " \
            f"A value of 0 will cause all searches to return no results!"

    def test_max_results_is_reasonable(self):
        """Test that MAX_RESULTS is within a reasonable range"""
        assert 1 <= config.MAX_RESULTS <= 20, \
            f"MAX_RESULTS should be between 1 and 20 (current: {config.MAX_RESULTS})"

    def test_chunk_size_is_positive(self):
        """Test that CHUNK_SIZE is a positive value"""
        assert config.CHUNK_SIZE > 0, "CHUNK_SIZE must be greater than 0"
        assert config.CHUNK_SIZE >= 100, "CHUNK_SIZE should be at least 100 characters"

    def test_chunk_overlap_is_valid(self):
        """Test that CHUNK_OVERLAP is valid"""
        assert config.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP cannot be negative"
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE, \
            "CHUNK_OVERLAP must be less than CHUNK_SIZE"

    def test_max_history_is_valid(self):
        """Test that MAX_HISTORY is a valid value"""
        assert config.MAX_HISTORY >= 0, "MAX_HISTORY cannot be negative"
        assert config.MAX_HISTORY <= 10, "MAX_HISTORY should not exceed 10 to avoid token bloat"

    def test_chroma_path_is_set(self):
        """Test that CHROMA_PATH is configured"""
        assert config.CHROMA_PATH, "CHROMA_PATH must be set"
        assert len(config.CHROMA_PATH) > 0, "CHROMA_PATH cannot be empty"

    def test_embedding_model_is_set(self):
        """Test that EMBEDDING_MODEL is configured"""
        assert config.EMBEDDING_MODEL, "EMBEDDING_MODEL must be set"
        assert len(config.EMBEDDING_MODEL) > 0, "EMBEDDING_MODEL cannot be empty"


class TestConfigIntegration:
    """Test configuration values work together correctly"""

    def test_all_required_settings_present(self):
        """Test that all required configuration attributes exist"""
        required_attrs = [
            'ANTHROPIC_API_KEY',
            'ANTHROPIC_MODEL',
            'EMBEDDING_MODEL',
            'CHUNK_SIZE',
            'CHUNK_OVERLAP',
            'MAX_RESULTS',
            'MAX_HISTORY',
            'CHROMA_PATH'
        ]

        for attr in required_attrs:
            assert hasattr(config, attr), f"Config missing required attribute: {attr}"

    def test_config_values_types(self):
        """Test that configuration values have correct types"""
        assert isinstance(config.ANTHROPIC_API_KEY, str), "ANTHROPIC_API_KEY must be a string"
        assert isinstance(config.ANTHROPIC_MODEL, str), "ANTHROPIC_MODEL must be a string"
        assert isinstance(config.EMBEDDING_MODEL, str), "EMBEDDING_MODEL must be a string"
        assert isinstance(config.CHUNK_SIZE, int), "CHUNK_SIZE must be an integer"
        assert isinstance(config.CHUNK_OVERLAP, int), "CHUNK_OVERLAP must be an integer"
        assert isinstance(config.MAX_RESULTS, int), "MAX_RESULTS must be an integer"
        assert isinstance(config.MAX_HISTORY, int), "MAX_HISTORY must be an integer"
        assert isinstance(config.CHROMA_PATH, str), "CHROMA_PATH must be a string"


def test_config_display():
    """Display current configuration values for debugging"""
    print("\n" + "="*60)
    print("CURRENT CONFIGURATION VALUES")
    print("="*60)
    print(f"ANTHROPIC_MODEL: {config.ANTHROPIC_MODEL}")
    print(f"EMBEDDING_MODEL: {config.EMBEDDING_MODEL}")
    print(f"CHUNK_SIZE: {config.CHUNK_SIZE}")
    print(f"CHUNK_OVERLAP: {config.CHUNK_OVERLAP}")
    print(f"MAX_RESULTS: {config.MAX_RESULTS} {'⚠️  WARNING: Set to 0!' if config.MAX_RESULTS == 0 else '✓'}")
    print(f"MAX_HISTORY: {config.MAX_HISTORY}")
    print(f"CHROMA_PATH: {config.CHROMA_PATH}")
    print(f"API_KEY_SET: {'Yes' if config.ANTHROPIC_API_KEY else 'No'}")
    print("="*60 + "\n")
