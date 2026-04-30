"""Tests for Long Context Adaptation Strategies."""

import json
import os
import subprocess
import sys
from typing import Dict, Any

import pytest

from context_adapter.tokenizer import PseudoTokenizer
from context_adapter.strategies import (
    sliding_window_truncation,
    segment_summary_concatenation,
    deterministic_retrieval_chunk_assembly,
)


class TestSlidingWindowTruncation:
    """Tests for Sliding Window / Tail-First Truncation strategy."""
    
    STRATEGY_NAME = "sliding_window_truncation"
    
    def test_truncation_necessary(self, long_text: str, small_limit: int):
        """Test case: 超长必截断 - text significantly exceeds limit."""
        tokenizer = PseudoTokenizer()
        tokens_in = tokenizer.count_tokens(long_text)
        assert tokens_in > small_limit, "Test text should exceed limit"
        
        result = sliding_window_truncation(long_text, small_limit, dry_run=False)
        
        assert result["strategy"] == self.STRATEGY_NAME
        assert result["truncated"] is True
        assert result["tokens_in"] == tokens_in
        assert result["tokens_out"] == small_limit
        assert result["tokens_out"] <= small_limit
        assert result["discarded_blocks"] >= 1
        assert result["output_text"] is not None
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0
        
        output_tokens = tokenizer.count_tokens(result["output_text"])
        assert output_tokens <= small_limit
    
    def test_boundary_no_truncation(self, boundary_text: str, context_limit: int):
        """Test case: 贴上限边界 - text is exactly at or below limit."""
        tokenizer = PseudoTokenizer()
        tokens_in = tokenizer.count_tokens(boundary_text)
        assert tokens_in == context_limit, "Test text should be at boundary"
        
        result = sliding_window_truncation(boundary_text, context_limit, dry_run=False)
        
        assert result["strategy"] == self.STRATEGY_NAME
        assert result["truncated"] is False
        assert result["tokens_in"] == tokens_in
        assert result["tokens_out"] == tokens_in
        assert result["discarded_blocks"] == 0
        assert result["output_text"] == boundary_text
        assert "no truncation" in result["explanation"].lower()
    
    def test_dry_run_mode(self, long_text: str, small_limit: int):
        """Test dry_run mode: output_text should be null."""
        result = sliding_window_truncation(long_text, small_limit, dry_run=True)
        
        assert result["strategy"] == self.STRATEGY_NAME
        assert result["truncated"] is True
        assert result["tokens_in"] > 0
        assert result["tokens_out"] == small_limit
        assert result["discarded_blocks"] >= 1
        assert result["output_text"] is None
        assert isinstance(result["explanation"], str)


class TestSegmentSummaryConcatenation:
    """Tests for Segment Summary Concatenation strategy."""
    
    STRATEGY_NAME = "segment_summary_concatenation"
    
    def test_truncation_necessary(self, long_text: str, small_limit: int):
        """Test case: 超长必截断 - text significantly exceeds limit."""
        tokenizer = PseudoTokenizer()
        tokens_in = tokenizer.count_tokens(long_text)
        assert tokens_in > small_limit, "Test text should exceed limit"
        
        result = segment_summary_concatenation(long_text, small_limit, dry_run=False)
        
        assert result["strategy"] == self.STRATEGY_NAME
        assert result["truncated"] is True
        assert result["tokens_in"] == tokens_in
        assert result["tokens_out"] <= small_limit
        assert result["tokens_out"] > 0
        assert result["discarded_blocks"] >= 0
        assert result["output_text"] is not None
        assert isinstance(result["explanation"], str)
        
        output_tokens = tokenizer.count_tokens(result["output_text"])
        assert output_tokens <= small_limit
    
    def test_boundary_no_truncation(self, boundary_text: str, context_limit: int):
        """Test case: 贴上限边界 - text is exactly at or below limit."""
        tokenizer = PseudoTokenizer()
        tokens_in = tokenizer.count_tokens(boundary_text)
        assert tokens_in == context_limit, "Test text should be at boundary"
        
        result = segment_summary_concatenation(boundary_text, context_limit, dry_run=False)
        
        assert result["strategy"] == self.STRATEGY_NAME
        assert result["truncated"] is False
        assert result["tokens_in"] == tokens_in
        assert result["tokens_out"] == tokens_in
        assert result["discarded_blocks"] == 0
        assert result["output_text"] == boundary_text
        assert "no summarization" in result["explanation"].lower()
    
    def test_dry_run_mode(self, long_text: str, small_limit: int):
        """Test dry_run mode: output_text should be null."""
        result = segment_summary_concatenation(long_text, small_limit, dry_run=True)
        
        assert result["strategy"] == self.STRATEGY_NAME
        assert result["truncated"] is True
        assert result["tokens_in"] > 0
        assert result["tokens_out"] <= small_limit
        assert result["output_text"] is None


class TestDeterministicRetrievalChunkAssembly:
    """Tests for Deterministic Retrieval-based Chunk Assembly strategy."""
    
    STRATEGY_NAME = "deterministic_retrieval_chunk_assembly"
    
    def test_truncation_necessary(self, long_text: str, small_limit: int):
        """Test case: 超长必截断 - text significantly exceeds limit."""
        tokenizer = PseudoTokenizer()
        tokens_in = tokenizer.count_tokens(long_text)
        assert tokens_in > small_limit, "Test text should exceed limit"
        
        result = deterministic_retrieval_chunk_assembly(
            long_text, small_limit, dry_run=False, seed=42
        )
        
        assert result["strategy"] == self.STRATEGY_NAME
        assert result["truncated"] is True
        assert result["tokens_in"] == tokens_in
        assert result["tokens_out"] <= small_limit
        assert result["tokens_out"] > 0
        assert result["discarded_blocks"] >= 1
        assert result["output_text"] is not None
        assert isinstance(result["explanation"], str)
        assert "seed" in result["explanation"].lower()
        
        output_tokens = tokenizer.count_tokens(result["output_text"])
        assert output_tokens <= small_limit
    
    def test_boundary_no_truncation(self, boundary_text: str, context_limit: int):
        """Test case: 贴上限边界 - text is exactly at or below limit."""
        tokenizer = PseudoTokenizer()
        tokens_in = tokenizer.count_tokens(boundary_text)
        assert tokens_in == context_limit, "Test text should be at boundary"
        
        result = deterministic_retrieval_chunk_assembly(
            boundary_text, context_limit, dry_run=False, seed=42
        )
        
        assert result["strategy"] == self.STRATEGY_NAME
        assert result["truncated"] is False
        assert result["tokens_in"] == tokens_in
        assert result["tokens_out"] == tokens_in
        assert result["discarded_blocks"] == 0
        assert result["output_text"] == boundary_text
        assert "no assembly" in result["explanation"].lower()
    
    def test_dry_run_mode(self, long_text: str, small_limit: int):
        """Test dry_run mode: output_text should be null."""
        result = deterministic_retrieval_chunk_assembly(
            long_text, small_limit, dry_run=True, seed=42
        )
        
        assert result["strategy"] == self.STRATEGY_NAME
        assert result["truncated"] is True
        assert result["tokens_in"] > 0
        assert result["tokens_out"] <= small_limit
        assert result["output_text"] is None
    
    def test_determinism_with_seed(self, long_text: str, small_limit: int):
        """Test that same seed produces same results."""
        result1 = deterministic_retrieval_chunk_assembly(
            long_text, small_limit, dry_run=False, seed=7
        )
        result2 = deterministic_retrieval_chunk_assembly(
            long_text, small_limit, dry_run=False, seed=7
        )
        
        assert result1["tokens_out"] == result2["tokens_out"]
        assert result1["discarded_blocks"] == result2["discarded_blocks"]
        assert result1["output_text"] == result2["output_text"]


class TestCLIFunctionality:
    """Tests for CLI functionality."""
    
    def run_cli(self, args: list) -> subprocess.CompletedProcess:
        """Helper to run CLI command."""
        cmd = [sys.executable, "-m", "context_adapter"] + args
        return subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    
    def test_cli_help(self):
        """Test CLI help output."""
        result = self.run_cli(["--help"])
        assert result.returncode == 0
        assert "Long Context Adaptation Tool" in result.stdout
    
    def test_cli_show_rules(self):
        """Test --show-rules output."""
        result = self.run_cli(["--show-rules"])
        assert result.returncode == 0
        
        rules = json.loads(result.stdout)
        assert isinstance(rules, dict)
        assert len(rules) > 0
    
    def test_cli_missing_file(self):
        """Test error when file doesn't exist."""
        result = self.run_cli(["/nonexistent/file.txt", "--strategy", "sliding_window"])
        assert result.returncode != 0
        
        output = json.loads(result.stdout)
        assert "error" in output
        assert "not found" in output["error"].lower()
    
    def test_cli_invalid_strategy(self):
        """Test error when strategy is invalid."""
        pass


class TestAcceptanceCriteria:
    """Acceptance tests for the project."""
    
    def run_cli(self, args: list, env: dict = None) -> subprocess.CompletedProcess:
        """Helper to run CLI command with optional env vars."""
        cmd = [sys.executable, "-m", "context_adapter"] + args
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        return subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=full_env
        )
    
    @pytest.mark.parametrize("strategy", [
        "sliding_window",
        "segment_summary",
        "deterministic_retrieval"
    ])
    def test_very_long_text_with_seed_7(
        self,
        temp_long_text_file: str,
        strategy: str,
        context_limit: int
    ):
        """
        Acceptance test: For >=5000 chars fixture, with CONTEXT_ADAPTER_SEED=7,
        all three strategies should output tokens <= limit, and dry-run estimate
        should match actual run.
        """
        env = {"CONTEXT_ADAPTER_SEED": "7"}
        
        dry_result = self.run_cli([
            temp_long_text_file,
            "--strategy", strategy,
            "--context-limit", str(context_limit),
            "--dry-run"
        ], env=env)
        
        assert dry_result.returncode == 0, f"Dry run failed: {dry_result.stderr}"
        dry_json = json.loads(dry_result.stdout)
        
        assert dry_json["truncated"] is True
        assert dry_json["tokens_out"] <= context_limit
        assert dry_json["output_text"] is None
        
        actual_result = self.run_cli([
            temp_long_text_file,
            "--strategy", strategy,
            "--context-limit", str(context_limit)
        ], env=env)
        
        assert actual_result.returncode == 0, f"Actual run failed: {actual_result.stderr}"
        actual_json = json.loads(actual_result.stdout)
        
        assert actual_json["tokens_out"] == dry_json["tokens_out"]
        assert actual_json["tokens_in"] == dry_json["tokens_in"]
        assert actual_json["truncated"] == dry_json["truncated"]
        assert actual_json["discarded_blocks"] == dry_json["discarded_blocks"]
        assert actual_json["strategy"] == dry_json["strategy"]
        
        tokenizer = PseudoTokenizer()
        if actual_json["output_text"]:
            actual_tokens = tokenizer.count_tokens(actual_json["output_text"])
            assert actual_tokens <= context_limit
    
    def test_text_length_requirement(self, very_long_text: str):
        """Verify test fixture is >= 5000 characters."""
        assert len(very_long_text) >= 5000, f"Test text should be >= 5000 chars, got {len(very_long_text)}"
