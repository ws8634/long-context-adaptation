"""CLI for Long Context Adaptation Library."""

import argparse
import json
import sys
from typing import Dict, Any, Optional

from context_adapter.tokenizer import PseudoTokenizer
from context_adapter.strategies import (
    sliding_window_truncation,
    segment_summary_concatenation,
    deterministic_retrieval_chunk_assembly,
)


STRATEGY_MAP = {
    "sliding_window": sliding_window_truncation,
    "segment_summary": segment_summary_concatenation,
    "deterministic_retrieval": deterministic_retrieval_chunk_assembly,
}

STRATEGY_NAMES = {
    "sliding_window": "sliding_window_truncation",
    "segment_summary": "segment_summary_concatenation",
    "deterministic_retrieval": "deterministic_retrieval_chunk_assembly",
}


def read_file(file_path: str) -> str:
    """Read a UTF-8 encoded file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except UnicodeDecodeError:
        raise ValueError(f"File is not valid UTF-8: {file_path}")


def process_text(
    text: str,
    strategy_name: str,
    context_limit: int,
    dry_run: bool
) -> Dict[str, Any]:
    """
    Process text using the specified strategy.
    
    Args:
        text: Input text
        strategy_name: Name of the strategy to use
        context_limit: Maximum token count
        dry_run: If True, don't produce output text
        
    Returns:
        Result dictionary from the strategy
    """
    if strategy_name not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available strategies: {', '.join(STRATEGY_MAP.keys())}"
        )
    
    strategy_func = STRATEGY_MAP[strategy_name]
    
    result = strategy_func(text, context_limit, dry_run)
    
    if result["strategy"] != STRATEGY_NAMES[strategy_name]:
        raise ValueError(f"Strategy name mismatch: expected {STRATEGY_NAMES[strategy_name]}, got {result['strategy']}")
    
    return result


def print_tokenizer_rules() -> None:
    """Print the tokenizer rules in JSON format."""
    rules = PseudoTokenizer.get_rules()
    print(json.dumps(rules, ensure_ascii=False, indent=2))


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Long Context Adaptation Tool - Process long text for LLM context windows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Strategies:
  sliding_window       Sliding Window / Tail-First Truncation
  segment_summary      Segment Summary Concatenation
  deterministic_retrieval  Deterministic Retrieval-based Chunk Assembly

Environment Variables:
  CONTEXT_ADAPTER_SEED  Seed for deterministic retrieval strategy (default: 0)
        """
    )
    
    parser.add_argument(
        "input_file",
        nargs='?',
        help="Path to input UTF-8 text file"
    )
    
    parser.add_argument(
        "--strategy", "-s",
        choices=list(STRATEGY_MAP.keys()),
        help="Strategy to use for context adaptation"
    )
    
    parser.add_argument(
        "--context-limit", "-l",
        type=int,
        default=PseudoTokenizer.DEFAULT_CONTEXT_LIMIT,
        help=f"Maximum token count (default: {PseudoTokenizer.DEFAULT_CONTEXT_LIMIT})"
    )
    
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Estimate only, output_text will be null, but all other fields populated"
    )
    
    parser.add_argument(
        "--show-rules",
        action="store_true",
        help="Show tokenizer rules and exit"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic retrieval strategy (overrides env var)"
    )
    
    args = parser.parse_args()
    
    if args.show_rules:
        print_tokenizer_rules()
        sys.exit(0)
    
    if not args.input_file:
        parser.error("The following arguments are required: input_file")
    
    if not args.strategy:
        parser.error("The following arguments are required: --strategy")
    
    if args.context_limit < 1:
        print(json.dumps({
            "error": "context_limit must be at least 1",
            "strategy": None,
            "truncated": None,
            "tokens_in": None,
            "tokens_out": None,
            "discarded_blocks": None,
            "output_text": None,
            "explanation": "Invalid context limit"
        }, ensure_ascii=False))
        sys.exit(1)
    
    try:
        text = read_file(args.input_file)
        
        original_seed_env = None
        if args.seed is not None and args.strategy == "deterministic_retrieval":
            import os
            original_seed_env = os.environ.get("CONTEXT_ADAPTER_SEED")
            os.environ["CONTEXT_ADAPTER_SEED"] = str(args.seed)
        
        result = process_text(
            text=text,
            strategy_name=args.strategy,
            context_limit=args.context_limit,
            dry_run=args.dry_run
        )
        
        if original_seed_env is not None:
            import os
            if original_seed_env is None:
                del os.environ["CONTEXT_ADAPTER_SEED"]
            else:
                os.environ["CONTEXT_ADAPTER_SEED"] = original_seed_env
        
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0)
        
    except ValueError as e:
        print(json.dumps({
            "error": str(e),
            "strategy": args.strategy,
            "truncated": None,
            "tokens_in": None,
            "tokens_out": None,
            "discarded_blocks": None,
            "output_text": None,
            "explanation": str(e)
        }, ensure_ascii=False))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "strategy": args.strategy,
            "truncated": None,
            "tokens_in": None,
            "tokens_out": None,
            "discarded_blocks": None,
            "output_text": None,
            "explanation": f"An unexpected error occurred: {str(e)}"
        }, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
