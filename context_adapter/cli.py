"""CLI for Long Context Adaptation Library."""

import argparse
import json
import os
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

STRATEGY_SUPPORTED_ARGS = {
    "sliding_window": ["context_limit", "dry_run"],
    "segment_summary": ["context_limit", "dry_run"],
    "deterministic_retrieval": ["context_limit", "dry_run", "seed"],
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


def build_params_summary(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a summary of effective parameters for error messages."""
    summary = {
        "input_file": args.input_file,
        "strategy": args.strategy,
        "context_limit": args.context_limit,
        "dry_run": args.dry_run,
    }
    if args.seed is not None:
        summary["seed"] = args.seed
    return summary


def print_error_json(
    error_message: str,
    strategy: Optional[str] = None,
    params_summary: Optional[Dict[str, Any]] = None,
    explanation: Optional[str] = None
) -> None:
    """Print error message as JSON with strategy and params summary."""
    output = {
        "error": error_message,
        "strategy": strategy,
        "params_summary": params_summary,
        "truncated": None,
        "tokens_in": None,
        "tokens_out": None,
        "discarded_blocks": None,
        "output_text": None,
        "explanation": explanation or error_message
    }
    print(json.dumps(output, ensure_ascii=False))


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
        params_summary = build_params_summary(args)
        print_error_json(
            error_message="context_limit must be at least 1",
            strategy=args.strategy,
            params_summary=params_summary,
            explanation="Invalid context limit"
        )
        sys.exit(1)
    
    if args.seed is not None and args.strategy != "deterministic_retrieval":
        params_summary = build_params_summary(args)
        print_error_json(
            error_message=f"--seed is not supported for strategy '{args.strategy}'. Only 'deterministic_retrieval' supports --seed.",
            strategy=args.strategy,
            params_summary=params_summary,
            explanation=f"Strategy '{STRATEGY_NAMES.get(args.strategy, args.strategy)}' does not accept seed parameter. "
                        f"Use --seed only with 'deterministic_retrieval' strategy."
        )
        sys.exit(1)
    
    original_seed_env = os.environ.get("CONTEXT_ADAPTER_SEED")
    seed_was_modified = False
    
    try:
        if args.seed is not None and args.strategy == "deterministic_retrieval":
            os.environ["CONTEXT_ADAPTER_SEED"] = str(args.seed)
            seed_was_modified = True
        
        text = read_file(args.input_file)
        
        result = process_text(
            text=text,
            strategy_name=args.strategy,
            context_limit=args.context_limit,
            dry_run=args.dry_run
        )
        
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0)
        
    except ValueError as e:
        params_summary = build_params_summary(args)
        print_error_json(
            error_message=str(e),
            strategy=args.strategy,
            params_summary=params_summary,
            explanation=str(e)
        )
        sys.exit(1)
    except Exception as e:
        params_summary = build_params_summary(args)
        print_error_json(
            error_message=f"Unexpected error: {str(e)}",
            strategy=args.strategy,
            params_summary=params_summary,
            explanation=f"An unexpected error occurred: {str(e)}"
        )
        sys.exit(1)
    finally:
        if seed_was_modified:
            if original_seed_env is None:
                if "CONTEXT_ADAPTER_SEED" in os.environ:
                    del os.environ["CONTEXT_ADAPTER_SEED"]
            else:
                os.environ["CONTEXT_ADAPTER_SEED"] = original_seed_env


if __name__ == "__main__":
    main()
