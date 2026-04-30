"""Long Context Adaptation Strategies."""

import os
import random
import re
from typing import Dict, Any, List, Optional, Tuple

from context_adapter.tokenizer import PseudoTokenizer, estimate_tokens


CHUNK_SIZE = 500


def sliding_window_truncation(
    text: str,
    context_limit: int,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Sliding Window / Tail-First Truncation Strategy.
    
    Preserves the most recent (tail) content when truncation is needed.
    This strategy simply truncates from the beginning, keeping the tail.
    
    Args:
        text: Input text to process
        context_limit: Maximum allowed token count
        dry_run: If True, don't modify the text, just estimate
        
    Returns:
        Dictionary with results including:
        - strategy: "sliding_window_truncation"
        - truncated: Whether truncation occurred
        - tokens_in: Estimated tokens in input
        - tokens_out: Estimated tokens in output
        - discarded_blocks: Number of discarded blocks
        - output_text: Processed text (may be None in dry_run)
        - explanation: Brief description of the operation
    """
    tokenizer = PseudoTokenizer(context_limit)
    tokens_in = tokenizer.count_tokens(text)
    truncated = tokens_in > context_limit
    discarded_blocks = 0
    
    if not truncated:
        return {
            "strategy": "sliding_window_truncation",
            "truncated": False,
            "tokens_in": tokens_in,
            "tokens_out": tokens_in,
            "discarded_blocks": 0,
            "output_text": None if dry_run else text,
            "explanation": "Text fits within context limit, no truncation needed."
        }
    
    tokens = tokenizer.tokenize(text)
    kept_tokens = tokens[-context_limit:]
    discarded_blocks = max(0, (len(tokens) - context_limit + CHUNK_SIZE - 1) // CHUNK_SIZE)
    
    tokens_out = len(kept_tokens)
    output_text = None
    if not dry_run:
        output_text = _reconstruct_from_tokens(text, tokens, kept_tokens)
    
    return {
        "strategy": "sliding_window_truncation",
        "truncated": True,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "discarded_blocks": discarded_blocks,
        "output_text": output_text,
        "explanation": f"Truncated from {tokens_in} to {tokens_out} tokens, preserving {discarded_blocks} tail blocks."
    }


def segment_summary_concatenation(
    text: str,
    context_limit: int,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Segment Summary Concatenation Strategy.
    
    Splits text into segments, takes the beginning of each segment as a "summary",
    then concatenates these summaries until reaching the context limit.
    
    Args:
        text: Input text to process
        context_limit: Maximum allowed token count
        dry_run: If True, don't modify the text, just estimate
        
    Returns:
        Dictionary with results
    """
    tokenizer = PseudoTokenizer(context_limit)
    tokens_in = tokenizer.count_tokens(text)
    truncated = tokens_in > context_limit
    discarded_blocks = 0
    
    if not truncated:
        return {
            "strategy": "segment_summary_concatenation",
            "truncated": False,
            "tokens_in": tokens_in,
            "tokens_out": tokens_in,
            "discarded_blocks": 0,
            "output_text": None if dry_run else text,
            "explanation": "Text fits within context limit, no summarization needed."
        }
    
    segments = _split_into_segments(text)
    discarded_blocks = len(segments)
    
    summary_parts = []
    total_tokens = 0
    summary_size = 50
    
    for i, segment in enumerate(segments):
        segment_tokens = tokenizer.tokenize(segment)
        if not segment_tokens:
            continue
        
        if i == 0 and len(segment_tokens) > summary_size:
            summary_tokens = segment_tokens[:summary_size]
        elif len(segment_tokens) > 20:
            summary_tokens = segment_tokens[:20]
        else:
            summary_tokens = segment_tokens
        
        summary_text = _reconstruct_from_tokens(segment, segment_tokens, summary_tokens)
        
        potential_summary = " ".join(summary_parts + [summary_text])
        potential_tokens = tokenizer.count_tokens(potential_summary)
        
        if potential_tokens <= context_limit:
            summary_parts.append(summary_text)
            total_tokens = potential_tokens
        else:
            break
    
    if not summary_parts:
        tokens = tokenizer.tokenize(text)
        summary_parts = [_reconstruct_from_tokens(text, tokens, tokens[:context_limit])]
        total_tokens = context_limit
    
    output_text = None if dry_run else "\n\n[SEGMENT SUMMARY]\n\n".join(summary_parts)
    
    return {
        "strategy": "segment_summary_concatenation",
        "truncated": True,
        "tokens_in": tokens_in,
        "tokens_out": total_tokens,
        "discarded_blocks": discarded_blocks - len(summary_parts),
        "output_text": output_text,
        "explanation": f"Summarized {len(segments)} segments, kept {len(summary_parts)} segment summaries ({total_tokens} tokens)."
    }


def deterministic_retrieval_chunk_assembly(
    text: str,
    context_limit: int,
    dry_run: bool = False,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Deterministic Retrieval-based Chunk Assembly Strategy.
    
    Splits text into chunks, then uses a deterministic random seed to select
    chunks for assembly. This ensures reproducibility while simulating a
    "retrieval" process.
    
    Args:
        text: Input text to process
        context_limit: Maximum allowed token count
        dry_run: If True, don't modify the text, just estimate
        seed: Random seed for deterministic selection. If None, reads from
              CONTEXT_ADAPTER_SEED environment variable, defaults to 0.
        
    Returns:
        Dictionary with results
    """
    if seed is None:
        seed = int(os.environ.get("CONTEXT_ADAPTER_SEED", "0"))
    
    tokenizer = PseudoTokenizer(context_limit)
    tokens_in = tokenizer.count_tokens(text)
    truncated = tokens_in > context_limit
    discarded_blocks = 0
    
    if not truncated:
        return {
            "strategy": "deterministic_retrieval_chunk_assembly",
            "truncated": False,
            "tokens_in": tokens_in,
            "tokens_out": tokens_in,
            "discarded_blocks": 0,
            "output_text": None if dry_run else text,
            "explanation": "Text fits within context limit, no assembly needed."
        }
    
    chunks = _split_into_chunks(text, CHUNK_SIZE)
    discarded_blocks = len(chunks)
    
    rng = random.Random(seed)
    chunk_indices = list(range(len(chunks)))
    rng.shuffle(chunk_indices)
    
    selected_chunks = []
    total_tokens = 0
    
    for idx in chunk_indices:
        chunk = chunks[idx]
        chunk_tokens = tokenizer.count_tokens(chunk)
        
        if total_tokens + chunk_tokens <= context_limit:
            selected_chunks.append((idx, chunk))
            total_tokens += chunk_tokens
        elif total_tokens == 0:
            tokens = tokenizer.tokenize(chunk)
            partial_chunk = _reconstruct_from_tokens(chunk, tokens, tokens[:context_limit])
            selected_chunks.append((idx, partial_chunk))
            total_tokens = context_limit
            break
    
    selected_chunks.sort(key=lambda x: x[0])
    selected_texts = [chunk for _, chunk in selected_chunks]
    
    output_text = None
    if not dry_run:
        output_text = "\n\n[CHUNK]\n\n".join(selected_texts)
    
    return {
        "strategy": "deterministic_retrieval_chunk_assembly",
        "truncated": True,
        "tokens_in": tokens_in,
        "tokens_out": total_tokens,
        "discarded_blocks": len(chunks) - len(selected_chunks),
        "output_text": output_text,
        "explanation": f"Selected {len(selected_chunks)} chunks from {len(chunks)} using seed={seed} ({total_tokens} tokens)."
    }


def _split_into_segments(text: str) -> List[str]:
    """Split text into logical segments based on paragraphs."""
    segments = re.split(r'\n\s*\n', text.strip())
    return [s.strip() for s in segments if s.strip()]


def _split_into_chunks(text: str, chunk_token_size: int) -> List[str]:
    """Split text into chunks of approximately chunk_token_size tokens."""
    tokenizer = PseudoTokenizer()
    tokens = tokenizer.tokenize(text)
    
    chunks = []
    for i in range(0, len(tokens), chunk_token_size):
        chunk_tokens = tokens[i:i + chunk_token_size]
        chunk_text = _reconstruct_from_tokens(text, tokens, chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks


def _reconstruct_from_tokens(original_text: str, all_tokens: List[str], selected_tokens: List[str]) -> str:
    """
    Reconstruct text from selected tokens by finding the corresponding substring.
    
    This is a simplified approach that finds the start and end positions.
    """
    if not selected_tokens:
        return ""
    
    tokenizer = PseudoTokenizer()
    
    start_idx = -1
    for i in range(len(all_tokens)):
        window_size = len(selected_tokens)
        if i + window_size <= len(all_tokens):
            if all_tokens[i:i + window_size] == selected_tokens:
                start_idx = i
                break
    
    if start_idx == -1:
        return " ".join(selected_tokens)
    
    end_idx = start_idx + len(selected_tokens)
    
    pos = 0
    token_positions = []
    
    text_pos = 0
    while text_pos < len(original_text):
        char = original_text[text_pos]
        
        if char.isspace():
            text_pos += 1
            continue
        
        if '\u4e00' <= char <= '\u9fff':
            token_positions.append((text_pos, text_pos + 1))
            text_pos += 1
            continue
        
        if char.isalpha():
            start = text_pos
            while text_pos < len(original_text) and original_text[text_pos].isalpha():
                text_pos += 1
            token_positions.append((start, text_pos))
            continue
        
        if char.isdigit():
            start = text_pos
            while text_pos < len(original_text) and original_text[text_pos].isdigit():
                text_pos += 1
            token_positions.append((start, text_pos))
            continue
        
        token_positions.append((text_pos, text_pos + 1))
        text_pos += 1
    
    if start_idx < len(token_positions) and end_idx <= len(token_positions):
        start_char = token_positions[start_idx][0]
        end_char = token_positions[end_idx - 1][1]
        return original_text[start_char:end_char]
    
    return " ".join(selected_tokens)
