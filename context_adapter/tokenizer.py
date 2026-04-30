"""Pseudo Tokenizer for Long Context Adaptation."""

import re
from typing import List, Dict, Any


class PseudoTokenizer:
    """
    A simple pseudo-tokenizer for token estimation.
    
    Default rules (fixed in code):
    - Chinese characters: each character is 1 token
    - English words: each word (sequence of a-z/A-Z) is 1 token
    - Numbers: each sequence of digits is 1 token
    - Whitespace: not counted
    - Other punctuation/symbols: each is 1 token
    """
    
    DEFAULT_CONTEXT_LIMIT = 4096
    
    def __init__(self, context_limit: int = None):
        self.context_limit = context_limit or self.DEFAULT_CONTEXT_LIMIT
    
    @staticmethod
    def get_rules() -> Dict[str, str]:
        """Return the tokenization rules as a dictionary."""
        return {
            "Chinese characters": "each character = 1 token",
            "English words": "each word (a-z/A-Z sequence) = 1 token",
            "Numbers": "each digit sequence = 1 token",
            "Whitespace": "not counted",
            "Other punctuation/symbols": "each = 1 token",
        }
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the text according to the fixed rules.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        tokens = []
        i = 0
        n = len(text)
        
        while i < n:
            char = text[i]
            
            if char.isspace():
                i += 1
                continue
            
            if '\u4e00' <= char <= '\u9fff':
                tokens.append(char)
                i += 1
                continue
            
            if char.isalpha():
                j = i
                while j < n and text[j].isalpha():
                    j += 1
                tokens.append(text[i:j])
                i = j
                continue
            
            if char.isdigit():
                j = i
                while j < n and text[j].isdigit():
                    j += 1
                tokens.append(text[i:j])
                i = j
                continue
            
            tokens.append(char)
            i += 1
        
        return tokens
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.tokenize(text))
    
    def estimate(self, text: str) -> int:
        """Alias for count_tokens."""
        return self.count_tokens(text)


def estimate_tokens(text: str) -> int:
    """
    Convenience function to estimate tokens using the default tokenizer.
    
    Args:
        text: Input text
        
    Returns:
        Number of tokens
    """
    tokenizer = PseudoTokenizer()
    return tokenizer.count_tokens(text)
