"""Pytest configuration and fixtures."""

import os
import tempfile
from typing import Generator

import pytest


@pytest.fixture
def context_limit() -> int:
    """Default context limit for tests."""
    return 100


@pytest.fixture
def small_limit() -> int:
    """Small context limit for truncation tests."""
    return 10


@pytest.fixture
def long_text(context_limit: int) -> str:
    """Long text that exceeds context limit by a significant amount."""
    chinese_text = "这是一段中文测试文本。" * 100
    english_text = "This is an English test text. " * 100
    mixed_text = chinese_text + "\n\n" + english_text
    return mixed_text


@pytest.fixture
def boundary_text(context_limit: int) -> str:
    """Text that is exactly at or slightly below context limit."""
    chinese_text = "中" * context_limit
    return chinese_text


@pytest.fixture
def very_long_text() -> str:
    """Very long text (>=5000 characters) for acceptance tests."""
    base_paragraph = "这是第一段测试文本。它包含一些中文内容，还有一些英文内容如This is a test。数字12345也应该被正确处理。标点符号！@#$%也需要考虑。"
    paragraphs = []
    for i in range(100):
        paragraphs.append(f"第{i+1}段: " + base_paragraph + f" 段落编号{i+1}的额外内容。")
    
    return "\n\n".join(paragraphs)


@pytest.fixture
def temp_long_text_file(very_long_text: str) -> Generator[str, None, None]:
    """Temporary file containing very long text."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', encoding='utf-8', delete=False) as f:
        f.write(very_long_text)
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_small_text_file(boundary_text: str) -> Generator[str, None, None]:
    """Temporary file containing boundary text."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', encoding='utf-8', delete=False) as f:
        f.write(boundary_text)
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def seed_7() -> Generator[None, None, None]:
    """Fixture to set CONTEXT_ADAPTER_SEED=7."""
    original_seed = os.environ.get("CONTEXT_ADAPTER_SEED")
    os.environ["CONTEXT_ADAPTER_SEED"] = "7"
    
    yield
    
    if original_seed is None:
        if "CONTEXT_ADAPTER_SEED" in os.environ:
            del os.environ["CONTEXT_ADAPTER_SEED"]
    else:
        os.environ["CONTEXT_ADAPTER_SEED"] = original_seed
