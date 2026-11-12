"""Focused tests for provider registry metadata and pricing."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from providers import (  # noqa: E402
    MODEL_METADATA,
    PROVIDERS,
    calc_cost,
    detect_provider,
    get_api_key,
)


def test_minimax_registry_and_detection():
    provider = PROVIDERS["minimax"]
    assert provider["type"] == "openai"
    assert provider["base_url"] == "https://api.minimax.io/v1"
    assert provider["anthropic_base_url"] == "https://api.minimax.io/anthropic"
    assert provider["default_model"] == "MiniMax-M3.1-Flash"
    assert provider["models"] == ["MiniMax-M3.1-Flash", "MiniMax-M3"]
    assert detect_provider("MiniMax-M3.1-Flash") == "minimax"
    assert detect_provider("MiniMax-M3") == "minimax"


def test_minimax_api_key_lookup(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    assert get_api_key("minimax", {}) == "test-key"
    assert get_api_key("minimax", {"minimax_api_key": "config-key"}) == "config-key"


def test_minimax_model_capabilities():
    flash = MODEL_METADATA["MiniMax-M3.1-Flash"]
    assert flash["input_modalities"] == ["text", "image", "video"]
    assert flash["thinking"] == ["adaptive"]
    assert flash["thinking_required"] is True
    assert flash["reasoning_efforts"] == [
        "low", "medium", "high", "xhigh", "max",
    ]
    assert flash["default_reasoning_effort"] == "max"
    assert MODEL_METADATA["MiniMax-M3"]["input_modalities"] == [
        "text", "image", "video",
    ]
    assert MODEL_METADATA["MiniMax-M3"]["thinking"] == ["adaptive", "disabled"]


def test_minimax_regional_endpoints():
    assert PROVIDERS["minimax"]["regional_endpoints"] == [
        {
            "region": "global_en",
            "openai_base_url": "https://api.minimax.io/v1",
            "anthropic_base_url": "https://api.minimax.io/anthropic",
            "docs_root": "https://platform.minimax.io/docs",
        },
        {
            "region": "cn_zh",
            "openai_base_url": "https://api.minimaxi.com/v1",
            "anthropic_base_url": "https://api.minimaxi.com/anthropic",
            "docs_root": "https://platform.minimaxi.com/docs",
        },
    ]


def test_minimax_cache_pricing_metadata():
    flash_pricing = MODEL_METADATA["MiniMax-M3.1-Flash"]["pricing"]
    m3_pricing = MODEL_METADATA["MiniMax-M3"]["pricing"]
    assert flash_pricing == {
        "input": 0.15,
        "output": 0.6,
        "cache_read": 0.015,
        "cache_write": None,
    }
    assert m3_pricing == {
        "input": 0.3,
        "output": 1.2,
        "cache_read": 0.06,
        "cache_write": None,
    }
    assert calc_cost("MiniMax-M3.1-Flash", 1000000, 1000000) == pytest.approx(0.75)
    assert calc_cost("MiniMax-M3", 1000000, 1000000) == pytest.approx(1.5)
