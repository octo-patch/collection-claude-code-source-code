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
    assert flash["thinking_always_on"] is True
    assert flash["thinking_defaults_by_api"] == {
        "anthropic_messages": "adaptive",
        "openai_chat_completions": "adaptive",
    }
    assert flash["reasoning_effort"] == [
        "low", "medium", "high", "xhigh", "max",
    ]
    assert flash["reasoning_effort_default"] == "max"
    assert flash["service_tier_supported"] is False
    m3 = MODEL_METADATA["MiniMax-M3"]
    assert m3["input_modalities"] == [
        "text", "image", "video",
    ]
    assert m3["thinking"] == ["adaptive", "disabled"]
    assert m3["thinking_always_on"] is False
    assert m3["thinking_defaults_by_api"] == {
        "anthropic_messages": "disabled",
        "openai_chat_completions": "adaptive",
    }
    assert m3["reasoning_effort"] == []
    assert m3["reasoning_effort_default"] is None
    assert m3["reasoning_effort_note"] == (
        "Effort fields are ignored by MiniMax-M3."
    )
    assert m3["service_tier_supported"] is True


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


def test_minimax_m3_tiered_pricing_metadata():
    tiers = MODEL_METADATA["MiniMax-M3"]["pricing_tiers"]
    assert tiers["selection_basis"] == "request_input_tokens"
    assert tiers["input_token_threshold"] == "512k"
    assert tiers["default_service_tier"] == "standard"
    assert tiers["standard"] == [
        {
            "input_token_condition": "<=512k",
            "pricing": {
                "input": 0.3,
                "output": 1.2,
                "cache_read": 0.06,
                "cache_write": None,
            },
        },
        {
            "input_token_condition": ">512k",
            "pricing": {
                "input": 0.6,
                "output": 2.4,
                "cache_read": 0.12,
                "cache_write": None,
            },
        },
    ]
    assert tiers["priority_multiplier"] == 1.5
    assert tiers["priority_multiplier_applies_to"] == [
        "input", "output", "cache_read",
    ]


def test_minimax_api_metadata_sources():
    provider = PROVIDERS["minimax"]
    assert provider["reasoning_effort_parameters_by_api"] == {
        "anthropic_messages": "output_config.effort",
        "openai_chat_completions": "reasoning_effort",
    }
    assert provider["official_sources"] == {
        "models": "https://platform.minimax.io/docs/guides/models-intro.md",
        "pricing": "https://platform.minimax.io/docs/pricing/overview.md",
        "anthropic_messages": (
            "https://platform.minimax.io/docs/api-reference/"
            "text-chat-anthropic.md"
        ),
        "openai_chat_completions": (
            "https://platform.minimax.io/docs/api-reference/"
            "text-chat-openai.md"
        ),
    }
