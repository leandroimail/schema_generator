import json
import re
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dictionary_generation import DataDictionary, find_dictionary_retry_targets, validate_dictionary_result


def test_dictionary_prompt_has_valid_json_example_and_balanced_fences():
    prompt = Path("prompts/dictionary_generation.md").read_text(encoding="utf-8")

    assert "//" not in prompt
    assert prompt.count("```") % 2 == 0

    match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", prompt)
    assert match is not None
    json.loads(match.group(1))


def test_data_dictionary_schema_matches_prompt_example():
    payload = {
        "table_name": "orders",
        "table_description": "Stores customer order transactions.",
        "fields": [
            {
                "field_name": "status",
                "data_type": "string",
                "field_description": "Current order status.",
                "example_value": "paid",
                "domain_values": ["paid", "cancelled"],
                "full_description": 'Current order status. - Domain: ["paid", "cancelled"]',
            },
            {
                "field_name": "amount",
                "data_type": "number",
                "field_description": "Order amount.",
                "example_value": 123.45,
                "full_description": "Order amount. - Example: 123.45",
            },
        ],
    }

    dictionary = DataDictionary.model_validate(payload)

    assert dictionary.table_name == "orders"
    assert dictionary.fields[0].domain_values == ["paid", "cancelled"]
    assert dictionary.fields[1].domain_values is None


def test_data_dictionary_schema_rejects_missing_required_field():
    payload = {
        "table_name": "orders",
        "table_description": "Stores customer order transactions.",
        "fields": [
            {
                "field_name": "amount",
                "data_type": "number",
                "example_value": 123.45,
                "full_description": "Order amount. - Example: 123.45",
            },
        ],
    }

    with pytest.raises(ValidationError):
        DataDictionary.model_validate(payload)


def test_validate_dictionary_result_detects_errors(tmp_path):
    error_file = tmp_path / "error.json"
    error_file.write_text(json.dumps({"error": "provider failed"}), encoding="utf-8")

    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text(json.dumps({"table_name": "orders"}), encoding="utf-8")

    assert validate_dictionary_result(tmp_path / "missing.json") == "missing result file"
    assert validate_dictionary_result(error_file) == "LLM error: provider failed"
    assert validate_dictionary_result(invalid_file).startswith("schema validation error:")


def test_find_dictionary_retry_targets_lists_invalid_and_missing_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CLIENTS", "openai_small,deepseek_small")
    monkeypatch.setenv("LLM_OPENAI_SMALL_PROVIDER", "openai")
    monkeypatch.setenv("LLM_OPENAI_SMALL_MODEL", "gpt-5.4-nano")
    monkeypatch.setenv("LLM_OPENAI_SMALL_API_KEY", "openai-key")
    monkeypatch.setenv("LLM_DEEPSEEK_SMALL_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_DEEPSEEK_SMALL_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("LLM_DEEPSEEK_SMALL_API_KEY", "deepseek-key")

    json_dir = tmp_path / "profile_a" / "json"
    json_dir.mkdir(parents=True)
    valid_payload = {
        "table_name": "orders",
        "table_description": "Stores customer order transactions.",
        "fields": [
            {
                "field_name": "amount",
                "data_type": "number",
                "field_description": "Order amount.",
                "example_value": 123.45,
                "domain_values": None,
                "full_description": "Order amount. - Example: 123.45",
            }
        ],
    }
    (json_dir / "openai_small_gpt-5.4-nano_parsed.json").write_text(
        json.dumps(valid_payload),
        encoding="utf-8",
    )

    retry_targets = find_dictionary_retry_targets(str(tmp_path), ["profile_a"])

    assert retry_targets == {"profile_a": ["deepseek_small"]}
