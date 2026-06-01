import json
import re
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dictionary_generation import DataDictionary


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
