import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dictionary_generation import DataDictionary
from llm import create_llm_client, load_llm_client_configs, process_text_with_llm, save_response_data


class FakeStructuredClient:
    model_name = "fake-model"
    client_name = "fake-client"

    async def async_parse(self, messages, response_model):
        assert messages[0]["role"] == "system"
        assert "conforms to the requested schema" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        return response_model.model_validate(
            {
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
                    }
                ],
            }
        )


def test_process_text_with_llm_uses_response_model_for_structured_calls():
    async def run_call():
        return await process_text_with_llm(
            FakeStructuredClient(),
            "Generate a data dictionary.",
            response_model=DataDictionary,
        )

    llm_name, generated_text, parsed_data, model_name = asyncio.run(run_call())

    assert llm_name == "fake-client"
    assert model_name == "fake-model"
    assert parsed_data["table_name"] == "orders"
    assert parsed_data["fields"][0]["domain_values"] == ["paid", "cancelled"]
    assert json.loads(generated_text) == parsed_data


def test_save_response_data_writes_schema_conformant_json(tmp_path):
    parsed_data = DataDictionary.model_validate(
        {
            "table_name": "orders",
            "table_description": "Stores customer order transactions.",
            "fields": [
                {
                    "field_name": "amount",
                    "data_type": "number",
                    "field_description": "Order amount.",
                    "example_value": 123.45,
                    "full_description": "Order amount. - Example: 123.45",
                }
            ],
        }
    ).model_dump(mode="json")

    _, json_path, _ = save_response_data(
        llm_name="fake/client",
        model_name="fake:model",
        text_id="unused",
        generated_text=json.dumps(parsed_data),
        parsed_data=parsed_data,
        prompt="prompt",
        base_output_dir=str(tmp_path),
    )

    saved = json.loads(Path(json_path).read_text(encoding="utf-8"))
    DataDictionary.model_validate(saved)
    assert Path(json_path).name == "fake_client_fake_model_parsed.json"


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS") != "1",
    reason="Set RUN_LIVE_LLM_TESTS=1 to spend API calls against configured providers.",
)
def test_live_llm_clients_return_dictionary_json():
    async def run_live_checks():
        configs = load_llm_client_configs()
        assert configs
        failures = []

        prompt = """
Generate a data dictionary for this single-column table.

## Data Profile
```json
{"table_name": "orders", "columns": [{"name": "status", "data_type": "string"}]}
```

## Data Sample
```csv
status
paid
cancelled
```
"""

        for config in configs:
            client = create_llm_client(config)
            await client.initialize_client()
            try:
                _, _, parsed_data, _ = await process_text_with_llm(
                    client,
                    prompt,
                    response_model=DataDictionary,
                )
            finally:
                await client.close_client()

            try:
                dictionary = DataDictionary.model_validate(parsed_data)
                assert dictionary.table_name
                assert dictionary.fields
            except Exception as exc:
                failures.append(f"{config.name} ({config.model_name}): {parsed_data.get('error', exc)}")

        if failures:
            pytest.fail("Live LLM checks failed:\n" + "\n".join(failures))

    asyncio.run(run_live_checks())
