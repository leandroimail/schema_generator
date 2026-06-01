import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dictionary_generation import DataDictionary
from llm import OpenAIClient, create_llm_client, load_llm_client_configs


def test_loads_multiple_named_llm_clients(monkeypatch):
    monkeypatch.setenv(
        "LLM_CLIENTS",
        "deepseek_small_flash,deepseek_small_pro,google_small_lite,google_small_flash,openai_small_mini,openai_small_nano",
    )
    monkeypatch.setenv("LLM_DEEPSEEK_SMALL_FLASH_NAME", "deepseek_small")
    monkeypatch.setenv("LLM_DEEPSEEK_SMALL_FLASH_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_DEEPSEEK_SMALL_FLASH_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("LLM_DEEPSEEK_SMALL_FLASH_API_KEY", "deepseek-key")
    monkeypatch.setenv("LLM_DEEPSEEK_SMALL_PRO_NAME", "deepseek_small")
    monkeypatch.setenv("LLM_DEEPSEEK_SMALL_PRO_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_DEEPSEEK_SMALL_PRO_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("LLM_DEEPSEEK_SMALL_PRO_API_KEY", "deepseek-key")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_LITE_NAME", "google_small")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_LITE_PROVIDER", "google")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_LITE_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_LITE_API_KEY", "google-key")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_FLASH_NAME", "google_small")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_FLASH_PROVIDER", "google")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_FLASH_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_FLASH_API_KEY", "google-key")
    monkeypatch.setenv("LLM_OPENAI_SMALL_MINI_NAME", "openai_small")
    monkeypatch.setenv("LLM_OPENAI_SMALL_MINI_PROVIDER", "openai")
    monkeypatch.setenv("LLM_OPENAI_SMALL_MINI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("LLM_OPENAI_SMALL_MINI_API_KEY", "openai-key")
    monkeypatch.setenv("LLM_OPENAI_SMALL_NANO_NAME", "openai_small")
    monkeypatch.setenv("LLM_OPENAI_SMALL_NANO_PROVIDER", "openai")
    monkeypatch.setenv("LLM_OPENAI_SMALL_NANO_MODEL", "gpt-5.4-nano")
    monkeypatch.setenv("LLM_OPENAI_SMALL_NANO_API_KEY", "openai-key")

    configs = load_llm_client_configs()

    assert [config.id for config in configs] == [
        "deepseek_small_flash",
        "deepseek_small_pro",
        "google_small_lite",
        "google_small_flash",
        "openai_small_mini",
        "openai_small_nano",
    ]
    assert [config.name for config in configs] == [
        "deepseek_small",
        "deepseek_small",
        "google_small",
        "google_small",
        "openai_small",
        "openai_small",
    ]
    assert [config.model_name for config in configs] == [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ]
    assert configs[2].base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert configs[0].use_json_schema is False


def test_selected_llm_client_filters_by_config_id(monkeypatch):
    monkeypatch.setenv("LLM_CLIENTS", "google_small_lite,google_small_flash")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_LITE_NAME", "google_small")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_LITE_PROVIDER", "google")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_LITE_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_LITE_API_KEY", "google-key")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_FLASH_NAME", "google_small")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_FLASH_PROVIDER", "google")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_FLASH_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_FLASH_API_KEY", "google-key")

    configs = load_llm_client_configs("google_small_flash")

    assert len(configs) == 1
    assert configs[0].id == "google_small_flash"
    assert configs[0].name == "google_small"
    assert configs[0].model_name == "gemini-3.5-flash"


def test_legacy_env_falls_back_to_small_current_models(monkeypatch):
    monkeypatch.delenv("LLM_CLIENTS", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_NAME", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_NAME", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL_NAME", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    configs = load_llm_client_configs()

    assert [config.name for config in configs] == ["openai", "gemini_native", "deepseek"]
    assert [config.model_name for config in configs] == [
        "gpt-5.4-nano",
        "gemini-3.5-flash",
        "deepseek-v4-flash",
    ]


def test_google_provider_uses_openai_compatible_client(monkeypatch):
    monkeypatch.setenv("LLM_CLIENTS", "google_small_flash")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_FLASH_NAME", "google_small")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_FLASH_PROVIDER", "google")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_FLASH_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("LLM_GOOGLE_SMALL_FLASH_API_KEY", "google-key")

    config = load_llm_client_configs()[0]
    client = create_llm_client(config)

    assert client.client_name == "google_small"
    assert client.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert client.model_name == "gemini-3.5-flash"


def test_openai_strict_json_schema_requires_all_properties():
    client = OpenAIClient(api_key="test", model_name="test")

    response_format = client._json_schema_response_format(DataDictionary)
    schema = response_format["json_schema"]["schema"]
    field_schema = schema["$defs"]["DictionaryField"]

    assert set(schema["required"]) == {"table_name", "table_description", "fields"}
    assert set(field_schema["required"]) == {
        "field_name",
        "data_type",
        "field_description",
        "example_value",
        "full_description",
        "domain_values",
    }
    assert schema["additionalProperties"] is False
    assert field_schema["additionalProperties"] is False
