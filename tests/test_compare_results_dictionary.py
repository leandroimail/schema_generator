"""Unit tests for src/compare_results_dictionary.py.

Focuses on the fixed behaviors:
  * model_name extraction via known client prefixes (Fix A)
  * table_name extraction with the new data_llm_dir layout (Fix B + C)
  * filtering of LLM error results in the load loop (Fix D)
  * JSON / CSV loaders
  * similarity / aggregation math (small inputs, no model download)
"""

import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
FIXTURES_DIR = TESTS_DIR / "fixtures"

sys.path.insert(0, str(SRC))

from compare_results_dictionary import (  # noqa: E402
    extract_table_and_model_names,
    find_all_files_in_directory,
    generate_output_json,
    load_csv_data,
    load_data_dictionary,
    load_json_data,
    calculate_similarities,
    compute_similarity_metrics,
    _load_embedding_model,
)


# ---------------------------------------------------------------------------
# extract_table_and_model_names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data_llm_dir,rel_path,expected_table,expected_model",
    [
        (
            "data/llm_results/dictionary_llm_results",
            "data/llm_results/dictionary_llm_results/bird__california_schools__frpm/json/openai_small_gpt-5.4-mini_parsed.json",
            "bird__california_schools__frpm",
            "gpt-5.4-mini",
        ),
        (
            "data/llm_results/dictionary_llm_results",
            "data/llm_results/dictionary_llm_results/bird__toxicology__bond/json/deepseek_small_deepseek-v4-flash_parsed.json",
            "bird__toxicology__bond",
            "deepseek-v4-flash",
        ),
        (
            "data/llm_results/dictionary_llm_results",
            "data/llm_results/dictionary_llm_results/bird__codebase_community__users/json/google_small_gemini-3.5-flash_parsed.json",
            "bird__codebase_community__users",
            "gemini-3.5-flash",
        ),
        (
            "data/llm_results/dictionary_llm_results",
            "data/llm_results/dictionary_llm_results/bird__codebase_community__users/json/google_small_gemini-3.1-flash-lite_parsed.json",
            "bird__codebase_community__users",
            "gemini-3.1-flash-lite",
        ),
        (
            "data/llm_results/dictionary_llm_results",
            "data/llm_results/dictionary_llm_results/bird__european_football_2__team/json/deepseek_small_deepseek-v4-pro_parsed.json",
            "bird__european_football_2__team",
            "deepseek-v4-pro",
        ),
        (
            "data/llm_results/dictionary_llm_results",
            "data/llm_results/dictionary_llm_results/bird__superhero__hero/json/openai_small_gpt-5.4-nano_parsed.json",
            "bird__superhero__hero",
            "gpt-5.4-nano",
        ),
    ],
)
def test_extract_table_and_model_names_six_models(
    data_llm_dir, rel_path, expected_table, expected_model
):
    table, model = extract_table_and_model_names(data_llm_dir, rel_path)
    assert table == expected_table
    assert model == expected_model


def test_extract_table_and_model_names_unrecognised_prefix_falls_back_to_stem():
    """Unknown client prefix should fall back to the file stem (post _parsed)."""
    table, model = extract_table_and_model_names(
        "data/llm_results/dictionary_llm_results",
        "data/llm_results/dictionary_llm_results/some__table/json/mystery_unknown-model_parsed.json",
    )
    assert table == "some__table"
    # Unknown prefix: the whole stem (minus _parsed) is returned.
    assert model == "mystery_unknown-model"


def test_extract_table_and_model_names_uses_index_one_with_new_layout():
    """Regression: the old code used split[2] which became 'json' under the new path."""
    table, _ = extract_table_and_model_names(
        "data/llm_results/dictionary_llm_results",
        "data/llm_results/dictionary_llm_results/bird__formula_1__races/json/openai_small_gpt-5.4-mini_parsed.json",
    )
    assert table != "json"
    assert table == "bird__formula_1__races"


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def test_load_json_data_returns_dict(tmp_path):
    p = tmp_path / "ok.json"
    p.write_text(json.dumps({"a": 1, "b": [1, 2]}))
    assert load_json_data(str(p)) == {"a": 1, "b": [1, 2]}


def test_load_json_data_returns_empty_on_error(tmp_path, caplog):
    p = tmp_path / "bad.json"
    p.write_text("{not valid")
    with caplog.at_level("ERROR", logger="__main__"):
        result = load_json_data(str(p))
    assert result == {}


def test_load_csv_data_reads_field_name_and_description(tmp_path):
    p = tmp_path / "dict.csv"
    p.write_text(
        "field_name,description\n"
        "id,Primary key\n"
        "name,Display name\n"
    )
    assert load_csv_data(str(p)) == {"id": "Primary key", "name": "Display name"}


def test_load_data_dictionary_dispatches_by_extension(tmp_path):
    json_p = tmp_path / "d.json"
    json_p.write_text(json.dumps({"id": "pk"}))
    csv_p = tmp_path / "d.csv"
    csv_p.write_text("field_name,description\nid,Primary key\n")

    assert load_data_dictionary(str(json_p)) == {"id": "pk"}
    assert load_data_dictionary(str(csv_p)) == {"id": "Primary key"}


def test_load_data_dictionary_unsupported_extension_returns_empty(tmp_path, caplog):
    p = tmp_path / "d.txt"
    p.write_text("anything")
    with caplog.at_level("WARNING", logger="__main__"):
        assert load_data_dictionary(str(p)) == {}


def test_find_all_files_in_directory_recursive(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "ignore.xml").write_text("<x/>")

    files = find_all_files_in_directory(str(tmp_path), [".json", ".csv"])
    assert len(files) == 2
    assert any(f.endswith("x.json") for f in files)
    assert any(f.endswith("b.json") for f in files)


# ---------------------------------------------------------------------------
# calculate_similarities (uses real numpy + sklearn; no model download)
# ---------------------------------------------------------------------------


def _vec(values):
    return np.array(values, dtype=float)


def test_calculate_similarities_identical_vectors_score_one():
    a = _vec([1, 0, 0])
    b = _vec([1, 0, 0])
    sims = calculate_similarities({"x": a}, {"x": b})
    assert sims == [{"field": "x", "score": pytest.approx(1.0)}]


def test_calculate_similarities_orthogonal_vectors_score_zero():
    a = _vec([1, 0])
    b = _vec([0, 1])
    sims = calculate_similarities({"x": a}, {"x": b})
    assert sims[0]["score"] == pytest.approx(0.0)


def test_calculate_similarities_ignores_missing_keys():
    a = _vec([1, 0])
    sims = calculate_similarities({"x": a}, {"y": _vec([1, 0])})
    assert sims == []


def test_calculate_similarities_scores_in_minus_one_one_range():
    a = _vec([1, 2, 3, 4])
    b = _vec([-1, -2, -3, -4])
    sims = calculate_similarities({"f": a}, {"f": b})
    assert sims[0]["score"] == pytest.approx(-1.0)
    assert -1.0 <= sims[0]["score"] <= 1.0


# ---------------------------------------------------------------------------
# generate_output_json
# ---------------------------------------------------------------------------


def test_generate_output_json_shape():
    out = generate_output_json(
        table_name="t",
        model_name="m",
        similarities=[{"field": "f", "score": 0.9}],
    )
    assert out == {
        "table_name": "t",
        "par-compare-models": "m",
        "similarities": [{"field": "f", "score": 0.9}],
    }


# ---------------------------------------------------------------------------
# compute_similarity_metrics
# ---------------------------------------------------------------------------


def test_compute_similarity_metrics_returns_empty_for_empty_input():
    assert compute_similarity_metrics([]) == {}


def test_compute_similarity_metrics_returns_all_keys():
    metrics = compute_similarity_metrics([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    expected_keys = {
        "mean", "std", "q25", "median", "q75", "d90", "d99", "min", "max", "count"
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["count"] == 10


def test_compute_similarity_metrics_values_are_correct():
    scores = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    metrics = compute_similarity_metrics(scores)
    # 11 values from 0.0 to 1.0 in steps of 0.1
    assert metrics["mean"] == pytest.approx(0.5)
    assert metrics["min"] == pytest.approx(0.0)
    assert metrics["max"] == pytest.approx(1.0)
    assert metrics["median"] == pytest.approx(0.5)
    # q25 / q75 for 11 evenly spaced values
    assert metrics["q25"] == pytest.approx(np.percentile(scores, 25))
    assert metrics["q75"] == pytest.approx(np.percentile(scores, 75))
    assert metrics["d90"] == pytest.approx(np.percentile(scores, 90))
    assert metrics["d99"] == pytest.approx(np.percentile(scores, 99))
    # ddof=1 sample std
    assert metrics["std"] == pytest.approx(np.std(scores, ddof=1))


def test_compute_similarity_metrics_single_value_zero_std():
    metrics = compute_similarity_metrics([0.42])
    assert metrics["mean"] == pytest.approx(0.42)
    assert metrics["std"] == 0.0
    assert metrics["median"] == pytest.approx(0.42)
    assert metrics["min"] == pytest.approx(0.42)
    assert metrics["max"] == pytest.approx(0.42)
    assert metrics["count"] == 1


def test_compute_similarity_metrics_handles_negative_scores():
    # cosine similarity range is [-1, 1]
    metrics = compute_similarity_metrics([-1.0, 0.0, 1.0])
    assert metrics["min"] == pytest.approx(-1.0)
    assert metrics["max"] == pytest.approx(1.0)
    assert metrics["mean"] == pytest.approx(0.0)
    assert metrics["median"] == pytest.approx(0.0)


def test_compute_similarity_metrics_drops_non_finite_scores():
    """NaN and +/-inf inputs should be dropped, not propagated to metrics."""
    metrics = compute_similarity_metrics([float("nan"), 0.2, 0.4, float("inf"), 0.6])
    assert metrics["count"] == 3
    assert metrics["min"] == pytest.approx(0.2)
    assert metrics["max"] == pytest.approx(0.6)
    for v in metrics.values():
        assert not (isinstance(v, float) and math.isnan(v))
        assert v != float("inf")


def test_compute_similarity_metrics_all_non_finite_returns_empty():
    assert compute_similarity_metrics([float("nan"), float("inf")]) == {}


# ---------------------------------------------------------------------------
# End-to-end with a fake SentenceTransformer (no model download)
# ---------------------------------------------------------------------------


class _FakeModel:
    """Returns a deterministic embedding based on the input string length."""

    def encode(self, text, **kwargs):
        # Accept and ignore extra kwargs (normalize_embeddings, batch_size, ...)
        # so the fake stays compatible with the real `SentenceTransformer.encode`
        # signature used by `calculate_embeddings` after the refactor.
        _ = kwargs
        if not isinstance(text, str) or not text.strip():
            return np.zeros(4, dtype=float)
        # Two identical texts -> same vector; different texts -> orthogonal vectors.
        h = abs(hash(text)) % 97
        v = np.zeros(4, dtype=float)
        v[h % 4] = 1.0
        if (h // 4) % 4 != h % 4:
            v[(h // 4) % 4] = 1.0
        return v


def test_main_end_to_end_with_fake_model(tmp_path, monkeypatch):
    """Drive main() with a fake embedding model and a tiny filesystem."""
    import re
    cfg_src = PROJECT_ROOT / "config.yaml"
    cfg = cfg_src.read_text(encoding="utf-8")

    table = "bird__unit__test"
    # Build a fake data tree inside tmp_path.
    dict_dir = tmp_path / "data" / "dictionaries"
    llm_root = tmp_path / "data" / "llm_results" / "dictionary_llm_results" / table / "json"
    out_dir = tmp_path / "data" / "distance_calculation"
    dict_dir.mkdir(parents=True)
    llm_root.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    # Baseline dictionary.
    (dict_dir / f"{table}.json").write_text(
        json.dumps(
            {
                "table_name": table,
                "table_description": "Baseline desc",
                "fields": [
                    {"field_name": "id", "field_description": "Primary key"},
                    {"field_name": "name", "field_description": "Display name"},
                ],
            }
        )
    )

    # Two LLM dictionaries, plus one with an error key (must be skipped).
    (llm_root / "openai_small_gpt-5.4-mini_parsed.json").write_text(
        json.dumps(
            {
                "table_name": table,
                "table_description": "LLM desc",
                "fields": [
                    {"field_name": "id", "full_description": "Primary key"},
                    {"field_name": "name", "full_description": "Display name"},
                ],
            }
        )
    )
    (llm_root / "deepseek_small_deepseek-v4-flash_parsed.json").write_text(
        json.dumps(
            {
                "table_name": table,
                "table_description": "LLM desc alt",
                "fields": [
                    {"field_name": "id", "full_description": "Primary key"},
                    {"field_name": "name", "full_description": "Display name"},
                ],
            }
        )
    )
    (llm_root / "google_small_gemini-3.5-flash_parsed.json").write_text(
        json.dumps({"error": "rate limit"})
    )

    # Build a minimal config that points to our temp tree.
    # data_llm_dir must point to the *parent* of the per-table directories,
    # matching the production layout (data/llm_results/dictionary_llm_results).
    new_cfg = (
        "list_of_data_dictionaries:\n"
        f"- name: {table}\n"
        f"  path: {dict_dir / (table + '.json')}\n"
        "data_llm_results_dictionary_generation:\n"
        f"  path: {llm_root.parent.parent}\n"
        "data_llm_results_distance_calculation:\n"
        f"  path: {out_dir}\n"
    )

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(new_cfg)

    from compare_results_dictionary import main

    # Patch the embedding model.
    monkeypatch.setattr(
        "compare_results_dictionary.SentenceTransformer", lambda *a, **kw: _FakeModel()
    )

    main(str(cfg_path))

    # Verify outputs: 2 LLM files survived (error one skipped), 1 baseline.
    out_files = list(out_dir.glob("output_*.json"))
    assert len(out_files) == 2
    summary = json.loads((out_dir / "all_similarities_results.json").read_text())
    assert table in summary["results"]
    # Exactly 2 models should be present (the error one must be filtered).
    models = set(summary["metrics_by_model"].keys())
    assert models == {"gpt-5.4-mini", "deepseek-v4-flash"}
    # The metrics payload should expose the full distribution for each model.
    for model_name, metrics in summary["metrics_by_model"].items():
        for key in ("mean", "std", "q25", "median", "q75", "d90", "d99", "min", "max", "count"):
            assert key in metrics, f"missing metric {key} for {model_name}"
        assert metrics["count"] > 0
        assert -1.0 <= metrics["min"] <= metrics["max"] <= 1.0
    # Per-table breakdown should also be present and use the same shape.
    table_metrics = summary["metrics_by_table_and_model"][table]
    for model_name, metrics in table_metrics.items():
        assert "mean" in metrics
        assert "median" in metrics
    # No scores should be NaN — fake embeddings give valid cosine similarities.
    for field_scores in summary["results"][table].values():
        for score in field_scores.values():
            assert not math.isnan(score)
            assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _load_embedding_model — config-driven model loader
# ---------------------------------------------------------------------------


def test_load_embedding_model_defaults_when_block_missing(monkeypatch):
    """No `embedding:` block (and `None` config) → safe defaults."""
    captured = {}

    def fake_st(model_name, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return _FakeModel()

    monkeypatch.setattr(
        "compare_results_dictionary.SentenceTransformer", fake_st
    )

    # Empty config.
    model, encode_kwargs = _load_embedding_model({})
    assert isinstance(model, _FakeModel)
    assert captured["model_name"] == "all-MiniLM-L6-v2"
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["cache_folder"] is None
    assert encode_kwargs == {"normalize_embeddings": False, "batch_size": 32}

    # `config=None` is also tolerated (defensive).
    captured.clear()
    _load_embedding_model(None)
    assert captured["model_name"] == "all-MiniLM-L6-v2"

    # Explicit empty mapping.
    captured.clear()
    _load_embedding_model({"embedding": {}})
    assert captured["model_name"] == "all-MiniLM-L6-v2"


def test_load_embedding_model_uses_config_values(monkeypatch):
    """All fields in the `embedding:` block are honored."""
    captured = {}

    def fake_st(model_name, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return _FakeModel()

    monkeypatch.setattr(
        "compare_results_dictionary.SentenceTransformer", fake_st
    )

    config = {
        "embedding": {
            "model_name": "BAAI/bge-small-en-v1.5",
            "device": "mps",
            "cache_dir": "/tmp/cache",
            "normalize_embeddings": True,
            "batch_size": 64,
        }
    }
    model, encode_kwargs = _load_embedding_model(config)
    assert captured["model_name"] == "BAAI/bge-small-en-v1.5"
    assert captured["kwargs"]["device"] == "mps"
    assert captured["kwargs"]["cache_folder"] == "/tmp/cache"
    assert encode_kwargs == {"normalize_embeddings": True, "batch_size": 64}


def test_load_embedding_model_rejects_non_mapping(monkeypatch):
    """Any non-mapping, non-None value under `embedding:` → ValueError.

    `None` is treated as "absent" (matches YAML/JSON `null`/`~` convention)
    and falls back to defaults; all other scalars/sequences are rejected.
    Covers str, bytes, int, float, bool, list, tuple, set.
    """
    calls = {"n": 0}

    def fake_st(*a, **kw):
        calls["n"] += 1
        return _FakeModel()

    monkeypatch.setattr(
        "compare_results_dictionary.SentenceTransformer", fake_st
    )

    bad_values = [
        "BAAI/bge-small-en-v1.5",   # str
        b"bytes-not-allowed",       # bytes
        42,                         # int
        3.14,                       # float
        True,                       # bool — must raise (not coalesce to {})
        False,                      # bool — also must raise
        ["a", "b"],                 # list
        ("a", "b"),                 # tuple
        {"a", "b"},                 # set
    ]
    for bad in bad_values:
        with pytest.raises(ValueError, match="must be a mapping"):
            _load_embedding_model({"embedding": bad})

    # Constructor must not have been called for any of the bad payloads.
    assert calls["n"] == 0

    # Positive control #1: a real dict is accepted (falls back to defaults).
    _load_embedding_model({"embedding": {}})
    assert calls["n"] == 1

    # Positive control #2: `None` (YAML/JSON null) is treated as "absent"
    # and silently uses defaults — same effect as omitting the key entirely.
    _load_embedding_model({"embedding": None})
    assert calls["n"] == 2


def test_load_embedding_model_handles_missing_cache_folder_kwarg(monkeypatch):
    """Older sentence-transformers without `cache_folder` → fallback branch."""
    calls = {"n": 0}

    def fake_st(model_name, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TypeError("unexpected kwarg cache_folder")
        # Second call (no cache_folder) records the kwargs for assertion.
        calls.setdefault("second_kwargs", kwargs)
        return _FakeModel()

    monkeypatch.setattr(
        "compare_results_dictionary.SentenceTransformer", fake_st
    )

    config = {"embedding": {"model_name": "X", "cache_dir": "/tmp"}}
    model, _ = _load_embedding_model(config)
    assert calls["n"] == 2
    assert isinstance(model, _FakeModel)
    # On the retry path `cache_folder` is NOT forwarded.
    assert "cache_folder" not in calls["second_kwargs"]


def test_load_embedding_model_logs_summary(monkeypatch, caplog):
    """Log line includes model_name, device, normalize, batch_size, cache_dir."""
    import logging

    monkeypatch.setattr(
        "compare_results_dictionary.SentenceTransformer",
        lambda *a, **kw: _FakeModel(),
    )
    config = {
        "embedding": {
            "model_name": "BAAI/bge-small-en-v1.5",
            "device": "cpu",
            "cache_dir": "/tmp/c",
            "normalize_embeddings": True,
            "batch_size": 16,
        }
    }
    with caplog.at_level(logging.INFO, logger="compare_results_dictionary"):
        _load_embedding_model(config)
    msg = " ".join(rec.getMessage() for rec in caplog.records)
    assert "BAAI/bge-small-en-v1.5" in msg
    assert "cpu" in msg
    assert "normalize=True" in msg
    assert "batch_size=16" in msg
    assert "/tmp/c" in msg
