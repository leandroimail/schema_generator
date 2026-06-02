"""Unit tests for run.py orchestrator subcommands.

These tests use monkeypatching to avoid real subprocess calls or LLM traffic.
"""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run  # noqa: E402


def _write_json(p: Path, payload: dict) -> None:
    p.write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def test_main_requires_subcommand():
    with pytest.raises(SystemExit):
        run.main()


@pytest.mark.parametrize(
    "argv,marker_key,marker_value",
    [
        (["run.py", "validate"], "validate", True),
        (["run.py", "retry-llm", "--dry-run"], "retry_dry_run", True),
        (["run.py", "compare"], "compare", True),
        (["run.py", "all", "--with-retry"], "all_with_retry", True),
        (["run.py", "bootstrap", "--update-config"], "bootstrap", True),
        (["run.py", "generate"], "generate", True),
        (["run.py", "generate", "--retry-errors"], "generate_retry", True),
        (["run.py", "generate", "--list-errors"], "generate_list", True),
        (["run.py", "pipeline", "--with-generate"], "pipeline_generate", True),
        (["run.py", "pipeline", "--with-bootstrap", "--with-generate", "--with-retry"], "pipeline_full", True),
    ],
)
def test_main_dispatches_subcommand(monkeypatch, argv, marker_key, marker_value):
    called: dict = {}

    def fake_func(args):
        called[marker_key] = marker_value
        return 0

    monkeypatch.setattr(run, "cmd_validate", fake_func)
    monkeypatch.setattr(run, "cmd_retry_llm", fake_func)
    monkeypatch.setattr(run, "cmd_compare", fake_func)
    monkeypatch.setattr(run, "cmd_all", fake_func)
    monkeypatch.setattr(run, "cmd_bootstrap", fake_func)
    monkeypatch.setattr(run, "cmd_generate", fake_func)
    monkeypatch.setattr(run, "cmd_pipeline", fake_func)

    monkeypatch.setattr("sys.argv", argv)
    rc = run.main()
    assert rc == 0
    assert called.get(marker_key) is True


def test_main_dispatches_validate(monkeypatch):
    called = {}
    monkeypatch.setattr(
        run, "cmd_validate", lambda args: (called.setdefault("validate", True), 0)[1]
    )
    monkeypatch.setattr("sys.argv", ["run.py", "validate"])
    rc = run.main()
    assert rc == 0
    assert called.get("validate") is True


def test_main_dispatches_retry_llm(monkeypatch):
    called = {}
    monkeypatch.setattr(
        run,
        "cmd_retry_llm",
        lambda args: (called.setdefault(args.dry_run, True), 0)[1],
    )
    monkeypatch.setattr("sys.argv", ["run.py", "retry-llm", "--dry-run"])
    rc = run.main()
    assert rc == 0
    assert called.get(True) is True


def test_main_dispatches_compare(monkeypatch):
    called = {}
    monkeypatch.setattr(
        run, "cmd_compare", lambda args: (called.setdefault("compare", True), 0)[1]
    )
    monkeypatch.setattr("sys.argv", ["run.py", "compare"])
    rc = run.main()
    assert rc == 0
    assert called.get("compare") is True


def test_main_dispatches_all(monkeypatch):
    called = {}
    monkeypatch.setattr(
        run,
        "cmd_all",
        lambda args: (called.setdefault("with_retry", args.with_retry), 0)[1],
    )
    monkeypatch.setattr("sys.argv", ["run.py", "all", "--with-retry"])
    rc = run.main()
    assert rc == 0
    assert called.get("with_retry") is True


def test_main_dispatches_bootstrap(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        run,
        "cmd_bootstrap",
        lambda args: (captured.setdefault("update_config", args.update_config), 0)[1],
    )
    monkeypatch.setattr("sys.argv", ["run.py", "bootstrap", "--update-config"])
    rc = run.main()
    assert rc == 0
    assert captured.get("update_config") is True


def test_main_dispatches_generate(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        run,
        "cmd_generate",
        lambda args: (
            captured.setdefault("retry_errors", args.retry_errors),
            captured.setdefault("list_errors", args.list_errors),
            0,
        )[-1],
    )
    monkeypatch.setattr("sys.argv", ["run.py", "generate", "--retry-errors"])
    rc = run.main()
    assert rc == 0
    assert captured.get("retry_errors") is True
    assert captured.get("list_errors") is False


def test_main_dispatches_pipeline(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        run,
        "cmd_pipeline",
        lambda args: (
            captured.setdefault("with_bootstrap", args.with_bootstrap),
            captured.setdefault("with_generate", args.with_generate),
            captured.setdefault("with_retry", args.with_retry),
            0,
        )[-1],
    )
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "pipeline", "--with-bootstrap", "--with-generate", "--with-retry"],
    )
    rc = run.main()
    assert rc == 0
    assert captured == {
        "with_bootstrap": True,
        "with_generate": True,
        "with_retry": True,
    }


# ---------------------------------------------------------------------------
# cmd_validate
# ---------------------------------------------------------------------------


def test_cmd_validate_reports_ok_and_errors(tmp_path, monkeypatch, capsys):
    data_llm_dir = tmp_path / "data" / "llm_results" / "dictionary_llm_results"
    (data_llm_dir / "tab_a" / "json").mkdir(parents=True)
    (data_llm_dir / "tab_b" / "json").mkdir(parents=True)
    _write_json(
        data_llm_dir / "tab_a" / "json" / "openai_small_gpt-5.4-mini_parsed.json",
        {"table_name": "tab_a", "fields": []},
    )
    _write_json(
        data_llm_dir / "tab_b" / "json" / "openai_small_gpt-5.4-nano_parsed.json",
        {"error": "rate limit"},
    )

    cfg = {
        "data_llm_results_dictionary_generation": {
            "path": str(data_llm_dir),
        }
    }
    monkeypatch.setattr(run, "load_config", lambda: cfg)

    rc = run.cmd_validate(argparse.Namespace())
    captured = capsys.readouterr()
    assert rc == 0
    assert "OK:  1" in captured.out
    assert "ERR: 1" in captured.out
    assert "rate limit" in captured.out


def test_cmd_validate_handles_malformed_json(tmp_path, monkeypatch, capsys):
    data_llm_dir = tmp_path / "data" / "llm_results" / "dictionary_llm_results"
    (data_llm_dir / "tab" / "json").mkdir(parents=True)
    (data_llm_dir / "tab" / "json" / "bad_parsed.json").write_text("{not valid")

    cfg = {
        "data_llm_results_dictionary_generation": {
            "path": str(data_llm_dir),
        }
    }
    monkeypatch.setattr(run, "load_config", lambda: cfg)

    rc = run.cmd_validate(argparse.Namespace())
    captured = capsys.readouterr()
    assert rc == 0
    assert "OK:  0" in captured.out
    assert "ERR: 1" in captured.out
    assert "parse error" in captured.out


def test_cmd_validate_handles_missing_config_dir(tmp_path, monkeypatch, capsys):
    cfg = {
        "data_llm_results_dictionary_generation": {
            "path": str(tmp_path / "does" / "not" / "exist"),
        }
    }
    monkeypatch.setattr(run, "load_config", lambda: cfg)

    rc = run.cmd_validate(argparse.Namespace())
    captured = capsys.readouterr()
    assert rc == 0
    assert "Found 0 parsed JSON files" in captured.out
    assert "OK:  0" in captured.out
    assert "ERR: 0" in captured.out


# ---------------------------------------------------------------------------
# cmd_compare (summary printing only)
# ---------------------------------------------------------------------------


def test_cmd_compare_prints_summary_on_success(tmp_path, monkeypatch, capsys):
    summary = {
        "results": {"tab1": {}, "tab2": {}},
        "average_by_model": {
            "deepseek-v4-flash": 0.61,
            "gpt-5.4-mini": 0.65,
        },
    }
    out_dir = tmp_path / "data" / "distance_calculation"
    out_dir.mkdir(parents=True)
    (out_dir / "all_similarities_results.json").write_text(json.dumps(summary))
    monkeypatch.setattr(run, "PROJECT_ROOT", tmp_path)

    fake_proc = MagicMock(returncode=0)
    monkeypatch.setattr(run.subprocess, "call", lambda *a, **kw: fake_proc.returncode)

    rc = run.cmd_compare(argparse.Namespace())
    captured = capsys.readouterr()
    assert rc == 0
    assert "Tables: 2" in captured.out
    assert "deepseek-v4-flash: 0.6100" in captured.out
    assert "gpt-5.4-mini: 0.6500" in captured.out


def test_cmd_compare_skips_summary_when_subprocess_fails(tmp_path, monkeypatch, capsys):
    fake_proc = MagicMock(returncode=1)
    monkeypatch.setattr(run.subprocess, "call", lambda *a, **kw: fake_proc.returncode)
    monkeypatch.setattr(run, "PROJECT_ROOT", tmp_path)

    rc = run.cmd_compare(argparse.Namespace())
    captured = capsys.readouterr()
    assert rc == 1
    # And no crash if the summary file is missing.
    assert "=== Summary ===" not in captured.out


def test_cmd_compare_skips_summary_when_file_missing(tmp_path, monkeypatch, capsys):
    fake_proc = MagicMock(returncode=0)
    monkeypatch.setattr(run.subprocess, "call", lambda *a, **kw: fake_proc.returncode)
    monkeypatch.setattr(run, "PROJECT_ROOT", tmp_path)

    rc = run.cmd_compare(argparse.Namespace())
    captured = capsys.readouterr()
    assert rc == 0
    assert "=== Summary ===" not in captured.out


# ---------------------------------------------------------------------------
# cmd_retry_llm
# ---------------------------------------------------------------------------


def test_cmd_retry_llm_dry_run_uses_list_errors(monkeypatch):
    captured = {}
    fake_proc = MagicMock(returncode=0)

    def fake_call(cmd, **kw):
        captured["cmd"] = cmd
        return fake_proc.returncode

    monkeypatch.setattr(run.subprocess, "call", fake_call)
    monkeypatch.setattr(run, "find_python", lambda: "/fake/python")

    rc = run.cmd_retry_llm(argparse.Namespace(dry_run=True))
    assert rc == 0
    assert "--list-errors" in captured["cmd"]
    assert "--retry-errors" not in captured["cmd"]


def test_cmd_retry_llm_real_uses_retry_errors(monkeypatch):
    captured = {}
    fake_proc = MagicMock(returncode=0)

    def fake_call(cmd, **kw):
        captured["cmd"] = cmd
        return fake_proc.returncode

    monkeypatch.setattr(run.subprocess, "call", fake_call)
    monkeypatch.setattr(run, "find_python", lambda: "/fake/python")

    rc = run.cmd_retry_llm(argparse.Namespace(dry_run=False))
    assert rc == 0
    assert "--retry-errors" in captured["cmd"]
    assert "--list-errors" not in captured["cmd"]


# ---------------------------------------------------------------------------
# cmd_bootstrap
# ---------------------------------------------------------------------------


def test_cmd_bootstrap_invokes_script_with_defaults(monkeypatch):
    captured = {}
    fake_proc = MagicMock(returncode=0)

    def fake_call(cmd, **kw):
        captured["cmd"] = cmd
        captured["cwd"] = kw.get("cwd")
        return fake_proc.returncode

    monkeypatch.setattr(run.subprocess, "call", fake_call)
    monkeypatch.setattr(run, "find_python", lambda: "/fake/python")

    rc = run.cmd_bootstrap(
        argparse.Namespace(
            update_config=False,
            dataset_url=None,
            sample_size=None,
            profile_sample_size=None,
        )
    )
    assert rc == 0
    assert captured["cmd"][0] == "/fake/python"
    assert captured["cmd"][1].endswith("src/bootstrap_bird_mini_dev.py")
    assert captured["cwd"] == run.PROJECT_ROOT
    assert "--update-config" not in captured["cmd"]


def test_cmd_bootstrap_passes_update_config_flag(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        run.subprocess,
        "call",
        lambda cmd, **kw: (captured.setdefault("cmd", cmd), 0)[1],
    )
    monkeypatch.setattr(run, "find_python", lambda: "/fake/python")

    rc = run.cmd_bootstrap(
        argparse.Namespace(
            update_config=True,
            dataset_url=None,
            sample_size=None,
            profile_sample_size=None,
        )
    )
    assert rc == 0
    assert "--update-config" in captured["cmd"]


def test_cmd_bootstrap_passes_optional_overrides(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        run.subprocess,
        "call",
        lambda cmd, **kw: (captured.setdefault("cmd", cmd), 0)[1],
    )
    monkeypatch.setattr(run, "find_python", lambda: "/fake/python")

    rc = run.cmd_bootstrap(
        argparse.Namespace(
            update_config=True,
            dataset_url="https://example.com/zip",
            sample_size=42,
            profile_sample_size=500,
        )
    )
    assert rc == 0
    assert "--update-config" in captured["cmd"]
    assert "--dataset-url" in captured["cmd"]
    assert "https://example.com/zip" in captured["cmd"]
    assert "--sample-size" in captured["cmd"]
    assert "42" in captured["cmd"]
    assert "--profile-sample-size" in captured["cmd"]
    assert "500" in captured["cmd"]


def test_cmd_bootstrap_propagates_subprocess_failure(monkeypatch):
    monkeypatch.setattr(run.subprocess, "call", lambda *a, **kw: 7)
    monkeypatch.setattr(run, "find_python", lambda: "/fake/python")

    rc = run.cmd_bootstrap(
        argparse.Namespace(
            update_config=False,
            dataset_url=None,
            sample_size=None,
            profile_sample_size=None,
        )
    )
    assert rc == 7


# ---------------------------------------------------------------------------
# cmd_generate
# ---------------------------------------------------------------------------


def test_cmd_generate_default_invokes_script(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        run.subprocess,
        "call",
        lambda cmd, **kw: (captured.setdefault("cmd", cmd), 0)[1],
    )
    monkeypatch.setattr(run, "find_python", lambda: "/fake/python")

    rc = run.cmd_generate(argparse.Namespace(retry_errors=False, list_errors=False))
    assert rc == 0
    assert captured["cmd"][1].endswith("src/dictionary_generation.py")
    assert "--retry-errors" not in captured["cmd"]
    assert "--list-errors" not in captured["cmd"]


def test_cmd_generate_with_retry_errors(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        run.subprocess,
        "call",
        lambda cmd, **kw: (captured.setdefault("cmd", cmd), 0)[1],
    )
    monkeypatch.setattr(run, "find_python", lambda: "/fake/python")

    rc = run.cmd_generate(argparse.Namespace(retry_errors=True, list_errors=False))
    assert rc == 0
    assert "--retry-errors" in captured["cmd"]
    assert "--list-errors" not in captured["cmd"]


def test_cmd_generate_with_list_errors(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        run.subprocess,
        "call",
        lambda cmd, **kw: (captured.setdefault("cmd", cmd), 0)[1],
    )
    monkeypatch.setattr(run, "find_python", lambda: "/fake/python")

    rc = run.cmd_generate(argparse.Namespace(retry_errors=False, list_errors=True))
    assert rc == 0
    assert "--list-errors" in captured["cmd"]
    assert "--retry-errors" not in captured["cmd"]


def test_cmd_generate_retry_takes_precedence_over_list(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        run.subprocess,
        "call",
        lambda cmd, **kw: (captured.setdefault("cmd", cmd), 0)[1],
    )
    monkeypatch.setattr(run, "find_python", lambda: "/fake/python")

    rc = run.cmd_generate(argparse.Namespace(retry_errors=True, list_errors=True))
    assert rc == 0
    assert "--retry-errors" in captured["cmd"]
    assert "--list-errors" not in captured["cmd"]


# ---------------------------------------------------------------------------
# cmd_all (orchestration)
# ---------------------------------------------------------------------------


def test_cmd_all_runs_validate_then_compare(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run, "cmd_validate", lambda a: (calls.append("validate") or 0)
    )
    monkeypatch.setattr(
        run, "cmd_compare", lambda a: (calls.append("compare") or 0)
    )
    monkeypatch.setattr(
        run, "cmd_retry_llm", lambda a: (calls.append("retry") or 0)
    )
    rc = run.cmd_all(argparse.Namespace(with_retry=False))
    assert rc == 0
    assert calls == ["validate", "compare"]


def test_cmd_all_with_retry_includes_retry_step(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run, "cmd_validate", lambda a: (calls.append("validate") or 0)
    )
    monkeypatch.setattr(
        run, "cmd_compare", lambda a: (calls.append("compare") or 0)
    )
    monkeypatch.setattr(
        run, "cmd_retry_llm", lambda a: (calls.append("retry") or 0)
    )
    rc = run.cmd_all(argparse.Namespace(with_retry=True))
    assert rc == 0
    assert calls == ["validate", "retry", "compare"]


def test_cmd_all_short_circuits_on_validate_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "cmd_validate", lambda a: (calls.append("v") or 1))
    monkeypatch.setattr(run, "cmd_compare", lambda a: (calls.append("c") or 0))
    monkeypatch.setattr(run, "cmd_retry_llm", lambda a: (calls.append("r") or 0))
    rc = run.cmd_all(argparse.Namespace(with_retry=True))
    assert rc == 1
    assert "c" not in calls
    assert "r" not in calls


def test_cmd_all_short_circuits_on_retry_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "cmd_validate", lambda a: (calls.append("v") or 0))
    monkeypatch.setattr(run, "cmd_retry_llm", lambda a: (calls.append("r") or 5))
    monkeypatch.setattr(run, "cmd_compare", lambda a: (calls.append("c") or 0))
    rc = run.cmd_all(argparse.Namespace(with_retry=True))
    assert rc == 5
    assert "c" not in calls


# ---------------------------------------------------------------------------
# cmd_pipeline (full orchestration)
# ---------------------------------------------------------------------------


def test_cmd_pipeline_without_flags_invokes_only_all(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run, "cmd_bootstrap", lambda a: (calls.append("bootstrap") or 0)
    )
    monkeypatch.setattr(
        run, "cmd_generate", lambda a: (calls.append("generate") or 0)
    )
    monkeypatch.setattr(run, "cmd_all", lambda a: (calls.append("all") or 0))
    rc = run.cmd_pipeline(argparse.Namespace(with_bootstrap=False, with_generate=False, with_retry=False))
    assert rc == 0
    assert calls == ["all"]


def test_cmd_pipeline_with_bootstrap_and_generate(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run, "cmd_bootstrap", lambda a: (calls.append("bootstrap") or 0)
    )
    monkeypatch.setattr(
        run, "cmd_generate", lambda a: (calls.append("generate") or 0)
    )
    monkeypatch.setattr(run, "cmd_all", lambda a: (calls.append("all") or 0))
    rc = run.cmd_pipeline(argparse.Namespace(with_bootstrap=True, with_generate=True, with_retry=False))
    assert rc == 0
    assert calls == ["bootstrap", "generate", "all"]


def test_cmd_pipeline_short_circuits_on_bootstrap_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run, "cmd_bootstrap", lambda a: (calls.append("bootstrap") or 3)
    )
    monkeypatch.setattr(
        run, "cmd_generate", lambda a: (calls.append("generate") or 0)
    )
    monkeypatch.setattr(run, "cmd_all", lambda a: (calls.append("all") or 0))
    rc = run.cmd_pipeline(argparse.Namespace(with_bootstrap=True, with_generate=True, with_retry=False))
    assert rc == 3
    assert "generate" not in calls
    assert "all" not in calls


def test_cmd_pipeline_short_circuits_on_generate_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run, "cmd_bootstrap", lambda a: (calls.append("bootstrap") or 0)
    )
    monkeypatch.setattr(
        run, "cmd_generate", lambda a: (calls.append("generate") or 4)
    )
    monkeypatch.setattr(run, "cmd_all", lambda a: (calls.append("all") or 0))
    rc = run.cmd_pipeline(argparse.Namespace(with_bootstrap=True, with_generate=True, with_retry=False))
    assert rc == 4
    assert calls == ["bootstrap", "generate"]


def test_cmd_pipeline_passes_with_retry_to_all(monkeypatch):
    captured = {}

    def fake_all(args):
        captured["with_retry"] = args.with_retry
        return 0

    monkeypatch.setattr(run, "cmd_bootstrap", lambda a: 0)
    monkeypatch.setattr(run, "cmd_generate", lambda a: 0)
    monkeypatch.setattr(run, "cmd_all", fake_all)
    rc = run.cmd_pipeline(argparse.Namespace(with_bootstrap=True, with_generate=True, with_retry=True))
    assert rc == 0
    assert captured["with_retry"] is True


def test_cmd_pipeline_bootstrap_uses_update_config(monkeypatch):
    captured = {}

    def fake_bootstrap(args):
        captured["update_config"] = args.update_config
        return 0

    monkeypatch.setattr(run, "cmd_bootstrap", fake_bootstrap)
    monkeypatch.setattr(run, "cmd_generate", lambda a: 0)
    monkeypatch.setattr(run, "cmd_all", lambda a: 0)
    rc = run.cmd_pipeline(argparse.Namespace(with_bootstrap=True, with_generate=True, with_retry=False))
    assert rc == 0
    assert captured["update_config"] is True


# ---------------------------------------------------------------------------
# find_python
# ---------------------------------------------------------------------------


def test_find_python_prefers_venv_python(tmp_path, monkeypatch):
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    monkeypatch.setattr(run, "PROJECT_ROOT", tmp_path)
    assert run.find_python() == str(venv_python)


def test_find_python_falls_back_to_sys_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "PROJECT_ROOT", tmp_path)
    assert run.find_python() == sys.executable


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_reads_yaml(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("a: 1\nb: hello\n")
    monkeypatch.setattr(run, "CONFIG_PATH", cfg_path)
    cfg = run.load_config()
    assert cfg == {"a": 1, "b": "hello"}
