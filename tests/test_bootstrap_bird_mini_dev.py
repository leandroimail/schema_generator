import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bootstrap_bird_mini_dev import (
    build_full_description,
    infer_domain_values,
    quote_identifier,
    safe_name,
)


def test_safe_name_normalizes_database_and_table_names():
    assert safe_name("European Football 2.Match") == "european_football_2_match"
    assert safe_name("Team_Attributes") == "team_attributes"


def test_quote_identifier_escapes_double_quotes():
    assert quote_identifier('weird"name') == '"weird""name"'


def test_build_full_description_prefers_domain_values_over_example():
    field = {
        "field_description": "Status flag.",
        "example_value": "A",
        "domain_values": ["A", "B"],
    }

    assert build_full_description(field) == 'Status flag. - Domain: ["A", "B"]'


def test_infer_domain_values_for_low_cardinality_column():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sample (status TEXT)")
    conn.executemany(
        "INSERT INTO sample (status) VALUES (?)",
        [("A",), ("A",), ("B",), (None,)],
    )

    values = infer_domain_values(
        conn,
        table_name="sample",
        field_name="status",
        row_count=4,
        max_distinct=5,
        max_ratio=1.0,
    )

    assert values == ["A", "B"]
