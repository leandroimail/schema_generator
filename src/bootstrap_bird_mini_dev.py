#!/usr/bin/env python3
"""Bootstrap BIRD Mini-Dev metadata into this project's dictionary pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import pandas as pd
import yaml


DEFAULT_DATASET_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
SYSTEM_TABLE_PREFIXES = ("sqlite_",)
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_PROFILE_SAMPLE_SIZE = 10000
DEFAULT_DOMAIN_MAX_DISTINCT = 30
DEFAULT_DOMAIN_MAX_RATIO = 0.2


@dataclass(frozen=True)
class BirdTable:
    db_id: str
    table_name: str
    sqlite_path: Path
    description_path: Path | None

    @property
    def artifact_name(self) -> str:
        return f"bird__{safe_name(self.db_id)}__{safe_name(self.table_name)}"


def safe_name(value: str) -> str:
    """Return a filesystem- and config-friendly identifier."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_minidev_available(dataset_root: Path, zip_path: Path, dataset_url: str) -> Path:
    """Ensure the extracted MINIDEV directory exists and return its path."""
    minidev_dir = dataset_root / "minidev" / "MINIDEV"
    if minidev_dir.exists():
        return minidev_dir

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"Downloading BIRD Mini-Dev from {dataset_url} to {zip_path}")
        urlretrieve(dataset_url, zip_path)

    print(f"Extracting MINIDEV files from {zip_path} to {dataset_root}")
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.startswith("minidev/MINIDEV/")]
        archive.extractall(dataset_root, members)

    if not minidev_dir.exists():
        raise FileNotFoundError(f"Expected MINIDEV directory not found: {minidev_dir}")
    return minidev_dir


def discover_tables(minidev_dir: Path) -> list[BirdTable]:
    """Discover all SQLite tables in all Mini-Dev databases."""
    db_root = minidev_dir / "dev_databases"
    if not db_root.exists():
        raise FileNotFoundError(f"BIRD dev_databases directory not found: {db_root}")

    tables: list[BirdTable] = []
    for sqlite_path in sorted(db_root.glob("*/*.sqlite")):
        db_id = sqlite_path.stem
        with sqlite3.connect(sqlite_path) as conn:
            rows = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()

        description_dir = sqlite_path.parent / "database_description"
        descriptions = {
            path.stem.casefold(): path
            for path in description_dir.glob("*.csv")
        } if description_dir.exists() else {}

        for (table_name,) in rows:
            if table_name.startswith(SYSTEM_TABLE_PREFIXES):
                continue
            tables.append(
                BirdTable(
                    db_id=db_id,
                    table_name=table_name,
                    sqlite_path=sqlite_path,
                    description_path=descriptions.get(table_name.casefold()),
                )
            )
    return tables


def load_dev_tables(minidev_dir: Path) -> dict[str, dict[str, Any]]:
    path = minidev_dir / "dev_tables.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["db_id"]: item for item in data}


def load_table_description(path: Path | None) -> dict[str, dict[str, str]]:
    """Load BIRD database_description CSV keyed by original column name."""
    if path is None or not path.exists():
        return {}

    text = read_text_with_encoding_fallback(path)
    reader = csv.DictReader(text.splitlines())
    rows = list(reader)

    result: dict[str, dict[str, str]] = {}
    for row in rows:
        original = clean_text(row.get("original_column_name", ""))
        if not original:
            continue
        result[original.casefold()] = {key: clean_text(value) for key, value in row.items()}
    return result


def read_text_with_encoding_fallback(path: Path) -> str:
    """Read BIRD metadata CSVs, which are not consistently UTF-8 encoded."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    try:
        from charset_normalizer import from_bytes
    except ImportError:
        return raw.decode("latin-1", errors="replace")

    best = from_bytes(raw).best()
    if best is None:
        return raw.decode("latin-1", errors="replace")
    return str(best)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\ufeff", "")).strip()


def sqlite_table_info(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(table_name)})").fetchall()
    return [
        {
            "cid": row[0],
            "name": row[1],
            "type": row[2] or "TEXT",
            "notnull": bool(row[3]),
            "default": row[4],
            "pk": row[5],
        }
        for row in rows
    ]


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def get_row_count(conn: sqlite3.Connection, table_name: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {quote_identifier(table_name)}").fetchone()[0])


def get_sample_df(conn: sqlite3.Connection, table_name: str, sample_size: int) -> pd.DataFrame:
    row_count = get_row_count(conn, table_name)
    limit = min(sample_size, row_count)
    if limit <= 0:
        return pd.DataFrame()
    return pd.read_sql_query(
        f"SELECT * FROM {quote_identifier(table_name)} ORDER BY RANDOM() LIMIT {limit}",
        conn,
    )


def get_example_value(conn: sqlite3.Connection, table_name: str, field_name: str) -> Any:
    row = conn.execute(
        f"""
        SELECT {quote_identifier(field_name)}
        FROM {quote_identifier(table_name)}
        WHERE {quote_identifier(field_name)} IS NOT NULL
        LIMIT 1
        """
    ).fetchone()
    return row[0] if row else None


def infer_domain_values(
    conn: sqlite3.Connection,
    table_name: str,
    field_name: str,
    row_count: int,
    max_distinct: int,
    max_ratio: float,
) -> list[Any] | None:
    if row_count == 0:
        return None

    count_sql = (
        f"SELECT COUNT(DISTINCT {quote_identifier(field_name)}) "
        f"FROM {quote_identifier(table_name)} "
        f"WHERE {quote_identifier(field_name)} IS NOT NULL"
    )
    distinct_count = int(conn.execute(count_sql).fetchone()[0])
    if distinct_count == 0 or distinct_count > max_distinct:
        return None
    if row_count > max_distinct and (distinct_count / row_count) > max_ratio:
        return None

    values_sql = (
        f"SELECT DISTINCT {quote_identifier(field_name)} "
        f"FROM {quote_identifier(table_name)} "
        f"WHERE {quote_identifier(field_name)} IS NOT NULL "
        f"ORDER BY {quote_identifier(field_name)} "
        f"LIMIT {max_distinct}"
    )
    return [row[0] for row in conn.execute(values_sql).fetchall()]


def bird_column_aliases(dev_table: dict[str, Any], table_name: str) -> dict[str, str]:
    """Map original column names to normalized BIRD column labels from dev_tables.json."""
    aliases: dict[str, str] = {}
    table_names = dev_table.get("table_names_original", [])
    try:
        table_idx = next(
            idx for idx, original in enumerate(table_names)
            if str(original).casefold() == table_name.casefold()
        )
    except StopIteration:
        return aliases

    originals = dev_table.get("column_names_original", [])
    normalized = dev_table.get("column_names", [])
    for original_item, normalized_item in zip(originals, normalized):
        if not isinstance(original_item, list) or not isinstance(normalized_item, list):
            continue
        if len(original_item) < 2 or len(normalized_item) < 2:
            continue
        if original_item[0] == table_idx and original_item[1] != "*":
            aliases[str(original_item[1]).casefold()] = clean_text(normalized_item[1])
    return aliases


def build_field_description(
    column: dict[str, Any],
    bird_description: dict[str, str],
    bird_alias: str,
    role_parts: list[str],
) -> str:
    description = clean_text(bird_description.get("column_description", ""))
    column_label = clean_text(bird_description.get("column_name", "")) or bird_alias
    value_description = clean_text(bird_description.get("value_description", ""))

    parts = []
    if description:
        parts.append(description)
    elif column_label and column_label.casefold() != column["name"].casefold():
        parts.append(f"{column['name']} represents {column_label}.")
    else:
        parts.append(f"{column['name']} column from the {column.get('table_name', 'table')} table.")

    if value_description:
        parts.append(f"Value notes: {value_description}")
    if role_parts:
        parts.append("Relational role: " + "; ".join(role_parts) + ".")
    return " ".join(parts)


def build_full_description(field: dict[str, Any]) -> str:
    parts = [field.get("field_description", "")]
    if field.get("domain_values"):
        parts.append(f"Domain: {json.dumps(field['domain_values'], ensure_ascii=False)}")
    elif field.get("example_value") is not None:
        parts.append(f"Example: {field['example_value']}")
    return " - ".join(part for part in parts if part)


def foreign_key_map(conn: sqlite3.Connection, table_name: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in conn.execute(f"PRAGMA foreign_key_list({quote_identifier(table_name)})").fetchall():
        from_col = row[3]
        target = f"{row[2]}.{row[4]}"
        result.setdefault(from_col, []).append(target)
    return result


def build_dictionary(
    table: BirdTable,
    dev_table: dict[str, Any],
    sample_size: int,
    domain_max_distinct: int,
    domain_max_ratio: float,
) -> dict[str, Any]:
    descriptions = load_table_description(table.description_path)
    aliases = bird_column_aliases(dev_table, table.table_name)

    with sqlite3.connect(table.sqlite_path) as conn:
        row_count = get_row_count(conn, table.table_name)
        fk_map = foreign_key_map(conn, table.table_name)
        columns = sqlite_table_info(conn, table.table_name)

        fields = []
        for column in columns:
            column["table_name"] = table.table_name
            name = column["name"]
            bird_description = descriptions.get(name.casefold(), {})
            role_parts = []
            if column["pk"]:
                role_parts.append("primary key")
            if name in fk_map:
                role_parts.append("foreign key to " + ", ".join(fk_map[name]))
            if column["notnull"]:
                role_parts.append("not null")

            field: dict[str, Any] = {
                "field_name": name,
                "data_type": clean_text(bird_description.get("data_format", "")) or column["type"],
                "field_description": build_field_description(
                    column=column,
                    bird_description=bird_description,
                    bird_alias=aliases.get(name.casefold(), ""),
                    role_parts=role_parts,
                ),
                "example_value": get_example_value(conn, table.table_name, name),
            }
            domain_values = infer_domain_values(
                conn,
                table.table_name,
                name,
                row_count,
                domain_max_distinct,
                domain_max_ratio,
            )
            if domain_values is not None:
                field["domain_values"] = domain_values
            field["full_description"] = build_full_description(field)
            fields.append(field)

    return {
        "table_name": table.table_name,
        "table_description": (
            f"Table {table.table_name} from BIRD Mini-Dev database {table.db_id}. "
            f"Metadata is enriched from BIRD database_description CSV files and SQLite introspection."
        ),
        "fields": fields,
    }


def build_profile(table: BirdTable, sample_df: pd.DataFrame, dictionary: dict[str, Any], row_count: int) -> dict[str, Any]:
    fields_by_name = {field["field_name"]: field for field in dictionary["fields"]}
    columns = []
    for column_name in sample_df.columns:
        series = sample_df[column_name]
        non_null = series.dropna()
        top_values = Counter(non_null.astype(str).head(1000)).most_common(10)
        columns.append(
            {
                "column_name": column_name,
                "data_type": fields_by_name.get(column_name, {}).get("data_type", str(series.dtype)),
                "sample_inferred_dtype": str(series.dtype),
                "null_count_in_sample": int(series.isna().sum()),
                "non_null_count_in_sample": int(non_null.shape[0]),
                "distinct_count_in_sample": int(non_null.nunique(dropna=True)),
                "example_value": fields_by_name.get(column_name, {}).get("example_value"),
                "top_values_in_sample": [
                    {"value": value, "count": count}
                    for value, count in top_values
                ],
            }
        )

    return {
        "source": "BIRD Mini-Dev SQLite introspection plus sampled table data",
        "database": table.db_id,
        "table_name": table.table_name,
        "row_count": row_count,
        "sample_row_count": int(sample_df.shape[0]),
        "columns": columns,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def update_config(config_path: Path, manifest: list[dict[str, Any]], outputs: dict[str, Path]) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    config["bird_mini_dev"] = {
        "dataset_root": str(outputs["dataset_root"]),
        "manifest_path": str(outputs["manifest_path"]),
        "dictionary_dir": str(outputs["dictionary_dir"]),
        "sample_dir": str(outputs["sample_dir"]),
        "profile_dir": str(outputs["profile_dir"]),
    }
    config["list_of_data_dictionaries"] = [
        {"name": item["artifact_name"], "path": item["dictionary_path"]}
        for item in manifest
    ]
    config["list_of_data_samples"] = [
        {"name": item["artifact_name"], "path": item["sample_path"]}
        for item in manifest
    ]
    config["list_of_data_samples_profiles"] = [
        {"name": item["artifact_name"], "path": item["profile_sample_path"]}
        for item in manifest
    ]
    config["list_of_profiles"] = [
        {"name": item["artifact_name"], "path": item["profile_path"]}
        for item in manifest
    ]
    config["list_full_data_samples"] = [
        {"name": item["artifact_name"], "path": item["profile_sample_path"]}
        for item in manifest
    ]
    config["list_of_data_in_codebase"] = [
        {"name": f"{item['db_id']}.{item['table_name']}"}
        for item in manifest
    ]

    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def validate_manifest(manifest: list[dict[str, Any]]) -> None:
    required_top = {"table_name", "table_description", "fields"}
    required_field = {"field_name", "data_type", "field_description", "full_description"}
    for item in manifest:
        dictionary = json.loads(Path(item["dictionary_path"]).read_text(encoding="utf-8"))
        missing_top = required_top - set(dictionary)
        if missing_top:
            raise ValueError(f"{item['dictionary_path']} missing keys: {sorted(missing_top)}")
        if not isinstance(dictionary["fields"], list) or not dictionary["fields"]:
            raise ValueError(f"{item['dictionary_path']} has no fields")
        for field in dictionary["fields"]:
            missing_field = required_field - set(field)
            if missing_field:
                raise ValueError(
                    f"{item['dictionary_path']} field {field.get('field_name')} "
                    f"missing keys: {sorted(missing_field)}"
                )
        for path_key in ("sample_path", "profile_sample_path", "profile_path"):
            path = Path(item[path_key])
            if not path.exists():
                raise FileNotFoundError(f"Missing generated artifact: {path}")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = project_root()
    dataset_root = (root / args.dataset_root).resolve()
    zip_path = (root / args.zip_path).resolve()
    output_root = root / args.output_root
    dictionary_dir = root / args.dictionary_dir
    sample_dir = output_root / "samples"
    profile_sample_dir = output_root / "profile_samples"
    profile_dir = output_root / "profiles"
    manifest_path = output_root / "manifest.json"

    minidev_dir = ensure_minidev_available(dataset_root, zip_path, args.dataset_url)
    dev_tables = load_dev_tables(minidev_dir)
    tables = discover_tables(minidev_dir)
    if not tables:
        raise RuntimeError(f"No SQLite tables discovered under {minidev_dir}")

    manifest: list[dict[str, Any]] = []
    for index, table in enumerate(tables, start=1):
        print(f"[{index}/{len(tables)}] Processing {table.db_id}.{table.table_name}")
        dictionary = build_dictionary(
            table=table,
            dev_table=dev_tables.get(table.db_id, {}),
            sample_size=args.sample_size,
            domain_max_distinct=args.domain_max_distinct,
            domain_max_ratio=args.domain_max_ratio,
        )

        with sqlite3.connect(table.sqlite_path) as conn:
            row_count = get_row_count(conn, table.table_name)
            sample_df = get_sample_df(conn, table.table_name, args.sample_size)
            profile_sample_df = get_sample_df(conn, table.table_name, args.profile_sample_size)

        dictionary_path = dictionary_dir / f"{table.artifact_name}.json"
        sample_path = sample_dir / f"{table.artifact_name}_sample_{sample_df.shape[0]}.parquet"
        profile_sample_path = profile_sample_dir / (
            f"{table.artifact_name}_sample_{profile_sample_df.shape[0]}.parquet"
        )
        profile_path = profile_dir / f"{table.artifact_name}_profile.json"

        write_json(dictionary_path, dictionary)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.to_parquet(sample_path, index=False)
        profile_sample_path.parent.mkdir(parents=True, exist_ok=True)
        profile_sample_df.to_parquet(profile_sample_path, index=False)
        write_json(profile_path, build_profile(table, profile_sample_df, dictionary, row_count))

        manifest.append(
            {
                "artifact_name": table.artifact_name,
                "db_id": table.db_id,
                "table_name": table.table_name,
                "sqlite_path": str(table.sqlite_path.relative_to(root)),
                "description_path": (
                    str(table.description_path.relative_to(root))
                    if table.description_path else None
                ),
                "dictionary_path": str(dictionary_path.relative_to(root)),
                "sample_path": str(sample_path.relative_to(root)),
                "profile_sample_path": str(profile_sample_path.relative_to(root)),
                "profile_path": str(profile_path.relative_to(root)),
                "row_count": row_count,
                "field_count": len(dictionary["fields"]),
            }
        )

    write_json(manifest_path, manifest)
    validate_manifest(manifest)

    if args.update_config:
        update_config(
            root / args.config_path,
            manifest,
            {
                "dataset_root": Path(args.dataset_root),
                "manifest_path": manifest_path.relative_to(root),
                "dictionary_dir": dictionary_dir.relative_to(root),
                "sample_dir": sample_dir.relative_to(root),
                "profile_dir": profile_dir.relative_to(root),
            },
        )

    db_count = len({item["db_id"] for item in manifest})
    print(f"Generated artifacts for {len(manifest)} tables across {db_count} databases.")
    print(f"Manifest: {manifest_path.relative_to(root)}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL)
    parser.add_argument("--dataset-root", default="benchmark/bird_mini_dev")
    parser.add_argument("--zip-path", default="benchmark/bird_mini_dev/minidev.zip")
    parser.add_argument("--output-root", default="data/bird_mini_dev")
    parser.add_argument("--dictionary-dir", default="data/bird_mini_dev/dictionaries")
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--profile-sample-size", type=int, default=DEFAULT_PROFILE_SAMPLE_SIZE)
    parser.add_argument("--domain-max-distinct", type=int, default=DEFAULT_DOMAIN_MAX_DISTINCT)
    parser.add_argument("--domain-max-ratio", type=float, default=DEFAULT_DOMAIN_MAX_RATIO)
    parser.add_argument("--update-config", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
