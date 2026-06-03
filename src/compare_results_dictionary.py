import os
import yaml
import json
import csv
import glob
import logging
from typing import Dict, List, Any, Union, Tuple, TypedDict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingConfig(TypedDict, total=False):
    """Schema for the `embedding:` block in `config.yaml`.

    All fields are optional — the helper `_load_embedding_model` fills in
    safe defaults for anything missing. See `reports/EMBEDDING_MODEL.md` §4
    for the field-by-field rationale and the recommended override
    (`BAAI/bge-small-en-v1.5`).
    """

    model_name: str
    device: str
    cache_dir: str
    normalize_embeddings: bool
    batch_size: int


class EncodeKwargs(TypedDict):
    """Kwargs forwarded to every `model.encode(...)` call.

    Sourced from `EmbeddingConfig` (`normalize_embeddings`, `batch_size`).
    """

    normalize_embeddings: bool
    batch_size: int

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the YAML configuration file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary containing the configuration.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        raise

def _load_embedding_model(
    config: Dict[str, Any],
) -> Tuple[SentenceTransformer, EncodeKwargs]:
    """
    Builds a SentenceTransformer from the `embedding:` block of the config
    and the kwargs to forward to `model.encode(...)`.

    Falls back to safe defaults (`all-MiniLM-L6-v2`, `cpu`, no normalization,
    `batch_size=32`) so the script keeps working when the block is missing —
    important for older configs and the test suite.

    Args:
        config: Full config dict loaded by `load_config`.

    Returns:
        A `(model, encode_kwargs)` tuple. `encode_kwargs` is a TypedDict
        with `normalize_embeddings` and `batch_size`, splatted into every
        `model.encode(...)` call.
    """
    emb_cfg: EmbeddingConfig = (config or {}).get("embedding")  # type: ignore[assignment]
    if emb_cfg is None:
        emb_cfg = {}
    if not isinstance(emb_cfg, dict):
        raise ValueError(
            f"'embedding' must be a mapping, got {type(emb_cfg).__name__}"
        )

    model_name = emb_cfg.get("model_name", "all-MiniLM-L6-v2")
    device = emb_cfg.get("device", "cpu")
    cache_dir = emb_cfg.get("cache_dir")
    normalize_embeddings = bool(emb_cfg.get("normalize_embeddings", False))
    batch_size = int(emb_cfg.get("batch_size", 32))

    logger.info(
        f"Loading embedding model: {model_name} "
        f"(device={device}, normalize={normalize_embeddings}, "
        f"batch_size={batch_size}, cache_dir={cache_dir})"
    )

    try:
        model = SentenceTransformer(
            model_name, device=device, cache_folder=cache_dir
        )
    except TypeError:
        # Older sentence-transformers versions don't accept `cache_folder`.
        model = SentenceTransformer(model_name, device=device)

    encode_kwargs: EncodeKwargs = {
        "normalize_embeddings": normalize_embeddings,
        "batch_size": batch_size,
    }
    return model, encode_kwargs

def load_json_data(file_path: str) -> Dict[str, Any]:
    """
    Loads data from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Dictionary with the JSON data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}

def load_csv_data(file_path: str) -> Dict[str, Any]:
    """
    Loads data from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Dictionary with the CSV data.
    """
    try:
        data = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Assuming there is a column for the field and another for the description
                field_name = row.get('field_name', '')
                description = row.get('description', '')
                if field_name:
                    data[field_name] = description
        return data
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}")
        return {}

def load_data_dictionary(file_path: str) -> Dict[str, Any]:
    """
    Loads a data dictionary based on the file extension.

    Args:
        file_path: Path to the file.

    Returns:
        Dictionary with the data.
    """
    extension = os.path.splitext(file_path)[1].lower()
    if extension == '.json':
        return load_json_data(file_path)
    elif extension == '.csv':
        return load_csv_data(file_path)
    else:
        logger.warning(f"Unsupported file format: {file_path}")
        return {}

def find_all_files_in_directory(directory: str, extensions: List[str] = ['.json', '.csv']) -> List[str]:
    """
    Finds all files with the specified extensions in a directory and its subfolders.

    Args:
        directory: Root directory for search.
        extensions: List of file extensions to look for.

    Returns:
        List of found file paths.
    """
    try:
        all_files = []
        for ext in extensions:
            pattern = os.path.join(directory, f"**/*{ext}")
            all_files.extend(glob.glob(pattern, recursive=True))
        return all_files
    except Exception as e:
        logger.error(f"Error searching for files in {directory}: {e}")
        return []

def calculate_embeddings(
    data: Dict[str, Any],
    model: SentenceTransformer,
    encode_kwargs: Union[EncodeKwargs, None] = None,
) -> Dict[str, np.ndarray]:
    """
    Calculates embeddings for table_description and all full_descriptions in fields.

    Args:
        data: Dictionary containing table_description and fields.
        model: Pre-loaded SentenceTransformer model.
        encode_kwargs: Optional `EncodeKwargs` TypedDict forwarded to
            `model.encode(...)` (e.g. `{"normalize_embeddings": True,
            "batch_size": 32}`). When `None` the call is made with no extra
            kwargs, preserving the pre-config behavior.

    Returns:
        Dictionary with keys as 'table_description' and each field_name, values as their embeddings.
    """
    kwargs: Dict[str, Any] = dict(encode_kwargs) if encode_kwargs else {}
    embeddings: Dict[str, np.ndarray] = {}
    try:
        # Embed table_description if present
        table_desc: str = data.get("table_description", "")
        if isinstance(table_desc, str) and table_desc.strip():
            embeddings["table_description"] = model.encode(table_desc, **kwargs)
        else:
            logger.warning("No valid table_description found.")
            embeddings["table_description"] = model.encode("", **kwargs)

        # Embed each field's full_description using field_name as key
        fields = data.get("fields", [])
        if isinstance(fields, list):
            for field in fields:
                field_name = field.get("field_name", "")
                field_desc = field.get("full_description", "")
                if not field_desc:
                    field_desc = field.get("field_description", "")
                if field_name:
                    if isinstance(field_desc, str) and field_desc.strip():
                        embeddings[field_name] = model.encode(field_desc, **kwargs)
                    else:
                        logger.warning(f"Field {field_name} has empty or invalid full_description.")
                        embeddings[field_name] = model.encode("", **kwargs)
        else:
            logger.warning("No valid fields list found in data.")
    except Exception as e:
        logger.error(f"Error calculating embeddings: {e}")
    return embeddings

def calculate_similarities(baseline_emb: Dict[str, np.ndarray], llm_emb: Dict[str, np.ndarray]) -> List[Dict[str, Union[str, float]]]:
    """
    Calculates cosine similarities between corresponding embeddings.

    Args:
        baseline_emb: Dictionary of fields and baseline embeddings.
        llm_emb: Dictionary of fields and LLM embeddings.

    Returns:
        List of dictionaries with fields and their similarity scores.
    """
    similarities = []
    for field in baseline_emb:
        if field in llm_emb:
            try:
                # Calculate cosine similarity between the two embeddings
                sim_score = cosine_similarity(
                    baseline_emb[field].reshape(1, -1),
                    llm_emb[field].reshape(1, -1)
                )[0][0]
                similarities.append({
                    "field": field,
                    "score": float(sim_score)
                })
            except Exception as e:
                logger.error(f"Error calculating similarity for field {field}: {e}")
                similarities.append({
                    "field": field,
                    "score": 0.0
                })
    return similarities

def generate_output_json(table_name: str, model_name: str, similarities: List[Dict[str, Union[str, float]]]) -> Dict[str, Any]:
    """
    Generates the output JSON structure.

    Args:
        table_name: Table name.
        model_name: Model name used for comparison.
        similarities: List of calculated similarities.

    Returns:
        Dictionary formatted for JSON output.
    """
    return {
        "table_name": table_name,
        "par-compare-models": model_name,
        "similarities": similarities
    }


def compute_similarity_metrics(scores: List[float]) -> Dict[str, float]:
    """
    Compute distribution metrics for a list of similarity scores.

    Reported metrics:
        - ``mean``: arithmetic mean
        - ``std``: sample standard deviation (ddof=1); 0.0 when fewer than 2 samples
        - ``q25``: 25th percentile
        - ``median``: 50th percentile
        - ``q75``: 75th percentile
        - ``d90``: 90th percentile
        - ``d99``: 99th percentile
        - ``min``: minimum score
        - ``max``: maximum score
        - ``count``: number of scores used

    Non-finite scores (NaN, +/-inf) are dropped before computing the metrics.

    Args:
        scores: List of cosine similarity scores in the [-1, 1] range.

    Returns:
        Dictionary with the requested metrics. Empty input yields an empty dict.
    """
    if not scores:
        return {}

    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return {}

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {}

    metrics: Dict[str, float] = {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "q25": float(np.percentile(finite, 25)),
        "median": float(np.percentile(finite, 50)),
        "q75": float(np.percentile(finite, 75)),
        "d90": float(np.percentile(finite, 90)),
        "d99": float(np.percentile(finite, 99)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "count": int(finite.size),
    }
    return metrics

def extract_table_and_model_names(data_llm_dir: str, file_path: str) -> Tuple[str, str]:
    """
    Extracts table and model names from the file path.

    Args:
        file_path: Path to the file.

    Returns:
        Tuple containing table name and model name.
    """
    try:
        base_name = os.path.basename(file_path)
        relative = file_path.replace(data_llm_dir, "").split(os.sep)
        # data_llm_dir is expected to be the full dictionary_llm_results path.
        # The table name is the first non-empty path segment after it.
        table_name = "unknown_table"
        for segment in relative:
            if segment:
                table_name = segment
                break

        stem = os.path.splitext(base_name)[0]
        if stem.endswith("_parsed"):
            stem = stem[: -len("_parsed")]
        known_prefixes = (
            "openai_small_",
            "deepseek_small_",
            "google_small_",
        )
        model_name = stem
        for prefix in known_prefixes:
            if stem.startswith(prefix):
                model_name = stem[len(prefix):]
                break

        return table_name, model_name
    except Exception as e:
        logger.error(f"Error extracting table/model names from file {file_path}: {e}")
        return "unknown_table", "unknown_model"

def main(config_path: str = "config.yaml"):
    """
    Main function that orchestrates the entire comparison process.
    """
    try:
        # 1. Load configuration
        config = load_config(config_path)

        # Extract directory paths from configuration
        data_llm_dir = config.get("data_llm_results_dictionary_generation", {}).get("path", "")
        baseline_dicts_config = config.get("list_of_data_dictionaries", [])

        # Extract the path to save the comparison results
        output_dir = config.get("data_llm_results_distance_calculation", {}).get("path", "")
        logger.info(f"Configured output directory: {output_dir}")

        if not data_llm_dir or not baseline_dicts_config:
            logger.error("Data directories are not properly configured.")
            return

        # Create output directory if it does not exist
        if output_dir:
            # Ensure it is an absolute path
            if not os.path.isabs(output_dir):
                output_dir = os.path.abspath(output_dir)

            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory created/verified: {output_dir}")
        else:
            logger.warning("Output directory not specified. Using current directory.")
            output_dir = "."

        # 2. Load baseline (ground truth) dictionaries
        logger.info("Loading baseline dictionaries...")
        baseline_dicts = {}
        for item in baseline_dicts_config:
            path = item.get('path', '')
            name = item.get('name', '')

            if not path or not name:
                logger.warning(f"Invalid configuration for baseline dictionary: {item}")
                continue

            data = load_data_dictionary(path)
            if data:
                baseline_dicts[name] = data

        # 3. Find and load LLM dictionaries recursively
        logger.info("Searching for LLM files...")
        llm_files = find_all_files_in_directory(data_llm_dir)

        logger.info("Loading LLM dictionaries...")
        llm_dicts = {}
        skipped_errors = 0
        for file_path in llm_files:
            data = load_data_dictionary(file_path)
            if not data:
                continue
            if isinstance(data, dict) and "error" in data:
                skipped_errors += 1
                logger.warning(f"Skipping {file_path}: contains 'error' key (LLM rate-limit/parse failure)")
                continue
            table_name, model_name = extract_table_and_model_names(data_llm_dir, file_path)
            if table_name not in llm_dicts:
                llm_dicts[table_name] = {}
            llm_dicts[table_name][model_name] = data
        if skipped_errors:
            logger.info(f"Skipped {skipped_errors} LLM result file(s) containing errors.")

        # 4. Initialize embeddings model
        model, encode_kwargs = _load_embedding_model(config)

        # 5. Calculate embeddings and similarities
        logger.info("Calculating embeddings and similarities...")
        # Refactored: results will be a dict as described above
        results: Dict[str, Dict[str, Dict[str, float]]] = {}
        saved_files_count = 0  # Counter for saved files

        for table_name, baseline_data in baseline_dicts.items():
            baseline_emb = calculate_embeddings(baseline_data, model, encode_kwargs)
            if table_name in llm_dicts:
                for model_name, llm_data in llm_dicts[table_name].items():
                    llm_emb = calculate_embeddings(llm_data, model, encode_kwargs)
                    similarities = []
                    baseline_fields = {f.get("field_name", ""): f.get("field_description", "") for f in baseline_data.get("fields", []) if "field_name" in f}
                    llm_fields = {f.get("field_name", ""): f.get("full_description", f.get("field_description", "")) for f in llm_data.get("fields", []) if "field_name" in f}
                    # Add table_description similarity
                    for sim in calculate_similarities(baseline_emb, llm_emb):
                        field = sim["field"]
                        score = sim["score"]
                        if table_name not in results:
                            results[table_name] = {}
                        if field not in results[table_name]:
                            results[table_name][field] = {}
                        results[table_name][field][model_name] = score

                        # Attach descriptions directly to each similarity entry
                        if field == "table_description":
                            baseline_desc = baseline_data.get("table_description", "")
                            llm_desc = llm_data.get("table_description", "")
                        else:
                            baseline_desc = baseline_fields.get(field, "")
                            llm_desc = llm_fields.get(field, "")
                        similarities.append({
                            "field": field,
                            "score": score,
                            "baseline_description": baseline_desc,
                            "llm_description": llm_desc
                        })

                    # Save individual result in the specified output directory (now with descriptions in similarities)
                    result = {
                        "table_name": table_name,
                        "par-compare-models": model_name,
                        "similarities": similarities
                    }
                    output_file = os.path.join(output_dir, f"output_{table_name}_{model_name}.json")
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                        logger.info(f"Result successfully saved at: {output_file}")
                        saved_files_count += 1
                    except Exception as e:
                        logger.error(f"Error saving individual result: {e}")
            else:
                logger.warning(f"No LLM data found for table {table_name}")

        # Save all results in a single file in the specified output directory (refactored structure)
        if results:
            # Calculate distribution metrics per table/model and per model
            metrics_by_table_model: Dict[str, Dict[str, Dict[str, float]]] = {}
            model_scores: Dict[str, List[float]] = {}

            # Collect scores per (table, model) and per model
            for table_name, fields in results.items():
                metrics_by_table_model[table_name] = {}
                # Get all models present in this table
                models_in_table = set()
                for field_data in fields.values():
                    for model_name in field_data.keys():
                        models_in_table.add(model_name)

                # For each model in this table, compute metrics over the field scores
                for model_name in models_in_table:
                    scores: List[float] = []
                    for field_data in fields.values():
                        if model_name in field_data:
                            scores.append(field_data[model_name])
                            model_scores.setdefault(model_name, []).append(field_data[model_name])

                    if scores:
                        metrics_by_table_model[table_name][model_name] = compute_similarity_metrics(scores)

            # Calculate overall metrics by model across all tables
            metrics_by_model: Dict[str, Dict[str, float]] = {
                model_name: compute_similarity_metrics(scores)
                for model_name, scores in model_scores.items()
                if scores
            }

            # Create final structured output with results and metrics
            final_results = {
                "results": results,
                "metrics_by_table_and_model": metrics_by_table_model,
                "metrics_by_model": metrics_by_model,
            }

            all_results_file = os.path.join(output_dir, "all_similarities_results.json")
            try:
                with open(all_results_file, 'w', encoding='utf-8') as f:
                    json.dump(final_results, f, indent=2, ensure_ascii=False)
                logger.info(f"All results with metrics ({len(results)} tables) saved at: {all_results_file}")
                saved_files_count += 1
            except Exception as e:
                logger.error(f"Error saving consolidated results file: {e}")

        logger.info(f"Processing completed. Total JSON files saved: {saved_files_count}")

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()
