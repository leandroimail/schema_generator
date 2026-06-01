import os
import yaml
import json
import pandas as pd
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import io

# Import our LLM client function
from llm import load_llm_client_configs, run_llm_clients, run_llm_clients_one
from pydantic import BaseModel, ConfigDict, Field
from py_markdown_table.markdown_table import markdown_table


JSONPrimitive = Union[str, int, float, bool, None]


class DictionaryField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_name: str = Field(description="Column name exactly as it appears in the input data.")
    data_type: str = Field(description="Inferred data type for the column.")
    field_description: str = Field(description="Semantic description of the column.")
    example_value: JSONPrimitive = Field(description="Representative scalar value from the sample, or null when unavailable.")
    full_description: str = Field(description="field_description plus Domain or Example detail.")
    domain_values: Optional[List[JSONPrimitive]] = Field(
        default=None,
        description="Known enum/restricted values. Null or omitted when the field has no restricted domain.",
    )


class DataDictionary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(description="Table name inferred from the profile/sample metadata.")
    table_description: str = Field(description="General semantic description and use of the table.")
    fields: List[DictionaryField] = Field(description="One entry for every column in the table.")


def output_file_stem(client_name: str, model_name: str) -> str:
    safe = "".join(char if char.isalnum() or char in "_.-" else "_" for char in f"{client_name}_{model_name}")
    return safe.strip("._") or "llm"


def parsed_result_path(profile_output_dir: str, client_name: str, model_name: str) -> Path:
    return Path(profile_output_dir) / "json" / f"{output_file_stem(client_name, model_name)}_parsed.json"


def validate_dictionary_result(path: Path) -> Optional[str]:
    if not path.exists():
        return "missing result file"

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        return f"invalid JSON: {exc}"
    except OSError as exc:
        return f"cannot read file: {exc}"

    if isinstance(data, dict) and data.get("error"):
        return f"LLM error: {data['error']}"
    if isinstance(data, dict) and data.get("raw_content"):
        return "raw_content fallback instead of parsed dictionary"

    try:
        DataDictionary.model_validate(data)
    except Exception as exc:
        return f"schema validation error: {exc}"

    return None


def find_dictionary_retry_targets(output_dir: str, profile_names: List[str]) -> Dict[str, List[str]]:
    retry_targets: Dict[str, List[str]] = {}
    configs = load_llm_client_configs()

    for profile_name in profile_names:
        profile_output_dir = os.path.join(output_dir, profile_name)
        for config in configs:
            result_path = parsed_result_path(profile_output_dir, config.name, config.model_name)
            error = validate_dictionary_result(result_path)
            if error:
                retry_targets.setdefault(profile_name, []).append(config.id)
                print(f"[retry] {profile_name} / {config.name} ({config.model_name}): {error}")

    return retry_targets


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}


def load_profile(profile_path: str) -> Dict[str, Any]:
    """
    Load profile data from JSON file.

    Args:
        profile_path (str): Path to the profile JSON file

    Returns:
        Dict[str, Any]: Profile data dictionary
    """
    try:
        with open(profile_path, 'r') as file:
            profile_data = json.load(file)
        return profile_data
    except (FileNotFoundError, FileExistsError) as e:
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            relative_path = os.path.join(base_dir, profile_path)
            with open(relative_path, 'r') as file:
                profile_data = json.load(file)
            return profile_data
        except Exception as e:
            print(f"Error loading profile from {profile_path}: {e}")
            return {}


def load_sample(sample_path: str) -> str:
    """
    Load sample data from Parquet file and convert to CSV string representation.

    Args:
        sample_path (str): Path to the parquet file

    Returns:
        str: CSV string representation of the sample data
    """
    try:
        # Read parquet file into pandas DataFrame
        df = pd.read_parquet(sample_path)

        data = df.to_dict(orient='records')
        return markdown_table(data).get_markdown()
    except Exception as e:
        print(f"Error loading sample from {sample_path}: {e}")
        return ""


def read_prompt_template(prompt_path: str) -> str:
    """
    Read prompt template from file.

    Args:
        prompt_path (str): Path to the prompt template file
        
    Returns:
        str: Prompt template content
    """
    try:
        with open(prompt_path, 'r') as file:
            prompt_template = file.read()
        return prompt_template
    except Exception as e:
        print(f"Error reading prompt template from {prompt_path}: {e}")
        return ""


async def generate_dictionaries(retry_errors_only: bool = False, list_errors_only: bool = False) -> None:
    """
    Main function to generate data dictionaries using LLMs.
    """
    # Load configuration
    config = load_config()
    if not config:
        print("Failed to load configuration. Exiting.")
        return

    # Get configuration sections
    profiles_config = config.get('list_of_profiles', [])
    samples_config = config.get('list_of_data_samples', [])
    output_dir = config.get('data_llm_results_dictionary_generation', {}).get('path', 'dictionary_llm_results')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load profiles
    profiles = {}
    for profile_item in profiles_config:
        name = profile_item.get('name')
        path = profile_item.get('path')
        if name and path:
            profile_data = load_profile(path)
            if profile_data:
                profiles[name] = profile_data
    
    if not profiles:
        print("No profiles loaded. Exiting.")
        return
    
    # Load samples
    samples = {}
    for sample_item in samples_config:
        name = sample_item.get('name')
        path = sample_item.get('path')
        if name and path:
            sample_data = load_sample(path)
            if sample_data:
                samples[name] = sample_data
    
    if not samples:
        print("No samples loaded. Exiting.")
        return
    
    # Read prompt template
    prompt_template = read_prompt_template("prompts/dictionary_generation.md")
    if not prompt_template:
        print("Failed to read prompt template. Exiting.")
        return
    
    retry_targets: Dict[str, List[str]] = {}
    if retry_errors_only or list_errors_only:
        retry_targets = find_dictionary_retry_targets(output_dir, list(profiles.keys()))
        total_targets = sum(len(targets) for targets in retry_targets.values())
        print(f"Found {total_targets} result(s) to rerun.")
        if list_errors_only:
            return
        if not retry_targets:
            return

    # Process each profile and sample combination
    for profile_name, profile_data in profiles.items():
        target_clients = retry_targets.get(profile_name)
        if retry_errors_only and not target_clients:
            continue

        # Find the sample with matching name if it exists
        sample_data = samples.get(profile_name, "")
        
        if not sample_data:
            # If no matching sample found, log warning but continue with empty sample
            print(f"No matching sample found for profile {profile_name}. Continuing with empty sample.")
        
        # Convert profile data to formatted string for prompt
        profile_str = json.dumps(profile_data, indent=2)
        
        # Create customized prompt by replacing placeholders
        prompt = prompt_template.replace("<profile>", profile_str).replace("<sample>", sample_data)
        
        # Create specific output directory for this profile
        profile_output_dir = os.path.join(output_dir, profile_name)
        os.makedirs(profile_output_dir, exist_ok=True)
        
        print(f"Processing dictionary generation for profile: {profile_name}")
        
        # Call the LLM clients to process the prompt
        if target_clients:
            for llm_name in target_clients:
                await run_llm_clients_one(
                    prompt=prompt,
                    base_output_dir=profile_output_dir,
                    llm_name=llm_name,
                    response_model=DataDictionary,
                )
        else:
            await run_llm_clients(prompt=prompt, base_output_dir=profile_output_dir, response_model=DataDictionary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data dictionaries using configured LLM clients.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Only rerun provider/profile combinations whose parsed JSON is missing, has error, or fails schema validation.",
    )
    parser.add_argument(
        "--list-errors",
        action="store_true",
        help="List provider/profile combinations that would be rerun, without calling LLM APIs.",
    )
    args = parser.parse_args()

    asyncio.run(generate_dictionaries(retry_errors_only=args.retry_errors, list_errors_only=args.list_errors))
