# Dictionary Generation Error Report

Generated at: 2026-06-01T19:53:13

## Summary

- Parsed JSON files scanned: 450
- Schema-valid files: 442
- Error files: 8

## Error Categories

- `openai_tpm_or_prompt_too_large`: 6
- `context_window_exceeded`: 1
- `provider_quota_exceeded`: 1

## Problem Files

- `data/llm_results/dictionary_llm_results/bird__codebase_community__posthistory/json/openai_small_gpt-5.4-mini_parsed.json`
  - Table: `bird__codebase_community__posthistory`
  - Artifact: `openai_small_gpt-5.4-mini`
  - Category: `openai_tpm_or_prompt_too_large`
- `data/llm_results/dictionary_llm_results/bird__codebase_community__posthistory/json/openai_small_gpt-5.4-nano_parsed.json`
  - Table: `bird__codebase_community__posthistory`
  - Artifact: `openai_small_gpt-5.4-nano`
  - Category: `openai_tpm_or_prompt_too_large`
- `data/llm_results/dictionary_llm_results/bird__codebase_community__posts/json/openai_small_gpt-5.4-mini_parsed.json`
  - Table: `bird__codebase_community__posts`
  - Artifact: `openai_small_gpt-5.4-mini`
  - Category: `openai_tpm_or_prompt_too_large`
- `data/llm_results/dictionary_llm_results/bird__codebase_community__posts/json/openai_small_gpt-5.4-nano_parsed.json`
  - Table: `bird__codebase_community__posts`
  - Artifact: `openai_small_gpt-5.4-nano`
  - Category: `openai_tpm_or_prompt_too_large`
- `data/llm_results/dictionary_llm_results/bird__european_football_2__match/json/google_small_gemini-3.1-flash-lite_parsed.json`
  - Table: `bird__european_football_2__match`
  - Artifact: `google_small_gemini-3.1-flash-lite`
  - Category: `context_window_exceeded`
- `data/llm_results/dictionary_llm_results/bird__european_football_2__match/json/google_small_gemini-3.5-flash_parsed.json`
  - Table: `bird__european_football_2__match`
  - Artifact: `google_small_gemini-3.5-flash`
  - Category: `provider_quota_exceeded`
- `data/llm_results/dictionary_llm_results/bird__european_football_2__match/json/openai_small_gpt-5.4-mini_parsed.json`
  - Table: `bird__european_football_2__match`
  - Artifact: `openai_small_gpt-5.4-mini`
  - Category: `openai_tpm_or_prompt_too_large`
- `data/llm_results/dictionary_llm_results/bird__european_football_2__match/json/openai_small_gpt-5.4-nano_parsed.json`
  - Table: `bird__european_football_2__match`
  - Artifact: `openai_small_gpt-5.4-nano`
  - Category: `openai_tpm_or_prompt_too_large`

## Errors By Table

- `bird__european_football_2__match`: 4
- `bird__codebase_community__posthistory`: 2
- `bird__codebase_community__posts`: 2

## Errors By Model Artifact

- `openai_small_gpt-5.4-mini`: 3
- `openai_small_gpt-5.4-nano`: 3
- `google_small_gemini-3.1-flash-lite`: 1
- `google_small_gemini-3.5-flash`: 1

## Detailed Errors

### `bird__codebase_community__posthistory`

- Artifact: `openai_small_gpt-5.4-mini`
  - Category: `openai_tpm_or_prompt_too_large`
  - File: `data/llm_results/dictionary_llm_results/bird__codebase_community__posthistory/json/openai_small_gpt-5.4-mini_parsed.json`
  - Detail: Error asynchronously parsing openai_small response: Error code: 429 - {'error': {'message': 'Request too large for gpt-5.4-mini in organization org-vLeyKkpB2AB7LQbKJlselVp9 on tokens per min (TPM): Limit 200000, Requested 225339. The input or output tokens must be reduced in order to run successfully. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens', 'param': None, 'code': 'r...
- Artifact: `openai_small_gpt-5.4-nano`
  - Category: `openai_tpm_or_prompt_too_large`
  - File: `data/llm_results/dictionary_llm_results/bird__codebase_community__posthistory/json/openai_small_gpt-5.4-nano_parsed.json`
  - Detail: Error asynchronously parsing openai_small response: Error code: 429 - {'error': {'message': 'Request too large for gpt-5.4-nano in organization org-vLeyKkpB2AB7LQbKJlselVp9 on tokens per min (TPM): Limit 200000, Requested 225339. The input or output tokens must be reduced in order to run successfully. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens', 'param': None, 'code': 'r...

### `bird__codebase_community__posts`

- Artifact: `openai_small_gpt-5.4-mini`
  - Category: `openai_tpm_or_prompt_too_large`
  - File: `data/llm_results/dictionary_llm_results/bird__codebase_community__posts/json/openai_small_gpt-5.4-mini_parsed.json`
  - Detail: Error asynchronously parsing openai_small response: Error code: 429 - {'error': {'message': 'Request too large for gpt-5.4-mini in organization org-vLeyKkpB2AB7LQbKJlselVp9 on tokens per min (TPM): Limit 200000, Requested 353175. The input or output tokens must be reduced in order to run successfully. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens', 'param': None, 'code': 'r...
- Artifact: `openai_small_gpt-5.4-nano`
  - Category: `openai_tpm_or_prompt_too_large`
  - File: `data/llm_results/dictionary_llm_results/bird__codebase_community__posts/json/openai_small_gpt-5.4-nano_parsed.json`
  - Detail: Error asynchronously parsing openai_small response: Error code: 429 - {'error': {'message': 'Request too large for gpt-5.4-nano in organization org-vLeyKkpB2AB7LQbKJlselVp9 on tokens per min (TPM): Limit 200000, Requested 353175. The input or output tokens must be reduced in order to run successfully. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens', 'param': None, 'code': 'r...

### `bird__european_football_2__match`

- Artifact: `google_small_gemini-3.1-flash-lite`
  - Category: `context_window_exceeded`
  - File: `data/llm_results/dictionary_llm_results/bird__european_football_2__match/json/google_small_gemini-3.1-flash-lite_parsed.json`
  - Detail: Error asynchronously parsing google_small response: Error code: 400 - [{'error': {'code': 400, 'message': 'The input token count exceeds the maximum number of tokens allowed (1048576).', 'status': 'INVALID_ARGUMENT'}}]. Response content: '{}...'
- Artifact: `google_small_gemini-3.5-flash`
  - Category: `provider_quota_exceeded`
  - File: `data/llm_results/dictionary_llm_results/bird__european_football_2__match/json/google_small_gemini-3.5-flash_parsed.json`
  - Detail: Error asynchronously parsing google_small response: Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/gener...
- Artifact: `openai_small_gpt-5.4-mini`
  - Category: `openai_tpm_or_prompt_too_large`
  - File: `data/llm_results/dictionary_llm_results/bird__european_football_2__match/json/openai_small_gpt-5.4-mini_parsed.json`
  - Detail: Error asynchronously parsing openai_small response: Error code: 429 - {'error': {'message': 'Request too large for gpt-5.4-mini in organization org-vLeyKkpB2AB7LQbKJlselVp9 on tokens per min (TPM): Limit 200000, Requested 3024906. The input or output tokens must be reduced in order to run successfully. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens', 'param': None, 'code': '...
- Artifact: `openai_small_gpt-5.4-nano`
  - Category: `openai_tpm_or_prompt_too_large`
  - File: `data/llm_results/dictionary_llm_results/bird__european_football_2__match/json/openai_small_gpt-5.4-nano_parsed.json`
  - Detail: Error asynchronously parsing openai_small response: Error code: 429 - {'error': {'message': 'Request too large for gpt-5.4-nano in organization org-vLeyKkpB2AB7LQbKJlselVp9 on tokens per min (TPM): Limit 200000, Requested 3024906. The input or output tokens must be reduced in order to run successfully. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens', 'param': None, 'code': '...

## Suggested Actions

- For `openai_tpm_or_prompt_too_large`, reduce prompt size before retrying; rerunning unchanged is expected to fail again.
- For `context_window_exceeded`, chunk the table by columns or compact the profile/sample.
- For `provider_quota_exceeded`, wait for quota reset or reduce request size/rate.
- For `invalid_model_json`, rerun that model after schema-hint improvements or use a stricter provider.

To list current retry targets:

```bash
uv run python src/dictionary_generation.py --list-errors
```

To rerun only current retry targets:

```bash
uv run python src/dictionary_generation.py --retry-errors
```
