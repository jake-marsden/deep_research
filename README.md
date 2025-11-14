## Generation & Evaluation Pipeline Guide

1. Navigate to **open_deep_research**
```
cd open_deep_research
```

2. Create a virtual environment and install dependencies
```
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml
uv sync
```

3. Configure system in `run_evaluate.py` (`open_deep_research` or `ROFC`)
```
# System selection: Switch between 'open_deep_research' and 'rofc'
SYSTEM_TO_EVALUATE = "open_deep_research"  # Options: "open_deep_research" or "rofc"

if SYSTEM_TO_EVALUATE == "rofc":
    from rofc.deep_researcher import deep_researcher_builder
else:
    from open_deep_research.deep_researcher import deep_researcher_builder
```

4. Configure models in `run_evaluate.py`
```
# NOTE: Configure the right dataset and evaluators
dataset_name = "Deep Research Bench"
evaluators = [eval_overall_quality, eval_relevance, eval_structure, eval_correctness, eval_groundedness, eval_completeness]
# NOTE: Configure the right parameters for the experiment, these will be logged in the metadata
max_structured_output_retries = 3
allow_clarification = False
max_concurrent_research_units = 10
search_api = "tavily" # NOTE: We use Tavily to stay consistent
max_researcher_iterations = 6
max_react_tool_calls = 10
summarization_model = "openai:gpt-4.1-mini"
summarization_model_max_tokens = 8192
research_model = "openai:gpt-5" # "anthropic:claude-sonnet-4-20250514"
research_model_max_tokens = 10000
compression_model = "openai:gpt-4.1"
compression_model_max_tokens = 10000
final_report_model = "openai:gpt-4.1"
final_report_model_max_tokens = 10000
```

5. Generate answers to `deep_research_bench`
```
python tests/run_evaluate.py
```

6. Extract the results
```
python tests/extract_langsmith_data.py --project-name "YOUR_EXPERIMENT_NAME" --model-name "you-model-name" --dataset-name "deep_research_bench"

# for example:
python tests/extract_langsmith_data.py --project-name "OPEN_DEEP_RESEARCH_GENSEE_openai:gpt-oss-120b_openai:gpt-oss-120b_openai:gpt-oss-120b_openai:gpt-oss-120b-3885ae6c" --model-name "gpt-oss-120b" --dataset-name "deep_research_bench"
```

7. Navigate to **deep_research_bench**
```
cd deep_research_bench
```

8. Save results to `data/test_data/raw_data/<model_name>.jsonl`

9. Configure target model in `run_benchmark.sh`
```
# Target model name list
TARGET_MODELS=("deep_research_bench_gpt-oss-120b") # Change to the name of the jsonl file you want to evaluate
```

10. View results
- RACE evaluation: `results/race/<model_name>/race_result.txt`
- FACT evaluation: `results/fact/<model_name>/fact_result.txt`