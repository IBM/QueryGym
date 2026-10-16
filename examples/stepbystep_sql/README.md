# Step-by-Step SQL Pipeline

This directory contains a complete implementation of a **Step-by-Step SQL** pipeline using QueryGym. Unlike direct Text-to-SQL approaches, this method breaks down SQL generation into incremental steps using various database operations (e.g., projection, filtering, joining, aggregation).

## Overview

The Step-by-Step SQL approach:
- Uses an agent-based system that interacts with the database environment
- Performs incremental operations to build up the final SQL query
- Leverages 19 predefined actions for data exploration and manipulation
- Provides detailed logging and tracing of the agent's decision-making process

## Directory Structure

```
stepbystep_sql/
├── config_parser.py              # CLI argument parser and configuration loader
├── pipeline.py                   # Core episode execution logic
├── run_stepbystep_sql.py        # Main entry point
├── stepbystep_sql_prompts.yaml  # System instructions and action definitions
└── README.md                     # This file
```

## Files Description

### `config_parser.py`
Handles command-line argument parsing and configuration management. Key parameters include:
- Model settings (model name, API base, temperature, top-p)
- Dataset settings (path, split, example indices)
- Agent settings (max steps, max LLM calls, message history limits)
- Execution settings (workers, log directory)

### `pipeline.py`
Contains the `run_episode()` function that:
1. Initializes the agent and environment for a single example
2. Runs the agent-environment interaction loop
3. Parses and executes actions from the agent
4. Handles errors and edge cases
5. Logs detailed trace information

### `run_stepbystep_sql.py`
Main entry point that:
1. Loads configuration
2. Initializes dataset and language model
3. Builds system instruction with available actions
4. Executes episodes in parallel
5. Aggregates and saves results

### `stepbystep_sql_prompts.yaml`
Contains:
- **actions**: Definitions of 19 available actions (get_overview, perform_projection, perform_filter, etc.)
- **system_instruction**: Prompt template for the agent with guidelines on how to respond

## Available Actions

The agent has access to 19 actions:

### Information Gathering
- `get_overview` - Get task overview
- `get_query` - Get user query
- `get_schema` - Get database schema
- `get_tables` - List all tables
- `get_columns` - List columns in a table
- `get_actions` - List available actions
- `get_operations` - List available operations

### Data Exploration
- `preview_table` - Preview first 5 rows
- `get_column_stats` - Get column statistics
- `get_unique_values` - Get unique values in a column
- `get_column_types` - Get column data types
- `get_sample_values` - Get sample values from a column

### Data Manipulation
- `perform_projection` - Project columns (SELECT)
- `perform_filter` - Filter rows (WHERE)
- `perform_join` - Join tables
- `perform_order_by` - Sort results
- `perform_limit` - Limit results
- `perform_aggregate` - Aggregate/group data
- `perform_union` - Union tables
- `perform_intersect` - Intersect tables

## Usage

### Basic Usage

```bash
python run_stepbystep_sql.py \
    --model meta-llama/llama-3-3-70b-instruct \
    --dataset-path ./bird \
    --dataset-split train
```

### Advanced Usage

```bash
python run_stepbystep_sql.py \
    --model meta-llama/llama-3-3-70b-instruct \
    --dataset-path ./bird \
    --dataset-split train \
    --start-index 0 \
    --end-index 100 \
    --workers 4 \
    --max-steps 15 \
    --max-llm-calls 20 \
    --temperature 0.0 \
    --log-dir ./results/stepbystep_sql \
    --prompt-definitions-file ./stepbystep_sql_prompts.yaml
```

### Process Specific Examples

```bash
python run_stepbystep_sql.py \
    --model meta-llama/llama-3-3-70b-instruct \
    --dataset-path ./bird \
    --dataset-split train \
    --example-ids 0 5 10 15 20
```

## Command-Line Arguments

### Model Settings
- `--model` (required): Language model name or path
- `--API-base`: API base URL (default: http://0.0.0.0:4000)
- `--temperature`: Sampling temperature (default: 0.0)
- `--top-p`: Nucleus sampling parameter (default: 0.95)

### Dataset Settings
- `--dataset-path`: Path to dataset directory (default: ./bird)
- `--dataset-split`: Dataset split to use (default: train)
- `--start-index`: Start index for examples (default: 0)
- `--end-index`: End index for examples (default: -1, processes all)
- `--example-ids`: Specific example IDs to process

### Agent Settings
- `--max-steps`: Maximum environment steps per episode (default: 10)
- `--max-llm-calls`: Maximum LLM calls per episode (default: 10)
- `--max-msg-history`: Maximum messages in agent history (default: 3000)
- `--max-tokens-input`: Maximum input tokens (default: 100000)

### Execution Settings
- `--workers`: Number of parallel workers (default: 1)
- `--log-dir`: Log directory path (default: ./results/stepbystep_sql)
- `--prompt-definitions-file`: Path to YAML prompts file (default: ./stepbystep_sql_prompts.yaml)

## Output

The pipeline generates:
- **Episode-specific logs**: Saved in `{log_dir}/{example_id}/`
  - `results.csv`: Episode metrics (reward, success, steps, etc.)
  - `trajectories.jsonl`: Detailed trace of agent-environment interactions
  - `chat_history.jsonl`: Complete message history
  - `debug.log`: Debug information

- **Aggregated results**: Saved in `{log_dir}/`
  - `results.csv`: Combined results for all episodes
  - Summary statistics and performance metrics

## Key Differences from Text-to-SQL

| Aspect | Text-to-SQL | Step-by-Step SQL |
|--------|-------------|------------------|
| Approach | Direct SQL generation | Incremental operations |
| Interaction | Single LLM call | Multiple agent-environment interactions |
| Feedback | Post-execution only | Step-by-step feedback |
| Debugging | Limited | Detailed trace available |
| Flexibility | Fixed output | Adaptive based on observations |

## Environment Variables

Create a `.env` file in the project root with:
```
LITELLM_API_KEY=your_api_key_here
```

## Dependencies

- `query_gym`: QueryGym environment and utilities
- `langchain`: For agent implementation and LLM interaction
- `pyyaml`: For loading prompt definitions
- `python-dotenv`: For environment variable management
- `tqdm`: For progress bars
- Standard libraries: `os`, `ast`, `traceback`, `concurrent.futures`

## Example Workflow

1. Agent receives initial observation (database schema, user query)
2. Agent decides on next action (e.g., `get_tables`)
3. Environment executes action and returns observation
4. Agent processes observation and decides next action
5. Repeat until query is answered or max steps reached
6. Final SQL is executed and evaluated

## Tips for Best Results

1. **Adjust max-steps**: Complex queries may need more steps
2. **Use parallel workers**: Speed up processing with `--workers`
3. **Monitor logs**: Check episode-specific logs for debugging
4. **Customize prompts**: Modify `stepbystep_sql_prompts.yaml` for different behaviors
5. **Temperature settings**: Use 0.0 for deterministic results, higher for exploration

## Troubleshooting

- **Parse errors**: Check action format in agent responses
- **Timeout errors**: Increase `timeout` in ChatOpenAI initialization
- **Memory issues**: Reduce `--max-tokens-input` or `--max-msg-history`
- **Slow execution**: Increase `--workers` for parallel processing

## References

- QueryGym Documentation: [Link to docs]
- Original Step-by-Step SQL Paper: [If applicable]
- LangChain Documentation: https://python.langchain.com/