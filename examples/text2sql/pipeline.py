# pipeline.py
"""
Pipeline Logic for Text-to-SQL
------------------------------
This module contains the core logic for running Text-to-SQL episodes using QueryGym.
It includes:
    - run_episode(): Executes a single example (episode) from the dataset.
    - Handles prompt generation, LLM invocation, SQL extraction, execution, and result logging.
"""

import traceback

from dotenv import load_dotenv

# Import QueryGym environment
from query_gym.envs import SQLiteQueryGym as QueryGymEnv

# Load environment variables from .env file (e.g., API keys)
load_dotenv()


def run_episode(config, logger, chat_model, dataset, i):
    """
    Executes a single Text-to-SQL episode.

    Steps:
    ------
    1. Initialize QueryGym environment for SQLite.
    2. Reset environment for the given example index.
    3. Build the prompt using schema, tables, and query placeholders.
    4. Invoke the language model to generate SQL.
    5. Extract SQL from the model response.
    6. Execute SQL in the environment and compute reward.
    7. Collect results and prepare JSON for logging.

    Args:
        config (dict): Configuration dictionary with model, dataset, and execution parameters.
        logger: Logger instance for recording results.
        chat_model: Language model instance (ChatOpenAI or HuggingFace).
        dataset: Dataset instance (UniteDataset or similar).
        i (int): Example index to process.

    Returns:
        tuple: (result_dict, json_dump)
            - result_dict: Summary of execution (reward, success, match metrics).
            - json_dump: Detailed log including prompt, LLM response, and environment info.
    """
    # 1. Initialize environment for a single episode
    env = QueryGymEnv(dataset=dataset, max_steps=1, logger=logger, allow_execute_sql=True)

    # 2. Reset environment for the given example
    observation = env.reset(i)

    # 3. Build the prompt using schema, tables, and query placeholders.
    schema = env._get_schema()  # Database schema representation
    query = env.dataset_item.query  # Natural language question
    tables = "\n".join([env._preview_table(t) for t in env.db['table_names']])  # Preview of tables
    prompt = (
        config['instruction']
        .replace("{schema}", schema)
        .replace("{tables}", tables)
        .replace("{query}", query)
    )

    # 4. Invoke language model with constructed prompt
    response = chat_model.invoke(prompt)

    # 5. Extract SQL from LLM response (between ```sql ... ```)
    try:
        pred_sql = response.content.split("```sql")[1].split("```")[0].strip()
    except IndexError:
        pred_sql = ""  # If SQL block is missing, fallback to empty string

    # Initialize execution flags
    sql_executed, sql_error = False, ""

    # 6. Execute predicted SQL in environment and compute reward
    try:
        observation, reward, done, truncated, info = env.step(["execute_sql", pred_sql])
        sql_executed = True
    except Exception as e:
        # Handle SQL execution failure
        reward, done, info = 0, False, {
            k: None for k in ["match", "non_empty_match", "subset_match", "row_length_correctness"]
        }
        sql_error = traceback.format_exception_only(e)[0]

    # 7. Prepare result dictionary for logging
    result_dict = {
        "example_id": i,
        "reward": reward,
        "success": done,
        "sql_executed": sql_executed,
        "sql_error": sql_error,
        "match": info['match'],
        "non_empty_match": info['non_empty_match'],
        "subset_match": info['subset_match'],
        "row_length_correctness": info['row_length_correctness']
    }

    # Prepare detailed JSON dump for debugging and analysis
    json_dump = env.dataset_item.info  # Base info from dataset
    json_dump.update(result_dict)  # Add execution results
    json_dump.update(response.additional_kwargs)  # Include extra LLM metadata
    json_dump["prompt"] = prompt  # Original prompt sent to LLM
    json_dump["llm_response"] = response.content  # Raw LLM response
    json_dump["action_execution"] = observation  # Environment observation after execution

    return result_dict, json_dump
