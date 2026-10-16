# run_stepbystep_sql.py
"""
Entry Point for Step-by-Step SQL Pipeline
------------------------------------------
This script:
    1. Loads configuration from CLI arguments using config_parser.
    2. Sets up logging for experiment tracking.
    3. Initializes dataset and language model.
    4. Runs Step-by-Step SQL episodes in parallel using ThreadPoolExecutor.
    5. Saves results and logs for analysis.

Usage:
------
python run_stepbystep_sql.py --model <model_name> --dataset-path <dataset-path>
"""

import os
import socket
import concurrent.futures
import traceback
from tqdm import tqdm
from dotenv import load_dotenv

# Import custom modules
from query_gym import setup_logger
from config_parser import get_config
from query_gym.dataset.unite_dataset import UniteDataset
from pipeline import run_episode

# QueryGym environment
from query_gym.envs import SQLiteQueryGym as QueryGymEnv

# Agent for LLM interaction
from langchain_openai import ChatOpenAI

# Load environment variables (e.g., API keys from .env)
load_dotenv()


# ------------------------------------------------------------
# Main Execution Function
# ------------------------------------------------------------
def run(config, log_dir, logger):
    """
    Runs the Step-by-Step SQL pipeline for all examples specified in config.

    Steps:
    ------
    1. Load dataset.
    2. Initialize language model (local HuggingFace or LITELLM API).
    3. Build system instruction with available actions.
    4. Determine example range.
    5. Execute episodes in parallel using ThreadPoolExecutor.
    6. Log results and handle errors.

    Args:
        config (dict): Configuration dictionary from CLI.
        log_dir (str): Directory path for saving logs.
        logger: Logger instance for recording results.
    """

    # 1. Load dataset
    dataset = UniteDataset(
        dataset_path=config['dataset_path'],
        mode=config['dataset_split']
    )

    # 2. Initialize language model based on environment
    if 'ccc' in socket.gethostname():
        # Running on CCC cluster → use HuggingFace model locally
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        llm = HuggingFacePipeline.from_model_id(
            model_id=config['model'],
            task="text-generation",
            pipeline_kwargs=dict(
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
                return_full_text=False
            ),
        )
        chat_model = ChatHuggingFace(llm=llm)
    else:
        # Use LITELLM API-based model for remote inference
        chat_model = ChatOpenAI(
            api_key= os.getenv("LITELLM_API_KEY", None),
            openai_api_base="/",
            model=config['model'],
            temperature=config['temperature'],
            top_p=config['top_p'],
            base_url=config['API_base'],
            # default_headers={}, #Add optional headers
            timeout=2000,
            max_retries=8
        )

    # 3. Build system instruction with available actions
    # Create a temporary environment to get ACTION_TYPES
    temp_env = QueryGymEnv(
        dataset=dataset,
        action_definition=config['prompt_definitions']['actions'],
        max_steps=1,
        logger=logger,
        allow_execute_sql=True
    )
    
    ad = config['prompt_definitions']['actions']
    available_actions = (
        f"There are {len(temp_env.ACTION_TYPES)} types of actions in this scenario. "
        f"The action name, description, usage, and example is given below.\n"
        + "\n".join([f"{i+1}. {a}: {ad[a.name]}" for i, a in enumerate(temp_env.ACTION_TYPES) if a.name != 'EXECUTE_SQL'])
    )
    instruction = config['prompt_definitions']['system_instruction'].replace(
        "{{get_actions}}", available_actions
    )
    temp_env.close()

    # 4. Determine example range
    if config['end_index'] == -1:
        config['end_index'] = len(dataset)
    run_examples = config['example_ids'] or range(config['start_index'], config['end_index'])

    # 5. Execute episodes in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=config['workers']) as executor:
        futures = [
            executor.submit(run_episode, config, log_dir, chat_model, dataset, i, instruction)
            for i in run_examples
        ]

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), desc="Running Episodes"):
            try:
                result = future.result()
                # Log results in tabular format
                logger._tabular = result
                logger.dump_tabular()
            except Exception:
                # Log any errors encountered during execution
                logger.log("================ERROR ENCOUNTERED===================")
                logger.log(traceback.format_exc())


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    # Load configuration from CLI arguments
    config = get_config()

    # Prepare logger name based on dataset and model
    dataset_name = f"{config['dataset_path'].split('/')[-1]}_{config['dataset_split']}"
    logger = setup_logger(
        f"stepbystep_sql_{dataset_name.replace('/', '_')}_{config['model'].split('/')[-1]}",
        variant=config,
        exp_id=os.getpid(),
        log_dir=config['log_dir']
    )

    # Get log directory for episode-specific logs
    log_dir = logger.get_snapshot_dir()

    # Run the pipeline
    run(config, log_dir, logger)

# Made with Bob
