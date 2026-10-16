
# run_text2sql.py
"""
Entry Point for Text-to-SQL Pipeline
-------------------------------------
This script:
    1. Loads configuration from CLI arguments using config_parser.
    2. Sets up logging for experiment tracking.
    3. Initializes dataset and language model.
    4. Runs Text-to-SQL episodes in parallel using ThreadPoolExecutor.
    5. Saves results and logs for analysis.

Usage:
------
python run_text2sql.py --model <model_name> --dataset-path <dataset-path> 
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
def run(config, logger):
    """
    Runs the Text-to-SQL pipeline for all examples specified in config.

    Steps:
    ------
    1. Load dataset.
    2. Initialize language model (local HuggingFace or LITELLM API).
    3. Determine example range.
    4. Execute episodes in parallel using ThreadPoolExecutor.
    5. Log results and handle errors.

    Args:
        config (dict): Configuration dictionary from CLI.
        logger: Logger instance for recording results.
    """

    # 1. Load dataset
    dataset = UniteDataset(dataset_path=config['dataset_path'], mode=config['dataset_split'])

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
            # default_headers={}, Add any optional headers
            timeout=2000,
            max_retries=8
        )

    # 3. Determine example range
    if config['end_index'] == -1:
        config['end_index'] = len(dataset)
    run_examples = config['example_ids'] or range(config['start_index'], config['end_index'])

    # 4. Execute episodes in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=config['workers']) as executor:
        futures = [executor.submit(run_episode, config, logger, chat_model, dataset, i) for i in run_examples]

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), desc="Running Episodes"):
            try:
                result, json_dump = future.result()
                # Log results in tabular format and save extra data
                logger.record_dict(result)
                logger.dump_tabular()
                logger.save_extra_data(json_dump, "response.jsonl", mode="jsonl")
            except Exception:
                # Log any errors encountered during execution
                logger.log("================ ERROR ENCOUNTERED ===================")
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
        f"text2sql_{dataset_name.replace('/', '_')}_{config['model'].split('/')[-1]}",
        variant=config,
        exp_id=os.getpid(),
        log_dir=config['log_dir']
    )

    # Run the pipeline
    run(config, logger)
