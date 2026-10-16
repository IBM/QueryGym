# config_parser.py
"""
Argument Parser for Step-by-Step SQL Pipeline
----------------------------------------------
Provides a modular way to parse CLI arguments and return configuration.
"""

import argparse
import yaml


def get_config():
    parser = argparse.ArgumentParser(
        description="Run Step-by-Step SQL pipeline using QueryGym. "
                    "This tool breaks down SQL generation into incremental steps "
                    "using various database operations and evaluates them against a database environment."
    )

    # -------------------------
    # Model and API settings
    # -------------------------
    parser.add_argument(
        '--model', type=str, required=True,
        help='Name or path of the language model to use (e.g., "meta-llama/llama-3-3-70b-instruct", or HuggingFace model ID).'
    )
    parser.add_argument(
        "--API-base", type=str, default="http://0.0.0.0:4000",
        help="Base URL for the API or LiteLLM proxy (default: http://0.0.0.0:4000)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Controls randomness in model output. Lower values = more deterministic (default: 0.0)."
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95,
        help="Nucleus sampling parameter for controlling diversity in generated text (default: 0.95)."
    )

    # -------------------------
    # Dataset and execution settings
    # -------------------------
    parser.add_argument(
        "--dataset-path", type=str, default="./bird",
        help="Path to the dataset directory (default: ./bird)."
    )
    parser.add_argument(
        "--dataset-split", type=str, default="train",
        help="Dataset split to use (e.g., 'train', 'dev', 'test'). Default: train."
    )
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="Start index for examples to process (default: 0)."
    )
    parser.add_argument(
        "--end-index", type=int, default=-1,
        help="End index for examples to process. Use -1 to process all remaining examples (default: -1)."
    )
    parser.add_argument(
        '--example-ids', type=int, nargs='+', default=None,
        help="Specific example IDs to process. Overrides start/end index if provided."
    )

    # -------------------------
    # Agent and environment settings
    # -------------------------
    parser.add_argument(
        "--max-steps", type=int, default=10,
        help="Maximum steps allowed in the environment per episode (default: 10)."
    )
    parser.add_argument(
        "--max-llm-calls", type=int, default=10,
        help="Maximum number of LLM calls allowed per episode (default: 10)."
    )
    parser.add_argument(
        "--max-msg-history", type=int, default=3000,
        help="Maximum number of messages to retain in agent history (default: 3000)."
    )
    parser.add_argument(
        "--max-tokens-input", type=int, default=100000,
        help="Maximum number of tokens allowed in message history (default: 100000)."
    )

    # -------------------------
    # Performance and logging
    # -------------------------
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers for multi-threaded execution (default: 1)."
    )
    parser.add_argument(
        "--log-dir", type=str, default="./results/stepbystep_sql",
        help="Directory where logs and results will be saved (default: ./results/stepbystep_sql)."
    )
    parser.add_argument(
        "--prompt-definitions-file", type=str, default="./stepbystep_sql_prompts.yaml",
        help="Path to YAML file containing system instructions and action definitions."
    )

    # Parse arguments
    args = parser.parse_args()
    config = vars(args)

    # Load prompt definitions from YAML file
    with open(config["prompt_definitions_file"]) as f:
        config['prompt_definitions'] = yaml.safe_load(f)

    # Prepare log directory name based on dataset and model
    dataset_name = f"{config['dataset_path'].split('/')[-1]}_{config['dataset_split']}"
    config['log_dir'] += f"/{dataset_name}_{config['model'].split('/')[-1]}"

    return config

# Made with Bob
