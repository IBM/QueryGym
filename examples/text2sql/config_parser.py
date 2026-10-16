# config_parser.py
"""
Argument Parser for Text-to-SQL Pipeline
----------------------------------------
Provides a modular way to parse CLI arguments and return configuration.
"""

import argparse

def get_config():
    parser = argparse.ArgumentParser(
        description="Run Text-to-SQL pipeline using QueryGym. "
                    "This tool converts natural language questions into SQL queries "
                    "and evaluates them against a database environment."
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
        help="Base URL for the API  or LiteLLM proxy (default: http://0.0.0.0:4000)."
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
        help="Dataset split to use (e.g., 'train', 'dev', 'test'). Default: dev."
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
    # Performance and logging
    # -------------------------
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers for multi-threaded execution (default: 1)."
    )
    parser.add_argument(
        "--log-dir", type=str, default="./results/text2sql",
        help="Directory where logs and results will be saved (default: ./results/text2sql)."
    )
    parser.add_argument(
        "--instruction-file", type=str, default="./text2sql_instruction.txt",
        help="Path to the instruction template file. "
             "Placeholders: {query} for NL question, {schema} for schema, {tables} for table previews."
    )

    # Parse arguments
    args = parser.parse_args()
    config = vars(args)

    # Load instruction template
    with open(config["instruction_file"]) as f:
        config['instruction'] = f.read()

    # Prepare log directory name based on dataset and model
    dataset_name = f"{config['dataset_path'].split('/')[-1]}_{config['dataset_split']}"
    config['log_dir'] += f"/{dataset_name}_{config['model'].split('/')[-1]}"

    return config
