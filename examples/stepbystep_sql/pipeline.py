# pipeline.py
"""
Pipeline Logic for Step-by-Step SQL
------------------------------------
This module contains the core logic for running Step-by-Step SQL episodes using QueryGym.
It includes:
    - run_episode(): Executes a single example (episode) from the dataset.
    - Handles agent-environment interaction, action parsing, execution, and result logging.
"""

import os
import ast
import traceback
from dotenv import load_dotenv
from langchain.output_parsers import RegexParser

# Import QueryGym environment
from query_gym.envs import SQLiteQueryGym as QueryGymEnv
from query_gym import setup_logger

# Import agent
from agents.gym_agent import GymnasiumAgent

# Load environment variables from .env file (e.g., API keys)
load_dotenv()


def run_episode(config, log_dir, chat_model, dataset, i, instruction):
    """
    Executes a single Step-by-Step SQL episode.

    Steps:
    ------
    1. Initialize logger for this specific episode.
    2. Check if episode already exists (skip if completed).
    3. Initialize agent and environment.
    4. Reset agent and environment for the given example index.
    5. Run agent-environment interaction loop:
       - Agent observes environment state
       - Agent generates action
       - Parse and execute action in environment
       - Collect reward and next observation
       - Continue until done/truncated or max LLM calls reached
    6. Log results and trace information.

    Args:
        config (dict): Configuration dictionary with model, dataset, and execution parameters.
        log_dir (str): Directory path for saving episode-specific logs.
        chat_model: Language model instance (ChatOpenAI or HuggingFace).
        dataset: Dataset instance (UniteDataset or similar).
        i (int): Example index to process.
        instruction (str): System instruction for the agent.

    Returns:
        dict: Result dictionary containing episode metrics and outcomes.
    """
    # 1. Check if episode already exists
    if os.path.exists(f"{log_dir}/{i}"):
        if os.path.exists(f"{log_dir}/{i}/results.csv") and os.path.getsize(f"{log_dir}/{i}/results.csv") > 0:
            raise Exception("Example Already Exists")

    # 2. Initialize episode-specific logger
    logger = setup_logger(
        f"stepbystep_sql_{config['model'].split('/')[-1]}",
        variant=None,
        log_dir=f"{log_dir}/{i}/",
        lightweight=True
    )

    # 3. Initialize agent with chat model and instruction
    agent = GymnasiumAgent(
        model=chat_model,
        instruction=instruction,
        max_msg_history=config['max_msg_history'],
        max_tokens_input=config['max_tokens_input'],
        logger=logger
    )

    # 4. Initialize environment
    env = QueryGymEnv(
        dataset=dataset,
        action_definition=config['prompt_definitions']['actions'],
        max_steps=config['max_steps'],
        logger=logger,
        allow_execute_sql=True
    )

    logger.log(f"******* Episode {i}: BEGINS *******")

    # 5. Reset agent and environment
    agent.reset()
    observation = env.reset(i)
    logger.trace(observation)
    agent.observe(observation)

    # 6. Initialize action parser
    action_parser = RegexParser(
        regex=r"Action: (.*)",
        output_keys=["action"],
        default_output_key="action"
    )

    # 7. Initialize tracking variables
    parse_error_counts = 0
    info = {
        'steps': 0,
        'operations': 0,
        'idx': i,
        'SQL': env.dataset_item.gt,
        'operation_fail': 0,
        'difficulty': 'UNKNOWN'
    }
    action_input = None

    # 8. Main agent-environment interaction loop
    for n_llm_calls in range(config['max_llm_calls']):
        try:
            # Agent generates action based on current observation
            action_str = agent.act()

            try:
                # Parse action from agent response
                action = action_parser.parse(action_str)["action"]
                action_input = ast.literal_eval(action)

                # Handle legacy action name mapping
                if "perform" in action_input[0]:
                    action_input[0] = "execute_sql"

                # Execute action in environment
                observation, reward, done, truncated, info = env.step(action_input)

                # Add feedback based on reward
                if reward == env.GOAL_REWARD:
                    observation += "\nThe user query is answered.\n"
                elif reward == env.STEP_REWARD:
                    observation += "\nThe user query is almost answered. Keep Going.\n"

            except Exception as e:
                # Handle action parsing or execution errors
                parse_error_counts += 1
                observation = (
                    f"Error: Suggested action failed with an exception.\n\n"
                    f"# Action\n\n```\n{action_str}\n```\n\n\n"
                    f"# Exception\n\n```\n{traceback.format_exception_only(e)[0]}\n```"
                )
                if action_input is None:
                    action_input = "PARSE ERROR"
                reward, done, truncated = 0, False, False

                # Update info with error details
                info.update({
                    'generated_sql': '',
                    'operation_success': None,
                    'steps': env.data['steps'],
                    'SQL': env.dataset_item.gt
                })

        except Exception as e:
            # Handle unexpected errors
            env.close()
            logger.log("================ERROR ENCOUNTERED===================")
            logger.log(traceback.format_exc())
            del logger
            raise Exception(f"Error occurred in Instance {i}\n{traceback.format_exc()}")

        # Log trace information
        logger.trace(
            o=observation,
            a=action_input,
            r=reward,
            d=done,
            t=truncated,
            i=info,
            a_str=action_str
        )

        # Agent observes new state
        agent.observe(observation)
        action_input = None

        # Check if episode is complete
        if done or truncated:
            break

    # 9. Record final metrics
    logger.record_tabular("parse_errors", parse_error_counts)
    logger.record_tabular("llm_calls", n_llm_calls + 1)
    logger.dump_trace()

    # 10. Prepare result dictionary
    result_dict = logger._tabular.copy()
    logger.dump_tabular()

    logger.log(f"******* Episode {i}: ENDS *******")

    # 11. Cleanup
    env.close()
    del logger

    return result_dict

# Made with Bob
