# Source: https://github.com/langchain-ai/langchain/blob/master/cookbook/gymnasium_agent_simulation.ipynb

import gymnasium as gym
import tenacity
from langchain_core.messages.utils import count_tokens_approximately
from openai import APIError, BadRequestError
from query_gym.utils.logging_util import setup_logger
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from httpx import RequestError
import tiktoken


class GymnasiumAgent:
    """
    A reinforcement learning agent that interacts with an environment using a language model (LLM).
    
    This agent:
    - Maintains a message history for context.
    - Observes environment states and sends them to the LLM.
    - Handles token limits by trimming message history.
    - Implements retry logic for API calls with exponential backoff.
    
    Attributes:
        model: The LLM model used for generating actions.
        system_message: Initial system instruction for the LLM.
        first_observation: The first observation received from the environment.
        max_tokens_input: Maximum tokens allowed for input to the LLM.
        message_history: List of messages exchanged with the LLM.
        ret: Cumulative reward (optional usage).
        max_history: Maximum number of messages to keep in history.
        first_two_msgs_len: Token count for system + first observation messages.
        logger: Logger instance for debugging and tracking.
    """

    def __init__(self, model, instruction, max_tokens_input=100000, max_msg_history=3000, logger=None):
        """
        Initialize the GymnasiumAgent.

        Args:
            model: The language model instance.
            instruction (str): System-level instruction for the LLM.
            max_tokens_input (int): Maximum token limit for input messages.
            max_msg_history (int): Maximum number of messages to retain in history.
            logger: Optional logger instance; if None, a default logger is created.
        """
        self.model = model
        self.system_message = SystemMessage(content=instruction)
        self.first_observation = None
        self.max_tokens_input = max_tokens_input
        self.message_history = []
        self.ret = 0
        self.max_history = max_msg_history
        self.first_two_msgs_len = 0
        self.logger = logger or setup_logger("gym_agent", variant=None)

    def dump_message(self, message):
        """
        Log a message to the logger for debugging and tracking.

        Args:
            message: A message object (SystemMessage, HumanMessage, AIMessage).
        """
        self.logger.dump_message_history({
            'agent_type': "gym_agent",
            'role': message.type,
            'content': message.text(),
            'meta_data': message.response_metadata,
            'additional_kwargs': message.additional_kwargs
        })

    def reset(self):
        """
        Reset the agent state for a new episode.
        Clears message history and logs the system message.
        """
        self.first_observation = None
        self.first_two_msgs_len = 0
        self.message_history = []
        self.dump_message(self.system_message)

    def observe(self, obs):
        """
        Add an observation from the environment to the message history.

        Args:
            obs (str): Observation text from the environment.
        """
        observation_object = HumanMessage(content=obs)

        if self.first_observation is None:
            # First observation: calculate token length for system + observation
            self.first_observation = observation_object
            self.first_two_msgs_len = count_tokens_approximately(
                [self.system_message, self.first_observation]
            ) + 100  # Add epsilon for approximation error
        else:
            # Maintain message history within max limit
            if len(self.message_history) >= self.max_history:
                self.message_history.pop(2)  # Remove oldest non-system message
            self.message_history.append(observation_object)

        self.dump_message(observation_object)

    def trim_before_retry(self, retry_state):
        """
        Adjust token length before retrying if context length exceeds limits.

        Args:
            retry_state: Tenacity retry state object.
        """
        self.logger.log("-----trim_before_retry-----")
        exception = retry_state.outcome.exception()
        self.logger.log(f"Exception: {exception.message}")

        # Handle context length errors from OpenAI API
        if isinstance(exception, BadRequestError) and "context length" in exception.message:
            requested_context = int(exception.message.split("you requested")[-1].split("tokens in the messages")[0])
            self.logger.log(f"self.max_tokens_input: {self.max_tokens_input}")
            self.logger.log(f"requested_context: {requested_context}")
            self.logger.log(f"self.first_two_msgs_len: {self.first_two_msgs_len}")
            self.first_two_msgs_len += requested_context - self.max_tokens_input
            self.logger.log(f"Updating self.first_two_msgs_len to {self.first_two_msgs_len}")
        self.logger.log("---------------------------")

    def invoke_llm(self):
        """
        Invoke the language model to generate an action based on message history.
        Implements retry logic for API errors.

        Returns:
            act_message: AIMessage object containing the LLM's response.
        """
        act_message = None

        # Retry up to 3 times with exponential backoff for API errors
        for attempt in tenacity.Retrying(
            stop=tenacity.stop_after_attempt(3),
            wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
            retry=tenacity.retry_if_exception_type((APIError, ConnectionError, RequestError)),
            before_sleep=self.trim_before_retry,
            reraise=True
        ):
            with attempt:
                # Trim message history to fit within token limits
                trimmed_message = trim_messages(
                    self.message_history,
                    strategy="last",  # Keep last N messages
                    token_counter=count_tokens_approximately,
                    max_tokens=self.max_tokens_input - self.first_two_msgs_len,
                    start_on=("human", "ai"),
                    end_on=("human", "ai"),
                    allow_partial=False
                )

                # Invoke the LLM with system message, first observation, and trimmed history
                act_message = self.model.invoke([self.system_message, self.first_observation] + trimmed_message)

                # Update token length if history was empty
                if len(trimmed_message) == 0 and 'token_usage' in act_message.response_metadata:
                    if 'prompt_tokens' in act_message.response_metadata['token_usage']:
                        self.first_two_msgs_len = act_message.response_metadata['token_usage']['prompt_tokens']

        return act_message

    def act(self):
        """
        Generate an action using the LLM and update message history.

        Returns:
            str: The content of the LLM's response (action).
        """
        act_message = self.invoke_llm()
        self.message_history.append(act_message)

        # Maintain message history within max limit
        if len(self.message_history) >= self.max_history:
            self.message_history.pop(2)

        self.dump_message(act_message)
        return act_message.content
