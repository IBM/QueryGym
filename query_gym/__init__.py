from gymnasium.envs.registration import register
from .utils.logging_util import setup_logger

register(
    id="query_gym_v1",
    entry_point="query_gym.envs:QueryGymEnv",
)

register(
    id="query_gym_sqlite_v1",
    entry_point="query_gym.envs:SQLiteQueryGym",
)
