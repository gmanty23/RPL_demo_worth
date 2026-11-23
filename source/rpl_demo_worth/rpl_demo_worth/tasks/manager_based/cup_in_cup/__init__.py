import gymnasium as gym
from .cup_in_cup_env_cfg import CupInCupEnvCfg, CupInCupRenderEnvCfg

gym.register(
    id="RPL-CupInCup-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": CupInCupEnvCfg},
)

gym.register(
    id="RPL-CupInCup-Render-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": CupInCupRenderEnvCfg},
)
