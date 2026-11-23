import gymnasium as gym
from .block_in_bowl_env_cfg import BlockInBowlEnvCfg, BlockInBowlRenderEnvCfg

gym.register(
    id="RPL-BlockInBowl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": BlockInBowlEnvCfg},
)

gym.register(
    id="RPL-BlockInBowl-Render-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": BlockInBowlRenderEnvCfg},
)
