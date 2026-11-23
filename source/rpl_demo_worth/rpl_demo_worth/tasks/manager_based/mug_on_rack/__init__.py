import gymnasium as gym
from .mug_on_rack_env_cfg import MugOnRackEnvCfg, MugOnRackRenderEnvCfg

gym.register(
    id="RPL-MugOnRack-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": MugOnRackEnvCfg},
)

gym.register(
    id="RPL-MugOnRack-Render-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": MugOnRackRenderEnvCfg},
)
