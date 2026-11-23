import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from ..common.assets import get_object_usd
from ..common.base_env_cfg import RplBaseEnvCfg, RplBaseSceneCfg
from .mdp import events as bib_events
from .mdp import terminations as bib_terms


def _make_rigid_cfg(prim_path: str, name: str, pos: tuple[float, float, float], use_render: bool) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=get_object_usd(name, use_render=use_render),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                linear_damping=0.5,
                angular_damping=0.5,
                sleep_threshold=0.01,
                stabilization_threshold=0.01,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
        ),
    )


@configclass
class BlockInBowlSceneCfg(RplBaseSceneCfg):
    block: RigidObjectCfg = _make_rigid_cfg("{ENV_REGEX_NS}/Block", "box", (0.45, -0.10, 0.002), False)
    bowl: RigidObjectCfg = _make_rigid_cfg("{ENV_REGEX_NS}/Bowl", "bowl", (0.55, 0.10, 0.002), False)


@configclass
class BlockInBowlEventCfg:
    reset_robot = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_objects = EventTerm(
        func=bib_events.reset_block_and_bowl,
        mode="reset",
        params={
            "block_cfg": SceneEntityCfg("block"),
            "bowl_cfg": SceneEntityCfg("bowl"),
        },
    )


@configclass
class BlockInBowlTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=bib_terms.block_in_bowl_success)
    dropped = DoneTerm(func=bib_terms.object_out_of_workspace)


@configclass
class BlockInBowlEnvCfg(RplBaseEnvCfg):
    """Physics copy: RPL-BlockInBowl-v0."""

    use_render_assets: bool = False
    scene: BlockInBowlSceneCfg = BlockInBowlSceneCfg(num_envs=1, env_spacing=2.5)
    events: BlockInBowlEventCfg = BlockInBowlEventCfg()
    terminations: BlockInBowlTerminationsCfg = BlockInBowlTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.block.spawn.usd_path = get_object_usd("box", use_render=self.use_render_assets)
        self.scene.bowl.spawn.usd_path = get_object_usd("bowl", use_render=self.use_render_assets)


@configclass
class BlockInBowlRenderEnvCfg(BlockInBowlEnvCfg):
    """Render copy: RPL-BlockInBowl-Render-v0 (single config switch)."""

    use_render_assets: bool = True
