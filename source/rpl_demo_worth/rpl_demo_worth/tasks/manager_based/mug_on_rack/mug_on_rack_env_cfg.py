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
from .mdp import events as mor_events
from .mdp import terminations as mor_terms


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
class MugOnRackSceneCfg(RplBaseSceneCfg):
    mug: RigidObjectCfg = _make_rigid_cfg("{ENV_REGEX_NS}/Mug", "mug", (0.45, -0.12, 0.002), False)
    mug_rack: RigidObjectCfg = _make_rigid_cfg("{ENV_REGEX_NS}/MugRack", "mug_rack", (0.55, 0.12, 0.002), False)


@configclass
class MugOnRackEventCfg:
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
        func=mor_events.reset_mug_and_rack,
        mode="reset",
        params={
            "mug_cfg": SceneEntityCfg("mug"),
            "rack_cfg": SceneEntityCfg("mug_rack"),
        },
    )


@configclass
class MugOnRackTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=mor_terms.mug_on_rack_success)
    dropped = DoneTerm(func=mor_terms.mug_out_of_workspace)


@configclass
class MugOnRackEnvCfg(RplBaseEnvCfg):
    """Physics copy: RPL-MugOnRack-v0."""

    use_render_assets: bool = False
    scene: MugOnRackSceneCfg = MugOnRackSceneCfg(num_envs=1, env_spacing=2.5)
    events: MugOnRackEventCfg = MugOnRackEventCfg()
    terminations: MugOnRackTerminationsCfg = MugOnRackTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.mug.spawn.usd_path = get_object_usd("mug", use_render=self.use_render_assets)
        self.scene.mug_rack.spawn.usd_path = get_object_usd("mug_rack", use_render=self.use_render_assets)


@configclass
class MugOnRackRenderEnvCfg(MugOnRackEnvCfg):
    """Render copy: RPL-MugOnRack-Render-v0 (single config switch)."""

    use_render_assets: bool = True
