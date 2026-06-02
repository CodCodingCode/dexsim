"""Tabletop scene for UR10e + Shadow Hand grasping.

A single InteractiveScene the imitation scripts reuse: ground plane, dome light,
a table, the combined arm+hand robot, and one manipulable object. The object is
spawned as a rigid body so BODex/DexGraspNet grasp trajectories can act on it.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from dexsim.assets import UR10E_SHADOW_CFG


@configclass
class TabletopGraspSceneCfg(InteractiveSceneCfg):
    """UR10e+Shadow over a table with one graspable object."""

    # ground + light
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.95)),
    )

    # table (props library)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0)),
    )

    # combined arm + hand (requires assets/ur10e_shadow.usd; see build script)
    robot = UR10E_SHADOW_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # a graspable object on the table -- swap for a BODex/DexGraspNet mesh.
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.3, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.83)),
    )
