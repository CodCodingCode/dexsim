"""Convert the MuJoCo LEFT Shadow Hand (mujoco_menagerie left_hand.xml) to USD.

Isaac ships only the RIGHT shadow hand, and a negative-scale mirror is rejected
by PhysX (reflection = non-positive determinant). The MuJoCo menagerie has a
proper LEFT model with valid right-handed frames, so we import THAT.

Output: assets/shadow_hand_left.usd  (bodies lh_*, e.g. lh_ffdistal; joints lh_*J*)
Then spawns it to verify stability and dumps joint/body names for env wiring.

  python scripts/build_left_hand_mjcf.py --headless
"""
from __future__ import annotations
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--xml", default=str(Path.home() /
    "DexGraspBench/third_party/mujoco_menagerie/shadow_hand/left_hand.xml"))
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

from dexsim import ASSETS_DIR
OUT = str(ASSETS_DIR / "shadow_hand_left.usd")

# enable the MJCF importer extension (name differs across Isaac versions)
from isaacsim.core.utils.extensions import enable_extension
ext_ok = False
for ext in ("isaacsim.asset.importer.mjcf", "omni.importer.mjcf"):
    try:
        if enable_extension(ext):
            ext_ok = ext
            break
    except Exception as e:
        print(f"  [{ext}] {e}")
print(f"[mjcf] importer extension: {ext_ok}")

import omni.kit.commands

# import config
cfg = None
for mod in ("isaacsim.asset.importer.mjcf", "omni.importer.mjcf"):
    try:
        _m = __import__(mod, fromlist=["_mjcf"])
        cfg = _m._mjcf.ImportConfig()
        break
    except Exception as e:
        print(f"  [{mod} ImportConfig] {e}")
if cfg is not None:
    cfg.set_fix_base(False)           # floating hand (mounted on the arm later)
    cfg.set_make_default_prim(True)
    cfg.set_import_inertia_tensor(True)
    try: cfg.set_self_collision(False)
    except Exception: pass

status, prim_path = omni.kit.commands.execute(
    "MJCFCreateAsset", mjcf_path=a.xml, import_config=cfg,
    prim_path="/shadow_hand_left", dest_path=OUT,
)
print(f"[mjcf] import status={status}  prim={prim_path}  -> {OUT}")

# verify it spawns + list names
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext, SimulationCfg

sim = SimulationContext(SimulationCfg(dt=1/120.0, device=a.device))
sim_utils.GroundPlaneCfg().func("/g", sim_utils.GroundPlaneCfg())
spawncfg = ArticulationCfg(
    prim_path="/World/LeftHand",
    spawn=sim_utils.UsdFileCfg(usd_path=OUT,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.5)),
    actuators={"f": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=3.0, damping=0.1)},
)
hand = Articulation(spawncfg)
sim.reset()
print(f"[mjcf] spawned: {hand.num_joints} joints, {hand.num_bodies} bodies")
print("  joints:", hand.data.joint_names)
print("  fingertip bodies:", [b for b in hand.data.body_names if "distal" in b])
maxv = 0.0
for _ in range(40):
    hand.set_joint_position_target(hand.data.default_joint_pos)
    hand.write_data_to_sim(); sim.step(); hand.update(1/120.0)
    maxv = max(maxv, float(hand.data.joint_vel[0].abs().max()))
print(f"[mjcf] 40-step max|vel| = {maxv:.3f} ({'STABLE' if maxv < 5 else 'unstable'})")
app.close()
