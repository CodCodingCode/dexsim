"""IK-solve BOTH arms to a palm-DOWN, fingers-forward (-X) pose over their windows
(left y=-0.30, right y=+0.30). With a left hand on the left arm and right hand on
the right, the two solutions come out mirror-symmetric -> a true L/R pair.
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a=p.parse_args([]); a.headless=True
app=AppLauncher(a).app
import numpy as np, torch, math, json
from isaaclab.sim import SimulationContext, SimulationCfg
from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg
from dexsim.render import studio
from dexsim.piano.ik import WristPoseIK
cfg=PianoEnvCfg(); sim=SimulationContext(SimulationCfg(dt=1/120.0,device="cuda"))
piano,left,right=studio.build_scene(cfg); sim.reset()
for _ in range(3): sim.step()
def quat2R(q):
    w,x,y,z=q; return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def R2quat(R):
    # stable 4-branch conversion (the old trace-only branch blew up near 180 deg,
    # where 1+trace ~ 0 -> w ~ 0 -> x/y/z = offdiag/(4w) exploded; that is exactly
    # the rotation the mirrored hand needs, so it produced a garbage target quat).
    t=R[0,0]+R[1,1]+R[2,2]
    if t>0:
        s=np.sqrt(t+1.0)*2; return np.array([0.25*s,(R[2,1]-R[1,2])/s,(R[0,2]-R[2,0])/s,(R[1,0]-R[0,1])/s])
    if R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2; return np.array([(R[2,1]-R[1,2])/s,0.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s])
    if R[1,1]>R[2,2]:
        s=np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2; return np.array([(R[0,2]-R[2,0])/s,(R[0,1]+R[1,0])/s,0.25*s,(R[1,2]+R[2,1])/s])
    s=np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2; return np.array([(R[1,0]-R[0,1])/s,(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,0.25*s])
def basis(u,v):
    u=u/np.linalg.norm(u); w=np.cross(u,v); w/=np.linalg.norm(w); return np.stack([u,np.cross(w,u),w],1)
ARM=["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"]
def palm_pose(rob):
    pid=rob.find_bodies("robot0_palm")[0][0]
    return (rob.data.body_pos_w[0,pid].cpu().numpy(),
            rob.data.body_quat_w[0,pid].cpu().numpy())

def palm_down_target(rob, ty):
    """Palm-DOWN, fingers-forward (-X) target (pos, quat) for one arm, built from
    its OWN measured fingertip frame (handedness-corrected)."""
    pid=rob.find_bodies("robot0_palm")[0][0]; ff=rob.find_bodies("robot0_ffdistal")[0][0]
    lf=rob.find_bodies("robot0_lfdistal")[0][0]; mf=rob.find_bodies("robot0_mfdistal")[0][0]
    g=lambda i: rob.data.body_pos_w[0,i].cpu().numpy()
    P,Q,F,L,M=g(pid),rob.data.body_quat_w[0,pid].cpu().numpy(),g(ff),g(lf),g(mf); R=quat2R(Q)
    fwd=(M-P)/np.linalg.norm(M-P); nrm=np.cross(L-F,M-P); nrm/=np.linalg.norm(nrm)
    if nrm[2]>0: nrm=-nrm        # downward (palmar-in-seed) face; see handedness note below
    f_loc=R.T@fwd; n_loc=R.T@nrm
    fW=np.array([-1.0,0,0]); nW=np.array([0,0,-1.0])
    R_t=basis(fW,nW)@basis(f_loc,n_loc).T
    return np.array([0.78,ty,0.86]), R2quat(R_t)

def run_ik(rob, tp_np, tq_np):
    names=rob.data.joint_names
    tp=torch.tensor([tp_np],device="cuda",dtype=torch.float32)
    tq=torch.tensor([tq_np],device="cuda",dtype=torch.float32)
    ik=WristPoseIK(rob,"robot0_palm",arm_only=True,max_step=0.04,max_ang_step=0.1,damping=0.06)
    for _ in range(400):
        q=ik.solve(tp,tq); rob.write_joint_state_to_sim(q,torch.zeros_like(q)); rob.write_data_to_sim(); sim.step()
    return {n:round(float(rob.data.joint_pos[0,names.index(n)]),3) for n in ARM}

# SOLVE LEFT, THEN MIRROR TO RIGHT (symmetry by construction, not by luck).
# Solving each arm independently let IK pick unrelated branches -> one upright, one
# toppled. Instead: solve the left palm-down, read what it ACHIEVED, reflect that pose
# across the keyboard centreline y=0 (S=diag(1,-1,1): pos.y->-pos.y, R->S R S), and drive
# the right arm to that mirror. Both arms then hold the same posture, mirror-imaged.
tpL,tqL=palm_down_target(left,-0.30)
res_left=run_ik(left,tpL,tqL)
PL,QL=palm_pose(left)                       # what the LEFT actually reached
S=np.diag([1.0,-1.0,1.0])
RL=quat2R(QL); R_mir=S@RL@S                  # mirror orientation across y=0
P_mir=np.array([PL[0],-PL[1],PL[2]])         # mirror position across y=0
res_right=run_ik(right,P_mir,R2quat(R_mir))
res={"left":res_left,"right":res_right}
json.dump(res,open("logs/both_arms.json","w"),indent=2); print(json.dumps(res,indent=2),flush=True)
app.close()
