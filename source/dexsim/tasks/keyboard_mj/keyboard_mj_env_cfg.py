"""Config for the MuJoCo typing task (two Shadow Hands on the MacBook keyboard).

Task: type a given text (a sequence of keystrokes) as fast as possible with
no stray key presses. Scored like the piano (correct / stray registrations),
plus characters per second. Pure RL from scratch, same PPO recipe as the
piano ([[dexsim-ppo-robopianist-goal]]): no demos, no BC.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from dexsim.mjcf.keyboard import NUM_KEYS

SIM_DT = 0.002            # scissor keys are light and stiff: 500 Hz physics
DECIMATION = 10           # 50 Hz policy -- typing needs faster control than
CONTROL_DT = SIM_DT * DECIMATION   # the 20 Hz piano policy (0.02 s)

# per hand: 20 hand actuators + 3 gantry axes (x, y, z)
HAND_ACT = 20
GANTRY_ACT = 3
PER_HAND_ACT = HAND_ACT + GANTRY_ACT
NUM_FINGERS = 10          # [L th ff mf rf lf, R th ff mf rf lf]

# lowercase / space / basic punctuation only: nothing needs shift, so the
# first curriculum is pure reach + press. Shifted characters are supported
# by the env (opposite-hand shift) -- put them in the corpus when ready.
DEFAULT_CORPUS = [
    "hello world",
    "the quick brown fox",
    "jumps over the lazy dog",
    "type faster",
    "robot hands",
    "piano to keyboard",
    "mujoco is fast",
    "asdf jkl;",
    "a sad lad",
    "fall hall",
    "sit still",
    "jade lake",
    "how are you",
    "fine thanks",
    "one two three",
]


@dataclass
class KeyboardMjEnvCfg:
    seed: int = 0
    sim_dt: float = SIM_DT
    decimation: int = DECIMATION
    control_dt: float = CONTROL_DT

    # --- text ---------------------------------------------------------------
    text: str = ""                    # fixed text; "" -> sample from corpus
    corpus: list = field(default_factory=lambda: list(DEFAULT_CORPUS))
    max_chars: int = 24               # truncate sampled texts to this
    # episode budget: this many seconds per character (+ a fixed head start)
    time_per_char_s: float = 1.0
    episode_head_s: float = 1.0
    goal_lookahead: int = 3           # upcoming keystrokes in the obs

    # --- scene (see dexsim.mjcf.keyboard_scene.KeyboardSceneCfg) ----------
    hover: float = 0.020              # home hover above the caps
    gantry_xy_limit: float = 0.20
    gantry_z_range: tuple = (-0.08, 0.06)

    # --- actions ------------------------------------------------------------
    action_space: int = 2 * PER_HAND_ACT                   # 46
    hand_action_scale: float = 0.8    # rad residual around pose G (== piano)
    gantry_xy_action_scale: float = 0.15   # m: absolute XY setpoint = scale * a
    gantry_z_action_scale: float = 0.04    # m: z setpoint in [-0.04, 0.04] about hover
    freeze_left_hand: bool = False    # curriculum: right hand only (left held)
    freeze_right_hand: bool = False

    # --- rewards ------------------------------------------------------------
    key_reward: float = 1.0           # per correct keystroke
    stray_penalty: float = 0.5        # per wrong keystroke
    hold_penalty: float = 0.02        # per step per non-target key held down
    reach_weight: float = 0.05        # dense: assigned fingertip -> target cap
    reach_scale: float = 0.10         # m: reach reward is 0 beyond this distance
    press_weight: float = 0.05        # dense: target key depression fraction
    time_penalty: float = 0.005       # per step: type FASTER
    complete_bonus: float = 5.0       # finished the text
    energy_weight: float = 0.0005
    jerk_weight: float = 0.0

    # --- observation --------------------------------------------------------
    obs_key_state: bool = True        # all 78 key depressions
    obs_prev_action: bool = True
    observation_space: int = 0        # computed
    state_space: int = 0

    def obs_dim(self) -> int:
        n = 0
        n += 2 * (24 + GANTRY_ACT)                # joint pos (per hand 24 + 3)
        n += 2 * (24 + GANTRY_ACT)                # joint vel
        n += NUM_FINGERS * 3                      # fingertips rel. target cap
        n += 2 * 3                                # palms rel. target cap
        n += self.goal_lookahead * (2 + 2 + 1 + NUM_FINGERS)   # upcoming keys
        n += 1                                    # shift held
        if self.obs_key_state:
            n += NUM_KEYS
        if self.obs_prev_action:
            n += self.action_space
        return n

    def critic_extra_dim(self) -> int:
        return 0

    def __post_init__(self):
        self.control_dt = self.sim_dt * self.decimation
        self.action_space = 2 * PER_HAND_ACT
        self.observation_space = self.obs_dim()
        self.state_space = self.observation_space

    def to_dict(self) -> dict:
        return asdict(self)
