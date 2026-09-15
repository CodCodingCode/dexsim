# CPU observation-layout test (no Isaac Sim needed)

Isaac Sim is Linux/Windows + NVIDIA only, so the real smoke test
(`../piano_env_smoke.py`) cannot run on a Mac. This harness stubs `isaaclab`
(`./isaaclab/`) just enough to import the REAL `PianoEnvCfg` and
`PianoEnv._get_observations`, then checks obs/critic sizes and that every
chunk of the policy obs is the expected slice of (fake) sim state.

    cd scripts/smoke/cpu_obs_test
    uv run --python 3.12 --with torch --with numpy --with gymnasium python test_obs_layout.py

It does NOT exercise physics, reward, or PPO -- for those, run the real smoke
test / `train_piano.py` on the Isaac box.
