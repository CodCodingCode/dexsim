"""Build + save the composed MuJoCo scene XML (for inspection / external tools).

The env compiles the scene in-memory (~0.3 s), so this is never *required* --
it exists for debugging the model in the MuJoCo viewer or simulate binary:

  .venv/bin/python scripts/mj/build_scene.py           # -> assets/mj/piano_scene.xml
  .venv/bin/python -m mujoco.viewer --mjcf=assets/mj/piano_scene.xml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))

from dexsim.mjcf.scene import save_scene_xml, DEFAULT_SCENE_XML  # noqa: E402
from dexsim.tasks.piano_mj import PianoMjEnvCfg  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_SCENE_XML))
    args = parser.parse_args()
    out = save_scene_xml(PianoMjEnvCfg(), args.out)
    print(f"[build_scene] wrote {out}")


if __name__ == "__main__":
    main()
