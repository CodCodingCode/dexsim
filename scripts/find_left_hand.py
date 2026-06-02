"""List the Isaac nucleus ShadowHand asset folder and search the library for any
LEFT shadow hand USD."""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import omni.client
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
print("ISAAC_NUCLEUS_DIR =", ISAAC_NUCLEUS_DIR)

def ls(url):
    res, entries = omni.client.list(url)
    if str(res) != "Result.OK":
        print(f"  [list failed: {res}] {url}")
        return []
    return [e.relative_path for e in entries]

for sub in ["Robots/ShadowHand", "Robots/ShadowHand/", "Robots"]:
    url = f"{ISAAC_NUCLEUS_DIR}/{sub}"
    print(f"\n== {url} ==")
    for name in ls(url):
        print("   ", name)

# recursive-ish search for 'left' under ShadowHand and Robots
import_root = f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowHand"
print("\n== searching for *left* under ShadowHand ==")
for name in ls(import_root):
    if "left" in name.lower() or "lh" in name.lower():
        print("   LEFT CANDIDATE:", name)
    if name.lower().endswith((".usd", ".usda")):
        print("   usd:", name)
app.close()
