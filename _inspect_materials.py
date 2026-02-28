"""Inspect fixture USD materials."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdShade
import os

files = [
    "exports/g1_kitchen_scene/fixtures/fridge_articulated.usd",
    "exports/g1_kitchen_scene/fixtures/dishwasher_articulated.usd",
    "exports/g1_kitchen_scene/fixtures/microwave_counter_articulated.usd",
]

for usd_file in files:
    name = os.path.basename(usd_file)
    print(f"\n=== {name} ===")
    stage = Usd.Stage.Open(usd_file)
    if stage is None:
        print("  FAILED to open")
        continue
    for p in stage.Traverse():
        tname = p.GetTypeName()
        if tname == "Material":
            print(f"  Mat: {p.GetPath()}")
        elif tname == "Shader":
            shader = UsdShade.Shader(p)
            sid = shader.GetShaderId()
            print(f"    Shader: {p.GetPath().name}  id={sid}")

app.close()
