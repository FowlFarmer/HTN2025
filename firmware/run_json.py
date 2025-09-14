from prodplotter import DeltaActuator
import json
import time
# Load JSON
with open("painting_vectorization/results/step8_stroke_ordering/starry.json", "r") as f:
    data = json.load(f)

act = DeltaActuator(observe=True)

# Count points
total_points = 0

for mask in data.get("mask_stroke_arrays", []):
    for stroke in mask.get("strokes", []):
        points = stroke.get("points", [])
        # act.downCold()
        # if(stroke.get("warmth_class", int) == 1):
        #     act.downWarm()
        # elif(stroke.get("warmth_class", int) == 0):
        act.upCold()
        for p in points:
            if not (isinstance(p, (list, tuple)) and len(p) == 2):
                continue
            total_points += 1
            x, y = float(p[0]), float(p[1])
            act.command(x, y)
            # time.sleep(0.05)  # small delay to visualize movement
        # if(stroke.get("warmth_class", int) == 1):
        #     act.upWarm()
        # elif(stroke.get("warmth_class", int) == 0):
            act.downCold()

print(f"Total points consumed: {total_points}")