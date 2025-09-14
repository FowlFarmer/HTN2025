from prodplotter import DeltaActuator
import json
import time
import sys
import os

# Add the text_to_speech module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'text_to_speech'))
from bob_ross_simple_tts import speak_in_bob_ross_voice

# Load JSON
with open("painting_vectorization/results/step8_stroke_ordering/starry.json", "r") as f:
    data = json.load(f)

act = DeltaActuator(observe=True)

# Bob Ross narration for the first three masks
bob_ross_narrations = [
    "Oh, look at this cheeky little cypress tree! It's flickering like a flame, wanting to draw our attention. Let's appreciate its fiery charm and give it a hug from afar, shall we?",
    "The sky is doing cartwheels today, with vibrant blue swirls unfolding like a symphony. Let's imagine we're painting with the clouds, adding our own touches to this joyous spectacle.",
    "Nestled in the quiet of night, this serene village seems to hug its residents to sleep. Let's paint it with gentle, loving strokes, reminding ourselves that quiet moments are just as valuable as louder ones. Consider getting your paint, brushes and palette ready as you relax and enjoy the entire painting process, gently guiding you through each step. Isn't it lovely?"
]

# Count points
total_points = 0

for mask_index, mask in enumerate(data.get("mask_stroke_arrays", [])):
    # Speak Bob Ross narration for the first three masks
    if mask_index < len(bob_ross_narrations):
        print(f"\n🎨 Mask {mask_index + 1}: {bob_ross_narrations[mask_index][:50]}...")
        speak_in_bob_ross_voice(bob_ross_narrations[mask_index])
        print("🎵 Narration complete, beginning to paint...")
    
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