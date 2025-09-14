from prodplotter import DeltaActuator
import json
import time
import sys
import os
import threading
import subprocess

# Add the text_to_speech and sentiment_analysis modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'text_to_speech'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sentiment_analysis'))
from bob_ross_simple_tts import speak_in_bob_ross_voice
from main import analyze_image_sentiment_from_url

# Load JSON
with open("painting_vectorization/results/step8_stroke_ordering/starry.json", "r") as f:
    data = json.load(f)

act = DeltaActuator(observe=True)

# Initialize music control
music_stop_event = threading.Event()

# Analyze the starry night image sentiment and start background music
print("🎭 Analyzing image sentiment for background music selection...")
starry_image_path = "painting_vectorization/examples/starrynight.jpg"
try:
    music_theme = analyze_image_sentiment_from_url(starry_image_path)
    music_file_path = os.path.join(os.path.dirname(__file__), '..', 'music_player', music_theme.filename)
    
    print(f"🎵 Selected music theme: {music_theme.theme_name}")
    print(f"🎼 Playing background music: {music_theme.filename}")
    
    # Start background music in a separate thread with lower volume and looping
    
    def play_background_music():
        try:
            while not music_stop_event.is_set():
                # Use afplay with volume control (macOS) - volume range is 0.0 to 1.0
                # We'll use a lower volume (0.3) for background music
                process = subprocess.Popen([
                    'afplay', music_file_path, '-v', '0.07'
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Wait for the music to finish or stop event
                while process.poll() is None and not music_stop_event.is_set():
                    time.sleep(0.1)
                
                # If stop event is set, terminate the process
                if music_stop_event.is_set():
                    process.terminate()
                    break
                    
                print("🔄 Looping background music...")
                
        except FileNotFoundError:
            print(f"Warning: Music file not found: {music_file_path}")
        except Exception as e:
            print(f"Warning: Could not play background music: {e}")
    
    music_thread = threading.Thread(target=play_background_music)
    music_thread.daemon = True  # Thread will close when main program exits
    music_thread.start()
    
except Exception as e:
    print(f"Warning: Could not analyze sentiment or start music: {e}")
    print("Continuing without background music...")

print("🎨 Starting painting process...")

# Bob Ross narration for the first three masks
bob_ross_narrations = [
    "Oh, look at this cheeky little cypress tree! It's flickering like a flame, wanting to draw our attention. Let's appreciate its fiery charm and give it a hug from afar, shall we?",
    "The sky is doing cartwheels today, with vibrant blue swirls unfolding like a symphony. Let's imagine we're painting with the clouds, adding our own touches to this joyous spectacle.",
    "Nestled in the quiet of night, this serene village seems to hug its residents to sleep. Let's paint it with gentle, loving strokes, reminding ourselves that quiet moments are just as valuable as louder ones. Consider getting your paint, brushes and palette ready as you relax and enjoy the entire painting process, gently guiding you through each step. Isn't it lovely?"
]

# Count points
total_points = 0

for mask_index, mask in enumerate(data.get("mask_stroke_arrays", [])):
    # Start Bob Ross narration thread for the first three masks (during drawing)
    narration_thread = None
    if mask_index < len(bob_ross_narrations):
        print(f"\n🎨 Mask {mask_index + 1}: Starting to paint while Bob Ross speaks...")
        narration_thread = threading.Thread(
            target=speak_in_bob_ross_voice, 
            args=(bob_ross_narrations[mask_index],)
        )
        narration_thread.daemon = True  # Thread will close when main program exits
        narration_thread.start()
    for stroke in mask.get("strokes", []):
        down = 0  
        points = stroke.get("points", [])
        for p in points:
            if(down == 1):
                if(stroke.get("warmth", 0) == 1):
                    print(f"🔥 Painting warm stroke with {len(points)} points...")
                    act.downCold()
                    # act.upCold()
                elif(stroke.get("warmth", 0) == 0):
                    print(f"❄️ Painting cold stroke with {len(points)} points...")
                    act.downCold()
                    # act.upHot()
                down += 1
            else:
                down += 1
            if not (isinstance(p, (list, tuple)) and len(p) == 2):
                continue
            total_points += 1
            x, y = float(p[0]), float(p[1])
            act.command(x, y)
            # time.sleep(0.05)  # small delay to visualize movement
        act.upHot()
        act.upCold()
    
    # Wait for narration to finish before moving to next mask
    if narration_thread and narration_thread.is_alive():
        print(f"⏳ Waiting for narration to complete for mask {mask_index + 1}...")
        narration_thread.join()
        print(f"✅ Narration complete for mask {mask_index + 1}")

print(f"Total points consumed: {total_points}")

# Stop background music when painting is complete
music_stop_event.set()
print("🎵 Stopping background music...")

print("🎨 Painting complete!")

act.return_home()