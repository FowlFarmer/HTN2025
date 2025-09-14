from prodplotter import DeltaActuator
import json
import time
import sys
import os
import threading
import subprocess

# Add the text_to_speech and cohere-multimodal modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'text_to_speech'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cohere-multimodal'))

from bob_ross_simple_tts import speak_in_bob_ross_voice

# Import painting profiles system
try:
    from painting_profiles import get_narrations_for_painting, get_music_for_painting, painting_manager
    PAINTING_PROFILES_AVAILABLE = True
    print("✅ Painting profiles system available!")
except ImportError as e:
    print(f"⚠️ Could not import painting profiles: {e}")
    PAINTING_PROFILES_AVAILABLE = False

# ===== CONFIGURATION: Change this to select different paintings =====
JSON_FILE = "starry.json"  # Options: starry.json, m3.json, eiffel.json, titanic.json, petronas.json, monalisa.json
# ================================================================

JSON_PATH = f"painting_vectorization/results/step8_stroke_ordering/{JSON_FILE}"

print(f"🎨 Loading painting data from: {JSON_FILE}")

# Load JSON
try:
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"❌ JSON file not found: {JSON_PATH}")
    print("Available JSON files should be in painting_vectorization/results/step8_stroke_ordering/")
    sys.exit(1)

act = DeltaActuator(observe=True)

# Initialize music control
music_stop_event = threading.Event()

# Get music selection based on painting profile
print("🎵 Selecting background music based on painting...")
if PAINTING_PROFILES_AVAILABLE:
    selected_music_file = get_music_for_painting(JSON_FILE)
    print(f"🎼 Profile-based music selection: {selected_music_file}")
else:
    # Fallback music selection
    selected_music_file = "Calm and Serenity.mp3"
    print(f"🎼 Default music selection: {selected_music_file}")

music_file_path = os.path.join(os.path.dirname(__file__), '..', 'music_player', selected_music_file)

try:
    print(f"🎵 Playing background music: {selected_music_file}")
    
    # Start background music in a separate thread with lower volume and looping
    
    def play_background_music():
        try:
            while not music_stop_event.is_set():
                # Use afplay with volume control (macOS) - volume range is 0.0 to 1.0
                # We'll use a lower volume (0.3) for background music
                process = subprocess.Popen([
                    'afplay', music_file_path, '-v', '0.15'
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

# Get Bob Ross narrations based on painting profile
print("\n🗣️ Selecting Bob Ross narrations based on painting...")

# Hardcoded fallback narrations (for safety)
fallback_narrations = [
    "Let's create something beautiful together on our canvas. Every stroke tells a story, and today we're telling yours.",
    "Remember, there are no mistakes in art, only happy accidents that lead us to unexpected beauty.",
    "Look how the colors dance together! Each element finds its perfect place in our composition, just like in life."
]

# Try to get profile-based narrations
bob_ross_narrations = fallback_narrations  # Default to fallback
narration_source = "hardcoded fallback"

if PAINTING_PROFILES_AVAILABLE:
    try:
        print(f"🤖 Getting profile-based narrations for {JSON_FILE}...")
        profile_narrations = get_narrations_for_painting(JSON_FILE, max_count=5)  # Get up to 5 narrations
        
        if profile_narrations and len(profile_narrations) >= 3:
            bob_ross_narrations = profile_narrations
            narration_source = f"painting profile ({JSON_FILE})"
            print(f"✅ Using profile-based narrations for {JSON_FILE}!")
        else:
            print(f"⚠️ Insufficient profile narrations, using fallback...")
            
    except Exception as e:
        print(f"⚠️ Error getting profile narrations: {e}")
        print("🔄 Falling back to default narrations...")

print(f"📝 Using {narration_source} narrations")

# Display the narrations that will be used
print(f"\n--- Bob Ross Narrations for {JSON_FILE} ---")
for i, narration in enumerate(bob_ross_narrations, 1):
    preview = narration[:80] + "..." if len(narration) > 80 else narration
    print(f"{i}. {preview}")

# Show painting profile info if available
if PAINTING_PROFILES_AVAILABLE:
    profile = painting_manager.get_profile_by_json(JSON_FILE)
    if profile:
        print(f"\n--- Painting Profile: {profile.name} ---")
        print(f"Music: {profile.music_file}")
        print(f"Narrations: {len(profile.narrations)} available")
        if profile.mask_descriptions:
            print(f"Masks: {len(profile.mask_descriptions)} described")

# Count points
total_points = 0

# Limit narrations to available masks
available_masks = len(data.get("mask_stroke_arrays", []))
narrations_to_use = bob_ross_narrations[:available_masks]

print(f"\n🎭 Found {available_masks} masks, using {len(narrations_to_use)} narrations")

for mask_index, mask in enumerate(data.get("mask_stroke_arrays", [])):
    # Start Bob Ross narration thread for available narrations
    narration_thread = None
    if mask_index < len(narrations_to_use):
        print(f"\n🎨 Mask {mask_index + 1}: Starting to paint while Bob Ross speaks...")
        
        # Show which narration is being used
        narration_text = narrations_to_use[mask_index]
        preview = narration_text[:100] + "..." if len(narration_text) > 100 else narration_text
        print(f"🗣️ Narration {mask_index + 1}: {preview}")
        
        narration_thread = threading.Thread(
            target=speak_in_bob_ross_voice, 
            args=(narration_text,)
        )
        narration_thread.daemon = True  # Thread will close when main program exits
        narration_thread.start()
    else:
        print(f"\n🎨 Mask {mask_index + 1}: Painting silently (no narration available)")
    
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
print(f"📝 Used: {narration_source}")
print(f"🎵 Music: {selected_music_file}")

act.return_home()