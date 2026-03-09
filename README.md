# HTN2025 — Painting Vectorization & Robotic Drawing System

Built for **Hack the North 2025**, this project transforms any digital painting into a sequence of robot-executable drawing strokes, complete with Bob Ross–style AI narration and mood-matched background music.

---

## Table of Contents

1. [What the Project Does](#1-what-the-project-does)
2. [Repository Structure](#2-repository-structure)
3. [How Each Part Works](#3-how-each-part-works)
   - [Painting Vectorization Pipeline (Steps 1–8)](#a-painting-vectorization-pipeline-steps-18)
   - [Firmware & Robot Control](#b-firmware--robot-control)
   - [AI Narration (Cohere Multimodal + TTS)](#c-ai-narration-cohere-multimodal--tts)
   - [Sentiment Analysis & Music Selection](#d-sentiment-analysis--music-selection)
   - [Camera Integration](#e-camera-integration)
4. [Data Flow](#4-data-flow)
5. [Configuration](#5-configuration)
6. [Tech Stack](#6-tech-stack)
7. [Entry Points & How to Run](#7-entry-points--how-to-run)

---

## 1. What the Project Does

The system takes any digital image (photograph, painting, etc.) and:

1. **Analyzes the image** — segments it into meaningful regions (sky, trees, buildings, etc.) using an AI model (SAM).
2. **Extracts strokes** — converts regions into simplified polylines that mimic brushstrokes.
3. **Scales to millimetres** — maps pixel coordinates to real-world millimetre coordinates on a physical 130 mm × 130 mm canvas.
4. **Classifies colours** — labels each stroke as warm (reds/yellows) or cool (blues/greens).
5. **Orders strokes** — uses a TSP-like algorithm to minimise how often the pen needs to be lifted.
6. **Drives a delta robot** — sends the stroke sequence to a physical robotic arm that paints on canvas.
7. **Narrates and plays music** — simultaneously reads Bob Ross–style commentary aloud and plays mood-matched ambient music.

---

## 2. Repository Structure

```
HTN2025/
├── painting_vectorization/          # Core image-to-strokes pipeline (8 steps)
│   ├── step1_preprocessing/         # Load, resize, and enhance the image
│   ├── step2_segmentation/          # Segment the image into distinct regions (SAM)
│   ├── step3_edge_extraction/       # Detect edges and contours inside each region
│   ├── step4_stroke_graph/          # Build a graph from the skeleton pixels
│   ├── step5_vectorization/         # Simplify paths with Ramer-Douglas-Peucker
│   ├── step6_sampling/              # Resample at uniform 1 mm spacing, scale to mm
│   ├── step7_color_detection/       # Warm/cool colour classification per stroke
│   ├── step8_stroke_ordering/       # TSP ordering, pen-lift instructions, JSON output
│   ├── examples/                    # Sample input images
│   ├── results/                     # Output visualizations & JSON stroke data
│   ├── models/                      # SAM model weights (downloaded separately)
│   ├── constants.py                 # Central config (canvas size, spacing, paths)
│   ├── process_image.py             # Orchestrator — runs all 8 steps end-to-end
│   └── visualization_utils.py       # Matplotlib helper functions
│
├── firmware/                        # Robot hardware interface
│   ├── run_json.py                  # Main demo entry point — loads JSON & drives robot
│   ├── prodplotter.py               # Delta robot actuator (kinematics + servo control)
│   ├── driver.py                    # Motor/stepper wrapper
│   ├── gcode_serial.py              # G-code serial communication with Arduino
│   ├── testplotter.py               # Software simulation (no hardware required)
│   └── repeater.ino                 # Arduino firmware for stepper motors
│
├── cohere-multimodal/               # AI multimodal painting analysis
│   ├── painting_profiles.py         # Pre-generated Bob Ross narrations & music
│   ├── painting_mask_analyzer.py    # Live Cohere Vision API integration
│   ├── starrynight_analyzer.py      # Specialised analysis for Starry Night
│   ├── add_painting_profile.py      # CLI tool to add a new painting profile
│   └── PAINTING_PROFILES_GUIDE.md   # Guide to the painting profiles system
│
├── text_to_speech/                  # Bob Ross voice synthesis
│   ├── bob_ross_simple_tts.py       # ElevenLabs API or macOS `say` fallback
│   └── text_to_speech.md            # API key setup guide
│
├── music_player/                    # Background music playback
│   └── music_player.py              # Pygame-based player with themed tracks
│
├── sentiment_analysis/              # Painting mood detection
│   └── main.py                      # Cohere text classification → music category
│
├── camera_integration/              # Voice-activated webcam capture
│   ├── voice_camera.py              # Speech recognition + OpenCV webcam
│   └── photos/                      # Captured images saved here
│
├── preprocessing/                   # Legacy preprocessing utilities
├── requirements.txt                 # All Python dependencies
├── INSTALLATION_GUIDE.md            # Full setup and troubleshooting guide
└── painting_vectorization_spec.md   # Detailed algorithm specification
```

---

## 3. How Each Part Works

### A. Painting Vectorization Pipeline (Steps 1–8)

The pipeline lives in `painting_vectorization/` and is orchestrated by `process_image.py`. Each step receives the output of the previous step and produces data for the next.

---

#### Step 1 — Image Preprocessing (`step1_preprocessing/image_preprocessing.py`)

**What it does:** Prepares the raw image for downstream ML models.

**How it works:**
- Loads the image with OpenCV (`cv2.imread`) and converts BGR → RGB (or RGBA → RGB).
- Resizes the image so the longest edge is at most 1024 px, recording the `scale_factor` so coordinates can be mapped back.
- Applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to sharpen local contrast without blowing out highlights.
- Optionally applies bilateral filtering to reduce noise while keeping edges sharp.
- Converts the result to a `float32` array in the range `[0, 1]` for compatibility with PyTorch models.

**Outputs:** Preprocessed image array, scale factor, original dimensions.

---

#### Step 2 — Segmentation (`step2_segmentation/segmentation.py`)

**What it does:** Splits the image into a set of meaningful binary masks (one per distinct visual element, e.g. sky, tree, building).

**How it works:**
- Loads the **SAM (Segment Anything Model)** ViT-H checkpoint from `models/sam_vit_h_4b8939.pth`.
- Generates many candidate mask proposals using SAM's automatic mask generator.
- Scores each candidate mask on three criteria: **area** (not too small, not too large), **edge strength** (does it follow real edges?), and **colour distinctiveness** (is it a visually separate region?).
- Merges masks that overlap more than 60 % (IoU > 0.6) to avoid duplicates.
- Keeps the top ~10–20 highest-scoring masks.

**Outputs:** A list of binary masks with salience scores and bounding boxes.

---

#### Step 3 — Edge Extraction (`step3_edge_extraction/edge_extraction.py`)

**What it does:** Finds the boundary lines inside each segmented region.

**How it works (outline-only mode, used for the demo):**
- Crops the original image to each mask's bounding box.
- Runs OpenCV contour extraction (`cv2.findContours`) on the mask.
- Keeps up to **4 contours** per mask to stay within demo time constraints.
- Returns contour point sequences (closed paths around region boundaries).

**How it works (traditional/full mode):**
- Applies bilateral filtering, then Canny edge detection on the grayscale crop.
- Runs **skeletonisation** (Zhang-Suen algorithm via `scikit-image`) to thin all edges to 1-pixel width centrelines.
- Returns the skeleton pixel map ready for graph construction.

**Outputs:** Contour point lists (outline mode) or skeleton pixel maps (traditional mode).

---

#### Step 4 — Stroke Graph Construction (`step4_stroke_graph/stroke_graph.py`)

**What it does:** Converts the raw pixel skeleton into a structured graph of strokes.

**How it works:**
- Uses the **sknw** library to parse skeleton pixels into a NetworkX graph where:
  - **Nodes** are endpoints and junction points.
  - **Edges** carry the pixel path between two nodes.
- Detects **Eulerian paths** (routes that traverse every edge exactly once) to minimise pen lifts.
- In outline mode, the closed contour paths from Step 3 are used directly as strokes without further graph analysis.

**Outputs:** A list of ordered pixel sequences representing individual strokes.

---

#### Step 5 — Vectorization (`step5_vectorization/vectorization.py`)

**What it does:** Reduces the number of points in each stroke while preserving its shape.

**How it works:**
- Applies the **Ramer-Douglas-Peucker (RDP)** algorithm (from the `rdp` library) to each pixel path.
- The tolerance is set to `max_error = 1.0 px` — points closer than 1 px to the simplified line are removed.
- A path of 10 000 pixels typically reduces to ~500 waypoints with less than 1 px maximum deviation.

**Outputs:** Simplified polylines (arrays of `[x_px, y_px]` points), compression ratio, average error.

---

#### Step 6 — Sampling & Scaling (`step6_sampling/sampling.py`)

**What it does:** Converts pixel coordinates to millimetres and resamples strokes at a uniform 1 mm interval.

**How it works:**
- Calculates `mm_per_px = CANVAS_SIZE_MM / image_width_px` (default canvas is 130 mm × 130 mm).
- Walks along each polyline using **arc-length parameterisation** (cumulative Euclidean distance along the path).
- Places a new sample point every `DEFAULT_SPACING_MM = 1.0 mm` along the arc.
- Applies each mask's crop offset so all strokes share the same coordinate space.

**Outputs:** Strokes as arrays of `[x_mm, y_mm]` waypoints in canvas-space millimetres.

---

#### Step 7 — Colour Detection (`step7_color_detection/color_detection.py`)

**What it does:** Labels each mask (and its strokes) as either **warm** or **cool**.

**How it works:**
- Converts the masked region from RGB to **CIELAB colour space** (`L*, a*, b*`).
- A pixel is classified as warm if `a*` (red–green axis) is high (red tones) or `b*` (blue–yellow axis) is high (yellow tones).
- A pixel is cool if both `a*` and `b*` are low (blues/cyans).
- The mask's overall classification is the majority vote with a **confidence score** (proportion of warm pixels).
- Neutral/greyscale regions are handled with a lower-confidence label.

**Outputs:** A warm/cool label (`0 = warm`, `1 = cool`) for each mask.

---

#### Step 8 — Stroke Ordering (`step8_stroke_ordering/stroke_ordering.py`)

**What it does:** Determines the optimal order to draw all strokes and produces the final hardware-ready JSON.

**How it works:**
- Flattens the nested mask → stroke hierarchy into a single list.
- Uses a **nearest-neighbour TSP heuristic**: starting from the home position, always move to the closest stroke endpoint next. This minimises total pen-travel distance.
- Inserts **pen-lift instructions** between strokes (lift height: 15 mm).
- Assembles the final JSON with:
  - All stroke waypoints in mm
  - Pen-lift events
  - Per-stroke metadata (colour, length, mask ID)
  - Pipeline metadata (canvas size, image name, timestamp)

**Outputs:** A JSON file saved to `results/step8_stroke_ordering/`. This file is the direct input to the firmware.

---

### B. Firmware & Robot Control (`firmware/`)

This layer reads the JSON produced by the pipeline and drives the physical robot.

| File | Role |
|------|------|
| `run_json.py` | **Main demo entry point.** Prompts for a JSON filename, initialises the delta robot, starts music and narration in background threads, then executes the stroke sequence. |
| `prodplotter.py` | **Delta robot actuator.** Implements delta-parallel-robot forward/inverse kinematics, controls three stepper motors, and operates a servo for pen lift/lower. |
| `driver.py` | **Motor wrapper.** Abstracts stepper direction/step signals, enforces hardware safety limits, and manages the home position calibration. |
| `gcode_serial.py` | **G-code layer.** Converts `[x, y, z]` move commands into G-code strings and writes them to the Arduino over a serial port. |
| `testplotter.py` | **Software simulation.** Mirrors the `DeltaActuator` API but prints moves instead of sending them to hardware. Useful for testing without a robot. |
| `repeater.ino` | **Arduino firmware.** Receives G-code over USB serial, drives stepper motor step/direction pins, and outputs PWM to a servo for the pen mechanism. |

**How `run_json.py` works end-to-end:**
1. Loads the JSON file from `painting_vectorization/results/step8_stroke_ordering/`.
2. Looks up the matching painting profile (narrations + music track) from `painting_profiles.py`.
3. Spawns a background thread that loops `afplay` to play ambient music at low volume.
4. Iterates through the stroke/pen-lift sequence:
   - For **pen-lift** events: raises the pen to 15 mm, moves to the next stroke start.
   - For **stroke** events: lowers the pen and moves through each `[x_mm, y_mm]` waypoint.
   - At narration trigger points: speaks a Bob Ross phrase via ElevenLabs TTS.
5. Returns the robot to the home position and stops the music when finished.

---

### C. AI Narration (Cohere Multimodal + TTS)

#### `cohere-multimodal/painting_profiles.py`

Stores **pre-generated painting profiles** for six paintings (Starry Night, M3 Building, Eiffel Tower, Titanic, Petronas Towers, Mona Lisa). Each profile contains:
- 3–4 Bob Ross–style narration strings (e.g. *"We start with this magnificent cypress tree…"*).
- The filename of the matching music track.
- Descriptions of what each mask represents.

Pre-generation is used in the demo because live Cohere analysis takes over a minute per image.

#### `cohere-multimodal/painting_mask_analyzer.py`

If live analysis is needed, this module:
- Sends the original image and each segmentation mask to **Cohere's Command-R-Plus Vision model**.
- Extracts `(object, colour)` pairs from the response.
- Generates Bob Ross–style descriptions dynamically.

#### `text_to_speech/bob_ross_simple_tts.py`

Converts narration text to speech:
- **Primary path:** Calls the **ElevenLabs API** with a Bob Ross–like voice preset (requires `ELEVENLABS_API_KEY` in the environment).
- **Fallback:** Uses the macOS built-in `say` command with a slower speech rate to approximate the relaxed Bob Ross cadence.

---

### D. Sentiment Analysis & Music Selection (`sentiment_analysis/`, `music_player/`)

#### `sentiment_analysis/main.py`

- Sends a description of the painting to the **Cohere text-classification API**.
- Classifies the mood into one of 10 categories (e.g. `AWE_AND_WONDER`, `CALM_AND_SERENITY`, `TENSION_AND_DRAMA`).
- Returns the category name so a matching music track can be selected.

#### `music_player/music_player.py`

- Uses **Pygame's** `pygame.mixer` module to play MP3 files.
- Supports volume control, looping, and cross-fade.
- Ten themed tracks are available (e.g. *Calm and Serenity.mp3*, *Melancholy Drift.mp3*).
- In `run_json.py`, music is played via `subprocess` + `afplay` (macOS) rather than Pygame directly, to avoid audio driver conflicts.

---

### E. Camera Integration (`camera_integration/`)

#### `voice_camera.py`

- Uses the **SpeechRecognition** library to continuously listen for the trigger phrase *"take photo"*.
- On trigger: captures a frame from the default webcam using OpenCV (`cv2.VideoCapture`).
- Saves the image to `camera_integration/photos/` with a timestamp filename.
- The saved photo can then be fed into `process_image.py` to kick off the full pipeline.

---

## 4. Data Flow

```
[Image File]
      │
      ▼  step1_preprocessing
  Preprocessed image (float32, ≤1024 px)
      │
      ▼  step2_segmentation  (SAM model)
  Binary masks[]  +  salience scores
      │
      ▼  step3_edge_extraction
  Contour paths[]  /  skeleton pixel maps
      │
      ▼  step4_stroke_graph  (sknw + NetworkX)
  Ordered pixel sequences (strokes)
      │
      ▼  step5_vectorization  (RDP algorithm)
  Simplified polylines  [x_px, y_px]
      │
      ▼  step6_sampling
  Uniform 1 mm waypoints  [x_mm, y_mm]
      │
      ▼  step7_color_detection  (CIELAB)
  Warm / cool label per stroke
      │
      ▼  step8_stroke_ordering  (TSP heuristic)
  stroke_ordering_<TIMESTAMP>.json
      │
      ├──▶ firmware/run_json.py
      │         ├── Delta robot executes strokes
      │         ├── Music plays in background thread
      │         └── TTS narrates at trigger points
      │
      └──▶ results/step8_stroke_ordering/  (visualizations)
```

---

## 5. Configuration

All global parameters are in **`painting_vectorization/constants.py`**. Changing a value there automatically propagates through the entire pipeline — no other files need editing.

| Constant | Default | Meaning |
|----------|---------|---------|
| `CANVAS_SIZE_MM` | `130` | Physical canvas size in mm (square) |
| `DEFAULT_SPACING_MM` | `1.0` | Distance between sampled waypoints |
| `ROBOT_CONFIG["pen_lift_height_mm"]` | `15.0` | Height to raise pen between strokes |
| `ROBOT_CONFIG["min_travel_distance_mm"]` | `2.0` | Minimum move distance before a pen lift is needed |
| `DEMO_DURATION_LIMIT_S` | `120.0` | Target demo duration (2 minutes) |

Output directories for each step are also defined in `OUTPUT_DIRS`.

---

## 6. Tech Stack

| Category | Libraries |
|----------|-----------|
| Image I/O & processing | OpenCV (`cv2`), Pillow |
| Numerical computing | NumPy, SciPy |
| Advanced image analysis | scikit-image (skeletonisation, CIELAB) |
| Machine learning | PyTorch, Segment Anything (SAM) |
| Graph algorithms | NetworkX, sknw |
| Path simplification | `rdp` (Ramer-Douglas-Peucker) |
| Visualisation | Matplotlib |
| AI / LLM | Cohere API (multimodal vision + text classification) |
| Text-to-speech | ElevenLabs API, macOS `say` |
| Audio playback | Pygame, macOS `afplay` |
| Voice input | SpeechRecognition, PyAudio |
| Hardware serial | PySerial |
| Environment config | python-dotenv |

**Python version:** 3.8+, recommended 3.9–3.10.

---

## 7. Entry Points & How to Run

### Process an image through the full pipeline

```bash
cd painting_vectorization

# Interactive — choose from a menu of images in examples/
python process_image.py

# Process a specific image directly
python process_image.py examples/monet.jpeg

# List available example images
python process_image.py --list

# Show recently generated results
python process_image.py --show-results
```

This produces a JSON file in `results/step8_stroke_ordering/` and PNG visualizations for each step.

### Run the robot demo

```bash
cd firmware
python run_json.py
# When prompted, enter a name such as: starry
# Available profiles: starry, m3, eiffel, titanic, petronas, monalisa
```

The script loads the matching JSON, plays music, speaks narrations, and drives the robot.

### Test TTS without the robot

```bash
python test_tts.py
python test_elevenlabs_tts.py
```

### Capture an image with voice command

```bash
cd camera_integration
python voice_camera.py
# Say "take photo" — the image is saved to camera_integration/photos/
```

### Run the full setup

See **`INSTALLATION_GUIDE.md`** for virtual environment setup, dependency installation, and the manual SAM model download (~2.4 GB).
