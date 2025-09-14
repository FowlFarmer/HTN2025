# Painting Profiles System Guide

## Overview

The Painting Profiles System provides hardcoded narrations and music selections for different paintings based on their JSON files. This system **replaces the fully functional Cohere multimodal AI** for demo performance reasons.

## ⚠️ **Why Hardcoded Profiles Instead of AI?**

**The Cohere multimodal AI works perfectly** and can:
- ✅ Analyze painting sentiment to select music
- ✅ Identify objects and colors from images + masks  
- ✅ Generate dynamic Bob Ross narrations

**But**: Each AI call takes 1+ minutes, consuming 1/3+ of demo time. For swift demonstrations, we use pre-generated high-quality content that delivers the same experience instantly.

## Current Profiles

### 1. Starry Night (`starry.json`)
- **Music**: `Melancholy Drift.mp3`
- **Narrations**: 3 narrations about cypress tree, swirling sky, and village
- **Masks**: Cypress tree, swirling sky, quiet village

### 2. M3 Building (`m3.json`)
- **Music**: `Inspiration and Creativity.mp3` 
- **Narrations**: 4 narrations for building components
- **Masks**:
  - Main building (blue glass tiles)
  - Brown concrete structural support
  - Green tree
  - Dark blue bridge

### 3. Eiffel Tower (`eiffel.json`)
- **Music**: `Love and Warmth.mp3`
- **Narrations**: 2 romantic, golden-themed narrations
- **Masks**:
  - Outer frame (shining golden)
  - Inner curves (darker gold)

### 4. Titanic (`titanic.json`)
- **Music**: `Fear and Unease.mp3`
- **Narrations**: 3 dramatic, respectful narrations about the tragic maritime moment
- **Masks**:
  - Part of ship not yet submerged (metallic black)
  - Bow of the ship (metallic black)
  - Surface of ship as it sinks (metallic black)

### 5. Petronas Towers (`petronas.json`)
- **Music**: `Awe and Wonder.mp3`
- **Narrations**: 3 inspiring narrations about architectural achievement and twin tower harmony
- **Masks**:
  - Left tower (silver)
  - Right tower (silver)
  - Left of the left tower (silver)

### 6. Mona Lisa (`monalisa.json`)
- **Music**: `Joyful Sunrise.mp3`
- **Narrations**: 3 Renaissance-themed narrations about the famous portrait
- **Masks**:
  - Woman's outline (dark dress)
  - Water body background (blue)
  - Face and neck (white)

## How to Use

### In run_json_with_profiles.py

1. **Set the JSON file** at the top of the script:
```python
JSON_FILE = "m3.json"  # Change to your desired painting
```

2. **The system automatically**:
   - Selects appropriate music
   - Loads matching narrations
   - Handles any number of masks
   - Falls back gracefully if profile not found

### Quick Test
```bash
cd cohere-multimodal
python painting_profiles.py
```

## Adding New Paintings

### Step 1: Prepare Your Content

For each new painting, gather:
- JSON filename (e.g., `"monalisa.json"`)
- Painting name (e.g., `"Mona Lisa"`)
- Music file from available options
- Bob Ross narrations (one per mask)
- Optional mask descriptions

### Step 2: Available Music Files

```
- Awe and Wonder.mp3
- Joyful Sunrise.mp3
- Melancholy Drift.mp3
- Empathy and Compassion.mp3
- Calm and Serenity.mp3
- Fear and Unease.mp3
- Inspiration and Creativity.mp3
- Love and Warmth.mp3
- Confusion and Curiosity.mp3
- Spirituality and Transcendence.mp3
```

### Step 3: Add Profile to painting_profiles.py

Edit the `_initialize_profiles()` method:

```python
# NEW PAINTING PROFILE
new_narrations = [
    "First mask narration...",
    "Second mask narration...",
    "Third mask narration..."
]

new_masks = [
    "Description of first mask",
    "Description of second mask",
    "Description of third mask"
]

self.profiles["newpainting.json"] = PaintingProfile(
    name="New Painting Name",
    json_file="newpainting.json", 
    music_file="Selected Music.mp3",
    narrations=new_narrations,
    mask_descriptions=new_masks
)
```

### Step 4: Test Your Profile

```bash
cd cohere-multimodal
python painting_profiles.py
```

### Step 5: Use in Production

Update `run_json_with_profiles.py`:
```python
JSON_FILE = "newpainting.json"
```

## Bob Ross Narration Guidelines

### Writing Style
- Use Bob Ross's gentle, encouraging tone
- Include painting metaphors and references
- Mention "happy little" elements
- Reference colors and techniques
- Keep 2-4 sentences per narration

### Example Structure
```
"[Opening observation about the element] [Bob Ross-style metaphor or description] [Encouraging instruction or reflection about the painting process]"
```

### Good Examples
- "Look at this magnificent blue glass building reaching toward the sky! Each panel catches the light like a happy little window to the world."
- "Here comes a cheerful green tree to soften our urban landscape! Trees are nature's way of saying 'hello' to the city."

## System Architecture

```
run_json_with_profiles.py
    ↓
painting_profiles.py (PaintingProfileManager)
    ↓
Profile Selection based on JSON filename
    ↓
Returns: narrations + music file
    ↓
Used in: TTS + background music
```

## Troubleshooting

### Profile Not Found
- Check JSON filename spelling
- Verify profile exists in `painting_profiles.py`
- System will use defaults if profile missing

### Music File Not Found
- Verify music file exists in `music_player/` directory
- Check exact filename spelling
- System will continue without music if file missing

### Too Few Narrations
- System adapts to available narrations
- Extra masks will paint silently
- Add more narrations to profile if needed

## Benefits

- ✅ **Fast execution** - No API calls (vs 1+ minute AI processing)
- ✅ **Demo-ready** - Instant response for live presentations
- ✅ **Reliable** - No network dependencies  
- ✅ **Customizable** - Tailored content per painting
- ✅ **Scalable** - Easy to add new paintings
- ✅ **Graceful fallbacks** - Handles missing profiles
- ✅ **Professional** - Consistent, high-quality narrations
- ✅ **AI-equivalent quality** - Curated content matches AI output quality

## Switching to Full AI Mode

To enable the complete Cohere multimodal pipeline:

1. **Uncomment AI calls** in:
   - `starrynight_analyzer.py` (lines ~126-143)
   - `painting_mask_analyzer.py` (multimodal sections)

2. **Set environment variables**:
   ```bash
   export COHERE_API_KEY="your_api_key_here"
   ```

3. **Accept processing time**:
   - 60-90 seconds per painting analysis
   - Dynamic, unique content generation
   - Real-time sentiment and object analysis

4. **Use cases for AI mode**:
   - Development and testing
   - Non-time-constrained scenarios  
   - When you want unique AI-generated content
   - Research and experimentation
