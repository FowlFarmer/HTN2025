# Painting Selection Guide for run_json.py

## Overview

The main `run_json.py` file now includes the complete painting profiles system. You can easily switch between any of the 6 available paintings by changing one line of code.

## How to Select a Painting

### 1. Open `run_json.py`

### 2. Find the Configuration Section (around line 25):

```python
# ===== CONFIGURATION: Change this to select different paintings =====
JSON_FILE = "starry.json"  # Options: starry.json, m3.json, eiffel.json, titanic.json, petronas.json, monalisa.json
# ================================================================
```

### 3. Change the JSON_FILE to your desired painting:

```python
JSON_FILE = "m3.json"         # M3 Building with Inspiration music
JSON_FILE = "eiffel.json"     # Eiffel Tower with Love and Warmth music  
JSON_FILE = "titanic.json"    # Titanic with Fear and Unease music
JSON_FILE = "petronas.json"   # Petronas Towers with Awe and Wonder music
JSON_FILE = "monalisa.json"   # Mona Lisa with Joyful Sunrise music
JSON_FILE = "starry.json"     # Starry Night with Melancholy Drift music (default)
```

### 4. Run the script:

```bash
cd firmware
python run_json.py
```

## What Happens Automatically

When you run `run_json.py`, the system automatically:

1. **Loads the correct JSON file** for stroke data
2. **Selects appropriate music** based on the painting
3. **Uses tailored Bob Ross narrations** specific to that artwork
4. **Adapts to the number of masks** in the painting
5. **Shows profile information** during startup

## Example Output

```
🎨 Loading painting data from: m3.json
✅ Painting profiles system available!
🎵 Selecting background music based on painting...
🎼 Profile-based music selection: Inspiration and Creativity.mp3
🎵 Playing background music: Inspiration and Creativity.mp3

🗣️ Selecting Bob Ross narrations based on painting...
🤖 Getting profile-based narrations for m3.json...
✅ Using profile-based narrations for m3.json!
📝 Using painting profile (m3.json) narrations

--- Bob Ross Narrations for m3.json ---
1. Look at this magnificent blue glass building reaching toward the sky! Each panel catches...
2. Now we're adding some lovely brown concrete support structures. These aren't just boring...
3. Here comes a cheerful green tree to soften our urban landscape! Trees are nature's way...
4. Finally, we're painting this elegant dark blue bridge connecting our spaces. Bridges are...

--- Painting Profile: M3 Building ---
Music: Inspiration and Creativity.mp3
Narrations: 4 available
Masks: 4 described

🎭 Found 4 masks, using 4 narrations
```

## Available Paintings

| JSON File | Painting | Music | Masks | Theme |
|-----------|----------|-------|-------|-------|
| `starry.json` | Starry Night | Melancholy Drift | 3 | Contemplative night scene |
| `m3.json` | M3 Building | Inspiration and Creativity | 4 | Modern architecture |
| `eiffel.json` | Eiffel Tower | Love and Warmth | 2 | Romantic Parisian landmark |
| `titanic.json` | Titanic | Fear and Unease | 3 | Dramatic maritime tragedy |
| `petronas.json` | Petronas Towers | Awe and Wonder | 3 | Architectural marvel |
| `monalisa.json` | Mona Lisa | Joyful Sunrise | 3 | Classic Renaissance portrait |

## Fallback System

If anything goes wrong:
- 🎵 **Music fallback**: Uses "Calm and Serenity.mp3"
- 🗣️ **Narration fallback**: Uses generic Bob Ross phrases
- 📁 **File fallback**: Clear error messages if JSON not found

## Benefits

- ✅ **One unified system** - No multiple file variants
- ✅ **Easy switching** - Change one line to switch paintings
- ✅ **Automatic adaptation** - Handles different numbers of masks
- ✅ **Professional output** - Shows what's being used
- ✅ **Robust fallbacks** - Graceful failure handling
- ✅ **Fast execution** - No API dependencies

## Quick Reference

To quickly switch paintings, just remember:
- **Starry Night**: `"starry.json"` 🌟
- **M3 Building**: `"m3.json"` 🏢  
- **Eiffel Tower**: `"eiffel.json"` 🗼
- **Titanic**: `"titanic.json"` 🚢
- **Petronas Towers**: `"petronas.json"` 🏙️
- **Mona Lisa**: `"monalisa.json"` 🖼️
