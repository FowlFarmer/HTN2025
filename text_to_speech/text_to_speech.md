# Text-to-Speech System

This folder contains a Bob Ross-themed text-to-speech system with two main components:

## Files Overview

### `bob_ross_cohere.py`
**Purpose**: Transform (component, color) pairs into Bob Ross-style descriptions using Cohere AI

**Main Function**: `transform_to_bob_ross_style(object_color_pairs, cohere_api_key=None)`
- **Input**: List of tuples like `[("apple", "red"), ("tree", "green"), ("sky", "blue")]`
- **Output**: Array of Bob Ross-style descriptions, one for each input pair
- **Features**:
  - Uses Cohere API to generate authentic Bob Ross descriptions
  - Includes fallback descriptions if API fails
  - Automatically handles API key from environment variables
  - Parses AI responses into individual descriptions

**Dependencies**: `cohere`, `python-dotenv`, `os`, `random`

### `bob_ross_simple_tts.py`
**Purpose**: Convert any text string to Bob Ross voice audio

**Main Function**: `speak_in_bob_ross_voice(text)`
- **Input**: Any text string
- **Output**: Plays Bob Ross-style audio through speakers
- **Features**:
  - Uses macOS "Fred" voice (calm, deep tone similar to Bob Ross)
  - Automatically adds Bob Ross phrases and styling
  - Generates temporary audio file and plays it
  - Cleans up temporary files after playback
  - Returns `True`/`False` for success/failure

**Dependencies**: `os`, `subprocess`, `random`

## Usage Examples

### Transform component-color pairs to descriptions:
```python
from bob_ross_cohere import transform_to_bob_ross_style

pairs = [("apple", "red"), ("mountain", "purple"), ("water", "blue")]
descriptions = transform_to_bob_ross_style(pairs)
# Returns: ["Now here we have a beautiful red apple...", "Let's paint a lovely purple mountain...", ...]
```

### Speak any text in Bob Ross voice:
```python
from bob_ross_simple_tts import speak_in_bob_ross_voice

speak_in_bob_ross_voice("Today we're painting happy little trees")
# Plays audio with Bob Ross styling and Fred voice
```

## System Requirements
- **macOS**: Uses built-in `say` command and `afplay` for audio
- **Python packages**: `cohere`, `python-dotenv` 
- **Environment**: `COHERE_API_KEY` environment variable for AI features

## Notes
- The TTS currently uses macOS "Fred" voice, not actual Bob Ross voice
- For true Bob Ross voice, would need Hugging Face model integration
- Both functions work independently and can be used separately