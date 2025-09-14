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
  - **Primary**: Uses ElevenLabs TTS with calm voice (JBFqnCBsd6RMkjVDRZzb)
  - **Fallback**: Uses macOS "Fred" voice if ElevenLabs fails
  - Automatically adds Bob Ross phrases and styling
  - Generates temporary audio file and plays it
  - Cleans up temporary files after playback
  - Returns `True`/`False` for success/failure
  - Requires ElevenLabs API key in environment variables

**Dependencies**: `os`, `subprocess`, `random`, `elevenlabs`, `python-dotenv`, `tempfile`

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
- **Python packages**: `cohere`, `python-dotenv`, `elevenlabs`
- **Environment Variables**: 
  - `COHERE_API_KEY` for AI features
  - `ELEVENLABS_API_KEY` for premium TTS (required for ElevenLabs)

## Environment Setup
To use ElevenLabs TTS, you **must** set your API key:
```bash
export ELEVENLABS_API_KEY="sk_cf566ff12821df1d2ee174f008db3df9d70188d1c3e0e3b6"
```
Or add it to your `.env` file:
```
ELEVENLABS_API_KEY=sk_cf566ff12821df1d2ee174f008db3df9d70188d1c3e0e3b6
```

## Notes
- **Primary TTS**: ElevenLabs with calm voice (JBFqnCBsd6RMkjVDRZzb)
- **Fallback TTS**: macOS "Fred" voice if ElevenLabs fails or unavailable
- **API Key Required**: ElevenLabs will only work if `ELEVENLABS_API_KEY` is set in environment
- Both functions work independently and can be used separately
- Automatic fallback ensures speech always works even if ElevenLabs service is down