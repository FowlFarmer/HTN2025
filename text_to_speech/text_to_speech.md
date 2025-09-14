# Text-to-Speech System

This folder contains a Bob Ross-themed text-to-speech system with two main components:

## Files Overview

### `bob_ross_simple_tts.py`
**Purpose**: Convert any text string to Bob Ross voice audio

**Main Function**: `speak_in_bob_ross_voice(text)`
- **Input**: Any text string
- **Output**: Plays Bob Ross-style audio through speakers
- **Features**:
  - **Primary**: Uses ElevenLabs TTS with calm voice (JBFqnCBsd6RMkjVDRZzb)
  - **Fallback**: Uses macOS "Fred" voice if ElevenLabs fails
  - Minimal Bob Ross styling for sequential narrations
  - Generates temporary audio file and plays it
  - Cleans up temporary files after playback
  - Returns `True`/`False` for success/failure
  - Requires ElevenLabs API key in environment variables

**Dependencies**: `os`, `subprocess`, `random`, `elevenlabs`, `python-dotenv`, `tempfile`

## Usage Examples

### Speak any text in Bob Ross voice:
```python
from bob_ross_simple_tts import speak_in_bob_ross_voice

speak_in_bob_ross_voice("We start with this magnificent cypress tree")
# Plays audio with ElevenLabs calm voice or macOS fallback
```

## System Requirements
- **macOS**: Uses built-in `say` command and `afplay` for audio
- **Python packages**: `python-dotenv`, `elevenlabs`
- **Environment Variables**: 
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
- **Sequential Narrations**: Designed for short, sequential painting step narrations
- **Minimal Styling**: Simple "Now, [text]" format instead of elaborate Bob Ross phrases
- Automatic fallback ensures speech always works even if ElevenLabs service is down