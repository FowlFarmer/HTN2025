# Emotion-Based Music Player

An intelligent music player that selects and plays audio tracks based on emotional themes and prompts. The system maps various emotional states to corresponding musical compositions, creating an immersive audio experience tailored to specific moods and feelings.

## Features

- **Theme-based Music Selection**: Choose from 10 different emotional themes
- **Case-insensitive Matching**: Flexible input handling for theme selection
- **Interactive Command Line Interface**: Easy-to-use terminal-based player
- **Robust Error Handling**: Graceful handling of missing files and user interruptions
- **Enum-based Theme Management**: Type-safe and exportable theme definitions

## Available Themes

The music player supports the following emotional themes:

1. **Awe and Wonder** - Inspiring and magnificent compositions
2. **Joy and Happiness** - Uplifting and cheerful melodies
3. **Melancholy and Sadness** - Contemplative and somber pieces
4. **Empathy and Compassion** - Warm and understanding tones
5. **Calm and Serenity** - Peaceful and tranquil soundscapes
6. **Fear and Unease** - Tense and suspenseful atmospheres
7. **Inspiration and Creativity** - Motivating and energizing tracks
8. **Love and Warmth** - Tender and affectionate compositions
9. **Confusion and Curiosity** - Mysterious and intriguing melodies
10. **Spirituality and Transcendence** - Ethereal and transcendent music

## Installation

### Prerequisites

```bash
pip install pygame
```

### Dependencies

The music player requires:
- Python 3.6+
- pygame library for audio playback
- pathlib (included in Python standard library)

## Usage

### Basic Usage

#### Interactive Mode

Run the music player in interactive mode:

```bash
python music_player.py
```

This will start an interactive session where you can:
- View all available themes
- Enter theme names to play music
- Type 'list' or 'themes' to see available options
- Type 'quit', 'exit', or 'q' to exit

#### Programmatic Usage

Import and use the music player in your own code:

```python
from music_player import play_music, list_available_themes, MusicTheme

# Play music by theme name
play_music("Calm and Serenity")

# List all available themes
list_available_themes()

# Use the enum for type-safe theme access
from music_player import MusicTheme

# Get all themes as a list
themes = MusicTheme.get_all_themes()
print(themes)

# Get theme mapping dictionary
mapping = MusicTheme.get_theme_mapping()
print(mapping)

# Access specific theme properties
theme = MusicTheme.CALM_AND_SERENITY
print(f"Theme: {theme.theme_name}")
print(f"File: {theme.filename}")
```

### Examples

#### Example 1: Playing Specific Themes

```python
from music_player import play_music

# Play calming music
play_music("Calm and Serenity")

# Play inspiring music (case-insensitive)
play_music("inspiration and creativity")

# Play joyful music
play_music("Joy and Happiness")
```

#### Example 2: Working with the MusicTheme Enum

```python
from music_player import MusicTheme, play_music

# Iterate through all available themes
for theme in MusicTheme:
    print(f"Theme: {theme.theme_name}")
    print(f"File: {theme.filename}")

# Get a specific theme
calm_theme = MusicTheme.CALM_AND_SERENITY
print(f"Playing: {calm_theme.theme_name}")
play_music(calm_theme.theme_name)

# Use enum for validation
def play_theme_safely(theme_enum):
    if isinstance(theme_enum, MusicTheme):
        play_music(theme_enum.theme_name)
    else:
        print("Invalid theme provided")

play_theme_safely(MusicTheme.AWE_AND_WONDER)
```

#### Example 3: Building a Custom Interface

```python
from music_player import MusicTheme, play_music
import random

def play_random_music():
    """Play a random theme from available options."""
    random_theme = random.choice(list(MusicTheme))
    print(f"Playing random theme: {random_theme.theme_name}")
    play_music(random_theme.theme_name)

def play_mood_music(mood_category):
    """Play music based on mood category."""
    mood_mappings = {
        'positive': [MusicTheme.JOY_AND_HAPPINESS, MusicTheme.AWE_AND_WONDER, MusicTheme.INSPIRATION_AND_CREATIVITY],
        'calm': [MusicTheme.CALM_AND_SERENITY, MusicTheme.SPIRITUALITY_AND_TRANSCENDENCE],
        'emotional': [MusicTheme.LOVE_AND_WARMTH, MusicTheme.EMPATHY_AND_COMPASSION],
        'intense': [MusicTheme.FEAR_AND_UNEASE, MusicTheme.CONFUSION_AND_CURIOSITY],
        'reflective': [MusicTheme.MELANCHOLY_AND_SADNESS]
    }
    
    if mood_category in mood_mappings:
        theme = random.choice(mood_mappings[mood_category])
        play_music(theme.theme_name)
    else:
        print(f"Unknown mood category: {mood_category}")

# Usage
play_random_music()
play_mood_music('positive')
play_mood_music('calm')
```

## API Reference

### Functions

#### `play_music(prompt: str)`

Plays music based on the given theme prompt.

**Parameters:**
- `prompt` (str): The theme/emotion prompt for music selection

**Returns:** None

**Behavior:**
- Initializes pygame mixer
- Searches for matching theme (case-insensitive)
- Loads and plays the corresponding audio file
- Handles playback controls and cleanup
- Provides user feedback during playback

#### `list_available_themes()`

Displays all available music themes in a numbered list.

**Parameters:** None

**Returns:** None

#### `main()`

Runs the interactive music player interface.

**Parameters:** None

**Returns:** None

### Classes

#### `MusicTheme(Enum)`

Enumeration of available music themes and their corresponding audio files.

**Class Methods:**

##### `get_theme_mapping() -> dict`

Returns a dictionary mapping theme names to filenames.

**Returns:** 
- `dict`: Dictionary with theme names as keys and filenames as values

##### `get_all_themes() -> list`

Returns a list of all available theme names.

**Returns:**
- `list`: List of theme name strings

**Enum Values:**

Each enum value contains:
- `theme_name` (str): Human-readable theme name
- `filename` (str): Corresponding audio file name

**Available Values:**
- `AWE_AND_WONDER`
- `JOY_AND_HAPPINESS`
- `MELANCHOLY_AND_SADNESS`
- `EMPATHY_AND_COMPASSION`
- `CALM_AND_SERENITY`
- `FEAR_AND_UNEASE`
- `INSPIRATION_AND_CREATIVITY`
- `LOVE_AND_WARMTH`
- `CONFUSION_AND_CURIOSITY`
- `SPIRITUALITY_AND_TRANSCENDENCE`

## Error Handling

The music player includes comprehensive error handling:

- **Missing Audio Files**: Displays error message with file path
- **Invalid Themes**: Shows available theme options
- **User Interruption**: Graceful handling of Ctrl+C during playback
- **Pygame Errors**: Catches and displays audio-related errors
- **Empty Input**: Prompts user with available themes

## File Structure

```
music_player/
├── music_player.py          # Main music player module
├── README.md               # This documentation file
├── Awe and Wonder.mp3      # Audio files for each theme
├── Joyful Sunrise.mp3
├── Melancholy Drift.mp3
├── Empathy and Compassion.mp3
├── Calm and Serenity.mp3
├── Fear and Unease.mp3
├── Inspiration and Creativity.mp3
├── Love and Warmth.mp3
├── Confusion and Curiosity.mp3
└── Spirituality and Transcendence.mp3
```

## Contributing

When adding new themes:

1. Add the audio file to the music_player directory
2. Add a new enum value to `MusicTheme` class
3. Update this documentation

## License

This project is part of the HTN2025 toolkit for emotion-based interactive experiences.
