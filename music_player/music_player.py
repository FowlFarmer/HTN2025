import os
import pygame
from pathlib import Path
from enum import Enum

class MusicTheme(Enum):
    """Enumeration of available music themes and their corresponding audio files."""
    AWE_AND_WONDER = ("Awe and Wonder", "Awe and Wonder.mp3")
    JOY_AND_HAPPINESS = ("Joy and Happiness", "Joyful Sunrise.mp3")
    MELANCHOLY_AND_SADNESS = ("Melancholy and Sadness", "Melancholy Drift.mp3")
    EMPATHY_AND_COMPASSION = ("Empathy and Compassion", "Empathy and Compassion.mp3")
    CALM_AND_SERENITY = ("Calm and Serenity", "Calm and Serenity.mp3")
    FEAR_AND_UNEASE = ("Fear and Unease", "Fear and Unease.mp3")
    INSPIRATION_AND_CREATIVITY = ("Inspiration and Creativity", "Inspiration and Creativity.mp3")
    LOVE_AND_WARMTH = ("Love and Warmth", "Love and Warmth.mp3")
    CONFUSION_AND_CURIOSITY = ("Confusion and Curiosity", "Confusion and Curiosity.mp3")
    SPIRITUALITY_AND_TRANSCENDENCE = ("Spirituality and Transcendence", "Spirituality and Transcendence.mp3")
    
    def __init__(self, theme_name, filename):
        self.theme_name = theme_name
        self.filename = filename
    
    @classmethod
    def get_theme_mapping(cls):
        """Return a dictionary mapping theme names to filenames."""
        return {theme.theme_name: theme.filename for theme in cls}
    
    @classmethod
    def get_all_themes(cls):
        """Return a list of all available theme names."""
        return [theme.theme_name for theme in cls]

def play_music(prompt):
    """
    Play music based on the given theme prompt.
    
    Args:
        prompt (str): The theme/emotion prompt for music selection
    """
    # Initialize pygame mixer
    pygame.mixer.init()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Get theme to filename mapping from enum
    theme_mapping = MusicTheme.get_theme_mapping()
    
    if not prompt.strip():
        print("Please enter a prompt")
        print("Available themes:")
        for theme in theme_mapping.keys():
            print(f"  - {theme}")
        return
    
    # Find matching theme (case-insensitive)
    selected_file = None
    for theme, filename in theme_mapping.items():
        if prompt.lower().strip() == theme.lower():
            selected_file = filename
            break
    
    if not selected_file:
        print(f"Theme '{prompt}' not found.")
        print("Available themes:")
        for theme in theme_mapping.keys():
            print(f"  - {theme}")
        return
    
    # Construct full path to audio file
    audio_path = script_dir / selected_file
    
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return
    
    try:
        print(f"Playing: {prompt}")
        print(f"File: {selected_file}")
        print("Press Ctrl+C to stop playback...")
        
        # Load and play the music
        pygame.mixer.music.load(str(audio_path))
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
            
        print("Playback finished.")
        
    except KeyboardInterrupt:
        print("\nPlayback stopped by user.")
        pygame.mixer.music.stop()
    except Exception as e:
        print(f"Error playing audio: {e}")
    finally:
        pygame.mixer.quit()

def list_available_themes():
    """List all available music themes."""
    themes = MusicTheme.get_all_themes()
    
    print("Available music themes:")
    for i, theme in enumerate(themes, 1):
        print(f"{i:2d}. {theme}")

def main():
    """Main function to run the music player interactively."""
    print("=== Emotion-Based Music Player ===")
    list_available_themes()
    print("\nEnter a theme name to play music, or 'quit' to exit.")
    
    while True:
        try:
            user_input = input("\nEnter theme: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() in ['list', 'themes']:
                list_available_themes()
            else:
                play_music(user_input)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()