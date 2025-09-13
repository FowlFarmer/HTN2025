import os
import pygame
from pathlib import Path

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
    
    # Theme to filename mapping
    theme_mapping = {
        "Awe and Wonder": "Awe and Wonder.mp3",
        "Joy and Happiness": "Joyful Sunrise.mp3",  # Using closest match
        "Melancholy and Sadness": "Melancholy Drift.mp3",  # Using closest match
        "Empathy and Compassion": "Empathy and Compassion.mp3",
        "Calm and Serenity": "Calm and Serenity.mp3",
        "Fear and Unease": "Fear and Unease.mp3",
        "Inspiration and Creativity": "Inspiration and Creativity.mp3",
        "Love and Warmth": "Love and Warmth.mp3",
        "Confusion and Curiosity": "Confusion and Curiosity.mp3",
        "Spirituality and Transcendence": "Spirituality and Transcendence.mp3"
    }
    
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
    themes = [
        "Awe and Wonder",
        "Joy and Happiness", 
        "Melancholy and Sadness",
        "Empathy and Compassion",
        "Calm and Serenity",
        "Fear and Unease",
        "Inspiration and Creativity",
        "Love and Warmth",
        "Confusion and Curiosity",
        "Spirituality and Transcendence"
    ]
    
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