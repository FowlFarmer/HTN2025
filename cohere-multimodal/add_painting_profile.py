#!/usr/bin/env python3
"""
Script to easily add new painting profiles
Use this as a template for adding new paintings
"""

from painting_profiles import painting_manager


def add_new_painting_profile(json_filename, painting_name, music_file, narrations, mask_descriptions=None):
    """
    Add a new painting profile to the system.
    
    Args:
        json_filename (str): e.g., "monalisa.json"
        painting_name (str): e.g., "Mona Lisa"
        music_file (str): e.g., "Love and Warmth.mp3"
        narrations (list): List of Bob Ross narration strings
        mask_descriptions (list): Optional descriptions of masks
    """
    painting_manager.add_profile(
        json_filename=json_filename,
        name=painting_name,
        music_file=music_file,
        narrations=narrations,
        mask_descriptions=mask_descriptions
    )
    
    print(f"✅ Added profile for {painting_name}")
    print(f"   JSON: {json_filename}")
    print(f"   Music: {music_file}")
    print(f"   Narrations: {len(narrations)}")
    if mask_descriptions:
        print(f"   Masks: {len(mask_descriptions)}")


def example_add_profiles():
    """Example of how to add profiles for other paintings."""
    
    # Example: Mona Lisa (you would replace with actual narrations)
    monalisa_narrations = [
        "Here we have the most famous smile in the world! Look at those gentle, mysterious eyes that seem to follow us wherever we go. Let's paint this enigmatic expression with soft, loving brushstrokes.",
        "The background shows beautiful, misty landscapes that fade into the distance. These aren't just hills - they're dreams and mysteries wrapped in layers of atmospheric perspective.",
        "Notice how the hands are positioned so gracefully, like they're holding secrets from centuries past. Every line tells a story of Renaissance mastery and timeless beauty."
    ]
    
    # Example: Titanic (you would replace with actual narrations)
    titanic_narrations = [
        "Look at this magnificent ship cutting through the dark waters! Even in tragedy, there's a certain majesty to this floating palace as it makes its journey through history.",
        "The ocean around us is vast and mysterious, painted in deep blues and blacks that remind us of the power and beauty of nature's forces.",
        "See how the lights twinkle like little stars against the darkness? Even in the most challenging moments, there are always points of light to guide us home."
    ]
    
    # You can uncomment these to add them to the system:
    # add_new_painting_profile(
    #     json_filename="monalisa.json",
    #     painting_name="Mona Lisa",
    #     music_file="Love and Warmth.mp3",
    #     narrations=monalisa_narrations
    # )
    
    # add_new_painting_profile(
    #     json_filename="titanic.json",
    #     painting_name="Titanic",
    #     music_file="Melancholy and Sadness.mp3",
    #     narrations=titanic_narrations
    # )
    
    print("Example profiles ready to be added (uncomment to activate)")


if __name__ == "__main__":
    print("=== Add Painting Profile Script ===\n")
    
    # Show current profiles
    print("Current profiles:")
    painting_manager.list_all_profiles()
    
    # Show examples
    example_add_profiles()
    
    print("\n=== Instructions ===")
    print("1. Use this script to add new painting profiles")
    print("2. Modify the narrations and music for each painting")
    print("3. Run the script to add them to the system")
    print("4. Update run_json_with_profiles.py to use the new JSON file")
    print("\nAvailable music files:")
    music_files = [
        "Awe and Wonder.mp3",
        "Joyful Sunrise.mp3", 
        "Melancholy Drift.mp3",
        "Empathy and Compassion.mp3",
        "Calm and Serenity.mp3",
        "Fear and Unease.mp3",
        "Inspiration and Creativity.mp3",
        "Love and Warmth.mp3",
        "Confusion and Curiosity.mp3",
        "Spirituality and Transcendence.mp3"
    ]
    for music in music_files:
        print(f"  - {music}")
