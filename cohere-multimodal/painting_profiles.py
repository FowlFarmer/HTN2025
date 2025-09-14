#!/usr/bin/env python3
"""
Painting Profiles System
Manages hardcoded narrations and music selections for different paintings
Based on specific JSON files and mask descriptions
"""

from enum import Enum
from pathlib import Path


class PaintingProfile:
    """Profile containing narrations and music for a specific painting."""
    
    def __init__(self, name, json_file, music_file, narrations, mask_descriptions=None):
        self.name = name
        self.json_file = json_file  # e.g., "starry.json"
        self.music_file = music_file  # e.g., "Melancholy Drift.mp3"
        self.narrations = narrations  # List of Bob Ross narrations
        self.mask_descriptions = mask_descriptions or []  # Optional mask descriptions
    
    def get_narrations(self, max_count=None):
        """Get narrations, optionally limited to max_count."""
        if max_count is None:
            return self.narrations
        return self.narrations[:max_count]
    
    def __str__(self):
        return f"PaintingProfile({self.name}, {self.json_file}, {len(self.narrations)} narrations)"


class PaintingProfileManager:
    """Manages all painting profiles and provides lookup functionality."""
    
    def __init__(self):
        self.profiles = {}
        self._initialize_profiles()
    
    def _initialize_profiles(self):
        """Initialize all hardcoded painting profiles."""
        
        # STARRY NIGHT PROFILE
        starry_narrations = [
            "Oh, look at this cheeky little cypress tree! It's flickering like a flame, wanting to draw our attention. Let's appreciate its fiery charm and give it a hug from afar, shall we?",
            "The sky is doing cartwheels today, with vibrant blue swirls unfolding like a symphony. Let's imagine we're painting with the clouds, adding our own touches to this joyous spectacle.",
            "Nestled in the quiet of night, this serene village seems to hug its residents to sleep. Let's paint it with gentle, loving strokes, reminding ourselves that quiet moments are just as valuable as louder ones. Consider getting your paint, brushes and palette ready as you relax and enjoy the entire painting process, gently guiding you through each step. Isn't it lovely?"
        ]
        
        starry_masks = [
            "Cypress tree - dark swirling flame-like shape",
            "Swirling blue sky with stars",
            "Quiet village with warm yellow lights"
        ]
        
        self.profiles["starry.json"] = PaintingProfile(
            name="Starry Night",
            json_file="starry.json",
            music_file="Melancholy Drift.mp3",
            narrations=starry_narrations,
            mask_descriptions=starry_masks
        )
        
        # M3 BUILDING PROFILE
        m3_narrations = [
            "Look at this magnificent blue glass building reaching toward the sky! Each panel catches the light like a happy little window to the world. The blue reminds me of a calm lake reflecting the heavens above.",
            "Now we're adding some lovely brown concrete support structures. These aren't just boring old concrete - they're the strong, steady foundation that holds our beautiful creation together. Like the trunk of a mighty oak tree, they give our building character and strength.",
            "Here comes a cheerful green tree to soften our urban landscape! Trees are nature's way of saying 'hello' to the city. This little fellow brings life and freshness to our architectural symphony, reminding us that creativity and nature can dance together beautifully.",
            "Finally, we're painting this elegant dark blue bridge connecting our spaces. Bridges are wonderful things - they bring people together, just like how art brings us all together in this moment. This deep blue creates a lovely contrast with our lighter elements, adding depth and harmony to our composition."
        ]
        
        m3_masks = [
            "Main building - blue glass tiles",
            "Brown concrete structural support",
            "Green tree",
            "Dark blue bridge"
        ]
        
        self.profiles["m3.json"] = PaintingProfile(
            name="M3 Building",
            json_file="m3.json",
            music_file="Inspiration and Creativity.mp3",
            narrations=m3_narrations,
            mask_descriptions=m3_masks
        )
        
        # EIFFEL TOWER PROFILE
        eiffel_narrations = [
            "Ah, look at this magnificent golden frame reaching toward the heavens! The Eiffel Tower's outer structure shines like a beacon of love and romance, each beam carefully crafted to capture the warm Parisian sunlight. It's like painting with liquid gold, isn't it wonderful?",
            "Now we're adding the beautiful inner curves with darker gold tones. These aren't just structural elements - they're the tower's gentle embrace, creating intimate spaces within this iron lady's heart. See how the deeper gold creates depth and mystery, like the warm shadows of a loving embrace on a perfect evening in Paris."
        ]
        
        eiffel_masks = [
            "Outer frame - shining golden",
            "Inner curves - darker gold"
        ]
        
        self.profiles["eiffel.json"] = PaintingProfile(
            name="Eiffel Tower",
            json_file="eiffel.json",
            music_file="Love and Warmth.mp3",
            narrations=eiffel_narrations,
            mask_descriptions=eiffel_masks
        )
        
        # TITANIC PROFILE
        titanic_narrations = [
            "Here we see this magnificent vessel in her final moments, painted in deep metallic black that reflects both the moonlight and the gravity of this historical moment. Even in tragedy, there's a certain dignity to this great ship - she was built with love and craftsmanship, and we paint her with that same respect and reverence.",
            "Now we're adding the proud bow section, still reaching forward with determination even as fate calls her name. See how the metallic black gives weight and substance to this part of the ship? It's a reminder that even in our darkest hours, we can find strength and beauty in the way we face our challenges.",
            "Finally, we paint the surface of the ship as she meets the cold Atlantic waters. The metallic black creates a solemn contrast with the dark sea, showing us that sometimes our most profound moments come when we're tested by forces beyond our control. But remember, even in this scene, we're creating something meaningful together - there are no accidents, only lessons painted with gentle understanding."
        ]
        
        titanic_masks = [
            "Part of ship not yet submerged - metallic black",
            "Bow of the ship - metallic black", 
            "Surface of ship as it sinks - metallic black"
        ]
        
        self.profiles["titanic.json"] = PaintingProfile(
            name="Titanic",
            json_file="titanic.json",
            music_file="Fear and Unease.mp3",
            narrations=titanic_narrations,
            mask_descriptions=titanic_masks
        )
        
        # PETRONAS TOWERS PROFILE
        petronas_narrations = [
            "Look at this magnificent left tower reaching toward the heavens! Painted in gleaming silver, it stands like a proud sentinel of modern achievement and architectural wonder. Each silver panel catches the Malaysian sunlight, creating a symphony of light that reminds us how human creativity can touch the sky itself. Isn't it just breathtaking?",
            "Now we're adding the equally stunning right tower, also dressed in that beautiful silver finish. See how these twin towers dance together in perfect harmony? They're not just buildings - they're a celebration of partnership and shared dreams reaching toward the clouds. The silver gives them such elegance and grace, like two gentle giants watching over their city.",
            "Finally, we paint the extension on the left side of the left tower, completing our silver architectural family. This additional element adds depth and character to our composition, showing us that even in grand designs, there's always room for thoughtful details. Together, these silver structures create a masterpiece of human ambition painted with respect and wonder."
        ]
        
        petronas_masks = [
            "Left tower - silver",
            "Right tower - silver",
            "Left of the left tower - silver"
        ]
        
        self.profiles["petronas.json"] = PaintingProfile(
            name="Petronas Towers",
            json_file="petronas.json",
            music_file="Awe and Wonder.mp3",
            narrations=petronas_narrations,
            mask_descriptions=petronas_masks
        )
        
        # MONA LISA PROFILE
        monalisa_narrations = [
            "Here we begin with the elegant silhouette of our mysterious lady, painted in a rich dark dress that speaks of Renaissance sophistication. See how the dark tones create such beautiful contrast and depth? This isn't just fabric - it's the foundation of one of history's most beloved portraits, and we're painting it with the same love and attention that made her famous.",
            "Now let's add the dreamy water body in the background, painted in those soft, mysterious blues. These aren't just any waters - they're the misty landscapes of Leonardo's imagination, creating an ethereal backdrop that makes our lady seem to float between reality and dreams. The blue gives such peaceful serenity to our composition, doesn't it?",
            "Finally, we paint her luminous face and graceful neck in gentle whites and soft tones. This is where the magic happens - that enigmatic smile, those knowing eyes that have captivated viewers for centuries. We're not just painting skin, we're capturing the essence of human mystery and beauty. Every brushstroke here tells the story of artistic mastery and timeless elegance."
        ]
        
        monalisa_masks = [
            "Woman's outline in dark dress",
            "Water body in background - blue",
            "Face and neck - white"
        ]
        
        self.profiles["monalisa.json"] = PaintingProfile(
            name="Mona Lisa",
            json_file="monalisa.json", 
            music_file="Joyful Sunrise.mp3",
            narrations=monalisa_narrations,
            mask_descriptions=monalisa_masks
        )
    
    def get_profile_by_json(self, json_filename):
        """
        Get painting profile by JSON filename.
        
        Args:
            json_filename (str): Name of the JSON file (e.g., "starry.json")
            
        Returns:
            PaintingProfile or None: The matching profile or None if not found
        """
        return self.profiles.get(json_filename)
    
    def get_narrations_for_json(self, json_filename, max_count=None):
        """
        Get narrations for a specific JSON file.
        
        Args:
            json_filename (str): Name of the JSON file
            max_count (int): Maximum number of narrations to return
            
        Returns:
            list: List of narration strings
        """
        profile = self.get_profile_by_json(json_filename)
        if profile:
            return profile.get_narrations(max_count)
        
        # Fallback to default narrations if profile not found
        print(f"⚠️ No profile found for {json_filename}, using default narrations")
        return self._get_default_narrations(max_count)
    
    def get_music_for_json(self, json_filename):
        """
        Get music filename for a specific JSON file.
        
        Args:
            json_filename (str): Name of the JSON file
            
        Returns:
            str: Music filename or default
        """
        profile = self.get_profile_by_json(json_filename)
        if profile:
            return profile.music_file
        
        # Fallback to default music
        print(f"⚠️ No profile found for {json_filename}, using default music")
        return "Calm and Serenity.mp3"
    
    def _get_default_narrations(self, max_count=None):
        """Get default narrations when no profile is found."""
        default_narrations = [
            "Let's create something beautiful together on our canvas. Every stroke tells a story, and today we're telling yours.",
            "Remember, there are no mistakes in art, only happy accidents that lead us to unexpected beauty.",
            "Look how the colors dance together! Each element finds its perfect place in our composition, just like in life."
        ]
        
        if max_count is None:
            return default_narrations
        return default_narrations[:max_count]
    
    def list_all_profiles(self):
        """List all available painting profiles."""
        print("Available Painting Profiles:")
        for json_file, profile in self.profiles.items():
            print(f"  {json_file} -> {profile.name}")
            print(f"    Music: {profile.music_file}")
            print(f"    Narrations: {len(profile.narrations)}")
            if profile.mask_descriptions:
                print(f"    Masks: {len(profile.mask_descriptions)}")
            print()
    
    def add_profile(self, json_filename, name, music_file, narrations, mask_descriptions=None):
        """
        Add a new painting profile.
        
        Args:
            json_filename (str): JSON file name
            name (str): Display name for the painting
            music_file (str): Music file name
            narrations (list): List of Bob Ross narrations
            mask_descriptions (list): Optional mask descriptions
        """
        profile = PaintingProfile(
            name=name,
            json_file=json_filename,
            music_file=music_file,
            narrations=narrations,
            mask_descriptions=mask_descriptions
        )
        
        self.profiles[json_filename] = profile
        print(f"✅ Added profile for {json_filename}: {name}")


# Global instance for easy access
painting_manager = PaintingProfileManager()


def get_narrations_for_painting(json_filename, max_count=3):
    """
    Convenience function to get narrations for a painting.
    
    Args:
        json_filename (str): JSON file name (e.g., "starry.json")
        max_count (int): Maximum narrations to return
        
    Returns:
        list: Bob Ross narrations
    """
    return painting_manager.get_narrations_for_json(json_filename, max_count)


def get_music_for_painting(json_filename):
    """
    Convenience function to get music for a painting.
    
    Args:
        json_filename (str): JSON file name
        
    Returns:
        str: Music filename
    """
    return painting_manager.get_music_for_json(json_filename)


if __name__ == "__main__":
    # Test the system
    print("=== Painting Profiles System Test ===\n")
    
    # List all profiles
    painting_manager.list_all_profiles()
    
    # Test getting narrations and music
    test_files = ["starry.json", "m3.json", "unknown.json"]
    
    for json_file in test_files:
        print(f"--- Testing {json_file} ---")
        
        # Get music
        music = get_music_for_painting(json_file)
        print(f"Music: {music}")
        
        # Get narrations
        narrations = get_narrations_for_painting(json_file, max_count=2)
        print(f"Narrations ({len(narrations)}):")
        for i, narration in enumerate(narrations, 1):
            preview = narration[:60] + "..." if len(narration) > 60 else narration
            print(f"  {i}. {preview}")
        print()
