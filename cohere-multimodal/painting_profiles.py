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
            "We start with this magnificent cypress tree reaching toward the stars.",
            "Now we add the swirling night sky with its dancing blue winds.",
            "Finally, we paint the peaceful village resting below."
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
            "We begin with the main blue glass building structure.",
            "Next we add the brown concrete support framework.", 
            "Now we paint a beautiful green tree for natural contrast.",
            "Finally we complete the dark blue bridge connection."
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
            "We start with the golden outer frame of this iconic tower.",
            "Now we add the darker gold inner curves and details."
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
            "We begin with the main hull of this historic vessel.",
            "Next we paint the proud bow section reaching forward.",
            "Finally we add the ship's surface meeting the Atlantic waters."
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
            "We start with the magnificent left tower in gleaming silver.",
            "Now we add the equally stunning right tower beside it.",
            "Finally we complete the extension on the left side."
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
            "We begin with the elegant dark dress of our mysterious lady.",
            "Next we add the dreamy blue water body in the background.",
            "Finally we paint her luminous face with that famous smile."
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
            "We start by creating the first layer of our painting.",
            "Now we add the next section with careful strokes.",
            "Finally we complete our artistic composition together."
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
