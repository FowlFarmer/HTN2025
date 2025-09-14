#!/usr/bin/env python3
"""
Starry Night specific analyzer for generating Bob Ross descriptions
Uses the Starry Night image with a specific color detection result
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from painting_mask_analyzer import analyze_painting_with_masks, analyze_masks_fallback
from text_to_speech.bob_ross_cohere import transform_to_bob_ross_style


def analyze_starrynight_with_specific_mask(cohere_api_key=None):
    """
    Analyze Starry Night with the specific color detection result.
    
    Args:
        cohere_api_key (str): Cohere API key (optional)
        
    Returns:
        tuple: (object_color_pairs, bob_ross_descriptions)
    """
    # Fixed paths for Starry Night analysis
    project_root = Path(__file__).parent.parent
    original_image = project_root / "painting_vectorization" / "examples" / "starrynight.jpg"
    specific_mask = project_root / "painting_vectorization" / "results" / "step7_color_detection" / "color_detection_results_20250914_042939.png"
    
    print(f"🌟 Analyzing Starry Night painting...")
    print(f"📷 Original image: {original_image.name}")
    print(f"🎭 Color mask: {specific_mask.name}")
    
    if not original_image.exists():
        raise FileNotFoundError(f"Starry Night image not found: {original_image}")
    
    if not specific_mask.exists():
        raise FileNotFoundError(f"Color detection mask not found: {specific_mask}")
    
    # COMMENTED OUT: API calls work correctly but take too long for demo timing
    # The Cohere API successfully generates dynamic object-color pairs, but the
    # network latency and processing time (5-15 seconds) is too slow for live demos
    # try:
    #     # Try with Cohere API first
    #     object_color_pairs = analyze_painting_with_masks(
    #         str(original_image), 
    #         str(specific_mask), 
    #         cohere_api_key
    #     )
    #     print(f"✅ Successfully analyzed with Cohere API")
    #     
    # except Exception as e:
    #     print(f"⚠️ Cohere API failed ({e}), using fallback analysis...")
    #     # Use fallback method
    #     object_color_pairs = analyze_masks_fallback(
    #         str(original_image), 
    #         str(specific_mask)
    #     )
    
    # Always use predefined object-color pairs for demo timing (API works but is slow)
    print(f"🔒 Using predefined object-color pairs for Starry Night (fast demo mode)")
    object_color_pairs = [
        ("sky", "blue"),
        ("cypress tree", "dark green"),
        ("village", "yellow"),
        ("hills", "purple"),
        ("stars", "white"),
        ("moon", "yellow")
    ]
    
    print(f"🎨 Extracted {len(object_color_pairs)} object-color pairs:")
    for i, (obj, color) in enumerate(object_color_pairs, 1):
        print(f"  {i}. {color} {obj}")
    
    # COMMENTED OUT: Bob Ross API generation works but takes too long for demo timing
    # The Cohere API successfully generates beautiful Bob Ross descriptions, but the
    # API calls add 10-20 seconds of delay which disrupts the painting demo flow
    # # Generate Bob Ross descriptions
    # print(f"\n🗣️ Generating Bob Ross descriptions...")
    # try:
    #     bob_ross_descriptions = transform_to_bob_ross_style(object_color_pairs, cohere_api_key)
    #     print(f"✅ Generated {len(bob_ross_descriptions)} Bob Ross descriptions")
    #     
    # except Exception as e:
    #     print(f"⚠️ Bob Ross generation failed ({e}), using simple fallback...")
    #     # Simple fallback descriptions
    #     bob_ross_descriptions = []
    #     for obj, color in object_color_pairs:
    #         simple_desc = f"Now here we have a beautiful {color} {obj}. Let's paint it with gentle, loving strokes."
    #         bob_ross_descriptions.append(simple_desc)
    
    # Always use the three hardcoded Bob Ross narrations for demo timing
    print(f"\n🔒 Using hardcoded Bob Ross narrations for Starry Night (fast demo mode)")
    bob_ross_descriptions = [
        "Oh, look at this cheeky little cypress tree! It's flickering like a flame, wanting to draw our attention. Let's appreciate its fiery charm and give it a hug from afar, shall we?",
        "The sky is doing cartwheels today, with vibrant blue swirls unfolding like a symphony. Let's imagine we're painting with the clouds, adding our own touches to this joyous spectacle.",
        "Nestled in the quiet of night, this serene village seems to hug its residents to sleep. Let's paint it with gentle, loving strokes, reminding ourselves that quiet moments are just as valuable as louder ones. Consider getting your paint, brushes and palette ready as you relax and enjoy the entire painting process, gently guiding you through each step. Isn't it lovely?"
    ]
    print(f"✅ Using {len(bob_ross_descriptions)} hardcoded Bob Ross descriptions")
    
    return object_color_pairs, bob_ross_descriptions


def get_starrynight_narrations(max_narrations=3, cohere_api_key=None):
    """
    Get Bob Ross narrations for Starry Night, limited to max_narrations.
    Returns hardcoded narrations for demo timing - API calls work but are too slow.
    
    Args:
        max_narrations (int): Maximum number of narrations to return
        cohere_api_key (str): Cohere API key (optional, not used due to timing)
        
    Returns:
        list: List of Bob Ross narration strings
        
    Note: The Cohere API integration works perfectly and generates beautiful
    dynamic narrations, but the 15-30 second processing time is too long for
    live painting demonstrations, so we use pre-generated hardcoded narrations.
    """
    # COMMENTED OUT: API calls work perfectly but are too slow for live demos
    # The full AI pipeline (image analysis + Bob Ross generation) works beautifully
    # but takes 15-30 seconds total, which is too long for live painting demonstrations
    # try:
    #     pairs, descriptions = analyze_starrynight_with_specific_mask(cohere_api_key)
    #     
    #     # Limit to max_narrations
    #     limited_descriptions = descriptions[:max_narrations]
    #     
    #     # If we don't have enough, add some general Bob Ross phrases
    #     while len(limited_descriptions) < max_narrations:
    #         fallback_phrases = [
    #             "Let's add some happy little details here and there. Remember, there are no mistakes, only happy accidents.",
    #             "Isn't this just delightful? We're creating something beautiful together, one gentle stroke at a time.",
    #             "Look how the colors dance together on our canvas. Each brushstroke tells its own little story."
    #         ]
    #         
    #         fallback_index = len(limited_descriptions) % len(fallback_phrases)
    #         limited_descriptions.append(fallback_phrases[fallback_index])
    #     
    #     return limited_descriptions[:max_narrations]
    #     
    # except Exception as e:
    #     print(f"❌ Error generating dynamic narrations: {e}")
    #     print("🔄 Falling back to hardcoded narrations...")
    
    # Always return the three hardcoded Bob Ross narrations for demo timing
    print(f"🔒 Returning hardcoded Bob Ross narrations for Starry Night (fast demo mode)")
    hardcoded_narrations = [
        "Oh, look at this cheeky little cypress tree! It's flickering like a flame, wanting to draw our attention. Let's appreciate its fiery charm and give it a hug from afar, shall we?",
        "The sky is doing cartwheels today, with vibrant blue swirls unfolding like a symphony. Let's imagine we're painting with the clouds, adding our own touches to this joyous spectacle.",
        "Nestled in the quiet of night, this serene village seems to hug its residents to sleep. Let's paint it with gentle, loving strokes, reminding ourselves that quiet moments are just as valuable as louder ones. Consider getting your paint, brushes and palette ready as you relax and enjoy the entire painting process, gently guiding you through each step. Isn't it lovely?"
    ]
    
    return hardcoded_narrations[:max_narrations]


if __name__ == "__main__":
    # Test the analyzer
    print("=== Testing Starry Night Analyzer ===\n")
    
    try:
        pairs, descriptions = analyze_starrynight_with_specific_mask()
        
        print(f"\n--- Results ---")
        print(f"Object-Color Pairs: {pairs}")
        print(f"\nBob Ross Descriptions:")
        for i, desc in enumerate(descriptions, 1):
            print(f"{i}. {desc}")
            
        print(f"\n--- Narrations for run_json.py ---")
        narrations = get_starrynight_narrations(3)
        for i, narration in enumerate(narrations, 1):
            print(f"{i}. {narration[:100]}...")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
