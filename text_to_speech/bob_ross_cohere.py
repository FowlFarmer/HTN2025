#!/usr/bin/env python3
"""
Bob Ross Text Transformation using Cohere API
Transforms any input text into Bob Ross-style speech, then uses TTS
"""

import os
import cohere
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def transform_to_bob_ross_style(object_color_pairs, cohere_api_key=None):
    """
    Transform array of (object, color) pairs into Bob Ross-style descriptions using Cohere API.
    
    Args:
        object_color_pairs (list): List of tuples containing (object, color) pairs
                                  e.g., [("apple", "red"), ("house", "orange"), ("water", "blue")]
        cohere_api_key (str): Cohere API key (if None, tries to get from environment)
        
    Returns:
        list: Array of Bob Ross-style descriptions, one for each input pair
    """
    # Get API key from parameter or environment variable
    if cohere_api_key is None:
        cohere_api_key = os.getenv('COHERE_API_KEY')
    
    if not cohere_api_key:
        raise ValueError("Cohere API key not provided. Set COHERE_API_KEY environment variable or pass as parameter.")
    
    # Initialize Cohere client
    co = cohere.Client(cohere_api_key)
    
    # Format the object-color pairs for the prompt
    pairs_text = ""
    for i, (obj, color) in enumerate(object_color_pairs, 1):
        pairs_text += f"{i}. {color} {obj}\n"
    
    # Create the prompt to transform object-color pairs into Bob Ross style
    prompt = f"""You are Bob Ross, the famous painter and TV host. For each object-color pair below, create a separate, humorous Bob Ross-style description. Bob Ross was known for:
- Speaking in a calm, gentle, soothing voice
- Using painting metaphors and references to "happy little trees", "happy accidents", etc.
- Being encouraging and positive
- Speaking slowly and thoughtfully
- Using phrases like "just like that", "there we go", "isn't that nice"
- Making everything sound peaceful and relaxing
- Adding humor and whimsy to descriptions

Object-color pairs to describe:
{pairs_text}

For each pair, write a separate Bob Ross description that's humorous and delightful. Format your response as:

1. [Description for first pair]
2. [Description for second pair]
3. [Description for third pair]
etc.

Make each description about 2-3 sentences long and full of Bob Ross charm and humor:"""

    try:
        # Generate Bob Ross-style text using Cohere
        response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=600,
            temperature=0.8,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
        
        # Extract the generated text
        full_response = response.generations[0].text.strip()
        
        # Parse the response into individual descriptions
        descriptions = []
        lines = full_response.split('\n')
        current_description = ""
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or 
                        line[0].isdigit() and '.' in line[:3]):
                # Save previous description if exists
                if current_description:
                    descriptions.append(current_description.strip())
                # Start new description (remove number prefix)
                current_description = line.split('.', 1)[1].strip() if '.' in line else line
            elif line and current_description:
                # Continue current description
                current_description += " " + line
        
        # Add the last description
        if current_description:
            descriptions.append(current_description.strip())
        
        # Ensure we have the right number of descriptions
        while len(descriptions) < len(object_color_pairs):
            # Fallback for missing descriptions
            missing_idx = len(descriptions)
            obj, color = object_color_pairs[missing_idx]
            fallback = add_simple_bob_ross_style_for_pair(obj, color)
            descriptions.append(fallback)
        
        # Trim to exact number needed
        descriptions = descriptions[:len(object_color_pairs)]
        
        return descriptions
        
    except Exception as e:
        print(f"Error with Cohere API: {e}")
        # Fallback to simple Bob Ross style transformation for each pair
        return [add_simple_bob_ross_style_for_pair(obj, color) for obj, color in object_color_pairs]

def add_simple_bob_ross_style_for_pair(obj, color):
    """
    Simple fallback function to add Bob Ross style for an object-color pair without API.
    
    Args:
        obj (str): The object name
        color (str): The color name
        
    Returns:
        str: Bob Ross-style description for the pair
    """
    import random
    
    bob_ross_intros = [
        "Now here we have a beautiful",
        "Let's paint a lovely",
        "Isn't this a happy little",
        "We're going to create a wonderful"
    ]
    
    bob_ross_middles = [
        "Just like that, we've made something special.",
        "There are no mistakes, only happy accidents.",
        "Isn't that just delightful?",
        "This brings such joy to the canvas."
    ]
    
    intro = random.choice(bob_ross_intros)
    middle = random.choice(bob_ross_middles)
    
    return f"{intro} {color} {obj}. {middle}"

def add_simple_bob_ross_style(text):
    """
    Simple fallback function to add Bob Ross style without API.
    
    Args:
        text (str): Original text
        
    Returns:
        str: Text with simple Bob Ross styling
    """
    bob_ross_phrases = [
        "Hello there, my friend.",
        "Let's take our time with this.",
        "Just like painting a happy little tree,",
        "There's no mistakes, only happy accidents.",
        "Isn't that just wonderful?",
        "Let's add some joy to this moment."
    ]
    
    import random
    intro = random.choice(bob_ross_phrases[:2])
    middle = random.choice(bob_ross_phrases[2:4])
    outro = random.choice(bob_ross_phrases[4:])
    
    return f"{intro} {middle} {text} {outro}"

def object_colors_to_bob_ross_speech(object_color_pairs, cohere_api_key=None, play_audio=True):
    """
    Complete pipeline: Transform object-color pairs to Bob Ross style and convert to speech.
    
    Args:
        object_color_pairs (list): List of (object, color) tuples to transform and speak
        cohere_api_key (str): Cohere API key (optional)
        play_audio (bool): Whether to play the audio immediately
        
    Returns:
        list: Array of Bob Ross-styled descriptions that were spoken
    """
    print(f"🎨 Original pairs: {object_color_pairs}")
    
    # Transform pairs using Cohere API
    try:
        bob_ross_descriptions = transform_to_bob_ross_style(object_color_pairs, cohere_api_key)
        print(f"🎨 Bob Ross descriptions:")
        for i, desc in enumerate(bob_ross_descriptions, 1):
            print(f"   {i}. {desc}")
    except Exception as e:
        print(f"⚠️  API transformation failed: {e}")
        bob_ross_descriptions = [add_simple_bob_ross_style_for_pair(obj, color) for obj, color in object_color_pairs]
        print(f"🎨 Fallback Bob Ross descriptions:")
        for i, desc in enumerate(bob_ross_descriptions, 1):
            print(f"   {i}. {desc}")
    
    if play_audio:
        # Import TTS only when needed to avoid pygame initialization
        from bob_ross_simple_tts import BobRossTTS
        
        # Initialize TTS and play each description
        print("🎙️  Converting to speech...")
        tts = BobRossTTS()
        for i, description in enumerate(bob_ross_descriptions, 1):
            print(f"🔊 Playing description {i}...")
            tts.text_to_speech_and_play(description)
            if i < len(bob_ross_descriptions):
                import time
                print("⏳ Waiting 2 seconds before next description...")
                time.sleep(2)
    
    return bob_ross_descriptions

def text_to_bob_ross_speech(input_text, cohere_api_key=None, play_audio=True):
    """
    Legacy function: Transform single text to Bob Ross style and convert to speech.
    
    Args:
        input_text (str): The original text to transform and speak
        cohere_api_key (str): Cohere API key (optional)
        play_audio (bool): Whether to play the audio immediately
        
    Returns:
        str: The Bob Ross-styled text that was spoken
    """
    print(f"🎨 Original text: '{input_text}'")
    
    # Transform text using simple Bob Ross style (since new function expects pairs)
    bob_ross_text = add_simple_bob_ross_style(input_text)
    print(f"🎨 Bob Ross style: '{bob_ross_text}'")
    
    if play_audio:
        # Import TTS only when needed to avoid pygame initialization
        from bob_ross_simple_tts import BobRossTTS
        
        # Initialize TTS and play the audio
        print("🎙️  Converting to speech...")
        tts = BobRossTTS()
        tts.text_to_speech_and_play(bob_ross_text)
    
    return bob_ross_text

def main():
    """Demo function to test the Bob Ross object-color transformation and TTS."""
    print("🎨" + "=" * 60 + "🎨")
    print("   Bob Ross AI Object-Color Transformation & Speech System")
    print("   'Let's turn your objects into happy little descriptions'")
    print("🎨" + "=" * 60 + "🎨")
    
    # Example usage with object-color pairs
    test_pairs = [
        ("apple", "red"),
        ("house", "orange"), 
        ("water", "blue"),
        ("tree", "green"),
        ("sunset", "purple")
    ]
    
    print(f"\n🎨 Testing Bob Ross transformations with pairs: {test_pairs}")
    
    try:
        # Test without audio first
        descriptions = object_colors_to_bob_ross_speech(test_pairs, play_audio=False)
        print(f"\n✅ Success! Generated {len(descriptions)} descriptions")
        
        # Show results
        print("\n🎨 Final Results:")
        for i, (pair, desc) in enumerate(zip(test_pairs, descriptions), 1):
            obj, color = pair
            print(f"{i}. {color.capitalize()} {obj}: {desc}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎨 Demo complete! To use with audio, set play_audio=True")
    print("🎨 To test: object_colors_to_bob_ross_speech([('apple', 'red'), ('house', 'orange')])")

if __name__ == "__main__":
    main()
