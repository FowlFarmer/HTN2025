#!/usr/bin/env python3
"""
Bob Ross Text Transformation using Cohere API
Transforms any input text into Bob Ross-style speech, then uses TTS
"""

import os
import cohere
from dotenv import load_dotenv

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


if __name__ == "__main__":
    test_pairs = [("apple", "red"), ("tree", "green"), ("sky", "blue")]
    descriptions = transform_to_bob_ross_style(test_pairs)
    
    print("Generated Bob Ross descriptions:")
    for i, desc in enumerate(descriptions, 1):
        print(f"{i}. {desc}")