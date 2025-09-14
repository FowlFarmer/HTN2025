#!/usr/bin/env python3
"""
Simple Bob Ross Text-to-Speech
One function that takes a string and outputs Bob Ross audio
"""

import os
import subprocess
import random

def speak_in_bob_ross_voice(text):
    """
    Takes a string input and speaks it in Bob Ross voice.
    
    Args:
        text (str): Text to speak in Bob Ross voice
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        styled_text = _add_bob_ross_style(text)
        output_path = "bob_ross_temp.aiff"
        voice = "Fred"
        
        cmd = ["say", "-v", voice, "-o", output_path, styled_text]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            play_cmd = ["afplay", output_path]
            subprocess.run(play_cmd)
            
            try:
                os.remove(output_path)
            except:
                pass
            
            return True
        else:
            print(f"Error with TTS: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error in Bob Ross TTS: {e}")
        return False

def _add_bob_ross_style(text):
    """
    Add Bob Ross-style phrases to make the speech more characteristic.
    
    Args:
        text (str): Original text
        
    Returns:
        str: Text with Bob Ross style additions
    """
    bob_ross_intros = [
        "Hello there, my friend.",
        "Well hello there, happy painter.",
        "Let's take our time with this.",
        "Now, isn't this just wonderful?"
    ]
    
    bob_ross_transitions = [
        "Just like painting a happy little tree,",
        "You know, there are no mistakes, only happy accidents, and",
        "Let's add some joy to this moment as",
        "With gentle brushstrokes of words,"
    ]
    
    bob_ross_outros = [
        "Just like that, we've created something beautiful together.",
        "Isn't that just delightful? Until next time, happy painting!",
        "There we go, another happy little moment shared.",
        "And remember, you have the power to create beauty wherever you go."
    ]
    
    intro = random.choice(bob_ross_intros)
    transition = random.choice(bob_ross_transitions)
    outro = random.choice(bob_ross_outros)
    
    return f"{intro} {transition} {text} {outro}"