#!/usr/bin/env python3
"""
Simple Bob Ross Text-to-Speech
One function that takes a string and outputs Bob Ross audio
Now with ElevenLabs integration and fallback to macOS TTS
"""

import os
import subprocess
import random
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ElevenLabs integration
try:
    from elevenlabs import ElevenLabs, play, save
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    print("⚠️ ElevenLabs not available. Install with: pip install elevenlabs")

def speak_in_bob_ross_voice(text):
    """
    Takes a string input and speaks it in Bob Ross voice.
    Uses ElevenLabs API with fallback to macOS TTS.
    
    Args:
        text (str): Text to speak in Bob Ross voice
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Try ElevenLabs first
    if _try_elevenlabs_tts(text):
        return True
    
    # Fallback to macOS TTS
    print("🔄 Falling back to macOS TTS...")
    return _fallback_macos_tts(text)

def _try_elevenlabs_tts(text):
    """
    Attempt to use ElevenLabs TTS with the calm voice.
    
    Args:
        text (str): Text to speak
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not ELEVENLABS_AVAILABLE:
        return False
    
    try:
        # Get API key from environment only
        api_key = os.getenv('ELEVENLABS_API_KEY')
        if not api_key:
            print("⚠️ ElevenLabs API key not found in environment variables")
            return False
        
        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=api_key)
        
        # Add Bob Ross styling
        styled_text = _add_bob_ross_style(text)
        
        # Generate speech with the calm voice
        print("🎙️ Generating speech with ElevenLabs...")
        audio = client.text_to_speech.convert(
            text=styled_text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # Calm voice ID
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        
        # Save to temporary file and play
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            save(audio, temp_file.name)
            
            # Play using afplay (macOS)
            play_cmd = ["afplay", temp_file.name]
            result = subprocess.run(play_cmd, capture_output=True)
            
            # Clean up
            try:
                os.remove(temp_file.name)
            except:
                pass
            
            if result.returncode == 0:
                print("✅ ElevenLabs TTS successful!")
                return True
            else:
                print(f"❌ Failed to play ElevenLabs audio: {result.stderr}")
                return False
        
    except Exception as e:
        print(f"❌ ElevenLabs TTS failed: {e}")
        return False

def _fallback_macos_tts(text):
    """
    Fallback to macOS built-in TTS system.
    
    Args:
        text (str): Text to speak
        
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
            
            print("✅ macOS TTS fallback successful!")
            return True
        else:
            print(f"❌ Error with macOS TTS: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error in macOS TTS fallback: {e}")
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