#!/usr/bin/env python3
"""
Bob Ross Text-to-Speech using a simpler approach
This version uses available libraries and provides a fallback implementation
"""

import os
import sys
import subprocess
import tempfile
import pygame
from pathlib import Path

class BobRossTTS:
    def __init__(self):
        """Initialize the Bob Ross TTS system."""
        print("🎨 Initializing Bob Ross TTS System 🎨")
        
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            print("Audio system initialized successfully!")
        except Exception as e:
            print(f"Warning: Could not initialize audio system: {e}")
    
    def synthesize_with_say_command(self, text, output_path="bob_ross_output.wav"):
        """
        Use macOS 'say' command to generate speech with a suitable voice.
        
        Args:
            text (str): Text to convert to speech
            output_path (str): Path to save the generated audio
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            print(f"Synthesizing with macOS 'say' command: '{text}'")
            
            # Use a deep, calm voice that's closest to Bob Ross
            # Available voices: Alex, Daniel, Fred, etc.
            voice = "Fred"  # Fred has a calm, deeper voice
            
            # Generate audio using macOS say command
            cmd = ["say", "-v", voice, "-o", output_path, text]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Audio generated successfully: {output_path}")
                return output_path
            else:
                print(f"Error with 'say' command: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error during speech synthesis: {e}")
            return None
    
    def download_and_use_hf_model(self, text, output_path="bob_ross_output.wav"):
        """
        Attempt to use the Hugging Face model directly.
        This is a placeholder for when the environment supports it.
        """
        print("Hugging Face model integration not available in current environment.")
        print("Using fallback method...")
        return self.synthesize_with_say_command(text, output_path)
    
    def play_audio(self, audio_path):
        """
        Play the generated audio file.
        
        Args:
            audio_path (str): Path to the audio file to play
        """
        try:
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return
            
            print("🔊 Playing Bob Ross voice...")
            
            # Try pygame first
            try:
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
                print("✅ Playback completed!")
                return
            except Exception as pygame_error:
                print(f"Pygame playback failed: {pygame_error}")
            
            # Fallback to system audio player
            print("Using system audio player...")
            if sys.platform == "darwin":  # macOS
                subprocess.run(["afplay", audio_path])
            elif sys.platform == "linux":
                subprocess.run(["aplay", audio_path])
            else:
                print("Unsupported platform for audio playback")
            
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def text_to_speech_and_play(self, text):
        """
        Convert text to speech and play it immediately.
        
        Args:
            text (str): Text to convert and play
        """
        # Create output file in current directory
        output_path = "bob_ross_voice.aiff"
        
        try:
            # Generate speech
            audio_path = self.synthesize_with_say_command(text, output_path)
            
            if audio_path and os.path.exists(audio_path):
                # Play the generated audio
                self.play_audio(audio_path)
                
                # Clean up
                try:
                    os.remove(audio_path)
                except:
                    pass
            else:
                print("❌ Failed to generate speech")
        
        except Exception as e:
            print(f"Error in text-to-speech process: {e}")

def read_text_file(file_path):
    """
    Read text from a file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: Content of the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            return content
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def add_bob_ross_style(text):
    """
    Add Bob Ross-style phrases to make the speech more characteristic.
    
    Args:
        text (str): Original text
        
    Returns:
        str: Text with Bob Ross style additions
    """
    bob_ross_intro = "Hello there, my friend. "
    bob_ross_outro = " Just like painting a happy little tree, we've created something beautiful together."
    
    return f"{bob_ross_intro}{text}{bob_ross_outro}"

'''
Test usage:

bob_ross_tts = BobRossTTS()
    text_file_path = "test.txt"
    text_content = read_text_file(text_file_path)
    
    if text_content:
        print(f"\n📖 Text from {text_file_path}: '{text_content}'")
        styled_text = add_bob_ross_style(text_content)
        bob_ross_tts.text_to_speech_and_play(styled_text)
'''