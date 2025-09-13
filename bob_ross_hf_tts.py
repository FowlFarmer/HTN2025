#!/usr/bin/env python3
"""
Bob Ross Text-to-Speech using Hugging Face XTTS Fine-tuned Model
Uses the drewThomasson/Xtts-FineTune-Bob-Ross model from Hugging Face
"""

import torch
import torchaudio
import soundfile as sf
from TTS.api import TTS
import pygame
import tempfile
import os
import sys

class BobRossTTS:
    def __init__(self):
        """Initialize the Bob Ross TTS system with the Hugging Face model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Load the Bob Ross fine-tuned XTTS model from Hugging Face
        print("Loading Bob Ross XTTS model from Hugging Face...")
        try:
            # Use the specific Bob Ross fine-tuned model
            # The model path for the Hugging Face model
            model_path = "drewThomasson/Xtts-FineTune-Bob-Ross"
            self.tts = TTS(model_path=model_path).to(self.device)
            print("Bob Ross fine-tuned model loaded successfully!")
        except Exception as e:
            print(f"Error loading Bob Ross model: {e}")
            print("Falling back to default XTTS model...")
            self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
    
    def synthesize_speech(self, text, output_path="bob_ross_output.wav"):
        """
        Convert text to speech using Bob Ross voice.
        
        Args:
            text (str): Text to convert to speech
            output_path (str): Path to save the generated audio
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            print(f"Synthesizing: '{text}'")
            
            # For XTTS, we need a reference audio file for voice cloning
            # Since we're using a fine-tuned model, we'll use the default speaker
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=None,  # Use the fine-tuned model's default voice
                language="en"
            )
            
            print(f"Audio generated and saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error during speech synthesis: {e}")
            return None
    
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
            
            print("Playing audio...")
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            print("Playback completed!")
            
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def text_to_speech_and_play(self, text):
        """
        Convert text to speech and play it immediately.
        
        Args:
            text (str): Text to convert and play
        """
        # Use a temporary file for audio output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Generate speech
            audio_path = self.synthesize_speech(text, temp_path)
            
            if audio_path:
                # Play the generated audio
                self.play_audio(audio_path)
            else:
                print("Failed to generate speech")
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

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
            return file.read().strip()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def main():
    """Main function to demonstrate Bob Ross TTS."""
    print("🎨 Bob Ross Text-to-Speech System 🎨")
    print("=" * 50)
    
    # Initialize the TTS system
    bob_ross_tts = BobRossTTS()
    
    # Read text from test.txt
    text_file_path = "test.txt"
    text_content = read_text_file(text_file_path)
    
    if text_content:
        print(f"Text to speak: '{text_content}'")
        print("\nGenerating Bob Ross voice...")
        
        # Convert text to speech and play
        bob_ross_tts.text_to_speech_and_play(text_content)
    else:
        print("No text found to convert to speech.")
        
        # Demo with default text
        demo_text = "Hello there, my friend. Let's paint some happy little trees together."
        print(f"Using demo text: '{demo_text}'")
        bob_ross_tts.text_to_speech_and_play(demo_text)

if __name__ == "__main__":
    main()
