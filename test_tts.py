#!/usr/bin/env python3
"""
Test script for ElevenLabs TTS integration
Tests the speak_in_bob_ross_voice function with a simple message
"""

import sys
import os

# Add the text_to_speech module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'text_to_speech'))

from bob_ross_simple_tts import speak_in_bob_ross_voice

def test_tts():
    """Test the TTS function with ElevenLabs and fallback"""
    test_text = "hello, i am at hack the north"
    
    print("🧪 Testing TTS with ElevenLabs (primary) and macOS fallback...")
    print(f"📝 Test text: '{test_text}'")
    print("🎙️ This will be styled with Bob Ross phrases and spoken")
    print("-" * 60)
    
    # Test the TTS function
    success = speak_in_bob_ross_voice(test_text)
    
    print("-" * 60)
    if success:
        print("✅ TTS test completed successfully!")
    else:
        print("❌ TTS test failed!")
    
    return success

if __name__ == "__main__":
    print("🎨 Bob Ross TTS Test Script")
    print("=" * 60)
    
    # Check if ElevenLabs API key is set
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if api_key:
        print("✅ ElevenLabs API key found - will try ElevenLabs first")
    else:
        print("⚠️  ElevenLabs API key not set - will use macOS TTS fallback")
    
    print()
    test_tts()
