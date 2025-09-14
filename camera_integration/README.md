# Voice-Activated Camera System

This system captures photos from your webcam when you speak a specific trigger phrase.

## Features

- **Voice Recognition**: Uses Google Speech Recognition to detect spoken commands
- **Webcam Integration**: Captures high-quality photos from your connected webcam
- **Customizable Trigger Phrase**: Set any phrase to trigger photo capture (default: "take photo")
- **Automatic Saving**: Photos are saved with timestamps in a dedicated folder
- **Preview Display**: Shows captured photos briefly after taking them

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have:
   - A working webcam connected to your computer
   - A working microphone
   - Internet connection (required for Google Speech Recognition)

## Usage

1. Run the voice camera system:
```bash
python voice_camera.py
```

2. When prompted, enter your desired trigger phrase or press Enter to use the default ("take photo")

3. The system will:
   - Initialize your webcam
   - Start listening for the trigger phrase
   - Take a photo whenever the phrase is detected
   - Save photos to the `photos/` directory

4. To stop the system:
   - Press 'q' and Enter, or
   - Press Ctrl+C

## Manual Controls

While the system is running, you can also:
- Type `photo` and press Enter to manually capture a photo
- Type `q` or `quit` and press Enter to exit

## Photo Storage

Photos are automatically saved in the `photos/` directory with filenames like:
- `photo_20250914_120345.jpg` (format: photo_YYYYMMDD_HHMMSS.jpg)

## Troubleshooting

If you encounter issues:

1. **Camera not working**: Make sure your webcam is connected and not being used by another application
2. **Microphone not working**: Check your microphone permissions and ensure it's properly connected
3. **Speech recognition errors**: Ensure you have an internet connection for Google Speech Recognition
4. **Installation issues**: Try installing packages individually:
   ```bash
   pip install opencv-python
   pip install SpeechRecognition
   pip install pyaudio
   ```

## Customization

You can modify the `VoiceActivatedCamera` class to:
- Change the default trigger phrase
- Adjust camera resolution
- Modify photo save location
- Add additional voice commands
