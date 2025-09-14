import cv2
import speech_recognition as sr
import threading
import time
import os
from datetime import datetime
import pyaudio

class VoiceActivatedCamera:
    def __init__(self, trigger_phrase="take photo", save_directory="photos"):
        """
        Initialize the voice-activated camera system.
        
        Args:
            trigger_phrase (str): The phrase that triggers photo capture
            save_directory (str): Directory to save captured photos
        """
        self.trigger_phrase = trigger_phrase.lower()
        self.save_directory = save_directory
        self.is_listening = False
        self.camera = None
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        print("Adjusting for ambient noise... Please wait.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        print("Ready for voice commands!")
    
    def initialize_camera(self):
        """Initialize the webcam."""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("Could not open webcam")
            
            # Set camera properties for better quality
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            print("Camera initialized successfully!")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def capture_photo(self):
        """Capture a photo from the webcam."""
        if not self.camera or not self.camera.isOpened():
            print("Camera not available!")
            return None
        
        try:
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture frame!")
                return None
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.jpg"
            filepath = os.path.join(self.save_directory, filename)
            
            # Save the photo
            cv2.imwrite(filepath, frame)
            print(f"Photo saved: {filepath}")
            
            # Show a preview window briefly
            cv2.imshow("Photo Captured", frame)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyAllWindows()
            
            return filepath
            
        except Exception as e:
            print(f"Error capturing photo: {e}")
            return None
    
    def listen_for_trigger(self):
        """Listen for the trigger phrase in a separate thread."""
        while self.is_listening:
            try:
                with self.microphone as source:
                    print(f"Listening for '{self.trigger_phrase}'...")
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                try:
                    # Recognize speech using Google Speech Recognition
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Heard: '{text}'")
                    
                    # Check if trigger phrase is in the recognized text
                    if self.trigger_phrase in text:
                        print(f"Trigger phrase detected! Taking photo...")
                        self.capture_photo()
                        
                except sr.UnknownValueError:
                    # Speech was unintelligible
                    pass
                except sr.RequestError as e:
                    print(f"Could not request results from speech recognition service: {e}")
                    
            except sr.WaitTimeoutError:
                # No speech detected within timeout
                pass
            except Exception as e:
                print(f"Error in speech recognition: {e}")
                time.sleep(1)
    
    def start_listening(self):
        """Start the voice-activated camera system."""
        if not self.initialize_camera():
            return
        
        self.is_listening = True
        
        # Start listening in a separate thread
        listen_thread = threading.Thread(target=self.listen_for_trigger)
        listen_thread.daemon = True
        listen_thread.start()
        
        print(f"Voice-activated camera started!")
        print(f"Say '{self.trigger_phrase}' to take a photo.")
        print("Press 'q' to quit or Ctrl+C to stop.")
        
        try:
            # Keep the main thread alive and allow manual quit
            while self.is_listening:
                key = input().strip().lower()
                if key == 'q' or key == 'quit':
                    break
                elif key == 'photo':
                    # Manual photo capture
                    self.capture_photo()
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_listening()
    
    def stop_listening(self):
        """Stop the voice-activated camera system."""
        print("\nStopping voice-activated camera...")
        self.is_listening = False
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        print("Camera system stopped.")

def main():
    """Main function to run the voice-activated camera."""
    print("=== Voice-Activated Camera System ===")
    print("This system will take photos when you say the trigger phrase.")
    
    # You can customize the trigger phrase here
    trigger_phrase = input("Enter trigger phrase (default: 'take photo'): ").strip()
    if not trigger_phrase:
        trigger_phrase = "take photo"
    
    # Initialize and start the camera system
    camera_system = VoiceActivatedCamera(trigger_phrase=trigger_phrase)
    
    try:
        camera_system.start_listening()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. A working webcam connected")
        print("2. A working microphone")
        print("3. Internet connection (for speech recognition)")
        print("4. Required packages installed:")
        print("   pip install opencv-python speechrecognition pyaudio")

if __name__ == "__main__":
    main()
