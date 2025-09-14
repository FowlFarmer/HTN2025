import speech_recognition as sr

def block_until_yes():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Listening for the word 'yes'...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)  # helps in noisy rooms
        while True:
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio).lower()
                print(f"Heard: {text}")
                if "yes" in text:
                    print("Detected 'yes'!")
                    break
            except sr.UnknownValueError:
                pass  # didn't catch speech, keep listening

if __name__ == "__main__":
    block_until_yes()