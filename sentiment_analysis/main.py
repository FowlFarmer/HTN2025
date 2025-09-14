import cohere
import requests
from PIL import Image
import base64
import io
import os
from dotenv import load_dotenv
from enum import Enum

load_dotenv()

class MusicTheme(Enum):
    """Enumeration of available music themes and their corresponding audio files."""
    AWE_AND_WONDER = ("Awe and Wonder", "Awe and Wonder.mp3")
    JOY_AND_HAPPINESS = ("Joy and Happiness", "Joyful Sunrise.mp3")
    MELANCHOLY_AND_SADNESS = ("Melancholy and Sadness", "Melancholy Drift.mp3")
    EMPATHY_AND_COMPASSION = ("Empathy and Compassion", "Empathy and Compassion.mp3")
    CALM_AND_SERENITY = ("Calm and Serenity", "Calm and Serenity.mp3")
    FEAR_AND_UNEASE = ("Fear and Unease", "Fear and Unease.mp3")
    INSPIRATION_AND_CREATIVITY = ("Inspiration and Creativity", "Inspiration and Creativity.mp3")
    LOVE_AND_WARMTH = ("Love and Warmth", "Love and Warmth.mp3")
    CONFUSION_AND_CURIOSITY = ("Confusion and Curiosity", "Confusion and Curiosity.mp3")
    SPIRITUALITY_AND_TRANSCENDENCE = ("Spirituality and Transcendence", "Spirituality and Transcendence.mp3")
    
    def __init__(self, theme_name, filename):
        self.theme_name = theme_name
        self.filename = filename


def analyze_image_sentiment_from_url(photo_path, cohere_api_key=None):
    """
    Analyze the sentiment/emotion of an image from a URL or local file path using Cohere API.
    
    Args:
        photo_path (str): URL or local file path of the image to analyze
        cohere_api_key (str, optional): Cohere API key. If not provided, will try to get from environment
        
    Returns:
        MusicTheme: The music theme that best matches the image's emotional content
    """
    try:
        if not cohere_api_key:
            cohere_api_key = os.getenv('COHERE_API_KEY')
            
        if not cohere_api_key:
            raise ValueError("Cohere API key not provided. Set COHERE_API_KEY environment variable or pass as parameter.")
        
        co = cohere.Client(cohere_api_key)
        
        if photo_path.startswith(('http://', 'https://')):
            print(f"Downloading image from URL: {photo_path}")
            response = requests.get(photo_path, timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        else:
            # Handle local file path
            from pathlib import Path
            file_path = Path(photo_path)
            if not file_path.is_absolute():
                # Convert relative path to absolute
                file_path = Path.cwd() / file_path
            
            print(f"Loading local image: {file_path}")
            if not file_path.exists():
                raise FileNotFoundError(f"Image file not found: {file_path}")
            
            image = Image.open(file_path)
        
        # Convert image to base64 for API (if needed)
        # For now, we'll use text-based analysis with image description
        
        # Create a detailed prompt for sentiment analysis
        prompt = f"""
        I need you to analyze the emotional sentiment and mood of an image located at: {photo_path}
        
        Please analyze the image and determine which of these emotional categories best describes the overall feeling and mood conveyed:
        
        1. AWE_AND_WONDER - Images that inspire amazement, grandeur, or breathtaking beauty
        2. JOY_AND_HAPPINESS - Images that convey cheerfulness, celebration, or positive energy
        3. MELANCHOLY_AND_SADNESS - Images that evoke sadness, longing, or somber moods
        4. EMPATHY_AND_COMPASSION - Images that show care, understanding, or human connection
        5. CALM_AND_SERENITY - Images that feel peaceful, tranquil, or meditative
        6. FEAR_AND_UNEASE - Images that create tension, anxiety, or discomfort
        7. INSPIRATION_AND_CREATIVITY - Images that spark imagination or artistic expression
        8. LOVE_AND_WARMTH - Images that convey affection, coziness, or intimate feelings
        9. CONFUSION_AND_CURIOSITY - Images that are puzzling, mysterious, or thought-provoking
        10. SPIRITUALITY_AND_TRANSCENDENCE - Images that evoke the sacred, divine, or transcendent
        
        Consider factors like:
        - Color palette and lighting
        - Subject matter and composition
        - Visual elements and their arrangement
        - Overall atmosphere and mood
        - Emotional impact on the viewer
        
        Respond with ONLY the category name (e.g., "AWE_AND_WONDER") that best matches the image's emotional content.
        """
        
        print("Analyzing image sentiment with Cohere Chat API...")
        response = co.chat(
            model='command-r-plus',
            message=prompt,
            temperature=0.3,
            max_tokens=50
        )
        
        predicted_sentiment = response.text.strip().upper()
        
        sentiment_mapping = {
            'AWE_AND_WONDER': MusicTheme.AWE_AND_WONDER,
            'JOY_AND_HAPPINESS': MusicTheme.JOY_AND_HAPPINESS,
            'MELANCHOLY_AND_SADNESS': MusicTheme.MELANCHOLY_AND_SADNESS,
            'EMPATHY_AND_COMPASSION': MusicTheme.EMPATHY_AND_COMPASSION,
            'CALM_AND_SERENITY': MusicTheme.CALM_AND_SERENITY,
            'FEAR_AND_UNEASE': MusicTheme.FEAR_AND_UNEASE,
            'INSPIRATION_AND_CREATIVITY': MusicTheme.INSPIRATION_AND_CREATIVITY,
            'LOVE_AND_WARMTH': MusicTheme.LOVE_AND_WARMTH,
            'CONFUSION_AND_CURIOSITY': MusicTheme.CONFUSION_AND_CURIOSITY,
            'SPIRITUALITY_AND_TRANSCENDENCE': MusicTheme.SPIRITUALITY_AND_TRANSCENDENCE
        }
        
        selected_theme = sentiment_mapping.get(predicted_sentiment)
        
        if not selected_theme:
            print(f"Warning: Unexpected sentiment response '{predicted_sentiment}'. Using default.")
            for key, theme in sentiment_mapping.items():
                if key in predicted_sentiment or predicted_sentiment in key:
                    selected_theme = theme
                    break
            
            if not selected_theme:
                selected_theme = MusicTheme.CALM_AND_SERENITY
        
        print(f"Image sentiment analysis complete!")
        print(f"Detected sentiment: {predicted_sentiment}")
        print(f"Selected music theme: {selected_theme.theme_name}")
        print(f"Associated music file: {selected_theme.filename}")
        
        return selected_theme
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from URL: {e}")
        return MusicTheme.CALM_AND_SERENITY
    except Exception as e:
        print(f"Error analyzing image sentiment: {e}")
        return MusicTheme.CALM_AND_SERENITY


def analyze_image_sentiment_with_vision(photo_path, cohere_api_key=None):
    """
    Alternative function using Cohere's vision capabilities (if available).
    
    Args:
        photo_path (str): URL or local file path of the image to analyze
        cohere_api_key (str, optional): Cohere API key
        
    Returns:
        MusicTheme: The music theme that best matches the image's emotional content
    """
    try:
        if not cohere_api_key:
            cohere_api_key = os.getenv('COHERE_API_KEY')
            
        if not cohere_api_key:
            raise ValueError("Cohere API key not provided.")
        
        co = cohere.Client(cohere_api_key)
        
        if photo_path.startswith(('http://', 'https://')):
            print(f"Downloading image from URL: {photo_path}")
            response = requests.get(photo_path, timeout=30)
            response.raise_for_status()
            image_data = response.content
        else:
            from pathlib import Path
            file_path = Path(photo_path)
            if not file_path.is_absolute():
                file_path = Path.cwd() / file_path
            
            print(f"Loading local image: {file_path}")
            if not file_path.exists():
                raise FileNotFoundError(f"Image file not found: {file_path}")
            
            with open(file_path, 'rb') as f:
                image_data = f.read()
        
        # Convert to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Create message for vision model
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze this image and determine its emotional sentiment. Choose ONE of these categories:
                    
AWE_AND_WONDER, JOY_AND_HAPPINESS, MELANCHOLY_AND_SADNESS, EMPATHY_AND_COMPASSION, 
CALM_AND_SERENITY, FEAR_AND_UNEASE, INSPIRATION_AND_CREATIVITY, LOVE_AND_WARMTH, 
CONFUSION_AND_CURIOSITY, SPIRITUALITY_AND_TRANSCENDENCE

Respond with only the category name."""
                },
                {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{image_base64}"
                }
            ]
        }
        
        response = co.chat(
            model='command-r-plus',
            message=message,
            temperature=0.3,
            max_tokens=50
        )
        
        predicted_sentiment = response.text.strip().upper()
        
        sentiment_mapping = {
            'AWE_AND_WONDER': MusicTheme.AWE_AND_WONDER,
            'JOY_AND_HAPPINESS': MusicTheme.JOY_AND_HAPPINESS,
            'MELANCHOLY_AND_SADNESS': MusicTheme.MELANCHOLY_AND_SADNESS,
            'EMPATHY_AND_COMPASSION': MusicTheme.EMPATHY_AND_COMPASSION,
            'CALM_AND_SERENITY': MusicTheme.CALM_AND_SERENITY,
            'FEAR_AND_UNEASE': MusicTheme.FEAR_AND_UNEASE,
            'INSPIRATION_AND_CREATIVITY': MusicTheme.INSPIRATION_AND_CREATIVITY,
            'LOVE_AND_WARMTH': MusicTheme.LOVE_AND_WARMTH,
            'CONFUSION_AND_CURIOSITY': MusicTheme.CONFUSION_AND_CURIOSITY,
            'SPIRITUALITY_AND_TRANSCENDENCE': MusicTheme.SPIRITUALITY_AND_TRANSCENDENCE
        }
        
        selected_theme = sentiment_mapping.get(predicted_sentiment, MusicTheme.CALM_AND_SERENITY)
        
        print(f"Image sentiment analysis complete!")
        print(f"Detected sentiment: {predicted_sentiment}")
        print(f"Selected music theme: {selected_theme.theme_name}")
        
        return selected_theme
        
    except Exception as e:
        print(f"Vision analysis failed, falling back to text analysis: {e}")
        return analyze_image_sentiment_from_url(photo_path, cohere_api_key)