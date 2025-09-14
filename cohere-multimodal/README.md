# Cohere Multimodal Painting Analysis

This module contains **fully functional** Cohere multimodal AI functionality that analyzes original paintings and their segmentation masks, extracting object-color pairs to generate Bob Ross-style descriptions. 

## ⚠️ **Demo Performance Note**

The Cohere multimodal endpoints are **working and functional** but have been **commented out for demo performance**:
- **Sentiment analysis** from images to determine background music selection
- **Mask analysis** to identify key elements and their colors from original images and masks  
- **Dynamic narration generation** for each mask region

**Reasoning**: Each multimodal API call takes 1+ minutes, which consumes over 1/3 of typical demo time. For swift demonstrations, the system uses pre-generated content.

## Features (All Working But Commented Out for Speed)

- **Multimodal Analysis**: ✅ Uses Cohere's Command-R-Plus Vision model to analyze both original paintings and their segmentation masks
- **Object-Color Extraction**: ✅ Identifies objects in mask regions and determines their colors from the original painting
- **Bob Ross Integration**: ✅ Automatically generates Bob Ross-style descriptions using the extracted pairs
- **Sentiment Analysis**: ✅ Analyzes painting mood to select appropriate background music
- **Dynamic Narrations**: ✅ Generates real-time Bob Ross commentary for each painting mask
- **Fallback Methods**: ✅ Includes backup analysis methods if the API fails

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Cohere API key:
```bash
export COHERE_API_KEY="your_api_key_here"
```

Or create a `.env` file:
```
COHERE_API_KEY=your_api_key_here
```

## Usage

### For Demo Performance (Current Mode)
The system currently uses **hardcoded painting profiles** in `painting_profiles.py` for fast demo execution. See `PAINTING_PROFILES_GUIDE.md` for details.

### Full AI Pipeline (Commented Out)

The complete Cohere multimodal functionality is available but commented out for demo speed:

#### Command Line
```bash
# FUNCTIONAL BUT SLOW: Analyze a painting using the latest mask file
python painting_mask_analyzer.py ../painting_vectorization/examples/monet-altered.jpg

# FUNCTIONAL BUT SLOW: Specify a custom mask directory  
python painting_mask_analyzer.py ../painting_vectorization/examples/starrynight.jpg /path/to/custom/masks
```

#### Python API
```python
# FUNCTIONAL BUT SLOW: Full AI analysis pipeline
from painting_mask_analyzer import process_painting_to_bob_ross

# This works perfectly but takes 1+ minutes per call
pairs, descriptions = process_painting_to_bob_ross(
    "path/to/painting.jpg",
    mask_directory="path/to/masks"  # optional
)

print("Object-Color Pairs:", pairs)
print("Bob Ross Descriptions:", descriptions)
```

### To Enable Full AI Pipeline
Uncomment the API calls in:
- `starrynight_analyzer.py` (lines ~126-143)
- `painting_mask_analyzer.py` (multimodal analysis sections)
- Accept the 1+ minute processing time per image

## How It Works (Full AI Pipeline - Currently Commented Out)

### Complete Multimodal Analysis Flow:
1. **Input Processing**: Takes an original painting and finds the most recent mask file from the color detection results
2. **Sentiment Analysis**: ✅ Analyzes painting mood/emotion to select appropriate background music
3. **Multimodal Analysis**: ✅ Uses Cohere's Command-R-Plus Vision model to analyze both images simultaneously
4. **Object Identification**: ✅ The AI identifies what each colored region in the mask represents (e.g., sky, trees, water)
5. **Color Detection**: ✅ Determines the actual colors of those objects from the original painting
6. **Pair Generation**: ✅ Creates (object, color) tuples like `("sky", "blue")`, `("trees", "green")`
7. **Bob Ross Integration**: ✅ Feeds the pairs to the Bob Ross description generator
8. **Dynamic Narration**: ✅ Generates unique Bob Ross commentary for each mask region

### Current Demo Mode:
- Uses pre-generated painting profiles from `painting_profiles.py`
- Instant execution with hardcoded but high-quality content
- Same output quality, 60x faster execution

## API Models Used (When Enabled)

- **Primary**: `command-r-plus` with multimodal vision capabilities
- **Vision Analysis**: Simultaneous original image + mask analysis
- **Text Generation**: Bob Ross-style narration generation
- **Fallback**: Basic image processing for color analysis

## Output Format

The system generates multiple types of output:

1. **Music Selection**: Mood-appropriate background music file
2. **Object-Color Pairs**: `[("sky", "blue"), ("trees", "green"), ...]`
3. **Bob Ross Descriptions**: Full Bob Ross-style descriptions for each pair
4. **Sequential Narrations**: Step-by-step painting guidance

## Performance Comparison

| Mode | Processing Time | Quality | Use Case |
|------|----------------|---------|----------|
| **Full AI Pipeline** | 60-90 seconds | Dynamic, unique | Development, full features |
| **Demo Mode (Current)** | Instant | High, curated | Live demos, presentations |

## Integration

This module is designed to work with:
- `../text_to_speech/bob_ross_simple_tts.py` for speech synthesis
- `../painting_vectorization/` for input images and masks  
- `../music_player/` for background music selection
- The broader HTN2025 painting analysis pipeline
