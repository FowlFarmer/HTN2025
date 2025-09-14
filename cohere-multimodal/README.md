# Cohere Multimodal Painting Analysis

This module uses Cohere's multimodal AI models to analyze original paintings and their segmentation masks, extracting object-color pairs that can be used to generate Bob Ross-style descriptions.

## Features

- **Multimodal Analysis**: Uses Cohere's Command-A Vision model to analyze both original paintings and their segmentation masks
- **Object-Color Extraction**: Identifies objects in mask regions and determines their colors from the original painting
- **Bob Ross Integration**: Automatically generates Bob Ross-style descriptions using the extracted pairs
- **Fallback Methods**: Includes backup analysis methods if the API fails

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

### Command Line

```bash
# Analyze a painting using the latest mask file
python painting_mask_analyzer.py ../painting_vectorization/examples/monet-altered.jpg

# Specify a custom mask directory
python painting_mask_analyzer.py ../painting_vectorization/examples/starrynight.jpg /path/to/custom/masks
```

### Python API

```python
from painting_mask_analyzer import process_painting_to_bob_ross

# Analyze painting and get Bob Ross descriptions
pairs, descriptions = process_painting_to_bob_ross(
    "path/to/painting.jpg",
    mask_directory="path/to/masks"  # optional
)

print("Object-Color Pairs:", pairs)
print("Bob Ross Descriptions:", descriptions)
```

## How It Works

1. **Input Processing**: Takes an original painting and finds the most recent mask file from the color detection results
2. **Multimodal Analysis**: Uses Cohere's Command-A Vision model to analyze both images simultaneously
3. **Object Identification**: The AI identifies what each colored region in the mask represents (e.g., sky, trees, water)
4. **Color Detection**: Determines the actual colors of those objects from the original painting
5. **Pair Generation**: Creates (object, color) tuples like `("sky", "blue")`, `("trees", "green")`
6. **Bob Ross Integration**: Feeds the pairs to the Bob Ross description generator

## API Models Used

- **Primary**: `command-r-plus` with multimodal capabilities
- **Fallback**: Basic image processing for color analysis

## Output Format

The system generates two types of output:

1. **Object-Color Pairs**: `[("sky", "blue"), ("trees", "green"), ...]`
2. **Bob Ross Descriptions**: Full Bob Ross-style descriptions for each pair

## Error Handling

- Graceful fallback to basic image processing if API fails
- Default object-color pairs if analysis fails completely
- Comprehensive error messages for debugging

## Integration

This module is designed to work with:
- `../text_to_speech/bob_ross_cohere.py` for generating descriptions
- `../painting_vectorization/` for input images and masks
- The broader HTN2025 painting analysis pipeline
