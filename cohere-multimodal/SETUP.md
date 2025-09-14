# Setup Guide for Cohere Multimodal Painting Analysis

## Quick Demo (No API Key Required)

To see the system in action without setting up API keys:

```bash
cd cohere-multimodal
python demo_without_api.py
```

This will show you:
- Mask region analysis results
- Generated object-color pairs using fallback methods
- Sample output format

## Full Setup with Cohere API

### 1. Get a Cohere API Key

1. Visit [https://cohere.ai/](https://cohere.ai/)
2. Sign up for an account
3. Navigate to the API section
4. Generate an API key

### 2. Configure the API Key

**Option A: Environment Variable**
```bash
export COHERE_API_KEY="your_api_key_here"
```

**Option B: .env File**
Create a `.env` file in the project root:
```
COHERE_API_KEY=your_api_key_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Test the Full System

```bash
# Test with a specific image
python painting_mask_analyzer.py ../painting_vectorization/examples/monet-altered.jpg

# Run comprehensive tests
python test_analyzer.py
```

## Usage Examples

### Command Line
```bash
# Basic usage with latest mask
python painting_mask_analyzer.py path/to/painting.jpg

# Specify custom mask directory
python painting_mask_analyzer.py path/to/painting.jpg path/to/masks/
```

### Python API
```python
from painting_mask_analyzer import process_painting_to_bob_ross

# Analyze and get Bob Ross descriptions
pairs, descriptions = process_painting_to_bob_ross(
    "painting.jpg",
    mask_directory="masks/"  # optional
)

print("Object-Color Pairs:", pairs)
for desc in descriptions:
    print(f"Bob Ross says: {desc}")
```

## Expected Output

The system will generate:

1. **Object-Color Pairs**: `[("sky", "blue"), ("trees", "green"), ...]`
2. **Bob Ross Descriptions**: 
   - "Now here we have a beautiful blue sky. Just like that, we've made something special."
   - "Let's paint a lovely green trees. Isn't that just delightful?"

## Troubleshooting

### API Key Issues
- **Error**: `invalid api token`
- **Solution**: Check that your API key is correctly set and valid

### Missing Mask Files
- **Error**: `No mask files found`
- **Solution**: Run the painting vectorization pipeline first to generate mask files

### Import Errors
- **Error**: `ModuleNotFoundError: No module named 'cohere'`
- **Solution**: Run `pip install -r requirements.txt`

## Integration with Other Components

This module integrates with:
- `../text_to_speech/bob_ross_cohere.py` - For generating Bob Ross descriptions
- `../painting_vectorization/` - For input images and masks
- The broader HTN2025 painting analysis pipeline

## Future Enhancements

- True multimodal analysis with image input to Cohere
- Better object recognition based on painting style
- Color accuracy improvements
- Integration with other painting analysis steps
