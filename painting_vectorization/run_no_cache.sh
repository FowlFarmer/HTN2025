#!/bin/bash
# Run Python with cache disabled

# Disable Python bytecode generation
export PYTHONDONTWRITEBYTECODE=1

# Clear existing cache
python3 clear_cache.py

# Run the pipeline
python3 process_image.py "$@"
