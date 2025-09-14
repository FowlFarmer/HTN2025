#!/usr/bin/env python3
"""
Verify that source files match expected content
"""
import hashlib
from pathlib import Path

def get_file_hash(filepath):
    """Get SHA256 hash of a file"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        return None

def verify_key_files():
    """Verify key files haven't been corrupted by caching"""
    
    key_files = {
        "step1_preprocessing/image_preprocessing.py": {
            "expected_lines": 58,
            "should_contain": ["def preprocess_image(image_path):"],
            "should_not_contain": ["quantize_colors", "sklearn", "noise reduction"]
        },
        "step2_segmentation/segmentation.py": {
            "expected_lines": 237,
            "should_contain": ["def segment_painting("],
            "should_not_contain": []
        }
    }
    
    print("🔍 Verifying key files...")
    
    for filepath, checks in key_files.items():
        file_path = Path(filepath)
        
        if not file_path.exists():
            print(f"❌ {filepath}: File not found")
            continue
            
        # Check line count
        with open(file_path, 'r') as f:
            lines = f.readlines()
            line_count = len(lines)
            content = ''.join(lines)
        
        print(f"\n📄 {filepath}:")
        print(f"   Lines: {line_count} (expected: {checks['expected_lines']})")
        
        if line_count != checks['expected_lines']:
            print(f"   ⚠️  Line count mismatch!")
        
        # Check required content
        for required in checks['should_contain']:
            if required in content:
                print(f"   ✅ Contains: {required}")
            else:
                print(f"   ❌ Missing: {required}")
        
        # Check forbidden content
        for forbidden in checks['should_not_contain']:
            if forbidden in content:
                print(f"   ❌ Contains forbidden: {forbidden}")
            else:
                print(f"   ✅ Clean of: {forbidden}")
        
        # Show file hash for tracking changes
        file_hash = get_file_hash(file_path)
        print(f"   🔑 Hash: {file_hash[:16]}...")

if __name__ == "__main__":
    verify_key_files()
