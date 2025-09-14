#!/usr/bin/env python3
"""
Clear Python cache files to prevent stale code execution
"""
import os
import shutil
from pathlib import Path

def clear_python_cache(root_dir="."):
    """Clear all Python cache files and directories"""
    root_path = Path(root_dir)
    
    # Find and remove __pycache__ directories
    pycache_dirs = list(root_path.rglob("__pycache__"))
    for cache_dir in pycache_dirs:
        print(f"Removing: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)
    
    # Find and remove .pyc files
    pyc_files = list(root_path.rglob("*.pyc"))
    for pyc_file in pyc_files:
        print(f"Removing: {pyc_file}")
        pyc_file.unlink(missing_ok=True)
    
    # Find and remove .pyo files (optimized bytecode)
    pyo_files = list(root_path.rglob("*.pyo"))
    for pyo_file in pyo_files:
        print(f"Removing: {pyo_file}")
        pyo_file.unlink(missing_ok=True)
    
    print("✅ Python cache cleared!")

if __name__ == "__main__":
    clear_python_cache()
