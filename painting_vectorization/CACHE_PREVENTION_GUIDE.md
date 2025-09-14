# 🚫 Cache Prevention Guide

## The Problem We Solved
Python caches bytecode in `__pycache__` directories and `.pyc` files. When you edit source files, sometimes the old cached version continues to run, causing confusion and bugs.

## Prevention Strategies

### 1. 🧹 Manual Cache Clearing
```bash
# Quick one-liner
find . -name "*.pyc" -delete && find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Or use our script
python clear_cache.py
```

### 2. 🛡️ Run with Cache Disabled
```bash
# Use our no-cache runner
./run_no_cache.sh examples/monet.jpeg

# Or manually set environment variable
export PYTHONDONTWRITEBYTECODE=1
python process_image.py examples/monet.jpeg
```

### 3. 🔍 Verify Files Before Running
```bash
# Check if files are as expected
python verify_files.py

# Should show:
# ✅ step1_preprocessing/image_preprocessing.py: 58 lines, clean
# ✅ step2_segmentation/segmentation.py: 237 lines, clean
```

### 4. 🔄 Development Workflow
```bash
# Recommended workflow when making changes:
1. python clear_cache.py          # Clear old cache
2. python verify_files.py         # Verify files are correct
3. ./run_no_cache.sh examples/monet.jpeg  # Run without caching
```

## 🚨 Warning Signs of Cache Issues

- **Unexpected behavior**: Code changes don't take effect
- **Old print statements**: Seeing output from deleted code
- **Import errors**: Functions that should exist are "not found"
- **Wrong line counts**: `wc -l file.py` shows different count than expected

## 🛠️ IDE-Specific Solutions

### VS Code / Cursor
- Restart the Python interpreter: `Cmd+Shift+P` → "Python: Restart Language Server"
- Clear workspace cache: Close and reopen the workspace

### PyCharm
- File → Invalidate Caches and Restart

### Terminal/Command Line
- Always use `./run_no_cache.sh` for critical runs
- Set `PYTHONDONTWRITEBYTECODE=1` in your shell profile

## 📋 Quick Reference Commands

```bash
# Clear cache
python clear_cache.py

# Verify files
python verify_files.py

# Run without cache
./run_no_cache.sh examples/monet.jpeg

# Check file integrity
wc -l step1_preprocessing/image_preprocessing.py  # Should be 58
grep -c "quantize_colors" step1_preprocessing/image_preprocessing.py  # Should be 0
```

## 🎯 Best Practices

1. **Always verify after major changes**: Run `verify_files.py`
2. **Use no-cache mode for testing**: Use `run_no_cache.sh`
3. **Clear cache before commits**: Run `clear_cache.py`
4. **Check line counts**: Verify files have expected size
5. **Watch for forbidden content**: Use grep to check for unwanted code

This prevents the "my changes aren't working" mystery we just solved!
