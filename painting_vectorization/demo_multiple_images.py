#!/usr/bin/env python3
"""
Demo: Process Multiple Images

This script demonstrates how to process multiple images and compare results.
"""

import sys
from pathlib import Path
import subprocess
import time

def run_command(cmd):
    """Run a command and capture output"""
    print(f"🔄 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"⚠️  Warnings: {result.stderr}")
    return result.returncode == 0

def main():
    print("🎨 Multi-Image Processing Demo")
    print("=" * 50)

    # List available images
    print("📂 Step 1: List available images")
    run_command("python process_image.py --list")
    print()

    # Process multiple images
    test_images = ["monet.jpeg", "starry_night.jpg"]

    for i, image in enumerate(test_images, 1):
        print(f"🖼️  Step {i+1}: Processing {image}")
        print("-" * 30)

        start_time = time.time()
        success = run_command(f"python process_image.py {image}")
        end_time = time.time()

        if success:
            print(f"✅ Completed in {end_time - start_time:.1f} seconds")
        else:
            print(f"❌ Failed to process {image}")
        print()

    # Show all results
    print("📊 Step 4: View all results")
    run_command("python process_image.py --show-results")

    print()
    print("🎉 Demo complete!")
    print("💡 Check the examples/ directory for all generated files")

    # List final results
    print("\n📁 Generated files:")
    examples_dir = Path("examples")
    for file_path in sorted(examples_dir.glob("*results*.png")):
        file_size = file_path.stat().st_size / (1024 * 1024)
        print(f"   📄 {file_path.name} ({file_size:.1f} MB)")

if __name__ == "__main__":
    main()