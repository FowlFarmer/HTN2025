#!/usr/bin/env python3
"""
Quick launcher for the brush simulator with the latest stroke results
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from brush_simulator import main, find_latest_stroke_results

def launch_with_latest():
    """Launch simulator with the specific stroke results file"""
    # Use the exact file you specified
    stroke_file = "painting_vectorization/results/step8_stroke_ordering/stroke_ordering_results_20250913_215941.json"
    
    # Check if file exists
    if not Path(stroke_file).exists():
        print(f"❌ Stroke file not found: {stroke_file}")
        # Try to find latest as fallback
        latest = find_latest_stroke_results()
        if latest:
            print(f"🔍 Using latest results instead: {latest}")
            stroke_file = latest
        else:
            print("❌ No stroke results found!")
            return
    
    print(f"🎨 Loading stroke results: {stroke_file}")
    
    # Import and run simulator
    from brush_simulator import BrushSimulator
    
    simulator = BrushSimulator(window_size=(1200, 900))
    simulator.brush_speed_mm_s = 40.0  # Start at reasonable speed
    
    try:
        simulator.run(stroke_file)
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted")
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🤖 Robotic Brush Simulator Launcher")
    print("=" * 50)
    launch_with_latest()

