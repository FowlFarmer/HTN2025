# Robotic Brush Movement Simulator

A real-time 2D bird's eye view simulation of robotic brush movement based on stroke ordering data from the painting vectorization pipeline.

## Features

### 🎨 **Visual Simulation**
- **Real-time brush movement** - Watch the brush move exactly as the robot would
- **Pen up/down visualization** - Green circle when drawing, red when traveling
- **Stroke color coding** - Warm strokes (red) vs cool strokes (blue)
- **Travel path display** - Gray lines show pen-up movements
- **Canvas coordinate system** - Matches real robot (160mm × 160mm)

### 🎮 **Interactive Controls**
- **Play/Pause** - Start/stop the simulation
- **Step Forward** - Execute one stroke at a time
- **Reset** - Return to starting position
- **Speed Control** - Adjust brush speed from 5-200 mm/s
- **Real-time feedback** - See position, speed, and progress

### 📊 **Real-World Accuracy**
- **Coordinate system** - Exact match to robot (0,0) at top-left
- **Canvas dimensions** - True 160mm × 160mm scaling
- **Speed simulation** - Realistic movement timing
- **Pen lift behavior** - Accurate travel vs drawing distinction

## Quick Start

### 1. **Auto-Run with Latest Results**
```bash
cd simulation
python brush_simulator.py
```
*Automatically finds and loads the most recent stroke ordering results*

### 2. **Load Specific File**
```bash
python brush_simulator.py ../painting_vectorization/results/step8_stroke_ordering/stroke_ordering_results_20250913_215941.json
```

### 3. **Custom Speed**
```bash
python brush_simulator.py --speed 60  # Start at 60 mm/s
```

## Controls

### 🖱️ **Mouse Controls**
- **Play Button** - Start/pause simulation
- **Reset Button** - Return to beginning
- **Step Button** - Execute next stroke
- **Speed+/Speed-** - Adjust brush speed

### ⌨️ **Keyboard Shortcuts**
- **SPACE** - Play/Pause
- **R** - Reset simulation
- **S** - Step forward one stroke
- **UP/DOWN** - Increase/decrease speed

## Display Elements

### 🎯 **Brush Indicator**
- **🟢 Green Circle** - Brush is down (drawing)
- **🔴 Red Circle** - Brush is up (traveling)
- **Yellow Outline** - Current brush position

### 🎨 **Stroke Colors**
- **🔴 Red Lines** - Warm color strokes
- **🔵 Blue Lines** - Cool color strokes  
- **⚪ Gray Lines** - Travel paths (pen up)

### 📊 **Status Display**
- **Brush Speed** - Current speed in mm/s
- **Position** - Real-time coordinates in mm
- **Progress** - Current stroke / total strokes
- **Pen State** - UP or DOWN
- **Simulation State** - PLAYING or PAUSED

## Technical Details

### 📐 **Coordinate System**
- **Origin (0,0)** - Top-left corner of canvas
- **X-axis** - Increases rightward
- **Y-axis** - Increases downward
- **Units** - Millimeters (matches robot)
- **Canvas Size** - 160mm × 160mm (configurable in constants.py)

### 🚀 **Performance**
- **60 FPS** - Smooth animation
- **Speed Multiplier** - 10x simulation speed by default
- **Real-time Updates** - Position, progress, and timing
- **Memory Efficient** - Streams stroke data progressively

### 🔧 **Configuration**
All robot parameters are imported from `painting_vectorization/constants.py`:
- Canvas dimensions
- Robot speeds
- Coordinate system settings

## Use Cases

### 🔍 **Development & Testing**
- **Visualize stroke ordering** - See the execution sequence
- **Debug travel optimization** - Spot inefficient movements
- **Test different speeds** - Find optimal painting speed
- **Validate coordinates** - Ensure proper scaling

### 🎨 **Demo & Presentation**
- **Show painting process** - Real-time visualization
- **Explain robot behavior** - Step-by-step execution
- **Compare strategies** - Different optimization results
- **Time estimation** - See actual painting duration

### 🛠️ **Hardware Preparation**
- **Path verification** - Ensure movements are correct
- **Speed calibration** - Test different speeds
- **Collision detection** - Visual safety checks
- **Timing analysis** - Optimize for 2-minute demo

## File Structure

```
simulation/
├── brush_simulator.py    # Main simulator application
├── README.md            # This documentation
└── requirements.txt     # Python dependencies (if needed)
```

## Dependencies

The simulator uses standard Python libraries:
- **pygame** - Graphics and user interface
- **json** - Loading stroke data
- **pathlib** - File operations
- **time** - Animation timing

Install pygame if needed:
```bash
pip install pygame
```

## Integration

### 📁 **Input Data**
Loads JSON files from:
```
painting_vectorization/results/step8_stroke_ordering/
```

Expected format:
```json
{
  "mask_stroke_arrays": [
    {
      "mask_id": 0,
      "strokes": [
        {
          "stroke_id": 1,
          "points": [[x1, y1], [x2, y2], ...],
          "warmth_name": "cool",
          "length_mm": 14.5
        }
      ]
    }
  ],
  "statistics": {
    "total_strokes": 1495,
    "total_length_mm": 6545.0
  }
}
```

### 🔗 **Configuration Import**
Automatically imports settings from:
```python
from painting_vectorization.constants import CANVAS_WIDTH_MM, CANVAS_HEIGHT_MM, ROBOT_CONFIG
```

This ensures the simulation exactly matches the real robot configuration.

## Future Enhancements

### 🎯 **Planned Features**
- **Brush pressure simulation** - Variable line thickness
- **Paint depletion** - Show brush running out of paint
- **Multi-color brushes** - Different colors for different masks
- **3D perspective** - Isometric view option
- **Export animation** - Save as video file
- **Performance metrics** - Detailed timing analysis

### 🔧 **Advanced Controls**
- **Scrub timeline** - Jump to any point in time
- **Variable speed** - Dynamic speed changes
- **Zoom/pan** - Detailed view of specific areas
- **Layer visualization** - Show mask boundaries

---

**Ready to simulate your robotic painting!** 🎨🤖

Run `python brush_simulator.py` to see your stroke data come to life!

