# Quick Start Guide - Brush Simulator

## 🚀 **Run the Simulation**

### **Option 1: Auto-Launch (Recommended)**
```bash
cd /Users/jonathan/Desktop/HTN2025/simulation
python run_simulation.py
```

### **Option 2: Direct Launch**
```bash
cd /Users/jonathan/Desktop/HTN2025/simulation
python brush_simulator.py ../painting_vectorization/results/step8_stroke_ordering/stroke_ordering_results_20250913_215941.json
```

### **Option 3: Auto-Find Latest**
```bash
cd /Users/jonathan/Desktop/HTN2025/simulation
python brush_simulator.py
```

## 🎮 **Controls**

### **Mouse Controls**
- **Play** - Start/pause the simulation
- **Reset** - Return to starting position  
- **Step** - Execute one stroke at a time
- **Speed+/Speed-** - Adjust brush speed (5-200 mm/s)

### **Keyboard Shortcuts**
- **SPACE** - Play/Pause
- **R** - Reset
- **S** - Step forward
- **UP/DOWN** - Speed control

## 📊 **What You'll See**

### **Your Monet Data:**
- **1,495 total strokes** across **5 masks**
- **6,217mm travel distance** with **960 pen lifts**
- **160mm × 160mm canvas** (matches real robot)
- **All cool colors** (blue strokes)

### **Visual Elements:**
- **🔵 Blue lines** - Painted strokes (all cool colors in your data)
- **⚪ Gray lines** - Travel paths (pen up)
- **🟢 Green circle** - Brush down (painting)
- **🔴 Red circle** - Brush up (traveling)

## ⚡ **Performance Tips**

- **Start slow** - Begin at 20-40 mm/s to see details
- **Speed up** - Use 100+ mm/s for full overview
- **Step mode** - Use 'S' key to see individual strokes
- **Real-time stats** - Watch position and progress

## 🎯 **Perfect for Hardware Prep**

This simulation shows **exactly** what your robot will do:
- **Precise coordinates** - Same coordinate system as robot
- **Actual pen lifts** - See where robot lifts/lowers pen
- **Travel optimization** - Visualize the stroke ordering
- **Speed testing** - Find optimal painting speed

---

**Ready to see your Monet painting come to life!** 🎨

The simulation will show all 1,495 strokes painting the masterpiece stroke by stroke, just like the real robot will do.

