#!/usr/bin/env python3
"""
Brush Movement Simulator - 2D Bird's Eye View

A real-time simulation of robotic brush movement based on stroke ordering data.
Shows exactly what the robot would paint from a top-down perspective.

Features:
- Real-time brush movement animation
- Adjustable speed controls
- Pen up/down visualization
- Travel path vs drawing path distinction
- Canvas coordinate system matching real robot
- Stroke-by-stroke playback controls
"""

import pygame
import json
import time
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Add project root to path for constants
sys.path.append(str(Path(__file__).parent.parent))
from painting_vectorization.constants import CANVAS_WIDTH_MM, CANVAS_HEIGHT_MM, ROBOT_CONFIG

class BrushSimulator:
    """
    Real-time brush movement simulator with bird's eye view
    """
    
    def __init__(self, window_size: Tuple[int, int] = (1000, 800)):
        """
        Initialize the brush simulator
        
        Args:
            window_size: (width, height) of the simulation window
        """
        pygame.init()
        
        # Window setup
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Robotic Brush Simulator - Bird's Eye View")
        
        # Canvas setup (in pixels)
        self.canvas_padding = 50
        self.canvas_pixel_width = min(window_size[0] - 2 * self.canvas_padding, 
                                    window_size[1] - 200)  # Leave space for controls
        self.canvas_pixel_height = self.canvas_pixel_width  # Square canvas
        
        # Canvas position on screen
        self.canvas_x = (window_size[0] - self.canvas_pixel_width) // 2
        self.canvas_y = self.canvas_padding
        
        # Scale factor: mm to pixels
        self.mm_to_pixel = self.canvas_pixel_width / CANVAS_WIDTH_MM
        
        # Colors
        self.colors = {
            'background': (240, 240, 240),
            'canvas': (255, 255, 255),
            'canvas_border': (100, 100, 100),
            'brush_down': (50, 150, 50),      # Green when drawing
            'brush_up': (200, 50, 50),        # Red when traveling
            'stroke_warm': (255, 100, 100),   # Warm strokes
            'stroke_cool': (100, 100, 255),   # Cool strokes
            'travel_path': (150, 150, 150),   # Travel lines (pen up)
            'current_pos': (255, 200, 0),     # Current brush position
            'ui_text': (50, 50, 50),
            'ui_bg': (220, 220, 220),
            'button': (180, 180, 180),
            'button_hover': (160, 160, 160)
        }
        
        # Simulation state
        self.stroke_data = None
        self.current_stroke_idx = 0
        self.current_point_idx = 0
        self.is_playing = False
        self.is_pen_down = False
        self.brush_speed_mm_s = 40.0  # Default speed
        self.simulation_speed_multiplier = 10.0  # Speed up simulation
        
        # Current brush position (in mm)
        self.brush_pos_mm = [CANVAS_WIDTH_MM / 2, CANVAS_HEIGHT_MM / 2]
        
        # Drawing surface for permanent strokes
        self.drawing_surface = pygame.Surface((self.canvas_pixel_width, self.canvas_pixel_height))
        self.drawing_surface.fill(self.colors['canvas'])
        
        # Font for UI
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # UI elements
        self.ui_y = self.canvas_y + self.canvas_pixel_height + 20
        self.buttons = self._create_buttons()
        
        # Animation timing
        self.last_update_time = time.time()
        self.stroke_start_time = None
        
        # Statistics
        self.total_strokes = 0
        self.completed_strokes = 0
        self.total_drawing_time = 0
        
    def _create_buttons(self) -> List[Dict]:
        """Create UI buttons"""
        button_width = 80
        button_height = 30
        button_spacing = 10
        start_x = 20
        
        buttons = [
            {
                'name': 'play_pause',
                'rect': pygame.Rect(start_x, self.ui_y, button_width, button_height),
                'text': 'Play',
                'action': self._toggle_play_pause
            },
            {
                'name': 'reset',
                'rect': pygame.Rect(start_x + (button_width + button_spacing), self.ui_y, button_width, button_height),
                'text': 'Reset',
                'action': self._reset_simulation
            },
            {
                'name': 'step',
                'rect': pygame.Rect(start_x + 2 * (button_width + button_spacing), self.ui_y, button_width, button_height),
                'text': 'Step',
                'action': self._step_forward
            },
            {
                'name': 'speed_up',
                'rect': pygame.Rect(start_x + 3 * (button_width + button_spacing), self.ui_y, 60, button_height),
                'text': 'Speed+',
                'action': self._increase_speed
            },
            {
                'name': 'speed_down',
                'rect': pygame.Rect(start_x + 3 * (button_width + button_spacing) + 70, self.ui_y, 60, button_height),
                'text': 'Speed-',
                'action': self._decrease_speed
            }
        ]
        
        return buttons
    
    def load_stroke_data(self, json_path: str) -> bool:
        """
        Load stroke ordering results from JSON file
        
        Args:
            json_path: Path to stroke ordering JSON file
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract mask stroke arrays
            if 'mask_stroke_arrays' not in data:
                print(f"❌ No mask_stroke_arrays found in {json_path}")
                return False
            
            # Flatten all strokes from all masks into execution order
            all_strokes = []
            for mask_array in data['mask_stroke_arrays']:
                all_strokes.extend(mask_array['strokes'])
            
            self.stroke_data = all_strokes
            self.total_strokes = len(all_strokes)
            
            # Extract statistics
            if 'statistics' in data:
                stats = data['statistics']
                print(f"📊 Loaded simulation data:")
                print(f"   • Total strokes: {stats.get('total_strokes', 0)}")
                print(f"   • Total masks: {stats.get('total_masks', 0)}")
                print(f"   • Drawing length: {stats.get('total_length_mm', 0):.1f}mm")
                print(f"   • Travel distance: {stats.get('estimated_travel_distance_mm', 0):.1f}mm")
                print(f"   • Pen lifts: {stats.get('total_pen_lifts', 0)}")
            
            self._reset_simulation()
            return True
            
        except Exception as e:
            print(f"❌ Error loading stroke data: {e}")
            return False
    
    def mm_to_screen(self, pos_mm: List[float]) -> Tuple[int, int]:
        """Convert millimeter coordinates to screen pixels"""
        # Note: Canvas origin (0,0) is at top-left as per constants.py
        screen_x = int(self.canvas_x + pos_mm[0] * self.mm_to_pixel)
        screen_y = int(self.canvas_y + pos_mm[1] * self.mm_to_pixel)
        return (screen_x, screen_y)
    
    def _toggle_play_pause(self):
        """Toggle play/pause state"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.stroke_start_time = time.time()
    
    def _reset_simulation(self):
        """Reset simulation to beginning"""
        self.current_stroke_idx = 0
        self.current_point_idx = 0
        self.is_playing = False
        self.is_pen_down = False
        self.brush_pos_mm = [CANVAS_WIDTH_MM / 2, CANVAS_HEIGHT_MM / 2]
        self.completed_strokes = 0
        self.total_drawing_time = 0
        
        # Clear drawing surface
        self.drawing_surface.fill(self.colors['canvas'])
        
        print("🔄 Simulation reset")
    
    def _step_forward(self):
        """Step forward by one stroke"""
        if self.stroke_data and self.current_stroke_idx < len(self.stroke_data):
            self._execute_next_stroke()
    
    def _increase_speed(self):
        """Increase brush speed"""
        self.brush_speed_mm_s = min(self.brush_speed_mm_s * 1.5, 200.0)
        print(f"🚀 Brush speed: {self.brush_speed_mm_s:.1f}mm/s")
    
    def _decrease_speed(self):
        """Decrease brush speed"""
        self.brush_speed_mm_s = max(self.brush_speed_mm_s / 1.5, 5.0)
        print(f"🐌 Brush speed: {self.brush_speed_mm_s:.1f}mm/s")
    
    def _execute_next_stroke(self):
        """Execute the next stroke in the sequence"""
        if not self.stroke_data or self.current_stroke_idx >= len(self.stroke_data):
            return False
        
        stroke = self.stroke_data[self.current_stroke_idx]
        points = stroke['points']
        
        if self.current_point_idx == 0:
            # Move to start of stroke (pen up)
            start_pos = points[0]
            self._move_to_position(start_pos, pen_down=False)
            self.is_pen_down = True  # Put pen down for drawing
            
        # Draw stroke point by point
        if self.current_point_idx < len(points):
            current_point = points[self.current_point_idx]
            
            # Update brush position first
            self.brush_pos_mm = current_point.copy()
            
            # Draw line segment only if we have a previous point
            if self.current_point_idx > 0:
                prev_point = points[self.current_point_idx - 1]
                self._draw_stroke_segment(prev_point, current_point, stroke)
            
            self.current_point_idx += 1
            
            if self.current_point_idx >= len(points):
                # Stroke complete
                self.current_stroke_idx += 1
                self.current_point_idx = 0
                self.completed_strokes += 1
                self.is_pen_down = False  # Lift pen after stroke
                
                print(f"✅ Completed stroke {self.completed_strokes}/{self.total_strokes}")
                
        return True
    
    def _move_to_position(self, target_mm: List[float], pen_down: bool = False):
        """Move brush to target position"""
        if pen_down:
            # Draw travel line
            start_screen = self.mm_to_screen(self.brush_pos_mm)
            end_screen = self.mm_to_screen(target_mm)
            pygame.draw.line(self.drawing_surface, self.colors['travel_path'], 
                           start_screen, end_screen, 1)
        
        self.brush_pos_mm = target_mm.copy()
    
    def _draw_stroke_segment(self, start_mm: List[float], end_mm: List[float], stroke: Dict):
        """Draw a stroke segment on the permanent surface"""
        start_screen = self.mm_to_screen(start_mm)
        end_screen = self.mm_to_screen(end_mm)
        
        # Choose color based on warmth
        if stroke.get('warmth_name') == 'warm':
            color = self.colors['stroke_warm']
        else:
            color = self.colors['stroke_cool']
        
        # Draw with appropriate thickness (simulate brush width)
        pygame.draw.line(self.drawing_surface, color, start_screen, end_screen, 2)
    
    def update(self):
        """Update simulation state"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        if self.is_playing and self.stroke_data:
            # Calculate how much to advance based on speed
            pixels_per_second = self.brush_speed_mm_s * self.mm_to_pixel * self.simulation_speed_multiplier
            
            # For now, step through strokes at a reasonable pace
            if current_time - (self.stroke_start_time or current_time) > 0.1:  # 10 FPS max
                if not self._execute_next_stroke():
                    self.is_playing = False  # Stop when done
                self.stroke_start_time = current_time
    
    def draw(self):
        """Render the simulation"""
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw canvas border
        canvas_rect = pygame.Rect(self.canvas_x - 2, self.canvas_y - 2, 
                                self.canvas_pixel_width + 4, self.canvas_pixel_height + 4)
        pygame.draw.rect(self.screen, self.colors['canvas_border'], canvas_rect)
        
        # Draw canvas
        canvas_rect = pygame.Rect(self.canvas_x, self.canvas_y, 
                                self.canvas_pixel_width, self.canvas_pixel_height)
        pygame.draw.rect(self.screen, self.colors['canvas'], canvas_rect)
        
        # Draw permanent strokes
        self.screen.blit(self.drawing_surface, (self.canvas_x, self.canvas_y))
        
        # Draw current brush position
        brush_screen_pos = self.mm_to_screen(self.brush_pos_mm)
        brush_color = self.colors['brush_down'] if self.is_pen_down else self.colors['brush_up']
        pygame.draw.circle(self.screen, brush_color, brush_screen_pos, 8)
        pygame.draw.circle(self.screen, (0, 0, 0), brush_screen_pos, 8, 2)
        
        # Add a small white dot in the center to make position clearer
        pygame.draw.circle(self.screen, (255, 255, 255), brush_screen_pos, 2)
        
        # Draw UI
        self._draw_ui()
        
        pygame.display.flip()
    
    def _draw_ui(self):
        """Draw user interface elements"""
        ui_y = self.ui_y + 40
        
        # Draw buttons
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            # Button background
            is_hover = button['rect'].collidepoint(mouse_pos)
            button_color = self.colors['button_hover'] if is_hover else self.colors['button']
            pygame.draw.rect(self.screen, button_color, button['rect'])
            pygame.draw.rect(self.screen, (100, 100, 100), button['rect'], 2)
            
            # Button text
            if button['name'] == 'play_pause':
                text = 'Pause' if self.is_playing else 'Play'
            else:
                text = button['text']
            
            text_surface = self.small_font.render(text, True, self.colors['ui_text'])
            text_rect = text_surface.get_rect(center=button['rect'].center)
            self.screen.blit(text_surface, text_rect)
        
        # Status information
        current_stroke_info = ""
        if self.stroke_data and self.current_stroke_idx < len(self.stroke_data):
            stroke = self.stroke_data[self.current_stroke_idx]
            current_stroke_info = f"Current: S{stroke.get('stroke_id', '?')} P{self.current_point_idx}/{len(stroke.get('points', []))}"
        
        info_texts = [
            f"Brush Speed: {self.brush_speed_mm_s:.1f} mm/s",
            f"Position: ({self.brush_pos_mm[0]:.1f}, {self.brush_pos_mm[1]:.1f}) mm",
            f"Stroke: {self.completed_strokes}/{self.total_strokes}",
            current_stroke_info,
            f"Pen: {'DOWN' if self.is_pen_down else 'UP'}",
            f"Status: {'PLAYING' if self.is_playing else 'PAUSED'}"
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = self.small_font.render(text, True, self.colors['ui_text'])
            self.screen.blit(text_surface, (20, ui_y + i * 20))
        
        # Canvas dimensions
        canvas_info = f"Canvas: {CANVAS_WIDTH_MM}mm × {CANVAS_HEIGHT_MM}mm"
        text_surface = self.small_font.render(canvas_info, True, self.colors['ui_text'])
        self.screen.blit(text_surface, (self.window_size[0] - 200, ui_y))
    
    def handle_event(self, event):
        """Handle pygame events"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = pygame.mouse.get_pos()
                for button in self.buttons:
                    if button['rect'].collidepoint(mouse_pos):
                        button['action']()
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self._toggle_play_pause()
            elif event.key == pygame.K_r:
                self._reset_simulation()
            elif event.key == pygame.K_s:
                self._step_forward()
            elif event.key == pygame.K_UP:
                self._increase_speed()
            elif event.key == pygame.K_DOWN:
                self._decrease_speed()
    
    def run(self, stroke_data_path: Optional[str] = None):
        """
        Run the simulation
        
        Args:
            stroke_data_path: Optional path to stroke data JSON file
        """
        print("🎨 Starting Brush Movement Simulator")
        print("📋 Controls:")
        print("   • SPACE: Play/Pause")
        print("   • R: Reset")
        print("   • S: Step forward")
        print("   • UP/DOWN: Adjust speed")
        print("   • Mouse: Click buttons")
        
        if stroke_data_path:
            if not self.load_stroke_data(stroke_data_path):
                print("❌ Failed to load stroke data")
                return
        else:
            print("💡 No stroke data provided. Load a JSON file with the Load button.")
        
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    self.handle_event(event)
            
            self.update()
            self.draw()
            clock.tick(60)  # 60 FPS
        
        pygame.quit()
        print("👋 Simulation ended")

def find_latest_stroke_results() -> Optional[str]:
    """Find the most recent stroke ordering results file"""
    results_dir = Path("painting_vectorization/results/step8_stroke_ordering")
    if not results_dir.exists():
        return None
    
    json_files = list(results_dir.glob("stroke_ordering_results_*.json"))
    if not json_files:
        return None
    
    # Sort by modification time, most recent first
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Robotic Brush Movement Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python brush_simulator.py                                    # Auto-find latest results
  python brush_simulator.py results.json                       # Load specific file
  python brush_simulator.py --speed 60                         # Set initial speed
        """
    )
    
    parser.add_argument('stroke_file', nargs='?', help='Stroke ordering JSON file')
    parser.add_argument('--speed', type=float, default=40.0, help='Initial brush speed (mm/s)')
    parser.add_argument('--window-size', type=int, nargs=2, default=[1000, 800], 
                       help='Window size (width height)')
    
    args = parser.parse_args()
    
    # Find stroke data file
    stroke_file = args.stroke_file
    if not stroke_file:
        stroke_file = find_latest_stroke_results()
        if stroke_file:
            print(f"🔍 Auto-detected latest results: {stroke_file}")
        else:
            print("⚠️  No stroke data found. Run without file to use manual controls.")
    
    # Create and run simulator
    simulator = BrushSimulator(window_size=tuple(args.window_size))
    simulator.brush_speed_mm_s = args.speed
    
    try:
        simulator.run(stroke_file)
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted")
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
