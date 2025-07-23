import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from color_detector import ColorDetector
from utils import calculate_orientation_angle

@dataclass
class TrackedObject:
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    angle: float
    prev_position: Optional[Tuple[float, float]] = None

class FrameTracker:
    def __init__(self, fps: float):
        self.detector = ColorDetector()
        self.fps = fps
        self.dt = 1.0 / fps
        self.tracked_objects: Dict[str, TrackedObject] = {}
    
    def update(self, frame: np.ndarray) -> Tuple[Dict[str, TrackedObject], np.ndarray]:
        """
        Update tracking for all colors in the frame.
        Returns tracked objects and visualization frame.
        """
        # Detect colors in current frame
        current_detections = self.detector.detect_colors(frame)
        
        # Update tracking for each color
        for color, current_pos in current_detections.items():
            if current_pos is None:
                if color in self.tracked_objects:
                    del self.tracked_objects[color]
                continue
            
            if color not in self.tracked_objects:
                # Initialize new tracked object
                self.tracked_objects[color] = TrackedObject(
                    position=current_pos,
                    velocity=(0.0, 0.0),
                    angle=-1.0,
                    prev_position=None
                )
            else:
                # Update existing tracked object
                prev_pos = self.tracked_objects[color].position
                dx = (current_pos[0] - prev_pos[0]) / self.dt
                dy = (current_pos[1] - prev_pos[1]) / self.dt
                angle = calculate_orientation_angle(dx, dy, self.tracked_objects[color].angle)
                
                self.tracked_objects[color] = TrackedObject(
                    position=current_pos,
                    velocity=(dx, dy),
                    angle=angle,
                    prev_position=prev_pos
                )
        
        # Create visualization
        vis_frame = self.detector.draw_detections(frame, current_detections)
        vis_frame = self._draw_tracking(vis_frame)
        
        return self.tracked_objects, vis_frame
    
    def _draw_tracking(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking information on the frame."""
        for color, obj in self.tracked_objects.items():
            if obj.prev_position is not None:
                # Draw motion vector
                cv2.arrowedLine(
                    frame,
                    (int(obj.prev_position[0]), int(obj.prev_position[1])),
                    (int(obj.position[0]), int(obj.position[1])),
                    (0, 255, 0),
                    2
                )
                
                # Draw velocity and angle
                info_text = f"vx: {obj.velocity[0]:.1f} vy: {obj.velocity[1]:.1f} θ: {obj.angle:.1f}°"
                cv2.putText(
                    frame,
                    info_text,
                    (int(obj.position[0]) + 10, int(obj.position[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
        
        return frame 