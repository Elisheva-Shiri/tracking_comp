import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from utils import COLOR_RANGES

class Detector:
    def __init__(self, min_contour_area: int = 100):
        self.min_contour_area = min_contour_area
        self.colors = list(COLOR_RANGES.keys())
    
    def _colors(self, frame: np.ndarray) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        Detect the most prominent object of each color in the frame.
        Returns a dictionary with color names as keys and (x, y) center coordinates as values.
        Returns None for colors where no object is detected.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results = {color: None for color in self.colors}
        
        for color in self.colors:
            # Create mask for the color
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for color_range in COLOR_RANGES[color]:
                color_mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
                mask = cv2.bitwise_or(mask, color_mask)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > self.min_contour_area:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = float(M["m10"] / M["m00"])
                        cy = float(M["m01"] / M["m00"])
                        results[color] = (cx, cy)
        
        return results
    
    def draw_detections(self, frame: np.ndarray, detections: Dict[str, Optional[Tuple[float, float]]]) -> np.ndarray:
        """Draw detected objects and their centers on the frame."""
        output = frame.copy()
        
        for color, center in detections.items():
            if center is not None:
                x, y = map(int, center)
                # Draw circle at center
                cv2.circle(output, (x, y), 5, (0, 255, 0), -1)
                # Draw color label
                cv2.putText(output, color, (x + 10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output 