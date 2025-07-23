import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from color_detector import ColorDetector
from utils import calculate_orientation_angle

@dataclass
class FlowTrackedObject:
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    angle: float
    flow_points: np.ndarray
    flow_vectors: np.ndarray
    confidence: float
    past_positions: List[Tuple[float, float]]
    filtered_velocity: Tuple[float, float]
    filtered_angle: float

class OpticalFlowTracker:
    def __init__(self, fps: float, use_gpu: bool = False, history_length: int = 3):
        self.detector = ColorDetector()
        self.fps = fps
        self.dt = 1.0 / fps
        self.use_gpu = use_gpu
        self.history_length = history_length
        
        # Initialize optical flow
        if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.flow = cv2.cuda_FarnebackOpticalFlow_create()
            self.use_cuda = True
        else:
            self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            self.use_cuda = False
            
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False)
        
        self.prev_frame = None
        self.tracked_objects: Dict[str, FlowTrackedObject] = {}
        
        # Parameters
        self.grid_size = 20  # pixels between flow points
        self.max_points = 100  # maximum flow points
        self.frame_shape = None
        self.flow_reset_threshold = 5.0  # pixels, for detecting abrupt changes
        self.ema_alpha = 0.3  # smoothing factor for filtering
    
    def _create_flow_points(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Create a grid of points for optical flow tracking."""
        h, w = frame_shape[:2]
        x = np.arange(self.grid_size, w - self.grid_size, self.grid_size)
        y = np.arange(self.grid_size, h - self.grid_size, self.grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.float32)
        return points[:self.max_points]
    
    def _clip_points_to_bounds(self, points: np.ndarray) -> np.ndarray:
        """Ensure points stay within image bounds."""
        if self.frame_shape is None:
            return points
        h, w = self.frame_shape[:2]
        points[:, 0] = np.clip(points[:, 0], 0, w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, h - 1)
        return points
    
    def update(self, frame: np.ndarray) -> Tuple[Dict[str, FlowTrackedObject], np.ndarray]:
        """Update tracking using optical flow and color detection."""
        self.frame_shape = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        frame_masked = cv2.bitwise_and(frame, frame, mask=fg_mask)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            self.flow_points = self._create_flow_points(frame.shape)
            return self.tracked_objects, frame
        
        # Calculate optical flow
        if self.use_cuda:
            prev_gpu = cv2.cuda_GpuMat(self.prev_frame)
            curr_gpu = cv2.cuda_GpuMat(gray)
            flow_gpu = self.flow.calc(prev_gpu, curr_gpu, None)
            flow = flow_gpu.download()
        else:
            flow = self.flow.calc(self.prev_frame, gray, None)
        
        # Update flow points with bounds checking
        flow_points_int = self._clip_points_to_bounds(self.flow_points).astype(int)
        flow_vectors = flow[flow_points_int[:, 1], flow_points_int[:, 0]]
        new_points = self.flow_points + flow_vectors
        new_points = self._clip_points_to_bounds(new_points)
        
        # Check for abrupt changes
        mean_flow = np.mean(np.linalg.norm(flow_vectors, axis=1))
        if mean_flow > self.flow_reset_threshold:
            self.flow_points = self._create_flow_points(frame.shape)
            new_points = self.flow_points.copy()
        
        # Detect colors on masked frame
        current_detections = self.detector.detect_colors(frame_masked)
        
        # Update tracking for each color
        for color, current_pos in current_detections.items():
            if current_pos is None:
                if color in self.tracked_objects:
                    del self.tracked_objects[color]
                continue
            
            # Find flow vectors near the detected object
            distances = np.linalg.norm(new_points - current_pos, axis=1)
            nearby_indices = np.where(distances < self.grid_size * 2)[0]
            
            if len(nearby_indices) > 0:
                nearby_points = self.flow_points[nearby_indices]
                nearby_vectors = new_points[nearby_indices] - nearby_points
                
                # Calculate velocity and confidence
                avg_velocity = np.mean(nearby_vectors, axis=0) / self.dt
                vector_variance = np.var(nearby_vectors, axis=0)
                confidence = 1.0 / (1.0 + np.mean(vector_variance))
                
                # Get previous filtered values
                prev_obj = self.tracked_objects.get(color, None)
                prev_velocity = prev_obj.filtered_velocity if prev_obj else (0.0, 0.0)
                prev_angle = prev_obj.filtered_angle if prev_obj else -1.0
                
                # Apply EMA filtering
                filtered_velocity = tuple(
                    self.ema_alpha * avg_velocity[i] + (1 - self.ema_alpha) * prev_velocity[i]
                    for i in range(2)
                )
                
                angle = calculate_orientation_angle(filtered_velocity[0], filtered_velocity[1], prev_angle)
                filtered_angle = self.ema_alpha * angle + (1 - self.ema_alpha) * prev_angle
                
                # Update past positions
                past_positions = ([current_pos] + 
                    (prev_obj.past_positions if prev_obj else []))[:self.history_length]
                
                self.tracked_objects[color] = FlowTrackedObject(
                    position=current_pos,
                    velocity=tuple(avg_velocity),
                    angle=angle,
                    flow_points=nearby_points,
                    flow_vectors=nearby_vectors,
                    confidence=confidence,
                    past_positions=past_positions,
                    filtered_velocity=filtered_velocity,
                    filtered_angle=filtered_angle
                )
        
        # Update for next frame
        self.prev_frame = gray
        self.flow_points = new_points
        
        # Create visualization
        vis_frame = self.detector.draw_detections(frame, current_detections)
        vis_frame = self._draw_tracking(vis_frame)
        
        return self.tracked_objects, vis_frame
    
    def _draw_tracking(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking information and flow vectors on the frame."""
        for color, obj in self.tracked_objects.items():
            # Draw flow vectors
            for point, vector in zip(obj.flow_points, obj.flow_vectors):
                start_point = tuple(map(int, point))
                end_point = tuple(map(int, point + vector))
                cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 1)
            
            # Draw trajectory
            for i in range(len(obj.past_positions) - 1):
                p1 = tuple(map(int, obj.past_positions[i]))
                p2 = tuple(map(int, obj.past_positions[i + 1]))
                cv2.line(frame, p1, p2, (255, 0, 0), 2)
            
            # Draw object info
            x, y = map(int, obj.position)
            info_text = (f"vx: {obj.filtered_velocity[0]:.1f} "
                        f"vy: {obj.filtered_velocity[1]:.1f} "
                        f"θ: {obj.filtered_angle:.1f}° "
                        f"conf: {obj.confidence:.2f}")
            cv2.putText(frame, info_text, (x + 10, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame