# Merged Detector System

This document describes the new merged detector system that integrates all tracking functionality into a single, unified interface.

## Overview

The `detector.py` file now contains all the tracking functionality that was previously spread across multiple files:

- `color_detector.py` → `ColorDetector` class
- `frame_tracker.py` → `FrameTracker` class  
- `optical_flow_tracker.py` → `OpticalFlowTracker` class

## Main Classes

### 1. Detector (Main Class)
The main `Detector` class provides a unified interface for all tracking methods.

```python
from detector import Detector

# Initialize with frame-based tracking
detector = Detector(tracking_method="frame", fps=30.0)

# Initialize with optical flow tracking
detector = Detector(tracking_method="optical_flow", fps=30.0, use_gpu=True)

# Update tracking
tracked_objects, processed_frame = detector.update(frame)

# Detect colors without tracking
detections = detector.detect_colors(frame)

# Draw detections
result_frame = detector.draw_detections(frame, detections)
```

#### Constructor Parameters
- `tracking_method`: "frame" or "optical_flow"
- `fps`: Frame rate for velocity calculations
- `use_gpu`: Whether to use GPU acceleration (for optical flow)
- `min_contour_area`: Minimum contour area for color detection

#### Main Methods
- `update(frame)`: Process frame and return tracking results
- `detect_colors(frame)`: Detect colors without tracking
- `draw_detections(frame, detections)`: Draw detection results
- `get_tracking_method()`: Get current tracking method
- `get_fps()`: Get current frame rate
- `is_gpu_enabled()`: Check GPU status
- `get_min_contour_area()`: Get contour area threshold
- `set_min_contour_area(area)`: Set contour area threshold

### 2. ColorDetector
Base color detection functionality used by all tracking methods.

```python
from detector import ColorDetector

detector = ColorDetector(min_contour_area=100)
detections = detector.detect_colors(frame)
result_frame = detector.draw_detections(frame, detections)
```

### 3. FrameTracker
Frame-based tracking using color detection and velocity calculation.

```python
from detector import FrameTracker

tracker = FrameTracker(fps=30.0)
tracked_objects, processed_frame = tracker.update(frame)
```

### 4. OpticalFlowTracker
Advanced optical flow-based tracking with motion analysis.

```python
from detector import OpticalFlowTracker

tracker = OpticalFlowTracker(fps=30.0, use_gpu=False)
tracked_objects, processed_frame = tracker.update(frame)
```

## Data Structures

### TrackedObject
Used by frame-based tracking:
```python
@dataclass
class TrackedObject:
    position: Tuple[float, float]      # Current (x, y) position
    velocity: Tuple[float, float]      # Current velocity (vx, vy)
    angle: float                       # Orientation angle in degrees
    prev_position: Optional[Tuple[float, float]]  # Previous position
```

### FlowTrackedObject
Used by optical flow tracking (extends TrackedObject):
```python
@dataclass
class FlowTrackedObject:
    position: Tuple[float, float]      # Current position
    velocity: Tuple[float, float]      # Raw velocity
    angle: float                       # Raw angle
    flow_points: np.ndarray           # Flow tracking points
    flow_vectors: np.ndarray          # Flow vectors
    confidence: float                  # Tracking confidence
    past_positions: List[Tuple[float, float]]  # Position history
    filtered_velocity: Tuple[float, float]     # Filtered velocity
    filtered_angle: float                       # Filtered angle
```

## Usage Examples

### Basic Color Detection
```python
from detector import Detector

detector = Detector(tracking_method="frame", fps=30.0)
detections = detector.detect_colors(frame)
print(f"Detected colors: {list(detections.keys())}")
```

### Frame-Based Tracking
```python
detector = Detector(tracking_method="frame", fps=30.0)

for frame in video_frames:
    tracked_objects, processed_frame = detector.update(frame)
    
    for color, obj in tracked_objects.items():
        print(f"{color}: pos=({obj.position[0]:.1f}, {obj.position[1]:.1f})")
        print(f"  velocity=({obj.velocity[0]:.1f}, {obj.velocity[1]:.1f})")
        print(f"  angle={obj.angle:.1f}°")
```

### Optical Flow Tracking
```python
detector = Detector(tracking_method="optical_flow", fps=30.0, use_gpu=False)

for frame in video_frames:
    tracked_objects, processed_frame = detector.update(frame)
    
    for color, obj in tracked_objects.items():
        print(f"{color}: pos=({obj.position[0]:.1f}, {obj.position[1]:.1f})")
        print(f"  filtered_vel=({obj.filtered_velocity[0]:.1f}, {obj.filtered_velocity[1]:.1f})")
        print(f"  filtered_angle={obj.filtered_angle:.1f}°")
        print(f"  confidence={obj.confidence:.2f}")
```

### Switching Tracking Methods
```python
# Start with frame tracking
detector = Detector(tracking_method="frame", fps=30.0)

# Process some frames
for i in range(10):
    tracked_objects, processed_frame = detector.update(frames[i])

# Switch to optical flow (creates new tracker internally)
detector = Detector(tracking_method="optical_flow", fps=30.0, use_gpu=False)

# Continue processing with new method
for i in range(10, 20):
    tracked_objects, processed_frame = detector.update(frames[i])
```

## Integration with Main System

The main.py file has been updated to use the new merged Detector class:

```python
# Old way (separate classes)
if config['method'] == 'frame':
    tracker = FrameTracker(fps)
else:
    tracker = OpticalFlowTracker(fps, use_gpu=(config['device'] == 'gpu'))

# New way (merged class)
tracking_method = "frame" if config['method'] == 'frame' else "optical_flow"
use_gpu = config['device'] == 'gpu'
tracker = Detector(tracking_method=tracking_method, fps=fps, use_gpu=use_gpu)
```

## Testing

Run the test script to verify the merged detector works correctly:

```bash
python test_detector.py
```

This will test:
- Color detection
- Frame-based tracking
- Optical flow tracking
- Detector utility methods

## Benefits of Merged Structure

1. **Unified Interface**: Single class for all tracking methods
2. **Easier Maintenance**: All tracking code in one place
3. **Better Integration**: Consistent API across tracking methods
4. **Simplified Imports**: Only need to import from `detector.py`
5. **Easier Testing**: Test all functionality in one place
6. **Better Documentation**: All related code documented together

## Migration Notes

- The old separate files (`color_detector.py`, `frame_tracker.py`, `optical_flow_tracker.py`) can be removed
- All imports should now reference `detector.py`
- The API remains the same, so existing code should work with minimal changes
- The main.py file has been updated to use the new structure

## File Structure

```
Comp/
├── detector.py              # Merged detector file (NEW)
├── main.py                  # Updated to use merged detector
├── data_logger.py           # No changes needed
├── utils.py                 # No changes needed
├── test_detector.py         # Test script for merged detector
├── README_MERGED.md         # This documentation
└── README.md                # Original README
```

The merged detector system provides a cleaner, more maintainable codebase while preserving all the original functionality.
