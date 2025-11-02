# Real-time Color Object Tracking System

This system provides real-time tracking of red, blue, and yellow objects using either frame-based tracking or optical flow tracking. It supports both CPU and GPU processing options and includes comprehensive data logging capabilities.

## Features

- Color detection for red, yellow, and blue objects
- Two tracking methods:
  - Frame-based tracking (center of mass)
  - Optical Flow tracking using OpenCV's DIS algorithm
- CPU/GPU processing options
- Real-time performance optimization
- Comprehensive data logging:
  - Original and processed videos
  - CSV file with tracking data
  - Detailed run logs
- Debug mode for real-time visualization

## Requirements

- Python 3.8 or higher
   * opencv-python>=4.8.0
   * numpy>=1.24.0
   * pandas>=2.0.0
   * tqdm>=4.65.0
   * python-dateutil>=2.8.2
   * torch>=2.0.0  # For GPU acceleration
   * torchvision>=0.15.0 

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python main.py
```

Follow the console prompts to:
1. Choose tracking method:
   - `Fr` for Frame-based tracking
   - `Op` for Optical Flow tracking
2. Select processing device:
   - `C` for CPU
   - `G` for GPU (if available)
3. Enable debug mode:
   - `Y` to show real-time visualization
   - `n` to run without visualization

## Output

The system creates a timestamped directory in the `data` folder containing:
- `original.mp4`: Original video feed
- `processed.mp4`: Video with tracking visualization
- `tracking_data.csv`: Tracking data for all detected objects
- `run_log.txt`: Detailed run information and logs

## CSV Data Format

The tracking data CSV includes the following columns:
- timestamp: Time of detection
- color: Detected color (red/yellow/blue)
- x, y: Object coordinates
- vx, vy: Velocity components
- angle: Movement orientation (0-360 degrees)
- fps: Current frame rate
- processing_time: Time taken to process the frame
- method: Tracking method used
- device: Processing device used
