import cv2
import time
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import threading
import sys
import select
import os

from utils import (
    get_user_inputs, create_output_directory, check_gpu_availability,
    get_frame_rate
)
from frame_tracker import FrameTracker, TrackedObject
from optical_flow_tracker import OpticalFlowTracker, FlowTrackedObject
from data_logger import DataLogger, TrackingData

@dataclass
class TrackingResult:
    tracked_objects: Dict[str, TrackedObject]
    processed_frame: np.ndarray
    processing_time: float

# Global flag for graceful shutdown
running = True

def check_for_quit():
    """Check for 'q' key press in terminal to quit the program."""
    global running
    
    print("Press 'q' in terminal to quit the program...")
    
    while running:
        # Check if there's input available (non-blocking)
        if sys.platform.startswith('win'):
            # Windows implementation
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                if key == 'q':
                    print("\nQuit command received from terminal.")
                    running = False
                    break
        else:
            # Unix/Linux implementation
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1).lower()
                if key == 'q':
                    print("\nQuit command received from terminal.")
                    running = False
                    break
        
        time.sleep(0.1)  # Small delay to prevent high CPU usage

def process_frame(tracker, frame: np.ndarray, fps: float) -> TrackingResult:
    """Process a single frame and return tracking results."""
    start_time = time.time()
    
    # Update tracking
    tracked_objects, processed_frame = tracker.update(frame)
    
    # Convert tracked objects to tracking data
    tracking_data = {}
    for color, obj in tracked_objects.items():
        tracking_data[color] = TrackingData(
            timestamp=time.time(),
            color=color,
            x=obj.position[0],
            y=obj.position[1],
            vx=obj.velocity[0],
            vy=obj.velocity[1],
            angle=obj.angle,
            fps=fps,
            processing_time=time.time() - start_time
        )
    
    return TrackingResult(
        tracked_objects=tracked_objects,
        processed_frame=processed_frame,
        processing_time=time.time() - start_time
    )

def main():
    global running
    
    # Get user configuration
    config = get_user_inputs()
    
    # Check if configuration is valid
    if not config:
        print("Error: Invalid configuration. Exiting.")
        return
    
    # Check GPU availability if requested
    if config['device'] == 'gpu' and not check_gpu_availability():
        print("Warning: GPU requested but not available. Falling back to CPU.")
        config['device'] = 'cpu'
    
    # Initialize video capture with selected camera
    camera_index = config['camera_index']
    print(f"\nInitializing camera {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return
    
    # Optimize camera settings for high-performance cameras
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for faster response
    
    # Get camera info for selected camera
    camera_info = config['camera_info']
    current_width = camera_info['width']
    current_height = camera_info['height']
    detected_fps = camera_info['fps']
    
    print(f"Detected camera properties: {current_width:.0f}x{current_height:.0f} @ {detected_fps:.1f} FPS")
    
    # For high-performance cameras, try to set optimal resolution
    # Don't force 640x360 if camera supports higher resolution
    if current_width < 640 or current_height < 360:
        # Only set to 640x360 if current resolution is lower
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        print("Setting resolution to 640x360")
    else:
        print(f"Using camera's native resolution: {current_width:.0f}x{current_height:.0f}")
    
    # Get actual resolution after setting
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"Final camera resolution: {actual_width:.0f}x{actual_height:.0f}")
    
    # Use detected frame rate from camera info
    fps = detected_fps
    print(f"Using detected frame rate: {fps:.1f} FPS")
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Initialize tracker based on user selection
    print("Initializing tracker...")
    if config['method'] == 'frame':
        tracker = FrameTracker(fps)
    else:
        tracker = OpticalFlowTracker(fps, use_gpu=(config['device'] == 'gpu'))
    
    # Initialize data logger
    print("Initializing data logger...")
    logger = DataLogger(output_dir, config['method'], config['device'])
    
    # Get first frame to initialize video writers
    print("Testing camera connection...")
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
    
    logger.initialize_video_writers(fps, frame.shape[:2][::-1])
    
    print(f"\nStarting tracking with Camera {camera_index}...")
    print(f"Target FPS: {fps:.1f}, Resolution: {actual_width:.0f}x{actual_height:.0f}")
    print("Press 'q' in terminal or 'q' in debug window to quit.")
    
    # Start quit monitoring thread
    quit_thread = threading.Thread(target=check_for_quit, daemon=True)
    quit_thread.start()
    
    frame_count = 0
    start_time = time.time()
    last_frame_time = start_time
    
    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                logger.log_error("Failed to read frame")
                break
            
            # Process frame
            result = process_frame(tracker, frame, fps)
            
            # Log results
            logger.log_frame(frame, result.processed_frame,
                           {color: TrackingData(
                               timestamp=time.time(),
                               color=color,
                               x=obj.position[0],
                               y=obj.position[1],
                               vx=obj.velocity[0],
                               vy=obj.velocity[1],
                               angle=obj.angle,
                               fps=fps,
                               processing_time=result.processing_time
                           ) for color, obj in result.tracked_objects.items()},
                           result.processing_time)
            
            # Show debug window if enabled
            if config['debug']:
                cv2.imshow('Tracking', result.processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nQuit command received from debug window.")
                    running = False
                    break
            
            # Frame rate control - optimized for high-performance cameras
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # For high FPS cameras, use adaptive frame rate control
            target_frame_time = 1.0 / fps
            if elapsed < target_frame_time:
                # Only sleep if we're ahead of schedule
                sleep_time = target_frame_time - elapsed
                if sleep_time > 0.001:  # Only sleep if more than 1ms
                    time.sleep(sleep_time)
            
            last_frame_time = time.time()
            
            # Print status every second
            if current_time - start_time >= 1.0:
                actual_fps = frame_count / (current_time - start_time)
                print(f"\rFPS: {actual_fps:.1f}, Processing time: {result.processing_time*1000:.1f}ms", end='')
                frame_count = 0
                start_time = current_time
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping tracking...")
    finally:
        # Cleanup
        running = False
        cap.release()
        cv2.destroyAllWindows()
        logger.close()
        print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    # Import msvcrt for Windows keyboard handling
    if sys.platform.startswith('win'):
        import msvcrt
    
    main() 