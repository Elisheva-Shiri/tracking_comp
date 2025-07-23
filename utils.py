import os
import cv2
import time
from datetime import datetime
import numpy as np
from typing import Tuple, Dict, Optional, List

# Color detection HSV ranges
COLOR_RANGES = {
    'red': [
        {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},  # Red lower range
        {'lower': np.array([160, 100, 100]), 'upper': np.array([180, 255, 255])}  # Red upper range
    ],
    'blue': [
        {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])}
    ],
    'yellow': [
        {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])}
    ]
}

def test_camera_with_backend(camera_index: int, backend) -> Optional[Dict]:
    """Test a specific camera index with a specific backend."""
    try:
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            # Set buffer size to 1 for faster response
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Try to read a frame with multiple attempts
            for attempt in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    cap.release()
                    return {
                        'index': camera_index,
                        'backend': backend,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'frame_shape': frame.shape
                    }
                time.sleep(0.2)  # Longer delay for HD cameras
            
            cap.release()
    except Exception as e:
        pass
    return None

def detect_available_cameras() -> List[int]:
    """Detect all available camera devices - enhanced for HD cameras with MSMF issues."""
    available_cameras = []
    
    print("Scanning for cameras with enhanced detection...")
    
    # Define backends to try (in order of preference for HD cameras)
    backends = [
        cv2.CAP_DSHOW,         # DirectShow (best for HD cameras on Windows)
        cv2.CAP_ANY,           # Default
        cv2.CAP_MSMF,          # Media Foundation (can have issues)
        cv2.CAP_V4L2,          # Video4Linux2 (Linux)
        cv2.CAP_AVFOUNDATION,  # AVFoundation (macOS)
    ]
    
    backend_names = {
        cv2.CAP_ANY: "Default",
        cv2.CAP_DSHOW: "DirectShow",
        cv2.CAP_MSMF: "Media Foundation", 
        cv2.CAP_V4L2: "Video4Linux2",
        cv2.CAP_AVFOUNDATION: "AVFoundation",
    }
    
    # First, try a quick scan to find cameras that can be opened
    print("Quick scan for available camera indices...")
    quick_found = []
    for i in range(31):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                quick_found.append(i)
                print(f"  Camera index {i} can be opened")
            cap.release()
        except:
            pass
    
    if not quick_found:
        print("No cameras found in quick scan. Trying manual input...")
        return get_manual_camera_input()
    
    print(f"Found {len(quick_found)} camera indices that can be opened.")
    
    # Now test each found camera with different backends
    for camera_index in quick_found:
        print(f"\nTesting camera index {camera_index}:")
        
        for backend in backends:
            backend_name = backend_names.get(backend, f"Backend_{backend}")
            print(f"  Trying {backend_name}...", end=' ')
            
            result = test_camera_with_backend(camera_index, backend)
            if result:
                available_cameras.append(camera_index)
                print(f"✓ Found: {result['width']:.0f}x{result['height']:.0f} @ {result['fps']:.1f} FPS")
                print(f"    Backend: {backend_name}, Frame shape: {result['frame_shape']}")
                break  # Found this camera, move to next index
            else:
                print("✗ Failed")
    
    if not available_cameras:
        print("\nNo cameras could read frames automatically.")
        print("This might be due to MSMF errors with HD cameras.")
        return get_manual_camera_input()
    
    print(f"\nTotal cameras found: {len(available_cameras)}")
    return available_cameras

def get_manual_camera_input() -> List[int]:
    """Get manual camera input when automatic detection fails."""
    print("\nManual camera detection mode:")
    print("This is common with HD cameras that have MSMF issues.")
    
    available_cameras = []
    
    while True:
        try:
            manual_index = input("Enter camera index to test (or 'done' to finish): ")
            if manual_index.lower() == 'done':
                break
            
            camera_index = int(manual_index)
            print(f"Testing camera index {camera_index}...")
            
            # Test with DirectShow first (best for HD cameras)
            result = test_camera_with_backend(camera_index, cv2.CAP_DSHOW)
            if result:
                available_cameras.append(camera_index)
                print(f"✓ Found with DirectShow: {result['width']:.0f}x{result['height']:.0f} @ {result['fps']:.1f} FPS")
            else:
                print(f"✗ Camera index {camera_index} not working with DirectShow")
                
        except ValueError:
            print("Invalid input. Please enter a number or 'done'.")
    
    return available_cameras

def get_camera_info(camera_index: int) -> Dict[str, float]:
    """Get camera information - enhanced for HD cameras."""
    # Try DirectShow first (best for HD cameras)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return {}
    
    # Set buffer size to 1 for faster response
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # If FPS is not available or seems wrong, use full detection
    if fps <= 0 or fps > 300:  # Unrealistic FPS values
        fps = get_frame_rate(cap)
    
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps
    }

def create_output_directory() -> str:
    """Create timestamped output directory for saving results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "data"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return os.path.join(base_dir, timestamp)

def calculate_orientation_angle(dx: float, dy: float, prev_angle: float = -1) -> float:
    """
    Calculate orientation angle in degrees (0-360) relative to X-axis.
    Returns -1 if no motion detected.
    """
    if abs(dx) < 0.1 and abs(dy) < 0.1:  # Threshold for no motion
        return prev_angle
    
    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 360
    return angle

def get_frame_rate(cap: cv2.VideoCapture) -> float:
    """Detect actual frame rate from video capture - full detection for high-performance cameras."""
    start_time = time.time()
    frame_count = 0
    max_frames = 100  # Full detection for high FPS cameras
    max_time = 3.0    # Full time for accurate detection
    
    while frame_count < max_frames and time.time() - start_time < max_time:
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1
    
    duration = time.time() - start_time
    fps = frame_count / duration if duration > 0 else 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to beginning
    return fps

def create_video_writer(output_path: str, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """Create video writer with appropriate codec."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)

def get_user_inputs() -> Dict[str, str]:
    """Get user inputs for camera selection, tracking method, processing device, and debug mode."""
    print("\n=== Motion Tracking System Configuration ===")
    
    # Detect available cameras
    print("\nDetecting available cameras...")
    available_cameras = detect_available_cameras()
    
    if not available_cameras:
        print("Error: No cameras detected!")
        return {}
    
    print(f"\nFound {len(available_cameras)} camera(s):")
    for i, cam_idx in enumerate(available_cameras):
        info = get_camera_info(cam_idx)
        if info:
            print(f"  {i+1}. Camera {cam_idx}: {info['width']:.0f}x{info['height']:.0f} @ {info['fps']:.1f} FPS")
    
    # Camera selection
    while True:
        try:
            camera_choice = input(f"\nSelect camera (1-{len(available_cameras)}): ")
            camera_idx = int(camera_choice) - 1
            if 0 <= camera_idx < len(available_cameras):
                selected_camera = available_cameras[camera_idx]
                break
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(available_cameras)}")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Get camera info for selected camera
    camera_info = get_camera_info(selected_camera)
    print(f"\nSelected Camera {selected_camera}: {camera_info['width']:.0f}x{camera_info['height']:.0f} @ {camera_info['fps']:.1f} FPS")
    
    # Tracking method selection
    while True:
        method = input("\nChoose tracking method (Fr for Frame-based, Op for Optical Flow): ").lower()
        if method in ['fr', 'op']:
            break
        print("Invalid input. Please enter 'Fr' or 'Op'")
    
    # Processing device selection
    while True:
        device = input("Select processing device (C for CPU, G for GPU): ").lower()
        if device in ['c', 'g']:
            break
        print("Invalid input. Please enter 'C' or 'G'")
    
    # Debug mode selection
    while True:
        debug = input("Enable debug mode to show video? (Y/n): ").lower()
        if debug in ['y', 'n', '']:
            break
        print("Invalid input. Please enter 'Y' or 'n'")
    
    return {
        'camera_index': selected_camera,
        'camera_info': camera_info,
        'method': 'frame' if method == 'fr' else 'optical_flow',
        'device': 'cpu' if device == 'c' else 'gpu',
        'debug': debug in ['y', '']
    }

def get_restart_inputs() -> Dict[str, str]:
    """Get tracking configuration for restart (without camera selection)."""
    print("\n=== New Run Configuration ===")
    
    # Tracking method selection
    while True:
        method = input("Choose tracking method (Fr for Frame-based, Op for Optical Flow): ").lower()
        if method in ['fr', 'op']:
            break
        print("Invalid input. Please enter 'Fr' or 'Op'")
    
    # Processing device selection
    while True:
        device = input("Select processing device (C for CPU, G for GPU): ").lower()
        if device in ['c', 'g']:
            break
        print("Invalid input. Please enter 'C' or 'G'")
    
    # Debug mode selection
    while True:
        debug = input("Enable debug mode to show video? (Y/n): ").lower()
        if debug in ['y', 'n', '']:
            break
        print("Invalid input. Please enter 'Y' or 'n'")
    
    return {
        'method': 'frame' if method == 'fr' else 'optical_flow',
        'device': 'cpu' if device == 'c' else 'gpu',
        'debug': debug in ['y', '']
    }

def check_gpu_availability() -> bool:
    """Check if GPU is available for processing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False 