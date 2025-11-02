#!/usr/bin/env python3
"""
Test script for the merged Detector class.
Tests both frame-based and optical flow tracking methods.
"""

import cv2
import numpy as np
from detector import Detector, TrackedObject, FlowTrackedObject

def test_color_detection():
    """Test basic color detection functionality."""
    print("Testing color detection...")
    
    # Create a test frame with colored objects
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a red circle
    cv2.circle(frame, (200, 200), 50, (0, 0, 255), -1)
    
    # Draw a blue rectangle
    cv2.rectangle(frame, (400, 150), (500, 250), (255, 0, 0), -1)
    
    # Draw a yellow triangle
    pts = np.array([[300, 350], [250, 450], [350, 450]], np.int32)
    cv2.fillPoly(frame, [pts], (0, 255, 255))
    
    # Test color detector
    detector = Detector(tracking_method="frame", fps=30.0)
    detections = detector.detect_colors(frame)
    
    print("Detections:", detections)
    
    # Draw detections
    result_frame = detector.draw_detections(frame, detections)
    
    # Save test result
    cv2.imwrite("test_color_detection.jpg", result_frame)
    print("Color detection test completed. Result saved as 'test_color_detection.jpg'")
    
    return detections

def test_frame_tracking():
    """Test frame-based tracking."""
    print("\nTesting frame-based tracking...")
    
    # Create detector with frame tracking
    detector = Detector(tracking_method="frame", fps=30.0)
    
    # Create test frames with moving object
    frames = []
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Move red circle from left to right
        x = 100 + i * 50
        cv2.circle(frame, (x, 240), 30, (0, 0, 255), -1)
        frames.append(frame)
    
    # Process frames
    for i, frame in enumerate(frames):
        tracked_objects, processed_frame = detector.update(frame)
        print(f"Frame {i}: {len(tracked_objects)} objects tracked")
        
        if tracked_objects:
            for color, obj in tracked_objects.items():
                print(f"  {color}: pos=({obj.position[0]:.1f}, {obj.position[1]:.1f}), "
                      f"vel=({obj.velocity[0]:.1f}, {obj.velocity[1]:.1f}), angle={obj.angle:.1f}°")
    
    print("Frame tracking test completed.")
    return detector

def test_optical_flow_tracking():
    """Test optical flow tracking."""
    print("\nTesting optical flow tracking...")
    
    try:
        # Create detector with optical flow tracking
        detector = Detector(tracking_method="optical_flow", fps=30.0, use_gpu=False)
        
        # Create test frames with moving object
        frames = []
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Move blue rectangle diagonally
            x = 100 + i * 30
            y = 100 + i * 20
            cv2.rectangle(frame, (x, y), (x + 60, y + 40), (255, 0, 0), -1)
            frames.append(frame)
        
        # Process frames
        for i, frame in enumerate(frames):
            tracked_objects, processed_frame = detector.update(frame)
            print(f"Frame {i}: {len(tracked_objects)} objects tracked")
            
            if tracked_objects:
                for color, obj in tracked_objects.items():
                    print(f"  {color}: pos=({obj.position[0]:.1f}, {obj.position[1]:.1f}), "
                          f"vel=({obj.filtered_velocity[0]:.1f}, {obj.filtered_velocity[1]:.1f}), "
                          f"angle={obj.filtered_angle:.1f}°, conf={obj.confidence:.2f}")
        
        print("Optical flow tracking test completed.")
        return detector
        
    except Exception as e:
        print(f"Optical flow tracking test failed: {e}")
        return None

def test_detector_methods():
    """Test detector utility methods."""
    print("\nTesting detector utility methods...")
    
    detector = Detector(tracking_method="frame", fps=30.0, min_contour_area=150)
    
    print(f"Tracking method: {detector.get_tracking_method()}")
    print(f"FPS: {detector.get_fps()}")
    print(f"GPU enabled: {detector.is_gpu_enabled()}")
    print(f"Min contour area: {detector.get_min_contour_area()}")
    
    # Test setting parameters
    detector.set_min_contour_area(200)
    print(f"Updated min contour area: {detector.get_min_contour_area()}")
    
    print("Detector utility methods test completed.")

def main():
    """Run all tests."""
    print("=== Testing Merged Detector Class ===\n")
    
    try:
        # Test color detection
        test_color_detection()
        
        # Test frame tracking
        test_frame_tracking()
        
        # Test optical flow tracking
        test_optical_flow_tracking()
        
        # Test detector methods
        test_detector_methods()
        
        print("\n=== All tests completed successfully! ===")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
