import os
import csv
import cv2
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TrackingData:
    timestamp: float
    color: str
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    fps: float
    processing_time: float

class DataLogger:
    def __init__(self, output_dir: str, method: str, device: str):
        self.output_dir = output_dir
        self.method = method
        self.device = device
        self.start_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CSV file
        self.csv_path = os.path.join(output_dir, 'tracking_data.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'color', 'x', 'y', 'vx', 'vy', 'angle',
            'fps', 'processing_time', 'method', 'device'
        ])
        
        # Initialize log file
        self.log_path = os.path.join(output_dir, 'run_log.txt')
        self.log_file = open(self.log_path, 'w')
        self._write_log_header()
        
        # Video writers
        self.original_writer = None
        self.processed_writer = None
    
    def _write_log_header(self):
        """Write header information to the log file."""
        self.log_file.write(f"Motion Tracking Run Log\n")
        self.log_file.write(f"=====================\n")
        self.log_file.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(f"Tracking Method: {self.method}\n")
        self.log_file.write(f"Processing Device: {self.device}\n")
        self.log_file.write(f"\nDetailed Log:\n")
        self.log_file.write(f"-------------\n\n")
    
    def initialize_video_writers(self, fps: float, frame_size: tuple):
        """Initialize video writers for original and processed frames."""
        original_path = os.path.join(self.output_dir, 'original.mp4')
        processed_path = os.path.join(self.output_dir, 'processed.mp4')
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.original_writer = cv2.VideoWriter(original_path, fourcc, fps, frame_size)
        self.processed_writer = cv2.VideoWriter(processed_path, fourcc, fps, frame_size)
    
    def log_frame(self, frame: np.ndarray, processed_frame: np.ndarray,
                 tracking_data: Dict[str, TrackingData], processing_time: float):
        """Log frame data and save videos."""
        # Save frames to video
        if self.original_writer is not None:
            self.original_writer.write(frame)
            self.processed_writer.write(processed_frame)
        
        # Log tracking data for each color
        for color, data in tracking_data.items():
            self.csv_writer.writerow([
                data.timestamp,
                color,
                data.x,
                data.y,
                data.vx,
                data.vy,
                data.angle,
                data.fps,
                processing_time,
                self.method,
                self.device
            ])
        
        # Log frame processing details
        self.log_file.write(
            f"Frame at {time.time() - self.start_time:.3f}s: "
            f"Processing time: {processing_time*1000:.1f}ms, "
            f"Detected colors: {list(tracking_data.keys())}\n"
        )
    
    def log_error(self, error_msg: str):
        """Log error messages."""
        self.log_file.write(f"ERROR: {error_msg}\n")
    
    def close(self):
        """Close all file handles and video writers."""
        if self.csv_file:
            self.csv_file.close()
        
        if self.log_file:
            duration = time.time() - self.start_time
            self.log_file.write(f"\nRun Summary:\n")
            self.log_file.write(f"Total Duration: {duration:.1f} seconds\n")
            self.log_file.close()
        
        if self.original_writer:
            self.original_writer.release()
        
        if self.processed_writer:
            self.processed_writer.release() 