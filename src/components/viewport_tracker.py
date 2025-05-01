import numpy as np
from sklearn.cluster import KMeans
import cv2
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import clustering_strategy

def calculate_region_of_interest(motion_boxes, frame_shape):
    """
    Calculate the primary region of interest based on motion boxes.

    Args:
        motion_boxes: List of motion detection bounding boxes
        frame_shape: Shape of the video frame (height, width)

    Returns:
        Tuple (x, y, w, h) representing the region of interest center point and dimensions
    """
    try:
        # If no motion is detected, use the center of the frame
        if not motion_boxes:
            height, width = frame_shape[:2]
            return (width // 2, height // 2, 0, 0)

        # Use clustering strategy for determining main area of interest: In this case kmeans clustering is used
        x, y, w, h = clustering_strategy(motion_boxes)

        return (x,y,w,h)
    except Exception as e:
        raise CustomException(e,sys)

    
def track_viewport(frames, motion_results, viewport_size,
                 ema_factor=0.15, kalman_q=0.01, kalman_r=0.5,
                 min_motion_thresh=0.02):
    """
    Track viewport position across frames with: Hybridisation of Kalman Filter & Exponential Moving Average.

    Args:
        frames: List of video frames
        motion_results: List of motion detection results for each frame
        viewport_size: Tuple (width, height) of the viewport
        ema_factor : Exponential Moving Average Factor
        kalman_q : Variance of Process noise
        kalman_r : Variance of Measurement Noise
        min_motion_thresh : Motion Threshold for Heavy or Light smoothing

    Returns:
        List of viewport positions for each frame as (x, y) center coordinates
    """
    logging.info("viewport_tracker.py entered")
    try:
        # Initialize frame's(height, width) , viewport's(height and width) & Initial state for EMA
        h, w = frames[0].shape[:2]
        vw, vh = viewport_size
        prev_ema = np.array([w/2, h/2])
        
        # Kalman Setup
        kalman = cv2.KalmanFilter(4, 2)
        
        # Initialize all required matrices for Kalman Filter : Transition(Model) matrix & Measurement Matrix
        kalman.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], dtype=np.float32)
        kalman.transitionMatrix = np.array([
            [1,0,0.3,0],
            [0,1,0,0.3],
            [0,0,0.8,0],
            [0,0,0,0.8]
        ], dtype=np.float32)
        
        # Initialize state vectors for Kalman Filter : Initial State ([x,y,del_x, del_y])
        kalman.statePre = np.zeros((4,1), dtype=np.float32)
        kalman.statePost = np.zeros((4,1), dtype=np.float32)
        kalman.statePre[0,0] = w/2  # Initial x position
        kalman.statePre[1,0] = h/2  # Initial y position
        
        # Initialize covariance matrices for Kalamn Filter: Process Noise Covariance & Measurement Noise Covariance
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * kalman_q
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * kalman_r
        
        # Initialize 1.Predicted Covariance 2. Corrected Covariance for state vector
        kalman.errorCovPre = np.eye(4, dtype=np.float32)
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        logging.info("Viewport Position estimation INITIATED")
        viewport_positions = []
        motion_intensity = 0
        
        for i, boxes in enumerate(motion_results):
            # Motion Analysis
            motion_area = sum(w*h for (_,_,w,h) in boxes)
            motion_intensity = 0.9*motion_intensity + 0.1*(motion_area/(w*h)) # Normalised motion with smoothing
            
            # Get Target ROI (with  clustering strategy)
            roi = calculate_region_of_interest(boxes, (h, w))
            target = np.array([roi[0]+roi[2]/2, roi[1]+roi[3]/2]) # calculate centre from roi
            logging.info(f"Target ROI for frame : {i + 1}/{len(frames)} calculated")
            
            # Adaptive EMA-Kalman Blending 
            if motion_intensity < min_motion_thresh:
                kalman.processNoiseCov = np.eye(4, dtype=np.float32) * kalman_q * 0.1 # Heavy Smoothing
                blend = 0.1 # trust Kalman more
            else:
                kalman.processNoiseCov = np.eye(4, dtype=np.float32) * kalman_q * (1 + motion_intensity*5) # Increased Responsivesness
                blend = 0.3 # trust EMA more
            
            # EMA Update
            prev_ema = ema_factor * target + (1-ema_factor) * prev_ema
            
            # Kalman Update
            kalman.predict()
            measurement = np.array([[np.float32(prev_ema[0])], [np.float32(prev_ema[1])]])
            corrected = kalman.correct(measurement)
            
            # Hybrid Output : Blending Kalman and EMA by weighted average
            kalman_pos = np.array([corrected[0,0], corrected[1,0]])
            final_pos = blend * prev_ema + (1-blend) * kalman_pos
            
            # Boundary Check : Ensure the viewport stays within the frame boundaries
            final_pos[0] = np.clip(final_pos[0], vw/2, w-vw/2)
            final_pos[1] = np.clip(final_pos[1], vh/2, h-vh/2)
            
            viewport_positions.append(tuple(final_pos))
        logging.info("Viewport positions Estimation COMPLETED!")
        return viewport_positions
    except Exception as e:
        raise CustomException(e,sys)