import cv2
import sys

from src.exception import CustomException
from src.logger import logging

def detect_motion(frames, frame_idx, threshold=25, min_area=100):
    """
    Detect motion in the current frame by comparing with previous frame.

    Args:
        frames: List of video frames
        frame_idx: Index of the current frame
        threshold: Threshold for frame difference detection
        min_area: Minimum contour area to consider

    Returns:
        List of bounding boxes for detected motion regions
    """
    logging.info("motion_detector.py entered and motion detection initiaded")
    
    try:
        # We need at least 2 frames to detect motion
        if frame_idx < 1 or frame_idx >= len(frames):
            return []

        # Get current and previous frame
        current_frame = frames[frame_idx]
        prev_frame = frames[frame_idx - 1]

        # Convert frames to grayscale
        current_frame_gray = cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
        prev_frame_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise from the current and previous frame
        current_frame_gray_blurred = cv2.GaussianBlur(current_frame_gray,(5,5),0)
        prev_frame_gray_blurred = cv2.GaussianBlur(prev_frame_gray,(5,5),0)
        
        # Calculate absolute difference between previous frame and current frame
        diff_gray = cv2.absdiff(prev_frame_gray_blurred,current_frame_gray_blurred)
        
        # Apply threshold to higlight differences 
        _, thresh = cv2.threshold(diff_gray,threshold,255,cv2.THRESH_BINARY)
        
        # Dilate the threshold image to fill in holes
        dilated = cv2.dilate(thresh,None,iterations=3)
        
        # Find countours in the thresholded image
        contours,_ = cv2.findContours(dilated, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area, ignoring countours with area less than minimum area & extract bounding box
        motion_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            (x,y,w,h) = cv2.boundingRect(contour)
            motion_boxes.append((x,y,w,h))
            
        return motion_boxes
    except Exception as e:
        raise CustomException(e,sys)
    

