import cv2
import sys

from src.exception import CustomException
from src.logger import logging

def process_video(video_path, target_fps=25, resize_dim=(1280, 720)):
    """
    Extract frames from a video at a specified frame rate.

    Args:
        video_path: Path to the video file
        target_fps: Target frames per second to extract
        resize_dim: Dimensions to resize frames to (width, height)

    Returns:
        List of extracted frames
    """
    logging.info("frame_processor.py entered and frame extraction initiated")
    
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame interval for the target FPS
        frame_interval = max(1, int(round(original_fps / target_fps)))

        frames = []
        frame_index = 0
        while True:
            # Read frames from the video capture object
            ret, frame = cap.read()
            if not ret:
                break
            # only keeping frames at the frame interval to achieve target_fps & resizing frames to resize_dim
            if frame_index % (frame_interval) == 0: 
                frame = cv2.resize(src=frame,dsize=resize_dim,interpolation=cv2.INTER_AREA) # cv2.INTER_AREA is good for shrinking images.
                frames.append(frame)        
            frame_index += 1
        
        cap.release()
        logging.info("Frame Extraction Completed")
        
        return frames
    except Exception as e:
        raise CustomException(e,sys)