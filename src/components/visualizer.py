import os
import cv2
import sys

from src.exception import CustomException
from src.logger import logging

def visualize_results(frames, motion_results, viewport_positions, viewport_size, output_dir,fps):
    """
    Create visualization of motion detection and viewport tracking results.

    Args:
        frames: List of video frames
        motion_results: List of motion detection results for each frame
        viewport_positions: List of viewport center positions for each frame
        viewport_size: Tuple (width, height) of the viewport
        output_dir: Directory to save visualization results
        fps : Frame per second
    """
    logging.info("visualizer.py entered")
    try:
        # Create output directory for frames
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        viewport_dir = os.path.join(output_dir, "viewport")
        os.makedirs(viewport_dir, exist_ok=True)

        # Get dimensions for the output video
        height, width = frames[0].shape[:2]

        # Create video writers
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = os.path.join(output_dir, "motion_detection_kmeans4_5fps_hybridsmoothing2.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        viewport_video_path = os.path.join(output_dir, "viewport_tracking_kmeans4_5fps_hybridsmoothing2.mp4")
        vp_width, vp_height = viewport_size
        viewport_writer = cv2.VideoWriter(
            viewport_video_path, fourcc, fps, (vp_width, vp_height)
        )

        for i, frame in enumerate(frames):
            frame_copy = frame.copy()
            #Draw bounding boxes around motion regions
            motion_boxes=motion_results[i]
            for box in motion_boxes:
                cv2.rectangle(frame_copy,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0),2)
            cv2.putText(img=frame_copy,text=f"Frame: {i+1}", org=(10, 30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.8,color=(0, 255, 0),thickness=2,lineType=cv2.LINE_AA)
            # Draw view port rectangle 
            (x,y) = viewport_positions[i]
            x1 = int(x - vp_width/2)
            y1 = int(y - vp_height/2)
            x2 = int(x + vp_width/2)
            y2 = int(y + vp_height/2)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # extract the viewport content
            viewport_frame = frame_copy[y1:y2, x1:x2]
            cv2.putText(img=viewport_frame,text=f"Frame: {i+1}", org=(10, 30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.8,color=(0, 255, 0),thickness=2,lineType=cv2.LINE_AA)
            
            # saving images
            filename = os.path.join(frames_dir, f"frame_{i+1:04d}.png")
            cv2.imwrite(filename, frame_copy)
            
            filename = os.path.join(viewport_dir, f"frame_{i+1:04d}.png")
            cv2.imwrite(filename, viewport_frame)      
            
            # writing frames to video writers
            video_writer.write(frame_copy)
            viewport_writer.write(viewport_frame)
            
        video_writer.release()
        viewport_writer.release()

        print(f"Visualization saved to {video_path}")
        print(f"Viewport video saved to {viewport_video_path}")
        print(f"Individual frames saved to {frames_dir} and {viewport_dir}")
        
        logging.info(f"Visualization saved to {video_path}")
        logging.info(f"Viewport video saved to {viewport_video_path}")
        logging.info(f"Individual frames saved to {frames_dir} and {viewport_dir}")
    
    except Exception as e:
        raise CustomException(e,sys)