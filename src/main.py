"""
HomeTeam Network - AI Engineer Take-Home Project
Sports Motion Detection & Viewport Tracking

This is the main entry point for the motion detection and viewport tracking program.
"""

import os
import sys
import argparse

from src.components.frame_processor import process_video
from src.components.motion_detector import detect_motion
from src.components.viewport_tracker import track_viewport
from src.components.visualizer import visualize_results
from src.exception import CustomException
from src.logger import logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sports Motion Detection & Viewport Tracking"
    )
    parser.add_argument(
        "--video", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--fps", type=int, default=5, help="Target frames per second")
    parser.add_argument(
        "--viewport_size",
        type=str,
        default="720x480",
        help="Size of viewport in format WIDTHxHEIGHT",
    )
    return parser.parse_args()


def main():
    """Main function to run the motion detection and viewport tracking pipeline."""
    logging.info("Viewport Tracking system is Initiated")
    try:
        # Parse arguments
        args = parse_args()
        
        # Parse viewport size
        viewport_width, viewport_height = map(int, args.viewport_size.split("x"))
        viewport_size = (viewport_width, viewport_height)
        logging.info(f"viewport_size received of format : {viewport_size}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)

        logging.info(f"Processing video: {args.video}")
        logging.info(f"Target frame rate received: {args.fps}") 

        # Step 1: Extract frames from video
        frames = process_video(args.video, args.fps)
        logging.info(f"Extracted {len(frames)} frames")

        # Step 2: Detect motion in frames
        logging.info("MOTION DETECTION STARTED")
        motion_results = []
        for i, frame in enumerate(frames):
            logging.info(f"Processing frame {i + 1}/{len(frames)} in motion_detector.py")

            # Pass the entire frames list and the current index to detect_motion
            motion_boxes = detect_motion(frames, i)
            motion_results.append(motion_boxes)
            logging.info(f" Motion Detection for frame {i + 1}/{len(frames)} completed")
        logging.info("MOTION DETECTION COMPLETED AND MOTION BOUNDING BOXES EXTRACTED")
        
        # Step 3: Track viewport based on motion detection
        viewport_positions = track_viewport(frames, motion_results, viewport_size)

        # Step 4: Visualize and save results
        visualize_results(
            frames, motion_results, viewport_positions, viewport_size, args.output,args.fps
        )
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    main()
    
#python src/main.py --video "data\sample_video_clip.mp4" --output "output" --fps 5  --viewport_size "720x480"

#conda create -p venv_viewport python==3.10
#conda activate venv_viewport/
#pip install -r requirements.txt
#python src/main.py --video "data\sample_video_clip.mp4" --output "output" --fps 5  --viewport_size "720x480"

