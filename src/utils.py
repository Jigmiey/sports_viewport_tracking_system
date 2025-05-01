import numpy as np
from sklearn.cluster import KMeans
import cv2
import sys

from src.exception import CustomException
from src.logger import logging

def clustering_strategy(bounding_boxes):
    """
    Calculate the primary region of interest based on bounding(motion) boxes using k-means clustering.

    Args:
        bounding_boxes: List of motion detection bounding boxes
        
    Returns:
        Tuple (x, y, w, h) representing the region of interest center point and dimensions
    """
    logging.info("clustering_strategy is getting accessed from the utils.py")
    
    try:
        # Compute centers of the bounding boxes
        centers = np.array([
            (x + w/2, y + h/2) for (x, y, w, h) in bounding_boxes
        ])

        # KMeans clustering : if the number of bounding boxes is less than number of clusters, then keep n_clusters = 1
        n_clusters=3
        if len(centers) < n_clusters:
            n_clusters = 1
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++',n_init=20,max_iter=300,random_state=42)
        labels = kmeans.fit_predict(centers)

        # Select cluster (here, the largest cluster)
        unique, counts = np.unique(labels, return_counts=True)
        largest_cluster = unique[np.argmax(counts)]
        
        # Get bounding boxes in the largest cluster
        selected_boxes = [
            bbox for bbox, label in zip(bounding_boxes, labels) if label == largest_cluster
        ]

        # Compute the final bounding box from the selected boxes from the larget cluster
        x_min = min(x for (x, y, w, h) in selected_boxes)
        y_min = min(y for (x, y, w, h) in selected_boxes)
        x_max = max(x + w for (x, y, w, h) in selected_boxes)
        y_max = max(y + h for (x, y, w, h) in selected_boxes)

        wfinal = x_max - x_min
        hfinal = y_max - y_min
        
        # Give some padding to the region of interest so that region of interest is not too tight
        padding_ratio =0.1 
        
        # add padding
        pad_w = int(round(wfinal * padding_ratio))
        pad_h = int(round(hfinal * padding_ratio))
        xfinal = x_min - pad_w//2
        yfinal = y_min - pad_h//2
        wfinal = wfinal + pad_w
        hfinal = hfinal + pad_h
        logging.info("Clustering Completed")
    except Exception as e:
        raise CustomException(e,sys)
    

    return (xfinal, yfinal, wfinal, hfinal)