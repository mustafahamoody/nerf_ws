import numpy as np
import torch
import cv2 
import json
import time

from nav.mat_utils import vec_to_rot_matrix, ...

def find_points_of_interest(image_rgb, render=False)
    # Converts image to grayscale.
    img = np.copy(image_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2Gray)

    # Use Scale-Invariant Feature Transform (SIFT) to detect keypoints
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)

    if render:
        # Draw keypoints on image
        feature_img = cv2.drawKeypoints(img, keypoints, None)
        cv2.imshow("Detected Keypoint Features", feature_img)  #  Display image
    else:
        feature_img = None


    # Get coordiantes of keypoints on the image as a numpy (int) array
    keypoint_coords = [tuple(keypoint.pt) for keypoint in keypoints]
    keypoint_coords = np.array(keypoint_coords).astype(int)
    
    # Remove duplicate keypoints
    keypoint_coords = np.unique(keypoint_coords, axis=0)

    extras = {
        'feature_image': feature_img
    }

    return keypoint_coords, extras

