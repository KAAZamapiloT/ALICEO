# src/align.py

import cv2 as cv
import numpy as np


def align_mono_to_rgb(
    rgb: np.ndarray,
    mono: np.ndarray,
    max_features: int = 500,
    good_match_ratio: float = 0.15
) -> np.ndarray:

    if rgb.dtype != np.uint8 or mono.dtype != np.uint8:
        raise ValueError("Images must have dtype uint8")

    # Convert RGB to grayscale for feature detection
    gray_rgb = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)

    # Initialize ORB detector
    orb = cv.ORB_create(max_features)

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray_rgb, None)
    kp2, des2 = orb.detectAndCompute(mono, None)

    # Fallback if features are not found
    if des1 is None or des2 is None:
        return cv.resize(mono, (rgb.shape[1], rgb.shape[0]))

    # Match descriptors using Hamming distance
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) < 10:
        return cv.resize(mono, (rgb.shape[1], rgb.shape[0]))

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep only top matches
    num_good_matches = int(len(matches) * good_match_ratio)
    matches = matches[:num_good_matches]

    # Extract matched points
    pts_rgb = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_mono = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv.findHomography(pts_mono, pts_rgb, cv.RANSAC)

    if H is None:
        return cv.resize(mono, (rgb.shape[1], rgb.shape[0]))

    # Warp mono image to RGB frame
    aligned_mono = cv.warpPerspective(
        mono,
        H,
        (rgb.shape[1], rgb.shape[0]),
        flags=cv.INTER_LINEAR
    )

    return aligned_mono
