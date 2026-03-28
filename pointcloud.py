import numpy as np
import cv2


def depth_to_pointcloud(depth, image, fx=500, fy=500, cx=None, cy=None):
    """
    Convert a depth map to a 3D point cloud using the pinhole camera model.

    Args:
        depth:  2D numpy array of depth values (from MiDaS, already scaled to metric)
        image:  Corresponding BGR image (same H x W)
        fx, fy: Focal lengths (use real intrinsics from nuScenes calibration)
        cx, cy: Principal point (defaults to image center if not provided)

    Returns:
        points: (N, 3) array of 3D points in camera space
        colors: (N, 3) array of corresponding BGR colors
    """
    # Smooth depth to reduce noise
    depth = cv2.GaussianBlur(depth.astype(np.float32), (5, 5), 0)

    h, w = depth.shape

    # Default principal point to image center if not provided
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    # Build pixel grid
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    z = depth  # depth is already metric (scaled externally)

    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy

    points = np.stack((X, Y, z), axis=-1).reshape(-1, 3)
    colors = image.reshape(-1, 3)

    # Remove invalid (zero depth) points
    mask = z.reshape(-1) > 0
    points = points[mask]
    colors = colors[mask]

    return points, colors