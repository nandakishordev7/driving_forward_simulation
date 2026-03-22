import numpy as np
import cv2

import numpy as np
import cv2

def depth_to_pointcloud(depth, image):

    depth = cv2.GaussianBlur(depth, (5, 5), 0)

    h, w = depth.shape

    fx, fy = 500, 500
    cx, cy = w / 2, h / 2

    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    z = depth * 0.02  

    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy

    points = np.stack((X, Y, z), axis=-1).reshape(-1, 3)
    colors = image.reshape(-1, 3)

    mask = z.reshape(-1) > 0
    points = points[mask]
    colors = colors[mask]

    return points, colors

    return np.array(points), np.array(colors)
