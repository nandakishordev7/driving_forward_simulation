import cv2
import numpy as np

from nu_scene_loader import NuScenesLoader
from depth import DepthEstimator
from pointcloud import depth_to_pointcloud
from visualize import visualize_pointcloud
from transform import transform_points

def main():

    # Load dataset
    loader = NuScenesLoader(dataroot="C://Projects//driving_forward//nu_scenes")
    nusc = loader.nusc

    sample = nusc.sample[0]
    images = loader.get_sample_images(sample['token'])

    depth_model = DepthEstimator()

    all_points = []
    all_colors = []

    for cam_name, image in images:
        print(f"Processing {cam_name}")

        # Get sample_data
        sample_data = nusc.get('sample_data', sample['data'][cam_name])

        # Calibration
        calib = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])

        # Depth
        _, _, depth_vis = depth_model.predict_from_array(image)

        # Convert to 3D (camera space)
        points, colors = depth_to_pointcloud(depth_vis, image)

        # Downsample (important)
        points = points[::2]
        colors = colors[::10]

        # 🔥 STEP 1: Camera → Ego
        points = transform_points(points, calib['rotation'], calib['translation'])

        # 🔥 STEP 2: Ego → Global
        points = transform_points(points, ego_pose['rotation'], ego_pose['translation'])

        all_points.append(points)
        all_colors.append(colors)

    # Merge all cameras
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    print(f"Total aligned points: {len(all_points)}")

    # Visualize
    visualize_pointcloud(all_points, all_colors)

if __name__ == "__main__":
    main()