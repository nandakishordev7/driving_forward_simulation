import open3d as o3d
import numpy as np

def visualize_pointcloud(points, colors):

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    colors = colors / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd.estimate_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcd)

    render_option = vis.get_render_option()

    render_option.point_size = 7.0

    render_option.background_color = np.array([0, 0, 0])

    vis.run()
    vis.destroy_window()
