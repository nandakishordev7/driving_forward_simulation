import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d
import cv2
import os


def visualize_pointcloud(points, colors, save_path="bev_output.png",
                         resolution=0.1, point_size=2.0,
                         z_min=-2.0, z_max=3.0,
                         bev_radius=50.0):
    """Visualize a 360° bird's eye view (BEV) of 3D points with RGB colors.
    """

    if len(points) == 0:
        print("[visualize] No points received.")
        return

    # ── 1. Z-height clamp  (removes sky projections and deep underground) ─────
    z_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    pts    = points[z_mask]
    cols   = colors[z_mask]
    print(f"[visualize] After Z clamp [{z_min}, {z_max}]: {len(pts)} / {len(points)} points kept")

    if len(pts) == 0:
        print("[visualize] Z clamp removed all points — widening to full Z range.")
        pts, cols = points, colors

    # ── 2. Radial clamp  (keeps only points within bev_radius of ego) ─────────
    r_mask = (np.abs(pts[:, 0]) <= bev_radius) & (np.abs(pts[:, 1]) <= bev_radius)
    pts    = pts[r_mask]
    cols   = cols[r_mask]
    print(f"[visualize] After radial clamp (±{bev_radius}m): {len(pts)} points")

    print(f"[visualize] X: {pts[:, 0].min():.1f} → {pts[:, 0].max():.1f}")
    print(f"[visualize] Y: {pts[:, 1].min():.1f} → {pts[:, 1].max():.1f}")
    print(f"[visualize] Z: {pts[:, 2].min():.1f} → {pts[:, 2].max():.1f}")

    # ════════════════════════════════════════════════════════════════════════
    #  Part A — BEV PNG
    # ════════════════════════════════════════════════════════════════════════
    x_range = (-bev_radius, bev_radius)
    y_range = (-bev_radius, bev_radius)
    W = int((x_range[1] - x_range[0]) / resolution)
    H = int((y_range[1] - y_range[0]) / resolution)
    bev_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Map XY → pixel  (X forward = right on image, Y left = up on image)
    px = ((pts[:, 0] - x_range[0]) / resolution).astype(int)
    py = ((y_range[1] - pts[:, 1]) / resolution).astype(int)
    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)

    # Sort: paint higher Z first so ground-level points end up on top
    z_vals    = pts[:, 2]
    order     = np.argsort(-z_vals)
    px, py    = px[order], py[order]
    rgb_cols  = cols[order][:, ::-1].astype(np.float32)  # BGR → RGB

    # Brightness: closer to ground = brighter
    z_sorted  = z_vals[order]
    z_rng     = z_sorted.max() - z_sorted.min()
    if z_rng > 0:
        brightness = 1.0 - 0.35 * (z_sorted - z_sorted.min()) / z_rng
    else:
        brightness = np.ones(len(z_sorted))
    rgb_cols = np.clip(rgb_cols * brightness[:, None], 0, 255).astype(np.uint8)

    r = max(1, int(point_size))
    for i in range(len(px)):
        cv2.circle(bev_rgb, (int(px[i]), int(py[i])), r, rgb_cols[i].tolist(), -1)

    # Grid lines every 10 m
    gc = (40, 40, 40)
    for x_m in range(int(x_range[0]), int(x_range[1]) + 1, 10):
        gx = int((x_m - x_range[0]) / resolution)
        if 0 <= gx < W:
            cv2.line(bev_rgb, (gx, 0), (gx, H - 1), gc, 1)
    for y_m in range(int(y_range[0]), int(y_range[1]) + 1, 10):
        gy = int((y_range[1] - y_m) / resolution)
        if 0 <= gy < H:
            cv2.line(bev_rgb, (0, gy), (W - 1, gy), gc, 1)

    # Ego at (0, 0) — centre of canvas
    ego_px = int((0 - x_range[0]) / resolution)
    ego_py = int((y_range[1] - 0) / resolution)

    car_l = int(4.5 / resolution / 2)
    car_w = int(2.0 / resolution / 2)
    cv2.rectangle(bev_rgb,
                  (ego_px - car_w, ego_py - car_l),
                  (ego_px + car_w, ego_py + car_l),
                  (0, 255, 0), -1)
    cv2.arrowedLine(bev_rgb,
                    (ego_px, ego_py),
                    (ego_px, max(0, ego_py - car_l - 15)),
                    (0, 220, 0), 3, tipLength=0.4)

    for r_m in [10, 20, 30, 40, 50]:
        cv2.circle(bev_rgb, (ego_px, ego_py), int(r_m / resolution), (55, 55, 55), 1)
        cv2.putText(bev_rgb, f"{r_m}m",
                    (ego_px + int(r_m / resolution) + 3, ego_py - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)

    # Matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    ax.imshow(bev_rgb, origin='upper',
              extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
    ax.set_xlabel("X  (forward →)  [m]", color='white', fontsize=11)
    ax.set_ylabel("Y  (← left  |  right →)  [m]", color='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.set_title("Full 360° Bird's Eye View — nuScenes Surround Camera Fusion",
                 color='white', fontsize=13, pad=12)
    legend_items = [
        patches.Patch(color='#00ff00', label='Ego vehicle (forward = up)'),
        patches.Patch(color='#aaaaaa', label='Scene points — image color'),
    ]
    ax.legend(handles=legend_items, loc='lower right',
              facecolor='#1a1a1a', edgecolor='#555', labelcolor='white', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[visualize] BEV PNG saved → {os.path.abspath(save_path)}")

    # ════════════════════════════════════════════════════════════════════════
    #  Part B — Interactive Open3D window (starts top-down, free to rotate)
    # ════════════════════════════════════════════════════════════════════════
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols[:, ::-1] / 255.0)  # BGR→RGB

<<<<<<< HEAD
    # Light outlier removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
=======
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = colors / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd.estimate_normals()
>>>>>>> 775816f4aae9684e557fc1a324901b78ec3ffd26

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="360° BEV — Left drag: rotate | Right drag: pan | Scroll: zoom",
        width=1280, height=960
    )
    vis.add_geometry(pcd)

    ro = vis.get_render_option()
    ro.point_size            = 2.0
    ro.background_color      = np.array([0.05, 0.05, 0.05])
    ro.show_coordinate_frame = True

<<<<<<< HEAD
    # Set camera to top-down BEV orientation
    ctr     = vis.get_view_control()
    pts_np  = np.asarray(pcd.points)
    centre  = pts_np.mean(axis=0)
    x_span  = pts_np[:, 0].max() - pts_np[:, 0].min()
    y_span  = pts_np[:, 1].max() - pts_np[:, 1].min()
    height  = max(x_span, y_span) * 1.1
=======
    render_option.point_size = 7.0
>>>>>>> 775816f4aae9684e557fc1a324901b78ec3ffd26

    cam_params = o3d.camera.PinholeCameraParameters()
    intr       = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(width=1280, height=960,
                        fx=800, fy=800, cx=640, cy=480)
    cam_params.intrinsic = intr

    R    = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    t    = centre.copy(); t[2] += height
    extr = np.eye(4); extr[:3, :3] = R; extr[:3, 3] = -R @ t
    cam_params.extrinsic = extr
    ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

    print("[visualize] Interactive window open. Drag to explore. Press Q to quit.\n")
    vis.run()
    vis.destroy_window()
