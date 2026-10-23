import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d


def visualize_pointcloud(
    points,
    colors,
    save_path="bev_output.png",
    resolution=0.1,
    point_size=2,
    z_min=-2.0,
    z_max=3.0,
    bev_radius=50.0,
):
    """Generate a 360-degree BEV image and an interactive Open3D view."""

    # 1. Validate inputs
    points = np.asarray(points)
    colors = np.asarray(colors)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")

    if colors.ndim != 2 or colors.shape != points.shape:
        raise ValueError(
            f"colors must have the same (N, 3) shape as points; "
            f"got {colors.shape} and {points.shape}"
        )

    if len(points) == 0:
        print("[visualize] No points received.")
        return

    if resolution <= 0 or bev_radius <= 0:
        raise ValueError("resolution and bev_radius must be positive.")

    # Remove invalid points and their corresponding colors
    valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points = points[valid].astype(np.float64)
    colors = colors[valid].astype(np.float64)

    if len(points) == 0:
        print("[visualize] No finite points available.")
        return

    # 2. Normalize colors to 0-255 if they are already in 0-1 range
    if colors.max() <= 1.0 and colors.min() >= 0.0:
        colors *= 255.0

    colors = np.clip(colors, 0, 255).astype(np.uint8)

    # Assumption: input colors are BGR, as in the original code.
    # Open3D requires RGB values normalized to [0, 1].

    # 3. Filter by height
    z_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    pts = points[z_mask]
    cols = colors[z_mask]

    print(f"[visualize] After Z filter: {len(pts)} / {len(points)}")

    if len(pts) == 0:
        print("[visualize] No points in the requested Z range.")
        print("[visualize] Using all finite points instead.")
        pts, cols = points, colors

    # 4. Filter to the BEV square
    xy_mask = (
        (np.abs(pts[:, 0]) <= bev_radius)
        & (np.abs(pts[:, 1]) <= bev_radius)
    )
    pts = pts[xy_mask]
    cols = cols[xy_mask]

    print(f"[visualize] Points inside BEV bounds: {len(pts)}")

    if len(pts) == 0:
        print("[visualize] No points inside BEV bounds.")
        return

    print(
        f"[visualize] X: {pts[:, 0].min():.1f} to {pts[:, 0].max():.1f}"
    )
    print(
        f"[visualize] Y: {pts[:, 1].min():.1f} to {pts[:, 1].max():.1f}"
    )
    print(
        f"[visualize] Z: {pts[:, 2].min():.1f} to {pts[:, 2].max():.1f}"
    )

    # ==========================================================
    # PART A: Bird's-eye-view PNG
    # ==========================================================
    W = int(round(2 * bev_radius / resolution))
    H = W

    # Keep the image in BGR format while drawing with OpenCV.
    bev_bgr = np.zeros((H, W, 3), dtype=np.uint8)

    # X increases to the right; positive Y appears toward the top.
    px = np.floor((pts[:, 0] + bev_radius) / resolution).astype(int)
    py = np.floor((bev_radius - pts[:, 1]) / resolution).astype(int)

    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)

    # Draw higher points first, so lower points are painted last.
    order = np.argsort(-pts[:, 2])
    z_sorted = pts[order, 2]
    cols_sorted = cols[order]
    px_sorted = px[order]
    py_sorted = py[order]

    z_range = np.ptp(z_sorted)
    if z_range > 0:
        brightness = 1.0 - 0.35 * (
            (z_sorted - z_sorted.min()) / z_range
        )
    else:
        brightness = np.ones(len(z_sorted))

    cols_bright = np.clip(
        cols_sorted.astype(np.float32) * brightness[:, None],
        0,
        255,
    ).astype(np.uint8)

    radius_px = max(1, int(point_size))

    for x, y, color in zip(px_sorted, py_sorted, cols_bright):
        cv2.circle(
            bev_bgr,
            (int(x), int(y)),
            radius_px,
            tuple(int(c) for c in color),
            -1,
        )

    # Grid lines every 10 metres
    grid_color = (40, 40, 40)

    for distance in range(-int(bev_radius), int(bev_radius) + 1, 10):
        gx = int(round((distance + bev_radius) / resolution))
        gy = int(round((bev_radius - distance) / resolution))

        if 0 <= gx < W:
            cv2.line(bev_bgr, (gx, 0), (gx, H - 1), grid_color, 1)

        if 0 <= gy < H:
            cv2.line(bev_bgr, (0, gy), (W - 1, gy), grid_color, 1)

    # Ego vehicle at the origin
    ego_px = int(round(bev_radius / resolution))
    ego_py = ego_px

    half_car_length = int(round(4.5 / (2 * resolution)))
    half_car_width = int(round(2.0 / (2 * resolution)))

    cv2.rectangle(
        bev_bgr,
        (ego_px - half_car_width, ego_py - half_car_length),
        (ego_px + half_car_width, ego_py + half_car_length),
        (0, 255, 0),
        -1,
    )

    # Forward arrow: assumes positive X is forward.
    cv2.arrowedLine(
        bev_bgr,
        (ego_px, ego_py),
        (min(W - 1, ego_px + half_car_length + 15), ego_py),
        (0, 220, 0),
        2,
        tipLength=0.4,
    )

    # Distance rings
    for distance in [10, 20, 30, 40, 50]:
        if distance > bev_radius:
            continue

        ring_radius = int(round(distance / resolution))
        cv2.circle(
            bev_bgr,
            (ego_px, ego_py),
            ring_radius,
            (55, 55, 55),
            1,
        )

        label_x = min(W - 45, ego_px + ring_radius + 3)
        cv2.putText(
            bev_bgr,
            f"{distance}m",
            (label_x, ego_py - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
        )

    # Convert BGR to RGB for Matplotlib
    bev_rgb = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(12, 12), facecolor="#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    ax.imshow(
        bev_rgb,
        origin="upper",
        extent=[-bev_radius, bev_radius,-bev_radius, bev_radius,],
        interpolation="nearest",
    )

    ax.set_xlabel("X [m]", color="white")
    ax.set_ylabel("Y [m]", color="white")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    ax.set_title(
        "360-degree Bird's-eye View — Point Cloud Fusion",
        color="white",
        fontsize=13,
    )

    ax.legend(
        handles=[
            patches.Patch(color="#00ff00", label="Ego vehicle"),
            patches.Patch(color="#aaaaaa", label="Colored scene points"),
        ],
        loc="lower right",
        facecolor="#1a1a1a",
        edgecolor="#555555",
        labelcolor="white",
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    print(f"[visualize] BEV image saved: {os.path.abspath(save_path)}")

    # ==========================================================
    # PART B: Interactive Open3D point cloud
    # ==========================================================
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Convert BGR [0,255] to RGB [0,1]
    rgb_colors = cols[:, ::-1].astype(np.float64) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)

    # Optional outlier removal
    if len(pts) >= 21:
        pcd, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0,
        )
        print(
            f"[visualize] Points after outlier removal: {len(pcd.points)}"
        )

    if len(pcd.points) == 0:
        print("[visualize] Open3D cloud is empty after filtering.")
        return

    # Do not overwrite pcd.points or pcd.colors after this point.
    vis = o3d.visualization.Visualizer()

    created = vis.create_window(
        window_name="3D Point Cloud — Rotate, Pan and Zoom",
        width=1280,
        height=960,
        visible=True,
    )

    if not created:
        print(
            "[visualize] Open3D window could not be created. "
            "Check GUI/display support."
        )
        return

    vis.add_geometry(pcd)

    render = vis.get_render_option()
    render.point_size = 2.0
    render.background_color = np.array([0.05, 0.05, 0.05])
    render.show_coordinate_frame = True

    # Initialize the camera using the actual filtered cloud
    view = vis.get_view_control()
    pts_np = np.asarray(pcd.points)
    center = pts_np.mean(axis=0)

    x_span = np.ptp(pts_np[:, 0])
    y_span = np.ptp(pts_np[:, 1])
    z_span = np.ptp(pts_np[:, 2])

    scale = max(x_span, y_span, z_span, 1.0)

    view.set_lookat(center)
    view.set_front([0, 0, -1])  # Look downward
    view.set_up([0, -1, 0])     # Orient the image axes
    view.set_zoom(0.7)

    print("[visualize] Open3D window opened. Press Q to close.")

    try:
        vis.run()
    finally:
        vis.destroy_window()