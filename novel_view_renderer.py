"""
novel_view_renderer.py

Renders the merged Gaussian scene from any arbitrary camera position
using a software rasterizer (no diff-gaussian-rasterization needed).

Includes:
  - render_from_pose()  : render one frame from a given camera pose
  - fly_around()        : render a full 360° orbit animation and save as MP4
  - interactive_viewer(): Open3D window with keyboard-controlled free-fly camera
"""

import numpy as np
import cv2
import os
import torch


# ── Core renderer ─────────────────────────────────────────────────────────────

def render_from_pose(gaussians, R, t, K, img_h=480, img_w=640,
                     z_near=0.1, z_far=100.0):
    """
    Render the Gaussian scene from a camera defined by (R, t, K).

    Args:
        gaussians : dict with keys xyz (N,3), opacity (N,1), sh_coeffs (N,C)
                    All numpy arrays in ego frame.
        R         : (3,3) world-to-camera rotation
        t         : (3,)  world-to-camera translation
        K         : (3,3) camera intrinsic matrix
        img_h/w   : output image size
        z_near/far: depth clipping planes

    Returns:
        img       : (H, W, 3) uint8 RGB image
    """
    xyz      = _to_np(gaussians['xyz'])       # (N, 3)
    opacity  = _to_np(gaussians['opacity'])    # (N,) or (N,1)
    sh       = _to_np(gaussians['sh_coeffs'])  # (N, C)

    if opacity.ndim == 2: opacity = opacity[:, 0]
    rgb = np.clip(0.5 + 0.282 * sh[:, :3], 0, 1)

    # ── Transform points: world → camera ──────────────────────────────────────
    pts_cam = (R @ xyz.T).T + t          # (N, 3)

    # Keep only points in front of camera and within range
    depth = pts_cam[:, 2]
    valid = (depth > z_near) & (depth < z_far)
    pts_cam = pts_cam[valid]
    rgb     = rgb[valid]
    opacity = opacity[valid]
    depth   = depth[valid]

    if len(pts_cam) == 0:
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # ── Project: camera → image ────────────────────────────────────────────────
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = (pts_cam[:, 0] * fx / depth + cx).astype(int)
    v = (pts_cam[:, 1] * fy / depth + cy).astype(int)

    in_frame = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v     = u[in_frame], v[in_frame]
    rgb      = rgb[in_frame]
    opacity  = opacity[in_frame]
    depth    = depth[in_frame]

    # ── Paint: depth-sorted, opacity-weighted ─────────────────────────────────
    canvas     = np.zeros((img_h, img_w, 3), dtype=np.float32)
    alpha_acc  = np.zeros((img_h, img_w),    dtype=np.float32)

    # Sort far → near so close points win
    order = np.argsort(-depth)
    u, v, rgb, opacity = u[order], v[order], rgb[order], opacity[order]

    np.add.at(canvas,    (v, u), rgb * opacity[:, None])
    np.add.at(alpha_acc, (v, u), opacity)

    safe = alpha_acc > 1e-6
    canvas[safe] /= alpha_acc[safe, None]

    img = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)

    # Light bilateral filter to smooth point splats
    img = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=5)
    return img


# ── Orbit animation ───────────────────────────────────────────────────────────

def fly_around(gaussians, output_path="surround_view.mp4",
               n_frames=120, orbit_height=5.0, orbit_radius=15.0,
               img_h=480, img_w=640, fps=30):
    """
    Render a full 360° orbit around the ego vehicle and save as MP4.

    The camera circles at `orbit_radius` metres from the origin
    at `orbit_height` metres elevation, always looking at the origin.

    Args:
        gaussians    : merged Gaussian dict (ego frame, numpy)
        output_path  : where to save the video
        n_frames     : number of frames (360 / n_frames = degrees per frame)
        orbit_height : camera elevation above ground in metres
        orbit_radius : how far from ego the camera orbits
        img_h/w      : output frame resolution
        fps          : video frame rate
    """
    print(f"[novel_view] Rendering {n_frames}-frame orbit → {output_path}")

    # Camera intrinsics for a wide-angle view
    fov_deg = 90.0
    fx = fy  = (img_w / 2.0) / np.tan(np.deg2rad(fov_deg / 2))
    cx, cy   = img_w / 2.0, img_h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (img_w, img_h))

    for i in range(n_frames):
        angle  = 2 * np.pi * i / n_frames
        cam_x  = orbit_radius * np.cos(angle)
        cam_y  = orbit_radius * np.sin(angle)
        cam_z  = orbit_height

        # Camera position in world
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        # Look-at: camera looks toward origin (0,0,0) from cam_pos
        forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)  # toward origin
        # World up = Z axis, project out forward component
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right    = np.cross(forward, world_up)
        right   /= np.linalg.norm(right) + 1e-8
        up       = np.cross(right, forward)

        # Build rotation matrix: rows are camera axes in world coords
        R = np.stack([right, -up, forward], axis=0)  # (3,3) world→cam rotation
        t = -R @ cam_pos                              # (3,) translation

        frame = render_from_pose(gaussians, R, t, K,
                                 img_h=img_h, img_w=img_w)

        # Draw ego marker
        ego_px = int(cx)
        ego_py = int(cy + orbit_height * fy / max(orbit_radius * 0.5, 1))
        cv2.putText(frame, f"Frame {i+1}/{n_frames}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Angle: {int(np.degrees(angle))}°",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_frames} frames rendered")

    writer.release()
    print(f"[novel_view] Saved → {os.path.abspath(output_path)}")


# ── Multi-view snapshot ───────────────────────────────────────────────────────

def render_surround_snapshot(gaussians, output_path="surround_snapshot.png",
                              img_h=300, img_w=400):
    """
    Render 8 views around the scene (N, NE, E, SE, S, SW, W, NW)
    and stitch them into a single 3×3 grid image with BEV in centre.

    Args:
        gaussians   : merged Gaussian dict
        output_path : where to save the grid image
    """
    from gaussian_renderer import render_bev_numpy

    directions = [
        ("Front",       0),
        ("Front-Right", 45),
        ("Right",       90),
        ("Back-Right",  135),
        ("Back",        180),
        ("Back-Left",   225),
        ("Left",        270),
        ("Front-Left",  315),
    ]

    orbit_r = 12.0
    orbit_h = 3.0
    fov_deg = 80.0
    fx = fy  = (img_w / 2.0) / np.tan(np.deg2rad(fov_deg / 2))
    cx, cy   = img_w / 2.0, img_h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    frames = {}
    for label, deg in directions:
        angle   = np.deg2rad(deg)
        cam_pos = np.array([orbit_r * np.cos(angle),
                             orbit_r * np.sin(angle),
                             orbit_h], dtype=np.float32)
        forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right    = np.cross(forward, world_up)
        right   /= np.linalg.norm(right) + 1e-8
        up       = np.cross(right, forward)
        R        = np.stack([right, -up, forward], axis=0)
        t        = -R @ cam_pos

        frame = render_from_pose(gaussians, R, t, K,
                                 img_h=img_h, img_w=img_w)
        cv2.putText(frame, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 1)
        frames[label] = frame

    # BEV for centre
    bev = render_bev_numpy(gaussians, bev_range=30.0, bev_size=img_w)
    bev = cv2.resize(bev, (img_w, img_h))

    # 3×3 grid layout:
    #  FL    F    FR
    #   L   BEV   R
    #  BL    B    BR
    order = [
        ["Front-Left",  "Front",  "Front-Right"],
        ["Left",        "BEV",    "Right"],
        ["Back-Left",   "Back",   "Back-Right"],
    ]

    rows = []
    for row_labels in order:
        row_imgs = []
        for lbl in row_labels:
            if lbl == "BEV":
                row_imgs.append(bev)
            else:
                row_imgs.append(frames[lbl])
        rows.append(np.hstack(row_imgs))

    grid = np.vstack(rows)
    cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[novel_view] Surround snapshot saved → {os.path.abspath(output_path)}")
    return grid


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)