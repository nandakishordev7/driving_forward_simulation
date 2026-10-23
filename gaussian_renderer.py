import numpy as np
import torch
import cv2
import os
try:
    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
    HAS_DGR = True
except ImportError:
    HAS_DGR = False

def _build_world_to_camera(R_wc, t_wc, device="cuda"):
    """
    Build 4x4 world->camera matrix from camera pose in world:
      X_cam = R_wc @ X_world + t_wc
    """
    W2C = torch.eye(4, device=device, dtype=torch.float32)
    W2C[:3, :3] = torch.from_numpy(R_wc).to(device=device, dtype=torch.float32)
    W2C[:3, 3] = torch.from_numpy(t_wc).to(device=device, dtype=torch.float32)
    return W2C


def _build_projection_from_K(K, H, W, znear=0.01, zfar=200.0, device="cuda"):
    """
    Approximate OpenGL-style projection matrix from camera intrinsics.
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    P = torch.zeros((4, 4), device=device, dtype=torch.float32)
    P[0, 0] = 2.0 * fx / W
    P[1, 1] = 2.0 * fy / H
    P[0, 2] = 2.0 * cx / W - 1.0
    P[1, 2] = 2.0 * cy / H - 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P

def render_camera_view(gaussians, K, R_wc, t_wc, H, W, device="cuda", bg=(0.0, 0.0, 0.0)):
    """
    Render merged Gaussians from a real camera view.

    gaussians keys:
      xyz        [N,3]
      opacity    [N,1] or [N]
      scales     [N,3]
      rotations  [N,4]
      sh_coeffs  [N,3]   # DC-only RGB fallback

    R_wc, t_wc:
      world->camera transform such that X_cam = R_wc @ X_world + t_wc
    """
    if not HAS_DGR:
        raise ImportError("diff_gauss is not installed or failed to import.")

    xyz = gaussians["xyz"]
    opacity = gaussians["opacity"]
    scales = gaussians["scales"]
    rotations = gaussians["rotations"]
    sh_coeffs = gaussians["sh_coeffs"]

    if isinstance(xyz, np.ndarray):
        xyz = torch.from_numpy(xyz).float().to(device)
    if isinstance(opacity, np.ndarray):
        opacity = torch.from_numpy(opacity).float().to(device)
    if isinstance(scales, np.ndarray):
        scales = torch.from_numpy(scales).float().to(device)
    if isinstance(rotations, np.ndarray):
        rotations = torch.from_numpy(rotations).float().to(device)
    if isinstance(sh_coeffs, np.ndarray):
        sh_coeffs = torch.from_numpy(sh_coeffs).float().to(device)

    xyz = xyz.contiguous()
    opacity = opacity.view(-1, 1).contiguous()
    scales = scales.contiguous()
    rotations = rotations.contiguous()

    # DC-only SH -> shape [N, 1, 3]
    if sh_coeffs.ndim == 2:
        shs = sh_coeffs[:, None, :].contiguous()
    else:
        shs = sh_coeffs.contiguous()

    means2D = torch.zeros_like(xyz, device=device, dtype=torch.float32)

    viewmatrix = _build_world_to_camera(R_wc, t_wc, device=device)
    projmatrix = _build_projection_from_K(K, H, W, device=device)

    # camera center in world coords:
    # X_cam = R_wc X_world + t_wc  =>  C_world = -R_wc^T t_wc
    cam_center = -torch.from_numpy(R_wc.T @ t_wc).to(device=device, dtype=torch.float32)

    tanfovx = float(W / (2.0 * max(K[0, 0], 1e-6)))
    tanfovy = float(H / (2.0 * max(K[1, 1], 1e-6)))

    settings = GaussianRasterizationSettings( # type: ignore
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor(bg, dtype=torch.float32, device=device),
        scale_modifier=2.5,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,                 # start with DC-only SH
        campos=cam_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=settings) # type: ignore

    rendered = rasterizer(
        means3D=xyz,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=None,
    )

    # This fork returns multiple outputs.
    # Keep unpacking flexible.
    if isinstance(rendered, tuple):
        if len(rendered) >= 4:
            color = rendered[0]
            depth = rendered[1]
            alpha = rendered[3]
            return {"rgb": color, "depth": depth, "alpha": alpha, "raw": rendered}
        else:
            return {"rgb": rendered[0], "raw": rendered} # type:ignore

    return {"rgb": rendered}

def render_bev_numpy(gaussians, bev_range=50.0, bev_size=512):
    xyz      = _to_np(gaussians['xyz'])         # (N, 3)
    opacity  = _to_np(gaussians['opacity'])    # (N, 1) or (N,)
    sh       = _to_np(gaussians['sh_coeffs'])  # (N, C)

    if opacity.ndim == 2:
        opacity = opacity[:, 0]
    opacity = opacity.astype(np.float32)

    # SH degree-0 → RGB: color = 0.5 + 0.282 * C0
    rgb = np.clip(0.5 + 0.282 * sh[:, :3], 0, 1).astype(np.float32)

    # 1. Spike suppression
    r_xy   = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)

    r_mask = r_xy <= bev_range
    z_mask = (xyz[:, 2] >= -2.5) & (xyz[:, 2] <= 2.5)
    o_mask = opacity > 0.05

    mask = r_mask & z_mask & o_mask
    xyz, rgb, opacity, r_xy = xyz[mask], rgb[mask], opacity[mask], r_xy[mask]

    # 2. Radial spike filter
    angle   = np.arctan2(xyz[:, 1], xyz[:, 0])
    n_bins  = 72
    bin_idx = ((angle + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    keep    = np.ones(len(xyz), dtype=bool)

    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() < 5:
            continue
        med = np.median(r_xy[sel])
        keep[sel & (r_xy > med * 1.8)] = False

    xyz, rgb, opacity = xyz[keep], rgb[keep], opacity[keep]
    print(f"[renderer] After spike filter: {len(xyz):,} Gaussians")

    # 3. Build canvas via alpha compositing
    H = W = bev_size
    canvas    = np.zeros((H, W, 3), dtype=np.float32)
    alpha_acc = np.zeros((H, W), dtype=np.float32)

    px = ((xyz[:, 0] + bev_range) / (2 * bev_range) * W).astype(int)
    py = ((bev_range - xyz[:, 1]) / (2 * bev_range) * H).astype(int)
    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)

    order = np.argsort(-xyz[:, 2])
    px, py  = px[order], py[order]
    rgb     = rgb[order]
    opacity = opacity[order]

    np.add.at(canvas, (py, px), rgb * opacity[:, None])
    np.add.at(alpha_acc, (py, px), opacity)

    safe = alpha_acc > 1e-6
    canvas[safe] /= alpha_acc[safe, None]
    canvas = np.clip(canvas, 0, 1)

    # 4. Gaussian blur to soften point artifacts
    canvas_u8 = (canvas * 255).astype(np.uint8)
    data_mask = (alpha_acc > 0.1).astype(np.uint8) * 255
    blurred   = cv2.GaussianBlur(canvas_u8, (3, 3), 0)
    canvas_u8 = np.where(data_mask[:, :, None] > 0, blurred, canvas_u8)

    # 5. Range rings + grid
    res = (2 * bev_range) / bev_size
    gc  = (45, 45, 45)
    for dist in [10, 20, 30, 40]:
        r_px = int(dist / res)
        cv2.circle(canvas_u8, (W // 2, H // 2), r_px, gc, 1)
        cv2.putText(
            canvas_u8, f"{dist}m",
            (W // 2 + r_px + 3, H // 2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1
        )

    for d in range(-int(bev_range), int(bev_range) + 1, 10):
        gx = int((d + bev_range) / res)
        gy = int((bev_range - d) / res)
        if 0 <= gx < W:
            cv2.line(canvas_u8, (gx, 0), (gx, H - 1), gc, 1)
        if 0 <= gy < H:
            cv2.line(canvas_u8, (0, gy), (W - 1, gy), gc, 1)

    # 6. Ego vehicle
    ex, ey = W // 2, H // 2
    car_l  = int(4.5 / res / 2)
    car_w  = int(2.0 / res / 2)
    cv2.rectangle(
        canvas_u8,
        (ex - car_w, ey - car_l),
        (ex + car_w, ey + car_l),
        (0, 255, 0), -1
    )
    cv2.arrowedLine(
        canvas_u8,
        (ex, ey), (ex, max(0, ey - car_l - 12)),
        (0, 200, 0), 2, tipLength=0.4
    )

    return canvas_u8


def render_bev_gpu(gaussians, bev_range=50.0, bev_size=512, device='cuda'):
    """GPU path — falls back to numpy if diff-gauss is missing."""
    if not HAS_DGR:
        return render_bev_numpy(
            {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
             for k, v in gaussians.items()},
            bev_range=bev_range, bev_size=bev_size
        )

    W = H = bev_size

    # Simple top-down virtual camera for BEV
    fx = fy = (W / 2.0) / np.tan(np.deg2rad(45.0))
    height = bev_range * 2.2
    near, far = 0.1, height * 3.0
    tanfovx = W / (2.0 * fx)
    tanfovy = H / (2.0 * fy)

    R = np.array([[1, 0, 0],
                  [0,-1, 0],
                  [0, 0,-1]], dtype=np.float32)
    T = -R @ np.array([0.0, 0.0, height], dtype=np.float32)

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = T
    w2c_t = torch.from_numpy(w2c).to(device=device, dtype=torch.float32)

    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 1.0 / tanfovx
    P[1, 1] = 1.0 / tanfovy
    P[2, 2] = far / (far - near)
    P[2, 3] = 1.0
    P[3, 2] = -(far * near) / (far - near)
    proj_t = torch.from_numpy(P).to(device=device, dtype=torch.float32)

    full_proj = (w2c_t.unsqueeze(0).bmm(proj_t.unsqueeze(0))).squeeze(0)
    cam_center = torch.from_numpy(-R.T @ T).to(device=device, dtype=torch.float32)

    rs = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, device=device),
        scale_modifier=2.5,
        viewmatrix=w2c_t,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=rs)

    means3D = gaussians['xyz'].to(device=device, dtype=torch.float32).contiguous()

    # For this API, keep means2D same shape as means3D.
    means2D = torch.zeros_like(means3D, device=device, dtype=torch.float32)

    sh = gaussians['sh_coeffs'].to(device=device, dtype=torch.float32)
    if sh.dim() == 2:
        sh = sh.unsqueeze(1)   # [N, 1, 3] for degree-0 SH
    sh = sh.contiguous()

    opacities = gaussians['opacity'].to(device=device, dtype=torch.float32)
    if opacities.dim() == 1:
        opacities = opacities.unsqueeze(1)
    opacities = opacities.contiguous()

    scales = gaussians['scales'].to(device=device, dtype=torch.float32).contiguous()
    rotations = gaussians['rotations'].to(device=device, dtype=torch.float32).contiguous()

    out = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=sh,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=None,
        extra_attrs=None
    )

    # slothfulxtx fork returns multiple outputs
    rendered = out[0]

    img = (rendered.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    return img



def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)