import numpy as np
import torch
import cv2
import os

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    HAS_DGR = True
except ImportError:
    HAS_DGR = False


def render_bev_numpy(gaussians, bev_range=50.0, bev_size=512):
    """
    Improved numpy BEV renderer with:
      - Confidence-weighted alpha compositing
      - Image-color from SH coefficients
      - Spike suppression via radial density filter
      - Near-ego fill via small radius boost
    """

    xyz      = _to_np(gaussians['xyz'])         # (N, 3)
    opacity  = _to_np(gaussians['opacity'])      # (N, 1) or (N,)
    sh       = _to_np(gaussians['sh_coeffs'])    # (N, C)

    if opacity.ndim == 2: opacity = opacity[:, 0]
    opacity = opacity.astype(np.float32)

    # SH degree-0 → RGB:  color = 0.5 + 0.282 * C0
    rgb = np.clip(0.5 + 0.282 * sh[:, :3], 0, 1).astype(np.float32)

    # ── 1. Spike suppression ──────────────────────────────────────────────────
    # Spikes happen when a camera's sky pixels get projected to extreme depths.
    # We suppress them by removing points whose radial distance in XY is much
    # larger than their local neighbourhood average (outlier removal in BEV).
    r_xy   = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)

    # Hard radial clamp
    r_mask = r_xy <= bev_range
    # Z clamp: ground plane ±2.5m  (removes sky projections)
    z_mask = (xyz[:, 2] >= -2.5) & (xyz[:, 2] <= 2.5)
    # Opacity threshold: remove near-zero confidence Gaussians
    o_mask = opacity > 0.05

    mask = r_mask & z_mask & o_mask
    xyz, rgb, opacity, r_xy = xyz[mask], rgb[mask], opacity[mask], r_xy[mask]

    # ── 2. Radial spike filter ────────────────────────────────────────────────
    # Divide BEV into angular bins; in each bin keep only points within
    # 1.5× the median radial distance — removes the spike tips
    angle     = np.arctan2(xyz[:, 1], xyz[:, 0])   # [-pi, pi]
    n_bins    = 72   # 5° bins
    bin_idx   = ((angle + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    keep      = np.ones(len(xyz), dtype=bool)

    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() < 5:
            continue
        med = np.median(r_xy[sel])
        # Remove points more than 1.8× the median radius in this angular slice
        keep[sel & (r_xy > med * 1.8)] = False

    xyz, rgb, opacity = xyz[keep], rgb[keep], opacity[keep]
    print(f"[renderer] After spike filter: {len(xyz):,} Gaussians")

    # ── 3. Build canvas via alpha compositing ─────────────────────────────────
    H = W = bev_size
    canvas     = np.zeros((H, W, 3), dtype=np.float32)
    alpha_acc  = np.zeros((H, W),    dtype=np.float32)

    # Pixel coords
    px = ((xyz[:, 0] + bev_range) / (2 * bev_range) * W).astype(int)
    py = ((bev_range - xyz[:, 1]) / (2 * bev_range) * H).astype(int)
    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)

    # Sort by Z so ground-level points (lowest Z) paint last = on top
    order = np.argsort(-xyz[:, 2])
    px, py   = px[order], py[order]
    rgb      = rgb[order]
    opacity  = opacity[order]

    # Weighted accumulation
    np.add.at(canvas,    (py, px), rgb * opacity[:, None])
    np.add.at(alpha_acc, (py, px), opacity)

    safe = alpha_acc > 1e-6
    canvas[safe] /= alpha_acc[safe, None]
    canvas = np.clip(canvas, 0, 1)

    # ── 4. Gaussian blur to soften point artifacts ────────────────────────────
    canvas_u8 = (canvas * 255).astype(np.uint8)
    # Only blur where we have data (avoid smearing into empty areas)
    data_mask = (alpha_acc > 0.1).astype(np.uint8) * 255
    blurred   = cv2.GaussianBlur(canvas_u8, (3, 3), 0)
    canvas_u8 = np.where(data_mask[:, :, None] > 0, blurred, canvas_u8)

    # ── 5. Range rings + grid ─────────────────────────────────────────────────
    res = (2 * bev_range) / bev_size
    gc  = (45, 45, 45)
    for dist in [10, 20, 30, 40]:
        r_px = int(dist / res)
        cv2.circle(canvas_u8, (W // 2, H // 2), r_px, gc, 1)
        cv2.putText(canvas_u8, f"{dist}m",
                    (W // 2 + r_px + 3, H // 2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)

    # Grid lines every 10m
    for d in range(-int(bev_range), int(bev_range) + 1, 10):
        gx = int((d + bev_range) / res)
        gy = int((bev_range - d) / res)
        if 0 <= gx < W: cv2.line(canvas_u8, (gx, 0),     (gx, H-1),  gc, 1)
        if 0 <= gy < H: cv2.line(canvas_u8, (0,  gy),    (W-1, gy),  gc, 1)

    # ── 6. Ego vehicle ────────────────────────────────────────────────────────
    ex, ey = W // 2, H // 2
    car_l  = int(4.5 / res / 2)
    car_w  = int(2.0 / res / 2)
    cv2.rectangle(canvas_u8,
                  (ex - car_w, ey - car_l),
                  (ex + car_w, ey + car_l),
                  (0, 255, 0), -1)
    cv2.arrowedLine(canvas_u8,
                    (ex, ey), (ex, max(0, ey - car_l - 12)),
                    (0, 200, 0), 2, tipLength=0.4)

    return canvas_u8


def render_bev_gpu(gaussians, bev_range=50.0, bev_size=512, device='cuda'):
    """GPU path — falls back to numpy if diff-gaussian-rasterization missing."""
    if not HAS_DGR:
        return render_bev_numpy(
            {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
             for k, v in gaussians.items()},
            bev_range=bev_range, bev_size=bev_size
        )
    # (GPU rasterization path unchanged — only reached if DGR installed)
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer)
    W = H = bev_size
    fx = fy = (W / 2.0) / np.tan(np.deg2rad(45))
    cx, cy  = W / 2.0, H / 2.0
    height  = bev_range * 2.2
    near, far = 0.1, height * 3
    tanfovx = W / (2 * fx); tanfovy = H / (2 * fy)
    R = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float32)
    T = -R @ np.array([0., 0., height])
    w2c = np.eye(4, dtype=np.float32); w2c[:3,:3]=R; w2c[:3,3]=T
    w2c_t = torch.from_numpy(w2c).to(device)
    P = np.zeros((4,4),dtype=np.float32)
    P[0,0]=1/tanfovx; P[1,1]=1/tanfovy
    P[2,2]=far/(far-near); P[2,3]=1.; P[3,2]=-(far*near)/(far-near)
    proj_t = torch.from_numpy(P).to(device)
    full_proj = (w2c_t.unsqueeze(0).bmm(proj_t.unsqueeze(0))).squeeze(0)
    cam_center = torch.from_numpy(-R.T@T).to(device)
    rs = GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=tanfovx, tanfovy=tanfovy,
        bg=torch.zeros(3,device=device), scale_modifier=1.0,
        viewmatrix=w2c_t, projmatrix=full_proj, sh_degree=0,
        campos=cam_center, prefiltered=False, debug=False)
    rasterizer = GaussianRasterizer(raster_settings=rs)
    sh = gaussians['sh_coeffs'].to(device)
    if sh.dim()==2: sh = sh.unsqueeze(1)
    rendered,_,_ = rasterizer(
        means3D=gaussians['xyz'].to(device),
        means2D=torch.zeros_like(gaussians['xyz'][:,:2]),
        shs=sh, colors_precomp=None,
        opacities=gaussians['opacity'].to(device),
        scales=gaussians['scales'].to(device),
        rotations=gaussians['rotations'].to(device),
        cov3D_precomp=None)
    img = (rendered.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
    return img


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)