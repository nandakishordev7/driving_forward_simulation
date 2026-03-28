"""
Self-supervised joint training of DepthNetwork + GaussianNetwork.

Loss:
  - Photometric loss: reconstruct each camera view from its neighbours
    using the predicted depth + relative pose from the trained PoseNet.
  - Smoothness loss: depth map edge-aware smoothness.
  - Opacity regularization: prevent all Gaussians collapsing to full opacity.

No depth GT, no LiDAR, no camera extrinsics needed during training.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from nuscenes.nuscenes import NuScenes

from gaussian_network import DepthNetwork, GaussianNetwork
from pose_network import PoseNet
from pose_dataset import NuScenesPoseDataset


# ── Config ────────────────────────────────────────────────────────────────────
DATAROOT      = "C://Projects//driving_forward//nu_scenes"
VERSION       = "v1.0-mini"
SAVE_DIR      = "C://Projects//driving_forward//checkpoints"
POSE_CKPT     = "C://Projects//driving_forward//checkpoints//pose_net_best.pth"
IMG_SIZE      = (192, 640)
BATCH_SIZE    = 4
NUM_EPOCHS    = 40
LR            = 5e-5
VAL_SPLIT     = 0.1
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Loss weights
W_PHOTO       = 1.0    # photometric reconstruction
W_SMOOTH      = 0.001  # depth smoothness
W_OPACITY     = 0.01   # opacity regularization
# ─────────────────────────────────────────────────────────────────────────────


# ── Photometric loss helpers ──────────────────────────────────────────────────

def ssim_loss(x, y):
    """Simplified SSIM loss (per-pixel, averaged)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sig_x  = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
    sig_y  = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
    sig_xy = F.avg_pool2d(x * y,  3, 1, 1) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2)
    return torch.clamp((1 - num / den) / 2, 0, 1)


def photometric_loss(pred, target):
    """0.85 * SSIM + 0.15 * L1 — standard self-supervised depth loss."""
    l1   = (pred - target).abs()
    ssim = ssim_loss(pred, target)
    return (0.85 * ssim + 0.15 * l1).mean()


def warp_image(img, depth, pose_mat, K):
    """
    Warp `img` (target) into the source frame using depth + pose.
    Args:
        img      : (B, 3, H, W)
        depth    : (B, H, W)
        pose_mat : (B, 4, 4) source-to-target transform
        K        : (B, 3, 3) camera intrinsics
    Returns:
        warped   : (B, 3, H, W)
    """
    B, _, H, W = img.shape
    device = img.device

    # Build pixel grid
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    ones = torch.ones_like(xs)
    pix  = torch.stack([xs, ys, ones], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    pix  = pix.reshape(B, 3, -1)   # (B, 3, H*W)

    # Unproject: pixel → camera 3D
    K_inv = torch.linalg.inv(K)
    cam   = K_inv @ pix             # (B, 3, H*W)
    cam   = cam * depth.reshape(B, 1, -1)

    # Transform to target camera
    ones4 = torch.ones(B, 1, H * W, device=device)
    cam_h = torch.cat([cam, ones4], dim=1)   # (B, 4, H*W)
    cam_t = pose_mat @ cam_h                  # (B, 4, H*W)
    cam_t = cam_t[:, :3, :]                   # (B, 3, H*W)

    # Project into target image
    proj  = K @ cam_t               # (B, 3, H*W)
    z     = proj[:, 2:3, :].clamp(min=1e-4)
    uv    = proj[:, :2, :] / z      # (B, 2, H*W)

    # Normalize to [-1, 1] for grid_sample
    uv[:, 0, :] = 2 * uv[:, 0, :] / (W - 1) - 1
    uv[:, 1, :] = 2 * uv[:, 1, :] / (H - 1) - 1
    grid = uv.permute(0, 2, 1).reshape(B, H, W, 2)

    warped = F.grid_sample(img, grid, mode='bilinear',
                           padding_mode='border', align_corners=True)
    return warped


def smooth_loss(depth, img):
    """Edge-aware depth smoothness loss."""
    grad_d_x = (depth[:, :, 1:] - depth[:, :, :-1]).abs()
    grad_d_y = (depth[:, 1:, :] - depth[:, :-1, :]).abs()
    grad_i_x = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean(dim=1)
    grad_i_y = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean(dim=1)
    w_x = torch.exp(-grad_i_x)
    w_y = torch.exp(-grad_i_y)
    return (grad_d_x * w_x).mean() + (grad_d_y * w_y).mean()


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"[train_gaussian] Loading nuScenes from {DATAROOT} ...")
    nusc    = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    dataset = NuScenesPoseDataset(nusc, img_size=IMG_SIZE)

    n_val   = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"[train_gaussian] Train: {n_train} | Val: {n_val} | Device: {DEVICE}")

    # ── Load frozen pose network ──────────────────────────────────────────────
    pose_net = PoseNet().to(DEVICE)
    ckpt     = torch.load(POSE_CKPT, map_location=DEVICE)
    pose_net.load_state_dict(ckpt['model_state'])
    pose_net.eval()
    for p in pose_net.parameters():
        p.requires_grad_(False)
    print(f"[train_gaussian] Loaded pose network (epoch {ckpt['epoch']})")

    # ── Depth + Gaussian networks ─────────────────────────────────────────────
    depth_net = DepthNetwork().to(DEVICE)
    gauss_net = GaussianNetwork().to(DEVICE)

    params    = list(depth_net.parameters()) + list(gauss_net.parameters())
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val  = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        depth_net.train()
        gauss_net.train()
        t_loss = 0.0

        for batch in train_loader:
            img1    = batch['img1'].to(DEVICE)      # (B, 3, H, W)
            img2    = batch['img2'].to(DEVICE)

            # Build camera intrinsics tensor (use fixed nuScenes CAM_FRONT approx)
            B = img1.shape[0]
            H, W = img1.shape[2], img1.shape[3]
            fx = fy = 800.0 * W / 1600
            cx, cy  = W / 2.0, H / 2.0
            K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                              dtype=torch.float32, device=DEVICE)
            K = K.unsqueeze(0).expand(B, -1, -1)

            # Depth predictions
            depth1  = depth_net(img1)   # (B, H, W)
            depth2  = depth_net(img2)

            # Relative pose from frozen pose network
            with torch.no_grad():
                from pose_network import pose_vec_to_matrix
                pose_vec  = pose_net(img1, img2)
                T_1to2    = pose_vec_to_matrix(pose_vec)   # (B, 4, 4)
                T_2to1    = torch.linalg.inv(T_1to2)

            # Photometric loss: warp img2 into img1's frame and vice versa
            img2_warped = warp_image(img2, depth1, T_1to2, K)
            img1_warped = warp_image(img1, depth2, T_2to1, K)

            loss_photo = (photometric_loss(img2_warped, img1) +
                          photometric_loss(img1_warped, img2)) / 2

            # Smoothness loss
            loss_smooth = (smooth_loss(depth1, img1) +
                           smooth_loss(depth2, img2)) / 2

            # Gaussian network on img1 — opacity regularization
            gauss1      = gauss_net(img1, depth1.detach(), K)
            loss_opacity = gauss1['opacity'].mean()

            loss = (W_PHOTO  * loss_photo  +
                    W_SMOOTH * loss_smooth +
                    W_OPACITY * loss_opacity)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            t_loss += loss.item()

        t_loss /= len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        depth_net.eval(); gauss_net.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                img1 = batch['img1'].to(DEVICE)
                img2 = batch['img2'].to(DEVICE)
                B    = img1.shape[0]
                H, W = img1.shape[2], img1.shape[3]
                fx = fy = 800.0 * W / 1600
                cx, cy  = W / 2.0, H / 2.0
                K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                                  dtype=torch.float32, device=DEVICE)
                K = K.unsqueeze(0).expand(B, -1, -1)
                depth1 = depth_net(img1)
                depth2 = depth_net(img2)
                from pose_network import pose_vec_to_matrix
                pose_vec = pose_net(img1, img2)
                T_1to2   = pose_vec_to_matrix(pose_vec)
                T_2to1   = torch.linalg.inv(T_1to2)
                img2_w   = warp_image(img2, depth1, T_1to2, K)
                img1_w   = warp_image(img1, depth2, T_2to1, K)
                v_loss  += photometric_loss(img2_w, img1).item() + \
                            photometric_loss(img1_w, img2).item()
        v_loss /= (2 * len(val_loader))

        scheduler.step()
        print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
              f"Train: {t_loss:.4f}  Val: {v_loss:.4f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save({
                'epoch'           : epoch,
                'depth_net_state' : depth_net.state_dict(),
                'gauss_net_state' : gauss_net.state_dict(),
                'val_loss'        : v_loss,
                'img_size'        : IMG_SIZE,
            }, os.path.join(SAVE_DIR, 'gaussian_net_best.pth'))
            print(f"  -> Saved best checkpoint (val={v_loss:.4f})")

    print(f"\n[train_gaussian] Done. Best val: {best_val:.4f}")


if __name__ == "__main__":
    train()