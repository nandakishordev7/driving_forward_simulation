"""
train_gaussian_baseline.py

Identical to the ORIGINAL train_gaussian.py — no RefineNet, no refinement loop.
Run this FIRST to get the baseline loss log.

Output: checkpoints/loss_log_baseline.json
        checkpoints/gaussian_net_baseline.pth
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from nuscenes.nuscenes import NuScenes

from gaussian_network import DepthNetwork, GaussianNetwork
from pose_network import PoseNet, pose_vec_to_matrix
from pose_dataset import NuScenesPoseDataset


# ── Config ────────────────────────────────────────────────────────────────────
DATAROOT   = "C://Projects//driving_forward//nu_scenes"
VERSION    = "v1.0-mini"
SAVE_DIR   = "C://Projects//driving_forward//checkpoints"
POSE_CKPT  = "C://Projects//driving_forward//checkpoints//pose_net_best.pth"
IMG_SIZE   = (192, 640)
BATCH_SIZE = 4
NUM_EPOCHS = 40
LR         = 5e-5
VAL_SPLIT  = 0.1
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

W_PHOTO    = 1.0
W_SMOOTH   = 0.001
W_OPACITY  = 0.01
# ─────────────────────────────────────────────────────────────────────────────


def ssim_loss(x, y):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x   = F.avg_pool2d(x, 3, 1, 1)
    mu_y   = F.avg_pool2d(y, 3, 1, 1)
    sig_x  = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
    sig_y  = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
    sig_xy = F.avg_pool2d(x * y,  3, 1, 1) - mu_x * mu_y
    num    = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den    = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2)
    return torch.clamp((1 - num / den) / 2, 0, 1)


def photometric_loss(pred, target):
    return (0.85 * ssim_loss(pred, target) + 0.15 * (pred - target).abs()).mean()


def warp_image(img, depth, pose_mat, K):
    B, _, H, W = img.shape
    device = img.device
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    ones  = torch.ones_like(xs)
    pix   = torch.stack([xs, ys, ones], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    pix   = pix.reshape(B, 3, -1)
    K_inv = torch.linalg.inv(K)
    cam   = K_inv @ pix
    cam   = cam * depth.reshape(B, 1, -1)
    ones4 = torch.ones(B, 1, H * W, device=device)
    cam_h = torch.cat([cam, ones4], dim=1)
    cam_t = pose_mat @ cam_h
    cam_t = cam_t[:, :3, :]
    proj  = K @ cam_t
    z     = proj[:, 2:3, :].clamp(min=1e-4)
    uv    = proj[:, :2, :] / z
    uv[:, 0, :] = 2 * uv[:, 0, :] / (W - 1) - 1
    uv[:, 1, :] = 2 * uv[:, 1, :] / (H - 1) - 1
    grid  = uv.permute(0, 2, 1).reshape(B, H, W, 2)
    return F.grid_sample(img, grid, mode='bilinear',
                         padding_mode='border', align_corners=True)


def smooth_loss(depth, img):
    grad_d_x = (depth[:, :, 1:]  - depth[:, :, :-1]).abs()
    grad_d_y = (depth[:, 1:, :]  - depth[:, :-1, :]).abs()
    grad_i_x = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean(dim=1)
    grad_i_y = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean(dim=1)
    return (grad_d_x * torch.exp(-grad_i_x)).mean() + \
           (grad_d_y * torch.exp(-grad_i_y)).mean()


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    nusc    = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    dataset = NuScenesPoseDataset(nusc, img_size=IMG_SIZE)
    n_val   = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)   # same split as refined run
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"[baseline] Train: {n_train} | Val: {n_val} | Device: {DEVICE}")

    # frozen pose net
    pose_net = PoseNet().to(DEVICE)
    ckpt     = torch.load(POSE_CKPT, map_location=DEVICE)
    pose_net.load_state_dict(ckpt['model_state'])
    pose_net.eval()
    for p in pose_net.parameters():
        p.requires_grad_(False)

    depth_net = DepthNetwork().to(DEVICE)
    gauss_net = GaussianNetwork().to(DEVICE)
    params    = list(depth_net.parameters()) + list(gauss_net.parameters())
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val  = float('inf')

    # loss log — matches keys expected by plot_real_comparison.py
    loss_log  = {
        "epochs"     : [],
        "train_loss" : [],
        "val_loss"   : [],
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        depth_net.train(); gauss_net.train()
        t_loss = 0.0

        for batch in train_loader:
            img1 = batch['img1'].to(DEVICE)
            img2 = batch['img2'].to(DEVICE)
            B    = img1.shape[0]
            H, W = img1.shape[2], img1.shape[3]
            fx = fy = 800.0 * W / 1600
            cx, cy  = W / 2.0, H / 2.0
            K = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]],
                              dtype=torch.float32, device=DEVICE)
            K = K.unsqueeze(0).expand(B, -1, -1)

            with torch.no_grad():
                pose_vec = pose_net(img1, img2)
                T_1to2   = pose_vec_to_matrix(pose_vec)
                T_2to1   = torch.linalg.inv(T_1to2)

            depth1 = depth_net(img1)
            depth2 = depth_net(img2)

            img2_w = warp_image(img2, depth1, T_1to2, K)
            img1_w = warp_image(img1, depth2, T_2to1, K)

            loss_photo  = (photometric_loss(img2_w, img1) +
                           photometric_loss(img1_w, img2)) / 2
            loss_smooth = (smooth_loss(depth1, img1) +
                           smooth_loss(depth2, img2)) / 2
            gauss1      = gauss_net(img1, depth1.detach(), K)
            loss_opa    = gauss1['opacity'].mean()

            loss = W_PHOTO * loss_photo + W_SMOOTH * loss_smooth + W_OPACITY * loss_opa
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            t_loss += loss.item()

        t_loss /= len(train_loader)

        # validation
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
                K = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]],
                                  dtype=torch.float32, device=DEVICE)
                K = K.unsqueeze(0).expand(B, -1, -1)
                pose_vec = pose_net(img1, img2)
                T_1to2   = pose_vec_to_matrix(pose_vec)
                T_2to1   = torch.linalg.inv(T_1to2)
                d1 = depth_net(img1)
                d2 = depth_net(img2)
                w2 = warp_image(img2, d1, T_1to2, K)
                w1 = warp_image(img1, d2, T_2to1, K)
                v_loss += (photometric_loss(w2, img1) +
                           photometric_loss(w1, img2)).item() / 2
        v_loss /= len(val_loader)
        scheduler.step()

        print(f"[baseline] Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
              f"Train: {t_loss:.4f}  Val: {v_loss:.4f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        loss_log["epochs"].append(epoch)
        loss_log["train_loss"].append(t_loss)
        loss_log["val_loss"].append(v_loss)
        json.dump(loss_log,
                  open(os.path.join(SAVE_DIR, "loss_log_baseline.json"), "w"),
                  indent=2)

        if v_loss < best_val:
            best_val = v_loss
            torch.save({
                'epoch'           : epoch,
                'depth_net_state' : depth_net.state_dict(),
                'gauss_net_state' : gauss_net.state_dict(),
                'val_loss'        : v_loss,
                'img_size'        : IMG_SIZE,
            }, os.path.join(SAVE_DIR, 'gaussian_net_baseline.pth'))
            print(f"  -> Saved (val={v_loss:.4f})")

    print(f"\n[baseline] Done. Best val: {best_val:.4f}")


if __name__ == "__main__":
    train()