"""
train_gaussian.py
Self-supervised joint training of DepthNetwork + GaussianNetwork + RefineNet (iterative geometry refinement).
Refinement loop per batch:
    D0 = DepthNetwork(img)
    for k in 0..K-1:
        f3           = GaussianNet.encode(img)      # cached — runs once
        Gk           = GaussianNet.decode(f3, Dk)
        render_err   = warp(img2, Dk) - img1        # photometric proxy
        Dk+1         = Dk + RefineNet(Dk, render_err)
    loss = weighted photometric sum across iterations
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from gaussian_network import DepthNetwork, GaussianNetwork
from nuscenes.nuscenes import NuScenes
from pose_dataset import NuScenesPoseDataset
from pose_network import PoseNet, pose_vec_to_matrix
from refine_net import RefineNet
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

# ── Config ────────────────────────────────────────────────────────────────────
DATAROOT = r"C:\Projects\driving_forward_conda\nu_scenes"
VERSION = "v1.0-mini"
SAVE_DIR = r"C:\Projects\driving_forward_conda\checkpoints"
POSE_CKPT = r"C:\Projects\driving_forward_conda\checkpoints\pose_net_best.pth"
IMG_SIZE = (192, 640)
BATCH_SIZE = 4
NUM_EPOCHS = 60
LR = 5e-5
VAL_SPLIT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loss weights
W_PHOTO = 1.0
W_SMOOTH = 0.001
W_OPACITY = 0.01

# Refinement config
N_REFINE_ITERS = 3# K — number of refinement steps per batch
GAMMA = 0.8  # loss weight decay for earlier iterations
# iter 0 weight = GAMMA^(K-1), iter K-1 weight = 1.0
# ─────────────────────────────────────────────────────────────────────────────
# ── Photometric loss helpers ──────────────────────────────────────────────────
def ssim_loss(x, y):
    C1, C2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sig_x = F.avg_pool2d(x**2, 3, 1, 1) - mu_x**2
    sig_y = F.avg_pool2d(y**2, 3, 1, 1) - mu_y**2
    sig_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
    return torch.clamp((1 - num / den) / 2, 0, 1)


def photometric_loss(pred, target):
    l1 = (pred - target).abs()
    ssim = ssim_loss(pred, target)
    return (0.85 * ssim + 0.15 * l1).mean()


def warp_image(img, depth, pose_mat, K):
    B, _, H, W = img.shape
    device = img.device

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    pix = pix.reshape(B, 3, -1)

    K_inv = torch.linalg.inv(K)
    cam = K_inv @ pix
    cam = cam * depth.reshape(B, 1, -1)

    ones4 = torch.ones(B, 1, H * W, device=device)
    cam_h = torch.cat([cam, ones4], dim=1)
    cam_t = pose_mat @ cam_h
    cam_t = cam_t[:, :3, :]

    proj = K @ cam_t
    z = proj[:, 2:3, :].clamp(min=1e-4)
    uv = proj[:, :2, :] / z

    uv[:, 0, :] = 2 * uv[:, 0, :] / (W - 1) - 1
    uv[:, 1, :] = 2 * uv[:, 1, :] / (H - 1) - 1
    grid = uv.permute(0, 2, 1).reshape(B, H, W, 2)

    warped = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return warped


def smooth_loss(depth, img):
    grad_d_x = (depth[:, :, 1:] - depth[:, :, :-1]).abs()
    grad_d_y = (depth[:, 1:, :] - depth[:, :-1, :]).abs()
    grad_i_x = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean(dim=1)
    grad_i_y = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean(dim=1)
    w_x = torch.exp(-grad_i_x)
    w_y = torch.exp(-grad_i_y)
    return (grad_d_x * w_x).mean() + (grad_d_y * w_y).mean()


# ── Training ──────────────────────────────────────────────────────────────────


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"[train_gaussian] Loading nuScenes from {DATAROOT} ...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    dataset = NuScenesPoseDataset(nusc, img_size=IMG_SIZE)

    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"[train_gaussian] Train: {n_train} | Val: {n_val} | Device: {DEVICE}")

    # ── Frozen pose network ───────────────────────────────────────────────────
    pose_net = PoseNet().to(DEVICE)
    ckpt = torch.load(POSE_CKPT, map_location=DEVICE)
    pose_net.load_state_dict(ckpt["model_state"])
    pose_net.eval()
    for p in pose_net.parameters():
        p.requires_grad_(False)
    print(f"[train_gaussian] Loaded pose network (epoch {ckpt['epoch']})")

    # ── Trainable networks ────────────────────────────────────────────────────
    depth_net = DepthNetwork().to(DEVICE)
    gauss_net = GaussianNetwork().to(DEVICE)
    refine_net = RefineNet(hidden=32, max_delta=2.0).to(DEVICE)

    params = (
        list(depth_net.parameters())
        + list(gauss_net.parameters())
        + list(refine_net.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val = float("inf")

    # ── Loss log (for plot_refinement_comparison.py) ──────────────────────────
    loss_log = {
        "epochs": [],
        "train_total": [],
        "baseline_val": [],  # val loss using D0 only (no refinement)
        "refine_d0_val": [],  # val loss at iteration 0
        "refine_d1_val": [],  # val loss at iteration 1
        "refine_d2_val": [],  # val loss at iteration 2
        "refine_d3_val": [],  # val loss at final iteration K
    }

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        depth_net.train()
        gauss_net.train()
        refine_net.train()

        epoch_loss = 0.0
        # accumulate per-iter losses across batches for logging
        iter_loss_accum = [0.0] * (N_REFINE_ITERS + 1)  # D0 .. DK

        for batch in train_loader:
            img1 = batch["img1"].to(DEVICE)
            img2 = batch["img2"].to(DEVICE)
            B = img1.shape[0]
            H, W = img1.shape[2], img1.shape[3]

            # camera intrinsics (approximate for nuScenes)
            fx = fy = 800.0 * W / 1600
            cx, cy = W / 2.0, H / 2.0
            K = torch.tensor(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                dtype=torch.float32,
                device=DEVICE,
            )
            K = K.unsqueeze(0).expand(B, -1, -1)

            # frozen pose
            with torch.no_grad():
                pose_vec = pose_net(img1, img2)
                T_1to2 = pose_vec_to_matrix(pose_vec)
                T_2to1 = torch.linalg.inv(T_1to2)

            # ── D0: initial depth ─────────────────────────────────────────────
            depth_k1 = depth_net(img1)
            depth_k2 = depth_net(img2)

            # cache image features — encoder runs only once per image
            f3_img1 = gauss_net.encode(img1)
            f3_img2 = gauss_net.encode(img2)

            # ── refinement loop ───────────────────────────────────────────────
            iter_losses = []  # photometric loss at each depth estimate

            for k in range(N_REFINE_ITERS + 1):
                # photometric loss at current depth Dk
                img2_warped = warp_image(img2, depth_k1, T_1to2, K)
                img1_warped = warp_image(img1, depth_k2, T_2to1, K)
                loss_photo_k = (
                    photometric_loss(img2_warped, img1)
                    + photometric_loss(img1_warped, img2)
                ) / 2
                iter_losses.append(loss_photo_k)

                # stop updating depth after last photometric eval
                if k == N_REFINE_ITERS:
                    break

                # render error proxy (detached — only guides geometry)
                render_err1 = (img2_warped - img1).detach()
                render_err2 = (img1_warped - img2).detach()

                # depth correction
                depth_k1 = refine_net(depth_k1, render_err1)
                depth_k2 = refine_net(depth_k2, render_err2)

            # ── weighted photometric loss: later iters weighted more ───────────
            # weight at step k = GAMMA^(K - k),  k=0..K
            K_steps = len(iter_losses)
            total_photo = sum(
                GAMMA ** (K_steps - 1 - k) * lk for k, lk in enumerate(iter_losses)
            )

            # smoothness on final refined depth
            loss_smooth = (
                smooth_loss(depth_k1, img1) + smooth_loss(depth_k2, img2)
            ) / 2

            # opacity regularisation on final Gaussians
            gauss_final = gauss_net.decode(f3_img1, depth_k1, K)
            loss_opacity = gauss_final["opacity"].mean()

            loss = (
                W_PHOTO * total_photo
                + W_SMOOTH * loss_smooth
                + W_OPACITY * loss_opacity
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            for k, lk in enumerate(iter_losses):
                iter_loss_accum[k] += lk.item()

        epoch_loss /= len(train_loader)
        iter_loss_accum = [v / len(train_loader) for v in iter_loss_accum]

        # ── Validation ────────────────────────────────────────────────────────
        depth_net.eval()
        gauss_net.eval()
        refine_net.eval()

        val_baseline = 0.0
        val_iter_accum = [0.0] * (N_REFINE_ITERS + 1)

        with torch.no_grad():
            for batch in val_loader:
                img1 = batch["img1"].to(DEVICE)
                img2 = batch["img2"].to(DEVICE)
                B = img1.shape[0]
                H, W = img1.shape[2], img1.shape[3]

                fx = fy = 800.0 * W / 1600
                cx, cy = W / 2.0, H / 2.0
                K = torch.tensor(
                    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                    dtype=torch.float32,
                    device=DEVICE,
                )
                K = K.unsqueeze(0).expand(B, -1, -1)

                pose_vec = pose_net(img1, img2)
                T_1to2 = pose_vec_to_matrix(pose_vec)
                T_2to1 = torch.linalg.inv(T_1to2)

                # baseline — D0 only, no refinement
                d1_base = depth_net(img1)
                d2_base = depth_net(img2)
                w2_base = warp_image(img2, d1_base, T_1to2, K)
                w1_base = warp_image(img1, d2_base, T_2to1, K)
                val_baseline += (
                    photometric_loss(w2_base, img1) + photometric_loss(w1_base, img2)
                ).item() / 2

                # refinement val
                depth_k1 = d1_base
                depth_k2 = d2_base
                f3_img1 = gauss_net.encode(img1)
                f3_img2 = gauss_net.encode(img2)

                for k in range(N_REFINE_ITERS + 1):
                    img2_w = warp_image(img2, depth_k1, T_1to2, K)
                    img1_w = warp_image(img1, depth_k2, T_2to1, K)
                    vl_k = (
                        photometric_loss(img2_w, img1) + photometric_loss(img1_w, img2)
                    ).item() / 2
                    val_iter_accum[k] += vl_k

                    if k == N_REFINE_ITERS:
                        break

                    render_err1 = img2_w - img1
                    render_err2 = img1_w - img2
                    depth_k1 = refine_net(depth_k1, render_err1)
                    depth_k2 = refine_net(depth_k2, render_err2)

        n_val_batches = len(val_loader)
        val_baseline /= n_val_batches
        val_iter_accum = [v / n_val_batches for v in val_iter_accum]
        val_final = val_iter_accum[-1]  # loss at DK

        scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────────
        print(
            f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
            f"Train: {epoch_loss:.4f}  "
            f"Val-base: {val_baseline:.4f}  "
            f"Val-D0: {val_iter_accum[0]:.4f}  "
            f"Val-D{N_REFINE_ITERS}: {val_final:.4f}  "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # fill per-iter log (pad to 4 slots for the plot script)
        iter_slots = val_iter_accum + [val_iter_accum[-1]] * (4 - len(val_iter_accum))

        loss_log["epochs"].append(epoch)
        loss_log["train_total"].append(epoch_loss)
        loss_log["baseline_val"].append(val_baseline)
        loss_log["refine_d0_val"].append(iter_slots[0])
        loss_log["refine_d1_val"].append(iter_slots[1])
        loss_log["refine_d2_val"].append(iter_slots[2])
        loss_log["refine_d3_val"].append(iter_slots[3])

        json.dump(
            loss_log, open(os.path.join(SAVE_DIR, "loss_log.json"), "w"), indent=2
        )

        # ── Save best checkpoint ──────────────────────────────────────────────
        if val_final < best_val:
            best_val = val_final
            torch.save(
                {
                    "epoch": epoch,
                    "depth_net_state": depth_net.state_dict(),
                    "gauss_net_state": gauss_net.state_dict(),
                    "refine_net_state": refine_net.state_dict(),
                    "val_loss": val_final,
                    "img_size": IMG_SIZE,
                },
                os.path.join(SAVE_DIR, "gaussian_net_best.pth"),
            )
            print(f"  -> Saved best checkpoint (val={val_final:.4f})")

    print(f"\n[train_gaussian] Done. Best val: {best_val:.4f}")
    print(f"[train_gaussian] Loss log → {SAVE_DIR}/loss_log.json")


if __name__ == "__main__":
    train()
