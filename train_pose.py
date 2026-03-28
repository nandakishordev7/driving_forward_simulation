import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from nuscenes.nuscenes import NuScenes

from pose_network import PoseNet
from pose_dataset import NuScenesPoseDataset


# ── Config ────────────────────────────────────────────────────────────────────
DATAROOT   = "C://Projects//driving_forward//nu_scenes"
VERSION    = "v1.0-mini"          # change to "v1.0-trainval" for full dataset
SAVE_DIR   = "C://Projects//driving_forward//checkpoints"
IMG_SIZE   = (192, 640)           # (H, W) — standard monodepth resolution
BATCH_SIZE = 8
NUM_EPOCHS = 30
LR         = 1e-4
VAL_SPLIT  = 0.1                  # 10% of pairs used for validation
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Loss weights — translation and rotation are different units so balance them
LAMBDA_T   = 1.0    # translation weight
LAMBDA_R   = 10.0   # rotation weight (radians are small, so upweight)
# ─────────────────────────────────────────────────────────────────────────────


def pose_loss(pred, gt):
    """
    Weighted L1 loss on translation + rotation separately.
    Args:
        pred: (B, 6)
        gt  : (B, 6)
    """
    t_loss = nn.functional.l1_loss(pred[:, :3], gt[:, :3])
    r_loss = nn.functional.l1_loss(pred[:, 3:], gt[:, 3:])
    return LAMBDA_T * t_loss + LAMBDA_R * r_loss, t_loss.item(), r_loss.item()


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"[train] Loading nuScenes {VERSION} from {DATAROOT} ...")
    nusc    = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    dataset = NuScenesPoseDataset(nusc, img_size=IMG_SIZE)

    n_val   = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"[train] Train pairs: {n_train} | Val pairs: {n_val}")
    print(f"[train] Device: {DEVICE}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = PoseNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_tl   = 0.0
        total_rl   = 0.0

        for batch in train_loader:
            img1    = batch['img1'].to(DEVICE)
            img2    = batch['img2'].to(DEVICE)
            pose_gt = batch['pose_gt'].to(DEVICE)

            optimizer.zero_grad()
            pred           = model(img1, img2)
            loss, tl, rl   = pose_loss(pred, pose_gt)
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tl   += tl
            total_rl   += rl

        avg_loss = total_loss / len(train_loader)
        avg_tl   = total_tl   / len(train_loader)
        avg_rl   = total_rl   / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                img1    = batch['img1'].to(DEVICE)
                img2    = batch['img2'].to(DEVICE)
                pose_gt = batch['pose_gt'].to(DEVICE)
                pred    = model(img1, img2)
                loss, _, _ = pose_loss(pred, pose_gt)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step()

        print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
              f"Train: {avg_loss:.4f} (t={avg_tl:.4f} r={avg_rl:.4f})  "
              f"Val: {val_loss:.4f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # ── Save best checkpoint ──────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(SAVE_DIR, "pose_net_best.pth")
            torch.save({
                'epoch'     : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss'  : val_loss,
                'img_size'  : IMG_SIZE,
            }, ckpt_path)
            print(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})")

    print(f"\n[train] Done. Best val loss: {best_val_loss:.4f}")
    print(f"[train] Checkpoint saved to: {SAVE_DIR}/pose_net_best.pth")


if __name__ == "__main__":
    train()