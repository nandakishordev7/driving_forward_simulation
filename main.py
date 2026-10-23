import os, cv2, numpy as np, torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyquaternion import Quaternion
from nu_scene_loader import NuScenesLoader
from gaussian_network import DepthNetwork, GaussianNetwork
from gaussian_renderer import render_bev_gpu, render_bev_numpy, HAS_DGR
from refine_net import RefineNet
from interactive_viewer import launch_viewer

DATAROOT = r"C:\Projects\driving_forward_conda\nu_scenes"
POSE_CKPT = r"C:\Projects\driving_forward_conda\driving_forward_simulation\checkpoints\pose_net_best.pth"
GAUSS_CKPT = r"C:\Projects\driving_forward_conda\driving_forward_simulation\checkpoints\gaussian_net_best.pth"
BEV_OUT = r"C:\Projects\driving_forward_conda\driving_forward_simulation\bev_output.png"
IMG_SIZE       = (192, 640)
BEV_RANGE      = 50.0
BEV_SIZE       = 768
STRIDE         = 2
N_REFINE_ITERS = 3       # must match what was used during training
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(img_bgr, img_size):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size[1], img_size[0]))
    t   = torch.from_numpy(img.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def get_cam_to_ego(nusc, sd_token):
    sd    = nusc.get('sample_data', sd_token)
    calib = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    R     = Quaternion(calib['rotation']).rotation_matrix.astype(np.float32)
    t     = np.array(calib['translation'], dtype=np.float32)
    K     = np.array(calib['camera_intrinsic'], dtype=np.float32)
    return R, t, K


def load_models():
    depth_net  = DepthNetwork().to(DEVICE)
    gauss_net  = GaussianNetwork().to(DEVICE)
    refine_net = RefineNet(hidden=32, max_delta=2.0).to(DEVICE)

    gc = torch.load(GAUSS_CKPT, map_location=DEVICE)
    depth_net.load_state_dict(gc['depth_net_state'])
    gauss_net.load_state_dict(gc['gauss_net_state'])
    refine_net.load_state_dict(gc['refine_net_state'])

    depth_net.eval()
    gauss_net.eval()
    refine_net.eval()
    print(f"[main] Models loaded (epoch {gc['epoch']}, val={gc['val_loss']:.4f})")
    return depth_net, gauss_net, refine_net


@torch.no_grad()
def process_camera(cam_name, img_bgr, nusc, sample,
                   depth_net, gauss_net, refine_net):

    sd_token            = sample['data'][cam_name]
    R_c2e, t_c2e, K_np = get_cam_to_ego(nusc, sd_token)
    H_in, W_in          = IMG_SIZE
    orig_h, orig_w      = img_bgr.shape[:2]

    # scale intrinsics to network input resolution
    K_t = torch.tensor(K_np, dtype=torch.float32, device=DEVICE)
    K_t[0] *= W_in / orig_w
    K_t[1] *= H_in / orig_h
    K_t = K_t.unsqueeze(0)

    img_t = preprocess(img_bgr, IMG_SIZE)

    # ── D0: initial depth ─────────────────────────────────────────────────────
    depth = depth_net(img_t)          # (1, H, W)

    # ── cache image features once ────────────────────────────────────────────
    f3 = gauss_net.encode(img_t)      # (1, 256, H/16, W/16)

    # ── iterative geometry refinement ────────────────────────────────────────
    # At inference we have no ground-truth neighbour frame, so render_err
    # is set to zeros. RefineNet still applies its learned spatial corrections
    # because the depth itself carries the signal through BN statistics.
    for k in range(N_REFINE_ITERS):
        render_err = torch.zeros_like(img_t)   # (1,3,H,W) — no GT available
        depth      = refine_net(depth, render_err)

    # ── final Gaussians from refined depth ───────────────────────────────────
    gauss = gauss_net.decode(f3, depth, K_t)

    img_resized = cv2.resize(img_bgr, (W_in, H_in))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_sh      = (img_rgb.reshape(-1, 3) - 0.5) / 0.282

    xyz       = gauss['xyz'][0].cpu().numpy()
    opacity   = gauss['opacity'][0].cpu().numpy()
    scales    = gauss['scales'][0].cpu().numpy()
    rotations = gauss['rotations'][0].cpu().numpy()

    idx = np.arange(0, xyz.shape[0], STRIDE)
    xyz, opacity   = xyz[idx], opacity[idx]
    scales, rotations = scales[idx], rotations[idx]
    sh_coeffs      = img_sh[idx]

    xyz_ego = (R_c2e @ xyz.T).T + t_c2e

    print(f"  [{cam_name}] {len(xyz_ego):,} Gaussians | "
          f"depth {depth[0].min().item():.2f}–{depth[0].max().item():.2f}m")

    return {
        'xyz'       : xyz_ego,
        'opacity'   : opacity,
        'scales'    : scales,
        'rotations' : rotations,
        'sh_coeffs' : sh_coeffs,
    }


def merge_gaussians(gauss_list):
    return {k: np.concatenate([g[k] for g in gauss_list], axis=0)for k in gauss_list[0]}

def save_bev(bev_img, save_path):
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    ax.imshow(bev_img, origin='upper',extent=[-BEV_RANGE, BEV_RANGE, -BEV_RANGE, BEV_RANGE]) # type: ignore
    ax.set_xlabel("X  (forward →)  [m]", color='white', fontsize=11)
    ax.set_ylabel("Y  (← left  |  right →)  [m]", color='white', fontsize=11)
    ax.tick_params(colors='white')
    for s in ax.spines.values():
        s.set_edgecolor('#444')
    ax.set_title("Full 360° BEV — Gaussian Splatting Render (with Refinement)",
                 color='white', fontsize=13, pad=10)
    ax.legend(handles=[
        patches.Patch(color='#00ff00', label='Ego vehicle (forward = up)'),
        patches.Patch(color='#aaaaaa', label='Gaussian primitives'),
    ], loc='lower right', facecolor='#1a1a1a', edgecolor='#555',
       labelcolor='white', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[main] BEV saved → {save_path}")


def main():
    loader = NuScenesLoader(dataroot=DATAROOT)
    nusc   = loader.nusc
    sample = nusc.sample[0]
    images = loader.get_sample_images(sample['token'])

    depth_net, gauss_net, refine_net = load_models()

    gauss_list = []
    for cam_name, img_bgr in images:
        print(f"\nProcessing {cam_name} ...")
        g = process_camera(cam_name, img_bgr, nusc, sample,
                           depth_net, gauss_net, refine_net)
        gauss_list.append(g)

    all_g = merge_gaussians(gauss_list)
    print(f"\n[main] Total Gaussians: {len(all_g['xyz']):,}")

    gauss_t = {k: torch.from_numpy(v).float().to(DEVICE)
               for k, v in all_g.items()}
    bev_img = (render_bev_gpu(gauss_t, bev_range=BEV_RANGE,
                               bev_size=BEV_SIZE, device=DEVICE)
               if HAS_DGR else
               render_bev_numpy(all_g, bev_range=BEV_RANGE, bev_size=BEV_SIZE))
    save_bev(bev_img, BEV_OUT)

    print("\n[main] Launching interactive 3D viewer ...")
    launch_viewer(all_g)

if __name__ == "__main__":
    main()