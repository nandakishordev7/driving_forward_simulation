"""
refine_net.py  —  Iterative Geometry Refinement Network

Loop (run K times after initial DepthNetwork forward pass):
    f3           = GaussianNet.encode(img)          # cached once
    µₖ           = Unproject(Dₖ, K_intrinsics)
    Gaussians    = GaussianNet.decode(f3, Dₖ)
    render_err   = warp(img2, Dₖ) - img1           # photometric error proxy
    ΔDₖ          = RefineNet(Dₖ, render_err)
    Dₖ₊₁        = Dₖ + ΔDₖ   (via softplus to keep positive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefineNet(nn.Module):
    """
    Thin residual depth-correction network.

    Inputs:
        depth_k   : (B, H, W)   current depth estimate
        render_err: (B, 3, H, W) photometric error  (Irender - Igt)

    Output:
        depth_next: (B, H, W)   refined depth (always positive)

    ~50K parameters — intentionally lightweight so it doesn't
    dominate training compute.
    """

    def __init__(self, hidden=32, max_delta=2.0):
        super().__init__()
        self.max_delta = max_delta

        self.net = nn.Sequential(
            # fuse depth (1ch) + render error (3ch)
            nn.Conv2d(4, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            # depthwise-separable block 1
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.Conv2d(hidden, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            # depthwise-separable block 2
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.Conv2d(hidden, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            # output head — bounded correction
            nn.Conv2d(hidden, 1, 1),
            nn.Tanh(),
        )

        # zero-init output so first iteration starts as identity
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, depth_k, render_err):
        """
        Args:
            depth_k   : (B, H, W) or (B, 1, H, W)
            render_err: (B, 3, H, W)
        Returns:
            depth_next: (B, H, W)
        """
        if depth_k.dim() == 3:
            depth_k = depth_k.unsqueeze(1)          # (B,1,H,W)

        x       = torch.cat([depth_k, render_err], dim=1)  # (B,4,H,W)
        delta   = self.net(x) * self.max_delta              # (B,1,H,W)
        refined = F.softplus(depth_k + delta)               # always positive
        return refined.squeeze(1)                           # (B,H,W)