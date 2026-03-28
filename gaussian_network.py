import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthEncoder(nn.Module):
    """
    Shared CNN encoder used by both DepthNetwork and GaussianNetwork.
    ResNet-style blocks, lightweight enough to run on a single GPU.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64,  64,  2)
        self.layer2 = self._make_layer(64,  128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

    @staticmethod
    def _make_layer(in_ch, out_ch, n_blocks, stride=1):
        layers = [nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )]
        for _ in range(n_blocks - 1):
            layers.append(nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        f0 = self.layer0(x)   # /4
        f1 = self.layer1(f0)  # /4
        f2 = self.layer2(f1)  # /8
        f3 = self.layer3(f2)  # /16
        f4 = self.layer4(f3)  # /32
        return f1, f2, f3, f4


class DepthDecoder(nn.Module):
    """
    U-Net decoder that produces a metric-scale depth map.
    """
    def __init__(self):
        super().__init__()
        self.up4 = self._up(512, 256)
        self.up3 = self._up(256 + 256, 128)
        self.up2 = self._up(128 + 128, 64)
        self.up1 = self._up(64  + 64,  32)
        self.head = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Softplus()   # positive depth, smooth gradients
        )

    @staticmethod
    def _up(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, feats, orig_h, orig_w):
        f1, f2, f3, f4 = feats
        x = self.up4(f4)
        x = self.up3(torch.cat([x, f3], dim=1))
        x = self.up2(torch.cat([x, f2], dim=1))
        x = self.up1(torch.cat([x, f1], dim=1))
        x = self.head(x)
        x = F.interpolate(x, size=(orig_h, orig_w),
                          mode='bilinear', align_corners=False)
        return x.squeeze(1)   # (B, H, W)


class DepthNetwork(nn.Module):
    """
    Self-supervised depth estimation network.
    Replaces MiDaS in the pipeline — outputs metric-consistent depth.
    """
    def __init__(self):
        super().__init__()
        self.encoder = DepthEncoder()
        self.decoder = DepthDecoder()

    def forward(self, img):
        """
        Args:
            img : (B, 3, H, W) normalised RGB [0, 1]
        Returns:
            depth : (B, H, W) positive depth values
        """
        h, w   = img.shape[2], img.shape[3]
        feats  = self.encoder(img)
        depth  = self.decoder(feats, h, w)
        return depth


# ── Gaussian parameter heads ───────────────────────────────────────────────

class GaussianNetwork(nn.Module):
    """
    Given a single RGB image + its depth map, predicts per-pixel
    3D Gaussian primitive parameters:
        - xyz        : 3D position  (from depth + camera intrinsics)
        - opacity    : (H*W, 1)
        - scales     : (H*W, 3)  log-space
        - rotations  : (H*W, 4)  quaternion
        - sh_coeffs  : (H*W, 3)  degree-0 spherical harmonics (colour)

    This is a simplified version of the Gaussian network in DrivingForward —
    it uses the shared encoder + lightweight parameter heads.
    """
    def __init__(self, sh_degree=0):
        super().__init__()
        self.encoder = DepthEncoder()

        # Parameter heads (applied at /8 resolution, then upsampled)
        self.opacity_head  = self._head(256, 1)
        self.scale_head    = self._head(256, 3)
        self.rot_head      = self._head(256, 4)
        sh_out = 3 * (sh_degree + 1) ** 2
        self.sh_head       = self._head(256, sh_out)

    @staticmethod
    def _head(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, img, depth, K):
        """
        Args:
            img   : (B, 3, H, W)
            depth : (B, H, W)
            K     : (B, 3, 3) camera intrinsics
        Returns dict with keys: xyz, opacity, scales, rotations, sh_coeffs
        """
        B, _, H, W = img.shape
        _, _, f3, _ = self.encoder(img)   # use /16 features

        def _decode(head, act=None):
            x = head(f3)
            x = F.interpolate(x, (H, W), mode='bilinear', align_corners=False)
            if act: x = act(x)
            return x.permute(0, 2, 3, 1).reshape(B, H * W, -1)

        opacity   = _decode(self.opacity_head,  torch.sigmoid)      # (B, H*W, 1)
        scales    = _decode(self.scale_head,    torch.exp)           # (B, H*W, 3)  positive
        rotations = _decode(self.rot_head)                           # (B, H*W, 4)
        rotations = F.normalize(rotations, dim=-1)                   # unit quaternion
        sh_coeffs = _decode(self.sh_head)                            # (B, H*W, C)

        # ── Unproject depth to 3D using camera intrinsics ─────────────────
        ys, xs = torch.meshgrid(
            torch.arange(H, device=img.device, dtype=torch.float32),
            torch.arange(W, device=img.device, dtype=torch.float32),
            indexing='ij'
        )
        xs = xs.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        ys = ys.unsqueeze(0).expand(B, -1, -1)

        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)

        Z = depth                             # (B, H, W)
        X = (xs - cx) * Z / fx
        Y = (ys - cy) * Z / fy

        xyz = torch.stack([X, Y, Z], dim=-1)  # (B, H, W, 3)
        xyz = xyz.reshape(B, H * W, 3)        # (B, H*W, 3)

        return {
            'xyz'       : xyz,
            'opacity'   : opacity,
            'scales'    : scales,
            'rotations' : rotations,
            'sh_coeffs' : sh_coeffs,
        }