import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseNet(nn.Module):
    """
    Lightweight PoseNet-style CNN that takes two stacked RGB images
    (6 channels) and predicts the relative 6-DoF pose between them.

    Output: [tx, ty, tz, rx, ry, rz]
        - translation (tx, ty, tz) in metres
        - rotation    (rx, ry, rz) as axis-angle in radians
    """

    def __init__(self):
        super(PoseNet, self).__init__()

        # -- Encoder: shared CNN backbone --
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(6, 32, kernel_size=7, stride=2, padding=3),   # /2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # /4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # /8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # /16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 5
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # /32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 6
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # /64
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # always 256x4x4 = 4096

        # -- Pose regression head --
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
        )

        # Separate heads for translation and rotation
        # (avoids one dominating the other during training)
        self.fc_translation = nn.Linear(128, 3)
        self.fc_rotation    = nn.Linear(128, 3)

        self._init_weights()

    def _init_weights(self):
        # Zero-init the final layers so the network starts near identity pose
        nn.init.zeros_(self.fc_translation.weight)
        nn.init.zeros_(self.fc_translation.bias)
        nn.init.zeros_(self.fc_rotation.weight)
        nn.init.zeros_(self.fc_rotation.bias)

    def forward(self, img1, img2):
        """
        Args:
            img1: (B, 3, H, W) reference frame
            img2: (B, 3, H, W) target frame
        Returns:
            pose: (B, 6)  [tx, ty, tz, rx, ry, rz]
        """
        x = torch.cat([img1, img2], dim=1)   # (B, 6, H, W)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.regressor(x)

        translation = self.fc_translation(x)
        rotation    = self.fc_rotation(x) * 0.01   # scale down rotation init

        pose = torch.cat([translation, rotation], dim=1)  # (B, 6)
        return pose


def pose_vec_to_matrix(pose_vec):
    """
    Convert a (B, 6) pose vector [tx, ty, tz, rx, ry, rz] to a
    (B, 4, 4) transformation matrix using axis-angle → rotation matrix.

    Args:
        pose_vec: torch.Tensor (B, 6)
    Returns:
        T: torch.Tensor (B, 4, 4)
    """
    B = pose_vec.shape[0]
    t = pose_vec[:, :3]          # (B, 3)
    r = pose_vec[:, 3:]          # (B, 3)  axis-angle

    angle = torch.norm(r, dim=1, keepdim=True).clamp(min=1e-6)   # (B, 1)
    axis  = r / angle                                              # (B, 3)

    cos_a = torch.cos(angle)     # (B, 1)
    sin_a = torch.sin(angle)     # (B, 1)
    K     = _skew_symmetric(axis)  # (B, 3, 3)

    # Rodrigues' rotation formula: R = I + sin(a)*K + (1-cos(a))*K^2
    I = torch.eye(3, device=pose_vec.device).unsqueeze(0).expand(B, -1, -1)
    R = I + sin_a.unsqueeze(-1) * K + (1 - cos_a).unsqueeze(-1) * (K @ K)

    T = torch.eye(4, device=pose_vec.device).unsqueeze(0).expand(B, -1, -1).clone()
    T[:, :3, :3] = R
    T[:, :3,  3] = t

    return T


def _skew_symmetric(v):
    """
    Build skew-symmetric matrix from (B, 3) vectors.
    Returns (B, 3, 3).
    """
    B  = v.shape[0]
    vx = v[:, 0]
    vy = v[:, 1]
    vz = v[:, 2]
    z  = torch.zeros(B, device=v.device)

    K = torch.stack([
         z,  -vz,  vy,
        vz,    z, -vx,
       -vy,   vx,   z,
    ], dim=1).view(B, 3, 3)

    return K