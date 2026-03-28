import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pyquaternion import Quaternion


class NuScenesPoseDataset(Dataset):
    """
    Dataset that yields consecutive camera frame pairs with their
    relative ground-truth 6-DoF pose from nuScenes metadata.

    Each item:
        img1       : (3, H, W) float32 tensor  — reference frame
        img2       : (3, H, W) float32 tensor  — target frame
        pose_gt    : (6,)      float32 tensor  — [tx, ty, tz, rx, ry, rz]
        cam_name   : str                        — camera identifier
    """

    def __init__(self, nusc, img_size=(192, 640), cameras=None):
        """
        Args:
            nusc      : NuScenes object (already initialised)
            img_size  : (H, W) to resize images to
            cameras   : list of camera names to include
                        (defaults to all 6 surround cameras)
        """
        self.nusc     = nusc
        self.img_size = img_size
        self.cameras  = cameras or [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT',
        ]
        self.pairs = self._build_pairs()
        print(f"[PoseDataset] Built {len(self.pairs)} training pairs "
              f"from {len(nusc.sample)} samples × {len(self.cameras)} cameras.")

    # ------------------------------------------------------------------
    def _build_pairs(self):
        """
        Walk through every sample and collect consecutive frame pairs
        for each camera.  A pair is (token_A, token_B, cam_name).
        """
        pairs = []
        for sample in self.nusc.sample:
            for cam in self.cameras:
                if cam not in sample['data']:
                    continue
                sd_token = sample['data'][cam]
                sd       = self.nusc.get('sample_data', sd_token)
                if sd['next'] == '':
                    continue   # no next frame
                pairs.append((sd_token, sd['next'], cam))
        return pairs

    # ------------------------------------------------------------------
    def _load_image(self, sd_token):
        sd       = self.nusc.get('sample_data', sd_token)
        img_path = self.nusc.get_sample_data_path(sd_token)
        img      = cv2.imread(img_path)
        img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img      = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img      = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)   # (3, H, W)

    # ------------------------------------------------------------------
    def _get_pose_matrix(self, sd_token):
        """
        Return the 4x4 world-to-camera transform for a sample_data token.
        """
        sd         = self.nusc.get('sample_data', sd_token)
        calib      = self.nusc.get('calibrated_sensor',
                                   sd['calibrated_sensor_token'])
        ego_pose   = self.nusc.get('ego_pose', sd['ego_pose_token'])

        # Sensor → ego
        R_sc = Quaternion(calib['rotation']).rotation_matrix
        t_sc = np.array(calib['translation'])

        # Ego → global
        R_eg = Quaternion(ego_pose['rotation']).rotation_matrix
        t_eg = np.array(ego_pose['translation'])

        # Build 4x4: sensor → global
        T_sc = np.eye(4); T_sc[:3, :3] = R_sc; T_sc[:3, 3] = t_sc
        T_eg = np.eye(4); T_eg[:3, :3] = R_eg; T_eg[:3, 3] = t_eg

        T_global = T_eg @ T_sc   # sensor-to-global
        return T_global

    # ------------------------------------------------------------------
    def _relative_pose_vec(self, T1, T2):
        """
        Compute relative transform T_rel = inv(T1) @ T2,
        then extract [tx, ty, tz, rx, ry, rz] (axis-angle).
        """
        T_rel = np.linalg.inv(T1) @ T2
        t     = T_rel[:3, 3]
        R     = T_rel[:3, :3]

        # Rotation matrix → axis-angle
        angle = np.arccos(
            np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
        )
        if abs(angle) < 1e-6:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ]) / (2 * np.sin(angle))

        r_vec = axis * angle
        return np.concatenate([t, r_vec]).astype(np.float32)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tok1, tok2, cam = self.pairs[idx]

        img1 = self._load_image(tok1)
        img2 = self._load_image(tok2)

        T1   = self._get_pose_matrix(tok1)
        T2   = self._get_pose_matrix(tok2)
        pose = self._relative_pose_vec(T1, T2)

        return {
            'img1'    : img1,
            'img2'    : img2,
            'pose_gt' : torch.from_numpy(pose),
            'cam_name': cam,
        }