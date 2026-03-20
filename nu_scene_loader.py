from nuscenes.nuscenes import NuScenes
import cv2
import os

class NuScenesLoader:
    def __init__(self, dataroot):
        self.nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)

    def get_sample_images(self, sample_token):
        sample = self.nusc.get('sample', sample_token)

        camera_list = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'
        ]

        images = []

        for cam in camera_list:
            cam_data = self.nusc.get('sample_data', sample['data'][cam])
            img_path = os.path.join(self.nusc.dataroot, cam_data['filename'])

            img = cv2.imread(img_path)
            images.append((cam, img))

        return images