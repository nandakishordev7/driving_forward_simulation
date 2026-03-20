import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self):
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.eval()

        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    # Existing function (for file path)
    def predict(self, image_path):
        img = cv2.imread(image_path)
        return self.predict_from_array(img)

    # new function (for array input)
    def predict_from_array(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        # Normalize for visualization
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)

        return img, depth, depth_norm