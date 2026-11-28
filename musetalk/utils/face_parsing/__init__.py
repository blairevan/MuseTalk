import torch
import time
import os
import cv2
import numpy as np
from PIL import Image
from .model import BiSeNet
from musetalk.utils.mask_utils import guard_jaw_dilation_mask
import torchvision.transforms as transforms

class FaceParsing():
    def __init__(self, left_cheek_width=80, right_cheek_width=80):
        self.net = self.model_init()
        self.preprocess = self.image_preprocess()
        # Ensure all size parameters are integers
        cone_height = 21
        tail_height = 12
        total_size = cone_height + tail_height
        
        # Create kernel with explicit integer dimensions
        kernel = np.zeros((total_size, total_size), dtype=np.uint8)
        center_x = total_size // 2  # Ensure center coordinates are integers
        
        # Cone part
        for row in range(cone_height):
            if row < cone_height//2:
                continue
            width = int(2 * (row - cone_height//2) + 1)
            start = int(center_x - (width // 2))
            end = int(center_x + (width // 2) + 1)
            kernel[row, start:end] = 1

        # Vertical extension part
        if cone_height > 0:
            base_width = int(kernel[cone_height-1].sum())
        else:
            base_width = 1
        
        for row in range(cone_height, total_size):
            start = max(0, int(center_x - (base_width//2)))
            end = min(total_size, int(center_x + (base_width//2) + 1))
            kernel[row, start:end] = 1
        self.kernel = kernel
        
        # Modify cheek erosion kernel to be flatter ellipse
        self.cheek_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (35, 3))
        
        # Add cheek area mask (protect chin area)
        self.cheek_mask = self._create_cheek_mask(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)
        
    def _create_cheek_mask(self, left_cheek_width=80, right_cheek_width=80):
        """Create cheek area mask (1/4 area on both sides)"""
        mask = np.zeros((512, 512), dtype=np.uint8)
        center = 512 // 2
        cv2.rectangle(mask, (0, 0), (center - left_cheek_width, 512), 255, -1)    # Left cheek
        cv2.rectangle(mask, (center + right_cheek_width, 0), (512, 512), 255, -1)  # Right cheek
        return mask

    def model_init(self, 
                   resnet_path='./models/face-parse-bisent/resnet18-5c106cde.pth', 
                   model_pth='./models/face-parse-bisent/79999_iter.pth'):
        net = BiSeNet(resnet_path)
        if torch.cuda.is_available():
            net.cuda()
            net.load_state_dict(torch.load(model_pth, weights_only=False)) 
        else:
            net.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu'), weights_only=False))
        net.eval()
        return net

    def image_preprocess(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def parse_classes(self, image, size=(512, 512)):
        """Return raw 19-class BiSeNet segmentation resized to ``image.size``.

        Args:
            image (PIL.Image.Image | str): Input RGB image or file path.
            size (tuple[int, int]): Internal BiSeNet evaluation resolution.

        Returns:
            PIL.Image.Image: Single-channel ``uint8`` class map matching ``image.size``.
        """
        if isinstance(image, str):
            with Image.open(image) as opened:
                image = opened.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        orig_size = image.size
        with torch.no_grad():
            resized = image.resize(size, Image.BILINEAR)
            img = torch.unsqueeze(self.preprocess(resized), 0)
            net_params = tuple(self.net.parameters()) if hasattr(self.net, "parameters") else ()
            if net_params:
                img = img.to(device=net_params[0].device)
            elif torch.cuda.is_available():
                img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0).astype(np.uint8)
        return Image.fromarray(parsing).resize(orig_size, Image.NEAREST)

    def __call__(self, image, size=(512, 512), mode="raw"):
        if isinstance(image, str):
            image = Image.open(image)

        width, height = image.size
        with torch.no_grad():
            image = image.resize(size, Image.BILINEAR)
            img = self.preprocess(image)
            if torch.cuda.is_available():
                img = torch.unsqueeze(img, 0).cuda()
            else:
                img = torch.unsqueeze(img, 0)
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
            # Add 14:neck, remove 10:nose and 7:8:9
            if mode == "neck":
                parsing[np.isin(parsing, [1, 11, 12, 13, 14])] = 255
                parsing[np.where(parsing!=255)] = 0
            elif mode == "jaw":
                face_region = np.isin(parsing, [1])*255
                face_region = face_region.astype(np.uint8)
                original_dilated = cv2.dilate(face_region, self.kernel, iterations=1)
                eroded = cv2.erode(original_dilated, self.cheek_kernel, iterations=2)
                face_region = cv2.bitwise_and(eroded, self.cheek_mask)
                face_region = cv2.bitwise_or(face_region, cv2.bitwise_and(original_dilated, ~self.cheek_mask))
                jaw_mask = np.zeros_like(parsing, dtype=np.uint8)
                jaw_mask[(face_region == 255) & (~np.isin(parsing, [10]))] = 255
                jaw_mask[np.isin(parsing, [11, 12, 13])] = 255
                parsing = guard_jaw_dilation_mask(parsing, jaw_mask)
            elif mode == "mouth":
                # Mouth interior only (class 11), used to locate generated teeth
                parsing[np.isin(parsing, [11])] = 255
                parsing[np.where(parsing!=255)] = 0
            else:
                parsing[np.isin(parsing, [1, 11, 12, 13])] = 255
                parsing[np.where(parsing!=255)] = 0

        parsing = Image.fromarray(parsing.astype(np.uint8))
        return parsing

if __name__ == "__main__":
    fp = FaceParsing()
    segmap = fp('154_small.png')
    segmap.save('res.png')
    
