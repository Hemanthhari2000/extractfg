import torch
import torchvision
from torchvision import transforms

from PIL import Image
import numpy as np
import cv2
import os


class ExtractFG():
    def __init__(self, input_image_path: str, output_image_path: str):
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(
            pretrained=True).eval()
        self.ROOT_DIR = os.getcwd()

    def get_devices():
        return torch.cuda.is_available()

    def resize_with_propotions(self, image, dims, fh=400):
        height_percent = fh/float(dims[1])
        width_size = int(float(dims[0]) * float(height_percent))
        return image.resize((width_size, fh), Image.NEAREST)

    def get_output_mask(self):
        input_image = Image.open(os.path.join(
            self.ROOT_DIR, self.input_image_path))
        input_image_size = input_image.size

        input_image = self.resize_with_propotions(
            input_image, input_image_size, fh=500)

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        if self.get_devices:
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')
            print("[INFO] Model Loaded to GPU")

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]

        output_predictions = output.argmax(0)
        mask = np.uint8(output_predictions.cpu() * 255)

        return input_image, mask

    def saveImage(self, image):
        cv2.imwrite(os.path.join(self.ROOT_DIR, self.output_image_path), image)

    def maskWithImage(self, inp_image, mask):
        masked_img = cv2.bitwise_and(inp_image, inp_image, mask=mask)
        rgb = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)
        bgra = cv2.cvtColor(rgb, cv2.COLOR_BGR2BGRA)
        alpha = bgra[:, :, 3]
        alpha[np.all(bgra[:, :, 0:3] == (0, 0, 0), 2)] = 0
        return bgra

    def getExtractedImage(self):
        input_image, mask = self.get_output_mask()
        input_image, mask = np.array(input_image), np.array(mask)
        masked_image = self.maskWithImage(input_image, mask)
        self.saveImage(masked_image)
        print("[INFO] Masked Image is saved")


# extract_fg = ExtractFG(input_image_path='uploads\input.png',
#                        output_image_path='masked.png')

# extract_fg.getExtractedImage()
