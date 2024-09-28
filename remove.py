import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

torch.set_float32_matmul_precision('high')
torch.jit.script = lambda f: f

device = "cuda" if torch.cuda.is_available() else "cpu"


def array_to_pil_image(image, size=(1024, 1024)):
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = Image.fromarray(image).convert('RGB')
    return image


class ImagePreprocessor():
    def __init__(self, resolution=(1024, 1024)) -> None:
        self.transform_image = transforms.Compose([
            # transforms.Resize(resolution),    # 1. keep consistent with the cv2.resize used in training 2. redundant with that in path_to_image()
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def proc(self, image):
        image = self.transform_image(image)
        return image


from transformers import AutoModelForImageSegmentation
weights_path = 'zhengpeng7/BiRefNet'
birefnet = AutoModelForImageSegmentation.from_pretrained(weights_path, trust_remote_code=True)
birefnet.to(device)
birefnet.eval()


def predict(image, resolution, weights_file):
    global weights_path
    global birefnet
    if weights_file != weights_path:
        # Load BiRefNet with chosen weights
        birefnet = AutoModelForImageSegmentation.from_pretrained(weights_file if weights_file is not None else 'zhengpeng7/BiRefNet', trust_remote_code=True)
        birefnet.to(device)
        birefnet.eval()

    resolution = f"{image.shape[1]}x{image.shape[0]}" if resolution == '' else resolution
    # Image is a RGB numpy array.
    resolution = [int(int(reso)//32*32) for reso in resolution.strip().split('x')]
    images = [image]
    image_shapes = [image.shape[:2] for image in images]
    images = [array_to_pil_image(image, resolution) for image in images]

    image_preprocessor = ImagePreprocessor(resolution=resolution)
    images_proc = []
    for image in images:
        images_proc.append(image_preprocessor.proc(image))
    images_proc = torch.cat([image_proc.unsqueeze(0) for image_proc in images_proc])

    with torch.no_grad():
        scaled_preds_tensor = birefnet(images_proc.to(device))[-1].sigmoid()   # BiRefNet needs an sigmoid activation outside the forward.
    preds = []
    for image_shape, pred_tensor in zip(image_shapes, scaled_preds_tensor):
        if device == 'cuda':
            pred_tensor = pred_tensor.cpu()
        preds.append(torch.nn.functional.interpolate(pred_tensor.unsqueeze(0), size=image_shape, mode='bilinear', align_corners=True).squeeze().numpy())
    image_preds = []
    for image, pred in zip(images, preds):
        image = image.resize(pred.shape[::-1])
        pred = np.repeat(np.expand_dims(pred, axis=-1), 3, axis=-1)
        image_preds.append((pred * image).astype(np.uint8))

    return image, image_preds[0]

img = cv2.imread("test.jpg")

img, img_new = predict(img, '', 'zhengpeng7/BiRefNet')

cv2.imwrite("testafter.jpg", img_new)