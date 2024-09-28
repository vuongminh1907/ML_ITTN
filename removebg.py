import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# Set torch precision
torch.set_float32_matmul_precision('high')
torch.jit.script = lambda f: f

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Helper function to convert array to PIL image
def array_to_pil_image(image, size=(1024, 1024)):
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = Image.fromarray(image).convert('RGB')
    return image

# Image preprocessor class
class ImagePreprocessor():
    def __init__(self, resolution=(1024, 1024)) -> None:
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def proc(self, image):
        return self.transform_image(image)

# Load models
weights_path = 'zhengpeng7/BiRefNet'
birefnet = AutoModelForImageSegmentation.from_pretrained(weights_path, trust_remote_code=True)
birefnet.to(device)
birefnet.eval()
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s_personface.pt")

# Prediction function
def predict(image, resolution, weights_file):
    global weights_path, birefnet, model
    
    #get the size of the image
    h, w, _ = image.shape

    # Update model weights if necessary
    if weights_file != weights_path:
        birefnet = AutoModelForImageSegmentation.from_pretrained(weights_file if weights_file is not None else 'zhengpeng7/BiRefNet', trust_remote_code=True)
        birefnet.to(device)
        birefnet.eval()
        weights_path = weights_file

    results = model(image)
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        if cls == 1:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1 = max(0, x1 - int((x2 - x1) * 0.1))
        y1 = max(0, y1 - int((y2 - y1) * 0.1))
        x2 = min(image.shape[1], x2 + int((x2 - x1) * 0.1))
        y2 = min(image.shape[0], y2 + int((y2 - y1) * 0.1))
        image = image[y1:y2, x1:x2]
        break
    cv2.imwrite("testbefore.jpg", image)


    resolution = f"{image.shape[1]}x{image.shape[0]}" if resolution == '' else resolution
    resolution = [int(int(reso)//32*32) for reso in resolution.strip().split('x')]
    images = [array_to_pil_image(image, resolution)]

    image_preprocessor = ImagePreprocessor(resolution=resolution)
    images_proc = torch.cat([image_preprocessor.proc(img).unsqueeze(0) for img in images])

    with torch.no_grad():
        scaled_preds_tensor = birefnet(images_proc.to(device))[-1].sigmoid()

    preds = [torch.nn.functional.interpolate(pred_tensor.unsqueeze(0), size=image.shape[:2], mode='bilinear', align_corners=True).squeeze().cpu().numpy()
             for pred_tensor in scaled_preds_tensor]

    image_preds = []
    for image, pred in zip(images, preds):
        image = image.resize(pred.shape[::-1])
        pred = np.repeat(pred[..., None], 3, axis=-1)
        pred_binary = (pred > 0.5).astype(np.uint8)
        white_background = np.ones_like(np.array(image)) * 255
        image_pred = np.where(pred_binary, np.array(image), white_background)
        image_preds.append(image_pred)



    return np.array(image), image_pred[0]

# Load and process image
img = cv2.imread("c1.jpeg")
img, img_new = predict(img, '', 'zhengpeng7/BiRefNet')
cv2.imwrite("testafter.jpg", img_new)
