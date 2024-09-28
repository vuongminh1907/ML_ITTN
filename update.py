import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from diffusers import StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
weights_path = 'zhengpeng7/BiRefNet'
birefnet = AutoModelForImageSegmentation.from_pretrained(weights_path, trust_remote_code=True)
birefnet.to(device)
birefnet.eval()
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s_personface.pt")

pipe_refiner = StableDiffusionInpaintPipeline.from_pretrained(
            'SG161222/Realistic_Vision_V5.1_noVAE', 
            torch_dtype=torch.float16,
        ).to("cuda") 

def detect_edges(mask, dilation_iterations=1):
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_mask, 100, 200)
    kernel = np.ones((7, 7), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=dilation_iterations)
    return dilated_edges

def array_to_pil_image(image, size=(1024, 1024)):
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = Image.fromarray(image).convert('RGB')
    return image

class ImagePreprocessor():
    def __init__(self, resolution=(1024, 1024)) -> None:
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def proc(self, image):
        return self.transform_image(image)

# Hàm này là remove background để đưa về nền trắng
def RemoveBackGround(image):
    global birefnet, model, pipe_refiner
    
    final_image = np.ones_like(image) * 255
    black_mask = np.ones_like(image)

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

    resolution = f"{image.shape[1]}x{image.shape[0]}"
    resolution = [int(int(reso)//32*32) for reso in resolution.strip().split('x')]
    images = [array_to_pil_image(image, resolution)]

    image_preprocessor = ImagePreprocessor(resolution=resolution)
    images_proc = torch.cat([image_preprocessor.proc(img).unsqueeze(0) for img in images])

    with torch.no_grad():
        scaled_preds_tensor = birefnet(images_proc.to(device))[-1].sigmoid()

    preds = [torch.nn.functional.interpolate(pred_tensor.unsqueeze(0), size=image.shape[:2], mode='bilinear', align_corners=True).squeeze().cpu().numpy()
             for pred_tensor in scaled_preds_tensor]

    image_preds = []
    masks = []
    for image, pred in zip(images, preds):
        image = image.resize(pred.shape[::-1])
        pred = np.repeat(pred[..., None], 3, axis=-1)
        pred_binary = (pred > 0.5).astype(np.uint8)
        white_background = np.ones_like(np.array(image)) * 255
        image_pred = np.where(pred_binary, np.array(image), white_background)
        image_preds.append(image_pred)
        masks.append(pred_binary * 255)
    
    final_image[y1:y2, x1:x2]=image_preds[0]
    black_mask[y1:y2, x1:x2]= masks[0]
    
    
    #load scheduler
    scheduler = EulerAncestralDiscreteScheduler()
    
    # blur mask
    mask_image = pipe_refiner.mask_processor.blur(black_mask, blur_factor=5)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    image_remove_person = pipe_refiner(
        prompt="",
        width=768,
        height=1024,
        image=image,
        num_inference_steps=30,
        guidance_scale=7,
        strength=1.0,
        mask_image=mask_image,
        scheduler=scheduler,
        padding_mask_crop = 32,
    )
    
    
    return final_image,image_remove_person  # Trả về ảnh nền trắng và ảnh mask 


# Hàm này và để paste ảnh sau khi swap vào ảnh background tức là ảnh gốc và trả về thêm ảnh mask viền
def PastingAndMask(image, background):
    global birefnet

    resolution = f"{image.shape[1]}x{image.shape[0]}"
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
        pred_binary = (pred > 0.5).astype(np.uint8)
        mask = np.repeat(pred_binary[..., None], 3, axis=-1)
        edges = detect_edges(mask*255)
        segmented_part = mask * np.array(image)
        alpha = mask
        combined_image = alpha * segmented_part + (1 - alpha) * background
        image_preds.append(combined_image.astype(np.uint8))
        break
        
    # Trả về ảnh sau khi paste và ảnh mask viền
    return image_preds[0],edges



##################################################################
# Try example of 2 function
# Load and process image
img = cv2.imread("image_swapp.png") # Image is swapped
background = cv2.imread("c1.jpeg")
background=cv2.resize(background, (768,1024))

##################################################################
# Remove background
remove_image, mask = RemoveBackGround(background)
cv2.imwrite("RemoveBackGround.jpg", remove_image)
cv2.imwrite("mask_before.jpg", mask)
# Pasting and mask
img_new, mask = PastingAndMask(img,background)
cv2.imwrite("testafter.jpg", img_new)
cv2.imwrite("mask.jpg", mask)import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from diffusers import StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
weights_path = 'zhengpeng7/BiRefNet'
birefnet = AutoModelForImageSegmentation.from_pretrained(weights_path, trust_remote_code=True)
birefnet.to(device)
birefnet.eval()
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s_personface.pt")

pipe_refiner = StableDiffusionInpaintPipeline.from_pretrained(
            'SG161222/Realistic_Vision_V5.1_noVAE', 
            torch_dtype=torch.float16,
        ).to("cuda") 

def detect_edges(mask, dilation_iterations=1):
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_mask, 100, 200)
    kernel = np.ones((7, 7), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=dilation_iterations)
    return dilated_edges

def array_to_pil_image(image, size=(1024, 1024)):
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = Image.fromarray(image).convert('RGB')
    return image

class ImagePreprocessor():
    def __init__(self, resolution=(1024, 1024)) -> None:
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def proc(self, image):
        return self.transform_image(image)

# Hàm này là remove background để đưa về nền trắng
def RemoveBackGround(image):
    global birefnet, model, pipe_refiner

    original_image = image.copy()
    
    final_image = np.ones_like(image) * 255
    black_mask = np.ones_like(image)

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

    resolution = f"{image.shape[1]}x{image.shape[0]}"
    resolution = [int(int(reso)//32*32) for reso in resolution.strip().split('x')]
    images = [array_to_pil_image(image, resolution)]

    image_preprocessor = ImagePreprocessor(resolution=resolution)
    images_proc = torch.cat([image_preprocessor.proc(img).unsqueeze(0) for img in images])

    with torch.no_grad():
        scaled_preds_tensor = birefnet(images_proc.to(device))[-1].sigmoid()

    preds = [torch.nn.functional.interpolate(pred_tensor.unsqueeze(0), size=image.shape[:2], mode='bilinear', align_corners=True).squeeze().cpu().numpy()
             for pred_tensor in scaled_preds_tensor]

    image_preds = []
    masks = []
    for image, pred in zip(images, preds):
        image = image.resize(pred.shape[::-1])
        pred = np.repeat(pred[..., None], 3, axis=-1)
        pred_binary = (pred > 0.5).astype(np.uint8)
        white_background = np.ones_like(np.array(image)) * 255
        image_pred = np.where(pred_binary, np.array(image), white_background)
        image_preds.append(image_pred)
        masks.append(pred_binary * 255)
    
    final_image[y1:y2, x1:x2]=image_preds[0]
    black_mask[y1:y2, x1:x2]= masks[0]

    

    
    
    #load scheduler
    scheduler = EulerAncestralDiscreteScheduler()

    #convert cv2 to pil



    


    
    
    # blur mask
    mask_image = pipe_refiner.mask_processor.blur(black_mask, blur_factor=5)
    

    generator = torch.Generator(device="cuda").manual_seed(seed)
    image_remove_person = pipe_refiner(
        prompt="",
        width=768,
        height=1024,
        image=image,
        num_inference_steps=30,
        guidance_scale=7,
        strength=1.0,
        mask_image=mask_image,
        scheduler=scheduler,
        padding_mask_crop = 32,
    )
    
    
    return final_image,image_remove_person  # Trả về ảnh nền trắng và ảnh mask 


# Hàm này và để paste ảnh sau khi swap vào ảnh background tức là ảnh gốc và trả về thêm ảnh mask viền
def PastingAndMask(image, background):
    global birefnet

    resolution = f"{image.shape[1]}x{image.shape[0]}"
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
        pred_binary = (pred > 0.5).astype(np.uint8)
        mask = np.repeat(pred_binary[..., None], 3, axis=-1)
        edges = detect_edges(mask*255)
        segmented_part = mask * np.array(image)
        alpha = mask
        combined_image = alpha * segmented_part + (1 - alpha) * background
        image_preds.append(combined_image.astype(np.uint8))
        break
        
    # Trả về ảnh sau khi paste và ảnh mask viền
    return image_preds[0],edges



##################################################################
# Try example of 2 function
# Load and process image
img = cv2.imread("image_swapp.png") # Image is swapped
background = cv2.imread("c1.jpeg")
background=cv2.resize(background, (768,1024))

##################################################################
# Remove background
remove_image, mask = RemoveBackGround(background)
cv2.imwrite("RemoveBackGround.jpg", remove_image)
cv2.imwrite("mask_before.jpg", mask)
# Pasting and mask
img_new, mask = PastingAndMask(img,background)
cv2.imwrite("testafter.jpg", img_new)
cv2.imwrite("mask.jpg", mask)