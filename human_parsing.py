from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image
import requests
import cv2

def calculate_intersection_area(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    inter_area = inter_width * inter_height

    return inter_area

def calculate_percentage_of_intersection(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    
    inter_area = calculate_intersection_area(box1, box2)
    
    if box1_area == 0:
        return 0  

    percentage = (inter_area / box1_area)
    return percentage

def check_reject_garment(image):

    #check the size
    w,h = image.size
    if w < 300 or h < 300:
        return "Reject cause the size is too small"
    
    #check having person
    human_model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s_personface.pt") 
    human_model.conf = 0.85
    results_human = human_model(image)
    if len(results_human.xyxy[0]) != 0:
       return "Reject cause there is person in the image"
    
    human_model.conf = 0.5
    results_human = human_model(image)

    if len(results_human.xyxy[0]) == 0:
       return "Accept"

    #check having accessories
    x1, y1, x2, y2, conf, cls = 0, 0, 0, 0, 0.0, 0
    for result in results_human.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        cls = int(cls)
        if cls == 0:
            x1, y1, x2, y2, conf = int(x1), int(y1), int(x2), int(y2), float(conf)
            break
        else:
            x1, y1, x2, y2, conf = 0, 0, 0, 0, 0.0
    box_human = [x1, y1, x2, y2]

    
    processor = AutoImageProcessor.from_pretrained("facebook/deformable-detr-detic")
    model = DeformableDetrForObjectDetection.from_pretrained("facebook/deformable-detr-detic")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]


    list_detect = ["bag","backpack","handbag","suitcase","purse","wallet","briefcase","luggage","baggage","ring","phone"]  
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        for i in list_detect:
            if i in model.config.id2label[label.item()]:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                box_class = [x1, y1, x2, y2]
                if calculate_percentage_of_intersection(box_class, box_human) > 0.5:
                    return "Reject cause there is accessories in the image" + i
                # print("Class: ", i)
                # print("Percentage overlap: ", calculate_percentage_of_intersection(box_class, box_human))

    return "Accept"

    

img_path = "tuixach.jpg"
image = Image.open(img_path)

print(check_reject_garment(image))
