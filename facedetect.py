import cv2
import torch
from PIL import Image
import numpy as np

# Model


img = cv2.imread("test.jpg")
#if the size is less than 640 then refuse

model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s_personface.pt") 

model.conf = 0.7
results = model(img)

for result in results.xyxy[0]:
    x1, y1, x2, y2, conf, cls = result
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    conf = float(conf)
    cls = int(cls)
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    img = cv2.putText(img, model.names[int(cls)], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    #save img
    cv2.imwrite("test.jpg", img)

        
    

    