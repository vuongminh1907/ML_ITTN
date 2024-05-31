import cv2
import torch
from PIL import Image
import numpy as np

# Model
model = torch.hub.load("ultralytics/yolov5", "custom", path="./model/yolov5s_personface.pt") 

# Use for webcam

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    cv2.imshow("YOLO", np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    if cv2.getWindowProperty("YOLO", cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()

        
    

    