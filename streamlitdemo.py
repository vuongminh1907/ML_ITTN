import cv2
import streamlit as st
import torch
from PIL import Image
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

image1 = cv2.imread("./face_database/Minh0610.jpg")
face1 = app.get(image1)[0]['embedding']

def is_face(embedding1, embedding2):
    cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    cosine_distance = 1 - cosine_similarity
    return cosine_distance < 0.5

def process_frame(frame):
    faces = app.get(frame)
    if len(faces) == 0:
        return frame
    face_embeddings = faces[0]['embedding']
    x, y, w, h = faces[0]['bbox']
    x, y, w, h = int(x), int(y), int(w), int(h)
    frame = cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
    if is_face(face1, face_embeddings):
        frame = cv2.putText(frame, "Trong", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    else:
        frame = cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame

def process_webcam(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    processed_image = process_frame(image)
    #processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)  
    return processed_image

st.set_page_config(
    page_title="Object Detection using YOLOv8",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded"    # Expanding sidebar by default
)

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        break
    frame = process_webcam(frame)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')