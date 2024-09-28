import gradio as gr
import cv2
import torch
from PIL import Image
import numpy as np
import os
import time
from utils import get_data_base, display_time
from modules.facemodel import Attendance

input_image = gr.Image(sources=["webcam"], streaming=True)
output_image = gr.Image()

def draw_image(input_image):
    cap=cv2.VideoCapture(input_image)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
gr.Interface(fn=draw_image, inputs=input_image, outputs=output_image).launch()
