import cv2
import torch
from PIL import Image
import numpy as np
import os
import time
from utils import get_data_base, display_time
from modules.facemodel import Attendance


folder = "./face_database"
model = Attendance()

face_database = get_data_base(folder, model)
#face_database = model.use_web_cam(face_database)

#model.check_attendence("./assets/oldman.png", face_database)
model.use_video("./assets/test.mp4", face_database)