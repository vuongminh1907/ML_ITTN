import os
import cv2
import time


def get_data_base(folder_dir,model):
    list_image = os.listdir(folder_dir)
    face_database = []
    for image in list_image:
        image_path = os.path.join(folder_dir, image)
        face = model.get_embedding(cv2.imread(image_path))
        face_database.append({"name": image.split(".")[0], "embedding": face, "time_in": None, "time_out": None})
    return face_database

def display_time(face_database):
    for face_data in face_database:
        print(f"Name: {face_data['name']}")
        if face_data['time_in'] is None:
            print("Time in: No attendance")
            print("Time out: No attendance")
            print("=====================================")
            continue
        print(f"Time in: {time.ctime(face_data['time_in'])}")
        print(f"Time out: {time.ctime(face_data['time_out'])}")
        print("=====================================")
