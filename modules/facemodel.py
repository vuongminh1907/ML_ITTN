import cv2
import torch
from PIL import Image
import numpy as np
from insightface.app import FaceAnalysis
import time

class Attendance:
    def __init__(self):
        self.model = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, image):
        face = self.model.get(image)
        if len(face) == 0:
            return None
        return face[0]['embedding']

    def is_face(self, embedding1, embedding2):
        cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        cosine_distance = 1 - cosine_similarity
        return cosine_distance < 0.5
    def use_web_cam(self, face_database):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            faces = self.model.get(frame)
            if len(faces) == 0:
                cv2.imshow("WebCam", frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                if cv2.getWindowProperty("WebCam", cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            for face in faces:
                face_embeddings = face['embedding']
                x, y, w, h = face['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)
                frame = cv2.rectangle(frame, (x, y), (w,h), (255, 0, 0), 2)
                for face_data in face_database:
                    if(self.is_face(face_data['embedding'], face_embeddings)):
                        #write the name of the person
                        frame = cv2.putText(frame, face_data['name'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        if face_data['time_in'] is None:
                            face_data['time_in'] = time.time()
                        else:
                            face_data['time_out'] = time.time()
                        break
            cv2.imshow("WebCam", frame)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty("WebCam", cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyAllWindows()
        return face_database
    
    def check_attendence(self,image_dir,face_database):
        image = cv2.imread(image_dir)
        face = self.get_embedding(image)
        if face is None:
            print("No face detected")
            return
        check = False
        for face_data in face_database:
            if(self.is_face(face_data['embedding'], face)):
                print("This person is in the class")
                print(f"Name: {face_data['name']}")
                check = True
                continue
        
        if not check:
            print("This person is not in the class")
            cv2.imwrite(f"./face_database/{time.time()}.jpg", image)
            print(f"Save {time.time()}.jpg") 

    def use_video(self,video_dir,face_database):
        #read video
        cap = cv2.VideoCapture(video_dir)
        #resize and fps video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            faces = self.model.get(frame)
            if len(faces) == 0:
                cv2.imshow("YOLO", frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            for face in faces:
                face_embeddings = face['embedding']
                x, y, w, h = face['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)
                frame = cv2.rectangle(frame, (x, y), (w,h), (255, 0, 0), 2)
                for face_data in face_database:
                    if(self.is_face(face_data['embedding'], face_embeddings)):
                        #write the name of the person
                        frame = cv2.putText(frame, face_data['name'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        if face_data['time_in'] is None:
                            face_data['time_in'] = time.time()
                        else:
                            face_data['time_out'] = time.time()
                        break
                else:
                    frame = cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.imshow("Video", frame)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
                break
