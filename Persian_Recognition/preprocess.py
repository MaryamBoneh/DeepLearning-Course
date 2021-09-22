from retinaface import RetinaFace
import matplotlib.pyplot as plt
import cv2
import os

for filename in os.listdir("/content/drive/MyDrive/persons/"):
    print(filename)
    faces = RetinaFace.extract_faces(img_path = "/content/drive/MyDrive/persons/" + filename, align = True)
    print(faces)
    for face in faces:
        plt.imshow(face)
        plt.show()
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        cv2.imwrite("/content/drive/MyDrive/persons/" + filename, face)