from retinaface import RetinaFace
import cv2
import os

c = 1
folder_address = "/content/drive/MyDrive/persons/"
for filename in os.listdir(folder_address):
    print(filename)
    faces = RetinaFace.extract_faces(img_path = folder_address + filename, align = True)
    for face in faces:
      c += 1
      face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
      cv2.imwrite(folder_address + "img" + str(c) + ".png", face)