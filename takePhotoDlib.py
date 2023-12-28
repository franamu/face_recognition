import dlib  # type: ignore
import cv2
import os
from PIL import Image
import numpy as np
from numpy import ndarray
from typing import List
from sklearn.metrics import accuracy_score  # type: ignore

ruta_script = os.path.dirname(os.path.abspath(__file__))

# path to model
models_path = os.path.join(ruta_script, 'models')
path_model = os.path.join(models_path, 'shape_predictor_68_face_landmarks.dat')
path_recognition_model = os.path.join(
    models_path, 'dlib_face_recognition_resnet_model_v1.dat')
face_descriptor_extractor = dlib.face_recognition_model_v1(
    path_recognition_model)

# método para obtener la cara
face_detector = dlib.get_frontal_face_detector()

# metodo que obtiene los 68 puntos para que funcione el algoritmo
points_detector = dlib.shape_predictor(path_model)

# rutas de imagenes para entrenar
paths = [os.path.join('assets', f)
         for f in os.listdir(os.path.join('assets'))]

camera = cv2.VideoCapture(0)

# entrenar muchas imagenes
index = {}  # type: ignore
idx = 0
face_descriptors = None

while True:
    connected, image = camera.read()
    if cv2.waitKey(1) & 0xFF == ord('t'):
        image_np = np.array(image, 'uint8')
        face_detection = face_detector(image_np, 1)
        for face in face_detection:
            l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(image_np, (l, t), (r, b), (0, 0, 255), 2)

            points = points_detector(image_np, face)
            for point in points.parts():
                cv2.circle(image_np, (point.x, point.y), 2, (0, 255, 0), 1)

            face_descriptor = face_descriptor_extractor.compute_face_descriptor(
                image_np, points)
            # print(type(face_descriptor))
            # print(len(face_descriptor))
            # print(face_descriptor)
            face_descriptor = [f for f in face_descriptor]
            # print(face_descriptor)
            face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
            # print(face_descriptor)
            # print(face_descriptor.shape)
            face_descriptor = face_descriptor[np.newaxis, :]
            # print(face_descriptor.shape)
            # print(face_descriptor)

            if face_descriptors is None:
                face_descriptors = face_descriptor
            else:
                face_descriptors = np.concatenate(
                    (face_descriptors, face_descriptor), axis=0)

            np.save('face_descriptors.npy', face_descriptors)
            print(face_descriptors)

    cv2.imshow('Video', image)
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
