import dlib  # type: ignore
import cv2
import os
from PIL import Image
import numpy as np
from numpy import ndarray
from typing import List
from sklearn.metrics import accuracy_score  # type: ignore
import time
from typing import Union

ruta_script = os.path.dirname(os.path.abspath(__file__))

# path to model
models_path = os.path.join(ruta_script, 'models')
path_model = os.path.join(models_path, 'shape_predictor_68_face_landmarks.dat')
path_recognition_model = os.path.join(
    models_path, 'dlib_face_recognition_resnet_model_v1.dat')
face_descriptor_extractor = dlib.face_recognition_model_v1(
    path_recognition_model)

loaded_face_descriptors = np.load('face_descriptors.npy')

# método para obtener la cara
face_detector = dlib.get_frontal_face_detector()

# metodo que obtiene los 68 puntos para que funcione el algoritmo
points_detector = dlib.shape_predictor(path_model)

threshold = 0.5
predictions = []
expected_outputs = []
camera = cv2.VideoCapture(0)

intervalo_segundos = 5
ultimo_tiempo_ejecucion: Union[int, float] = time.time()
tiempo_actual = time.time()

while (True):
    connected, image = camera.read()
    # Ejecutar el bucle for cada 5 segundos
    if tiempo_actual - ultimo_tiempo_ejecucion >= intervalo_segundos:
        image_np = np.array(image, 'uint8')
        face_detection = face_detector(image_np, 1)
        for face in face_detection:
            l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(image, (l, t), (r, b), (0, 255, 255), 2)
            points = points_detector(image_np, face)
            face_descriptor = face_descriptor_extractor.compute_face_descriptor(
                image_np, points)
            face_descriptor = [f for f in face_descriptor]
            face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
            face_descriptor = face_descriptor[np.newaxis, :]

            distances = np.linalg.norm(
                face_descriptor - loaded_face_descriptors, axis=1)
            min_index = np.argmin(distances)
            min_distance = distances[min_index]
            print(min_distance <= threshold)
            if min_distance <= threshold:
                print('sos vos fran')
                name_pred = 'Francisco Amuchastegui'
            else:
                print('no identificado')
                name_pred = 'Not identificado'

            # Actualizar el tiempo de la última ejecución
            ultimo_tiempo_ejecucion = tiempo_actual

    cv2.imshow("Face", image)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
