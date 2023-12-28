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
paths = [os.path.join('yalefaces', 'train', f)
         for f in os.listdir(os.path.join('yalefaces', 'train'))]

# entrenar muchas imagenes de un repositorio
index = {}
idx = 0
face_descriptors = None

for path in paths:
    # print(path)
    image = Image.open(path).convert('RGB')
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

        index[idx] = path
        idx += 1
    # cv2_imshow(image_np)


threshold = 0.5
predictions = []
expected_outputs = []

paths_test_images = [os.path.join('yalefaces', 'test', f)
                     for f in os.listdir(os.path.join('yalefaces', 'test'))]

for path in paths_test_images:
    image = Image.open(path).convert('RGB')
    image_np = np.array(image, 'uint8')
    face_detection = face_detector(image_np, 1)
    for face in face_detection:
        points = points_detector(image_np, face)
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(
            image_np, points)
        face_descriptor = [f for f in face_descriptor]
        face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
        face_descriptor = face_descriptor[np.newaxis, :]

        distances = np.linalg.norm(face_descriptor - face_descriptors, axis=1)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        if min_distance <= threshold:
            name_pred = os.path.split(index[int(min_index)])[
                1].split('.')[0].replace('subject', '')
        else:
            name_pred = 'Not identificado'

        name_real = os.path.split(path)[1].split('.')[
            0].replace('subject', '')

        predictions.append(name_pred)
        expected_outputs.append(name_real)

        cv2.putText(image_np, 'Pred: ' + str(name_pred), (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        cv2.putText(image_np, 'Exp : ' + str(name_real), (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        cv2.imshow('Imagen con detecciones', image_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

predictions = np.array(predictions).tolist()
expected_outputs = np.array(expected_outputs).tolist()
print(accuracy_score(expected_outputs, predictions))
