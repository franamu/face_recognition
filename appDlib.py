import dlib  # type: ignore
import cv2
import os
from PIL import Image
import numpy as np
from numpy import ndarray
from typing import List
from sklearn.metrics import accuracy_score

# raiz path
ruta_script = os.path.dirname(os.path.abspath(__file__))

# path to assets
assets_path = os.path.join(ruta_script, 'assets')

# path to model
models_path = os.path.join(ruta_script, 'models')
path_model = os.path.join(models_path, 'shape_predictor_68_face_landmarks.dat')


# detecting facial points
face_detector = dlib.get_frontal_face_detector()
points_detector = dlib.shape_predictor(path_model)
image_path = os.path.join(assets_path, 'descarga.png')

image = cv2.imread(image_path)

face_detection = face_detector(image, 1)

for face in face_detection:
    points = points_detector(image, face)
    for point in points.parts():
        cv2.circle(image, (point.x, point.y), 2, (0, 225, 0), 2)
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 255), 2)


# entrenar muchas imagenes de un repositorio
path_recognition_model = os.path.join(
    models_path, 'dlib_face_recognition_resnet_model_v1.dat')

face_descriptor_extractor = dlib.face_recognition_model_v1(
    path_recognition_model)

index = {}
idx = 0
face_descriptors: List[ndarray] = []

paths = [os.path.join('yalefaces', 'train', f)
         for f in os.listdir(os.path.join('yalefaces', 'train'))]

for path in paths:
    image = Image.open(path).convert('RGB')
    image_np = np.array(image, 'uint8')
    face_detection = face_detector(image_np, 1)
    for face in face_detection:
        points = points_detector(image_np, face)
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(image_np, (l, t), (r, b), (0, 255, 255), 2)
        for point in points.parts():
            cv2.circle(image_np, (point.x, point.y), 2, (0, 225, 0), 2)
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(
            image_np, points)
        face_descriptor = [f for f in face_descriptor]
        # print(face_descriptor)
        face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
        face_descriptor = face_descriptor[np.newaxis, :]

        if not np.any(face_descriptors):
            face_descriptors = face_descriptor
        else:
            face_descriptors = np.concatenate(
                (face_descriptors, face_descriptor), axis=0)

        index[idx] = path
        idx += 1

        # cv2.imshow('Imagen con detecciones', image_np)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# compare faces to get the difference
print(np.linalg.norm(face_descriptors[131] - face_descriptors[130]))

# compare one image with all the data set and get de minimum value

print(np.linalg.norm(face_descriptors[131] - face_descriptors, axis=1))

threshold = 0.5
predictions = []
expected_outputs = []

# detecting faces with test images
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
            name_pred = os.path.split(index[min_index])[
                1].split('.')[0].replace('subject', '')
        else:
            name_pred = 'Not identificado'

        name_real = os.path.split(path)[
            1].split('.')[0].replace('subject', '')

        predictions.append(name_pred)
        expected_outputs.append(name_real)

        cv2.putText(image_np, 'Pred: ' + name_pred, (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        cv2.putText(image_np, 'Exp: ' + name_pred, (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        # cv2.imshow('Imagen con detecciones', image_np)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

predictions = np.array(predictions).tolist()
expected_outputs = np.array(expected_outputs).tolist()

print(accuracy_score(expected_outputs, predictions))
