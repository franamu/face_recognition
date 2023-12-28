from PIL import Image
import cv2
import numpy as np
import os
from typing import List
from sklearn.metrics import accuracy_score  # type: ignore

# Extract zip
# path_script = os.path.dirname(os.path.abspath(__file__))
# path_dataset = os.path.join(path_script, 'dataset')
# path_zip_dataset = os.path.join(path_dataset, 'yalefaces.zip')
# zip_object = zipfile.ZipFile(file=path_zip_dataset, mode='r')
# zip_object.extractall('./')
# zip_object.close()


def get_image_data():
    paths = [os.path.join('yalefaces', 'train', f)
             for f in os.listdir(os.path.join('yalefaces', 'train'))]

    faces = []
    ids = []
    for path in paths:
        image = Image.open(path).convert('L')
        image_np = np.array(image, 'uint8')
        id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
        ids.append(id)
        faces.append(image_np)

    return np.array(ids), faces


ids, faces = get_image_data()

print(ids)
# default value 8 (8 row and columns) (8 + 8 = 64 histograms)
lbph_classifier = cv2.face.LBPHFaceRecognizer_create(
    radius=4, neighbors=14, grid_x=9, grid_y=9)
lbph_classifier.train(faces, ids)
lbph_classifier.write('lbph_classifier.yml')

# recongizing faces
lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()

lbph_face_classifier.read(os.path.join('lbph_classifier.yml'))

test_image = os.path.join('yalefaces/test/subject06.happy.gif')

image = Image.open(test_image).convert('L')
image_np = np.array(image, 'uint8')

prediction = lbph_face_classifier.predict(image_np)
print(prediction)

# evualuating the face classifier

paths = [os.path.join('yalefaces', 'test', f)
         for f in os.listdir(os.path.join('yalefaces', 'test'))]
predictions: List[int] = []
expected_outputs: List[int] = []

for path in paths:
    image = Image.open(path).convert('L')
    image_np = np.array(image, 'uint8')
    prediction, _ = lbph_face_classifier.predict(image_np)
    expected_output = int(os.path.split(path)[1].split('.')[
                          0].replace('subject', ''))

    predictions.append(prediction)
    expected_outputs.append(expected_output)

predictions_np: np.ndarray = np.array(predictions)
expected_outputs_np: np.ndarray = np.array(expected_outputs)

print(predictions_np)
print(expected_outputs_np)
print(accuracy_score(expected_outputs_np, predictions_np))
