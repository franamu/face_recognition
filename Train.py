from PIL import Image
import cv2
import numpy as np
import os
from typing import List


def get_image_data():
    paths = [os.path.join('assets', f)
             for f in os.listdir(os.path.join('assets'))]

    faces = []
    ids = []
    for path in paths:
        name_file = os.path.splitext(os.path.basename(path))[0]
        names_parts = name_file.split('_')
        print(name_file)
        print(names_parts)
        # Verificar si el nombre del archivo sigue el patrón esperado
        if len(names_parts) == 4:
            # Leer la imagen y convertirla a escala de grises
            image = Image.open(path).convert('L')
            image_np = np.array(image, 'uint8')
            ids.append(int(names_parts[2]))
            faces.append(image_np)
        else:
            print(f"El nombre del archivo no sigue el patrón esperado: {
                  name_file}")

    return np.array(ids), faces


ids, faces = get_image_data()

print(faces)

# default value 8 (8 row and columns) (8 + 8 = 64 histograms)
lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(faces, ids)
lbph_classifier.write('models/lbph_classifier.yml')
