import cv2
import os


# Ruta de la carpeta "assets"
assets_folder = "assets"

# Crear la carpeta si no existe
if not os.path.exists(assets_folder):
    os.makedirs(assets_folder)


face_detector = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml")
width, height = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)
counter = 0

while True:
    connected, image = camera.read()
    if cv2.waitKey(1) & 0xFF == ord('t'):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = face_detector.detectMultiScale(
            image_gray, scaleFactor=1.5, minSize=(100, 100))
        for (x, y, w, h) in detections:
            image_face = cv2.resize(
                image_gray[y:y + w, x:x + h], (width, height))
            image_path = os.path.join('assets', str(counter) + '.jpg')
            cv2.imwrite(image_path, image_face)
        counter += 1

    cv2.imshow('Video', image)
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
