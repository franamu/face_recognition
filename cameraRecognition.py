import cv2

face_detector = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()  # type: ignore
face_recognizer.read("models/lbph_classifier.yml")
width, height = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

threshold = 90

while (True):
    connected, image = camera.read()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = face_detector.detectMultiScale(
        image_gray, scaleFactor=1.5, minSize=(100, 100))
    for (x, y, w, h) in detections:
        image_face = cv2.resize(image_gray[y:y + w, x:x + h], (width, height))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = face_recognizer.predict(image_face)
        name = ""
        color = (0, 255, 0)
        if id == 1:
            if confidence <= threshold:
                name = 'Francisco Amuchastegui'
                color = (0, 255, 0)
        elif id == 2:
            if confidence <= threshold:
                name = 'Lionel Messi'
                color = (0, 255, 0)
        elif id == 3:
            if confidence <= threshold:
                name = 'Jose Cardozo'
                color = (0, 255, 0)
        elif id == 4:
            if confidence <= threshold:
                name = 'Ramon Amuchastegui'
                color = (0, 255, 0)
        else:
            name = 'No identificado'
            color = (0, 255, 0)

        cv2.putText(image, name, (x, y + (w+30)), font, 2, color)
        cv2.putText(image, str(confidence),
                    (x, y + (h+50)), font, 1, color)

    cv2.imshow("Face", image)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
