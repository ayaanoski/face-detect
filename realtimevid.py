import cv2
import mediapipe as mp
import face_recognition
import time
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
pTime = 0

# Mediapipe face detection
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

# Load the known faces and their encodings
# Add the images of the people you want to recognize
known_face_encodings = []
known_face_names = []

# Example: Load and encode faces
face_1 = face_recognition.load_image_file("E:\\ETHOS\\faces\\ayaan.jpeg")
face_1_encoding = face_recognition.face_encodings(face_1)[0]
known_face_encodings.append(face_1_encoding)
known_face_names.append("ayaan")

face_2 = face_recognition.load_image_file("E:\\ETHOS\\faces\\babu.jpeg")
face_2_encoding = face_recognition.face_encodings(face_2)[0]
known_face_encodings.append(face_2_encoding)
known_face_names.append("anwesha")

face_1 = face_recognition.load_image_file("E:\\ETHOS\\faces\\mandi.jpeg")
face_1_encoding = face_recognition.face_encodings(face_1)[0]
known_face_encodings.append(face_1_encoding)
known_face_names.append("mandi")

face_2 = face_recognition.load_image_file("E:\\ETHOS\\faces\\krishna.jpeg")
face_2_encoding = face_recognition.face_encodings(face_2)[0]
known_face_encodings.append(face_2_encoding)
known_face_names.append("krishna")


# Add more faces as needed in the same way

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        face_locations = []
        face_encodings = []

        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )

            # Extract face location for face recognition
            face_locations.append(
                (bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0])
            )

        # Perform face recognition on each detected face
        if face_locations:
            face_encodings = face_recognition.face_encodings(img, face_locations)

            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding
                )
                name = "Unknown"

                # Find the known face with the smallest distance
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                # Draw bounding box around face
                cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 255), 2)
                cv2.putText(
                    img,
                    name,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 255),
                    2,
                )

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2
    )

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
