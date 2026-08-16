import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

print("Loading FaceNet...")
embedder = FaceNet()

print("Loading MTCNN...")
detector = MTCNN()

print("Loading embeddings...")

with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = np.array(data["embeddings"])
known_names = np.array(data["names"])

print("Opening webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Face Recognition Started")
print("Press Q to Quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb)

    for face in faces:

        x, y, w, h = face["box"]

        x = max(0, x)
        y = max(0, y)

        face_img = rgb[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        face_img = cv2.resize(face_img, (160, 160))

        face_img = np.expand_dims(face_img, axis=0)

        embedding = embedder.embeddings(face_img)[0]

        similarity = cosine_similarity(
            [embedding],
            known_embeddings
        )[0]

        best_index = np.argmax(similarity)
        best_score = similarity[best_index]

        if best_score > 0.75:
            name = known_names[best_index]
        else:
            name = "Unknown"

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{name} {best_score:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Deep Learning Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()