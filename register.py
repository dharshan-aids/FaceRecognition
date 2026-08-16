print("Program started")
import cv2
import os

# Ask for person's name
person_name = input("Enter person's name: ")

# Create folder
dataset_path = "dataset"
person_path = os.path.join(dataset_path, person_name)

if not os.path.exists(person_path):
    os.makedirs(person_path)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

count = 0

print("Look at the camera...")
print("Collecting face images...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    print("Faces detected:", len(faces))

    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Crop face
        face = frame[y:y+h, x:x+w]

        # Resize
        face = cv2.resize(face, (160,160))

        # Save image
        filename = os.path.join(person_path, f"{count}.jpg")
        cv2.imwrite(filename, face)

        count += 1

        cv2.putText(frame,
                    f"Images: {count}/100",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

    cv2.imshow("Register Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= 100:
        break

cap.release()
cv2.destroyAllWindows()

print("Registration Complete!")