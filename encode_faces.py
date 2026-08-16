import os
import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet

# Load FaceNet model
embedder = FaceNet()

dataset_path = "dataset"

embeddings = []
names = []

for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (160, 160))

        image = np.expand_dims(image, axis=0)

        embedding = embedder.embeddings(image)[0]

        embeddings.append(embedding)
        names.append(person_name)

print("Total embeddings:", len(embeddings))

data = {
    "embeddings": embeddings,
    "names": names
}

with open("embeddings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Embeddings saved successfully!")