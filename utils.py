import cv2
import os
import numpy as np

IMG_SIZE = 224

def load_data(dataset_path):
    categories = ["covid", "normal"]
    data = []
    labels = []

    for label, category in enumerate(categories):
        path = os.path.join(dataset_path, category)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            data.append(img)
            labels.append(label)

    data = np.array(data) / 255.0
    labels = np.array(labels)

    return data, labels