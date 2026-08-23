import os

import cv2
import numpy as np


IMG_SIZE = 224


def load_data(dataset_path):

    categories = [
        "covid",
        "normal"
    ]

    data = []
    labels = []

    for label, category in enumerate(categories):

        path = os.path.join(
            dataset_path,
            category
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset folder not found: {path}"
            )

        print(f"\nLoading: {category}")

        image_count = 0

        for img_name in os.listdir(path):

            img_path = os.path.join(
                path,
                img_name
            )

            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)

            if img is None:
                print(
                    f"Skipping invalid image: {img_path}"
                )
                continue

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            img = cv2.resize(
                img,
                (IMG_SIZE, IMG_SIZE)
            )

            img = img.astype(
                np.float32
            ) / 255.0

            data.append(img)
            labels.append(label)

            image_count += 1

        print(
            f"{category}: {image_count} images"
        )

    if len(data) == 0:
        raise ValueError(
            "No valid images were found in the dataset."
        )

    data = np.array(
        data,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.float32
    )

    print("\n==============================")
    print("DATASET INFORMATION")
    print("==============================")

    print(
        "Total images:",
        len(data)
    )

    print(
        "Image shape:",
        data.shape
    )

    print(
        "COVID images:",
        int(np.sum(labels == 0))
    )

    print(
        "Normal images:",
        int(np.sum(labels == 1))
    )

    return data, labels