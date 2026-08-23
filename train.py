import os

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping
)

from utils import load_data

from models import (
    build_resnet,
    build_efficientnet,
    build_densenet
)

DATASET_PATH = "dataset"

MODEL_DIR = "models"

EPOCHS = 10

BATCH_SIZE = 8

RANDOM_STATE = 42

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

print("\nLoading dataset...")

X, y = load_data(
    DATASET_PATH
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\n==============================")
print("DATA SPLIT")
print("==============================")

print(
    "Training images:",
    len(X_train)
)

print(
    "Testing images:",
    len(X_test)
)

model_builders = {

    "resnet": build_resnet,

    "efficientnet": build_efficientnet,

    "densenet": build_densenet
}


results = {}

for name, build_model in model_builders.items():

    print("\n")
    print("=" * 60)
    print(f"TRAINING {name.upper()}")
    print("=" * 60)

    model = build_model()

    model.compile(

        optimizer=Adam(
            learning_rate=1e-4
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"{name}.h5"
    )

    checkpoint = ModelCheckpoint(

        model_path,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1
    )

    early_stopping = EarlyStopping(

        monitor="val_loss",

        patience=3,

        restore_best_weights=True,

        verbose=1
    )

    history = model.fit(

        X_train,

        y_train,

        validation_split=0.20,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            checkpoint,
            early_stopping
        ],

        verbose=1
    )

    import tensorflow as tf

    best_model = tf.keras.models.load_model(
        model_path
    )

    probabilities = best_model.predict(
        X_test,
        verbose=0
    ).ravel()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    print("\n")
    print("-" * 60)
    print(f"{name.upper()} TEST RESULTS")
    print("-" * 60)

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "COVID",
                "Normal"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

best_model_name = max(
    results,
    key=results.get
)

best_accuracy = results[
    best_model_name
]

import shutil

source_model = os.path.join(
    MODEL_DIR,
    f"{best_model_name}.h5"
)

best_model_path = os.path.join(
    MODEL_DIR,
    "best_model.h5"
)

shutil.copy2(
    source_model,
    best_model_path
)

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

for name, accuracy in results.items():

    print(
        f"{name:<15} : {accuracy:.4f}"
    )


print("\nBEST MODEL:")
print(
    best_model_name.upper()
)

print("\nBEST TEST ACCURACY:")
print(
    f"{best_accuracy * 100:.2f}%"
)

print("\nSaved as:")
print(
    best_model_path
)

print("\nTraining complete!")