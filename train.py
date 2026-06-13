from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from utils import load_data
from models import build_resnet, build_efficientnet, build_densenet

DATASET_PATH = "dataset"

X, y = load_data(DATASET_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "resnet": build_resnet(),
    "efficientnet": build_efficientnet(),
    "densenet": build_densenet()
}

best_acc = 0
best_model_name = None

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint = ModelCheckpoint(
        f"{name}.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=8,
        callbacks=[checkpoint]
    )

    val_acc = max(history.history["val_accuracy"])

    print(f"{name} Best Val Accuracy: {val_acc}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_name = name

print("\nBEST MODEL:", best_model_name)