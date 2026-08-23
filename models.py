import tensorflow as tf

from tensorflow.keras.applications import (
    ResNet50,
    EfficientNetB0,
    DenseNet121
)

from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout
)

from tensorflow.keras.models import Model


IMG_SIZE = 224


def build_resnet():
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    output = Dense(
        1,
        activation="sigmoid"
    )(x)

    return Model(
        inputs=base.input,
        outputs=output
    )


def build_efficientnet():
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    output = Dense(
        1,
        activation="sigmoid"
    )(x)

    return Model(
        inputs=base.input,
        outputs=output
    )


def build_densenet():
    base = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    output = Dense(
        1,
        activation="sigmoid"
    )(x)

    return Model(
        inputs=base.input,
        outputs=output
    )