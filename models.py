from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_resnet():
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    return Model(base.input, out)

def build_efficientnet():
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    return Model(base.input, out)

def build_densenet():
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    return Model(base.input, out)