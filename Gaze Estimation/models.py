from enum import Enum

import keras
from keras import layers


class ModelType(Enum):
    Test = 0,
    Simple = 1


def get_model(model_type: ModelType, image_shape, info_shape: int):
    model = None
    if model_type == ModelType.Test:
        image_input = keras.Input(
            shape=image_shape, name="image"
        )

        info_input = keras.Input(
            shape=(info_shape,), name="info"
        )

        image_features = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(image_input)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
        image_features = layers.Dropout(0.3)(image_features)

        image_features = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
        image_features = layers.Dropout(0.5)(image_features)

        image_features = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
        image_features = layers.Dropout(0.5)(image_features)

        x = layers.Flatten()(image_features)
        x = layers.concatenate([x, info_input])
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        pixel_prediction = layers.Dense(2, name="pixel_prediction")(x)

        model = keras.Model(
            inputs=[image_input, info_input],
            outputs={"pixel_prediction": pixel_prediction}
        )
    elif model_type == ModelType.Simple:
        pass

    return model
