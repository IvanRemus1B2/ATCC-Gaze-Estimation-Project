from enum import Enum

import keras
from keras import layers
from keras import regularizers


# TODO:Get 4.5 cm

class ModelType(Enum):
    Test_VGG_4M = 0,
    Test_VGG_4M_Regularized_ELU = 1,
    Test_VGG_1M_Regularized_ELU = 2,
    Test_VGG_4M_2 = 3,
    Simple = 10


def get_model(model_type: ModelType, image_shape, info_shape: int):
    model = None
    image_input = keras.Input(
        shape=image_shape, name="image"
    )
    info_input = keras.Input(
        shape=(info_shape,), name="info"
    )
    x = None
    if model_type == ModelType.Test_VGG_4M:
        image_features = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(image_input)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
        image_features = layers.Dropout(0.1)(image_features)

        image_features = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
        image_features = layers.Dropout(0.1)(image_features)

        image_features = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(4, 4))(image_features)
        image_features = layers.Dropout(0.2)(image_features)

        x = layers.Flatten()(image_features)
        x = layers.concatenate([x, info_input])
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
    elif model_type == ModelType.Test_VGG_4M_Regularized_ELU:
        elu_layer = layers.ELU()
        image_features = layers.Conv2D(32, (3, 3), padding='same', activation=elu_layer,
                                       kernel_regularizer=regularizers.L2(1e-2))(image_input)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(32, (3, 3), padding='same', activation=elu_layer,
                                       kernel_regularizer=regularizers.L2(1e-2))(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
        image_features = layers.Dropout(0.5)(image_features)

        image_features = layers.Conv2D(64, (3, 3), padding='same', activation=elu_layer,
                                       kernel_regularizer=regularizers.L2(1e-2))(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(64, (3, 3), padding='same', activation=elu_layer,
                                       kernel_regularizer=regularizers.L2(1e-2))(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
        image_features = layers.Dropout(0.5)(image_features)

        image_features = layers.Conv2D(128, (3, 3), padding='same', activation=elu_layer,
                                       kernel_regularizer=regularizers.L2(1e-2))(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(128, (3, 3), padding='same', activation=elu_layer,
                                       kernel_regularizer=regularizers.L2(1e-2))(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
        image_features = layers.Dropout(0.5)(image_features)

        x = layers.Flatten()(image_features)
        x = layers.concatenate([x, info_input])
        x = layers.Dense(128, activation=elu_layer, kernel_regularizer=regularizers.L2(1e-2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    elif model_type == ModelType.Test_VGG_1M_Regularized_ELU:
        elu_layer = layers.ELU()
        image_features = layers.GaussianNoise(stddev=0.1)(image_input)
        image_features = layers.Conv2D(32, (3, 3), padding='same', activation=elu_layer, kernel_initializer='he_normal',
                                       kernel_regularizer=regularizers.L2(1e-2))(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(32, (3, 3), padding='same', activation=elu_layer, kernel_initializer='he_normal',
                                       kernel_regularizer=regularizers.L2(1e-2))(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
        image_features = layers.Dropout(0.5)(image_features)

        image_features = layers.Conv2D(64, (3, 3), padding='same', activation=elu_layer, kernel_initializer='he_normal',
                                       kernel_regularizer=regularizers.L2(1e-2))(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(64, (3, 3), padding='same', activation=elu_layer, kernel_initializer='he_normal',
                                       kernel_regularizer=regularizers.L2(1e-2))(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
        image_features = layers.Dropout(0.5)(image_features)

        image_features = layers.Conv2D(128, (3, 3), padding='same', activation=elu_layer,
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=regularizers.L2(1e-2))(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(128, (3, 3), padding='same', activation=elu_layer,
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=regularizers.L2(1e-2))(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(4, 4))(image_features)
        image_features = layers.Dropout(0.5)(image_features)

        x = layers.Flatten()(image_features)
        x = layers.concatenate([x, info_input])
        x = layers.Dense(128, activation=elu_layer, kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.L2(1e-2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    elif model_type == ModelType.Test_VGG_4M_2:
        elu_layer = layers.ELU()
        image_features = layers.Conv2D(32, (3, 3), padding='same', activation=elu_layer)(image_input)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(32, (3, 3), padding='same', activation=elu_layer)(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
        image_features = layers.Dropout(0.2)(image_features)

        image_features = layers.Conv2D(64, (3, 3), padding='same', activation=elu_layer)(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(64, (3, 3), padding='same', activation=elu_layer)(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
        image_features = layers.Dropout(0.2)(image_features)

        image_features = layers.Conv2D(128, (3, 3), padding='same', activation=elu_layer)(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(64, (3, 3), padding='same', activation=elu_layer)(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(4, 4))(image_features)
        image_features = layers.Dropout(0.2)(image_features)

        x = layers.Flatten()(image_features)
        x = layers.concatenate([x, info_input])
        x = layers.Dense(128, activation=elu_layer)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation=elu_layer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32, activation=elu_layer)(x)
        x = layers.BatchNormalization()(x)
    elif model_type == ModelType.Simple:
        image_features = layers.GaussianNoise(stddev=0.025)(image_input)
        image_features = layers.Conv2D(64, (7, 7), padding='same', activation=layers.ELU(),
                                       kernel_initializer='he_normal')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(128, (5, 5), padding='same', activation=layers.ELU(),
                                       kernel_initializer='he_normal')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(3, 3))(image_features)

        image_features = layers.Conv2D(64, (5, 5), padding='same', activation=layers.ELU(),
                                       kernel_initializer='he_normal')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(32, (3, 3), padding='same', activation=layers.ELU(),
                                       kernel_initializer='he_normal', kernel_regularizer=regularizers.L2(1e-2))(
            image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(3, 3))(image_features)
        image_features = layers.Dropout(0.2)(image_features)

        x = layers.Flatten()(image_features)
        x = layers.concatenate([x, info_input])
        x = layers.Dense(256, activation=layers.ELU(), kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.L2(1e-2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation=layers.ELU(), kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.L2(1e-2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation=layers.ELU(), kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.L2(1e-2))(x)
        x = layers.BatchNormalization()(x)

    pixel_prediction = layers.Dense(2, name="pixel_prediction")(x)

    model = keras.Model(
        inputs=[image_input, info_input],
        outputs={"pixel_prediction": pixel_prediction}
    )

    return model
