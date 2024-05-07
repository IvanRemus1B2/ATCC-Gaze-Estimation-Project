from keras.models import Sequential
from keras import datasets, layers, models
from keras import utils
from keras import regularizers
from keras.layers import Dense, Dropout, BatchNormalization
from enum import Enum


class ModelType(Enum):
    Simple = 0,
    SimpleKR = 1,
    SimpleKRLeakyRelu = 2,
    SimpleSeparableConv = 3,


def get_model(model_type: ModelType, no_classes: int):
    model = Sequential()
    if model_type == ModelType.Simple:
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(no_classes, activation='softmax'))  # num_classes = 10
    elif model_type == ModelType.SimpleKR:
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.BatchNormalization())
        model.add(
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.3))

        model.add(
            layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(
            layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.3))

        model.add(
            layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(
            layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(no_classes, activation='softmax'))
    elif model_type == ModelType.SimpleKRLeakyRelu:
        pass
    elif model_type == ModelType.SimpleSeparableConv:
        pass

    return model
