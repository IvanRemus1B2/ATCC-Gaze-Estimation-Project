from enum import Enum

import keras
from keras import layers
from keras import regularizers

import tensorflow as tf

from keras_cv.layers import RandAugment


class ModelArchitectureType(Enum):
    Test_VGG_4M = 0,
    Test_VGG_4M_Regularized_ELU = 1,
    Test_VGG_1M_Regularized_ELU = 2,
    Test_VGG_4M_2 = 3,
    ResNet_4M_RELU = 4,
    ResNet_4M_ELU = 5,
    ResNet_5M_ELU_RA = 6,
    ResNet_5M_RELU_RA = 7,
    ResNet_25M_ELU_RA = 8,
    ResNet_25M_RELU_RA = 9,
    Simple = 10


class ModelType(Enum):
    Basic = 0,
    PretrainedFaceDetection = 1


# ResNet architecture
# From Kaggle Resnet+Keras(amazing)
def identity_block(X: tf.Tensor, level: int, block: int, filters: list[int],
                   activation_type: str, kernel_initializer: str = 'glorot_uniform') -> tf.Tensor:
    """
    Creates an identity block (see figure 3.1 from readme)

    Input:
        X - input tensor of shape (m, height_prev, width_prev, chan_prev)
        level - integer, one of the 5 levels that our networks is conceptually divided into (see figure 3.1 in the readme file)
              - level names have the form: conv2_x, conv3_x ... conv5_x
        block - each conceptual level has multiple blocks (1 identity and several convolutional blocks)
                block is the number of this block within its conceptual layer
                i.e. first block from level 2 will be named conv2_1
        filters - a list on integers, each of them defining the number of filters in each convolutional layer

    Output:
        X - tensor (m, height, width, chan)
    """

    # layers will be called conv{level}_iden{block}_{convlayer_number_within_block}'
    conv_name = f'conv{level}_{block}' + '_{layer}_{type}'

    # unpack number of filters to be used for each conv layer
    f1, f2, f3 = filters

    # the shortcut branch of the identity block
    # takes the value of the block input
    X_shortcut = X

    # first convolutional layer (plus batch norm & relu activation, of course)
    X = layers.Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1),
                      padding='valid', name=conv_name.format(layer=1, type='conv'),
                      kernel_initializer=kernel_initializer)(X)

    X = layers.BatchNormalization(axis=3, name=conv_name.format(layer=1, type='bn'))(X)
    X = layers.Activation(activation_type, name=conv_name.format(layer=1, type=activation_type))(X)

    # second convolutional layer
    X = layers.Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', name=conv_name.format(layer=2, type='conv'),
                      kernel_initializer=kernel_initializer)(X)
    X = layers.BatchNormalization(axis=3, name=conv_name.format(layer=2, type='bn'))(X)
    X = layers.Activation(activation_type)(X)

    # third convolutional layer
    X = layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1),
                      padding='valid', name=conv_name.format(layer=3, type='conv'),
                      kernel_initializer=kernel_initializer)(X)
    X = layers.BatchNormalization(axis=3, name=conv_name.format(layer=3, type='bn'))(X)

    # add shortcut branch to main path
    X = layers.Add()([X, X_shortcut])

    # relu activation at the end of the block
    X = layers.Activation(activation_type, name=conv_name.format(layer=3, type=activation_type))(X)

    return X


def convolutional_block(X: tf.Tensor, level: int, block: int, filters: list[int],
                        activation_type: str, kernel_initializer: str = 'glorot_uniform',
                        s: tuple[int, int] = (2, 2)) -> tf.Tensor:
    """
    Creates a convolutional block (see figure 3.1 from readme)

    Input:
        X - input tensor of shape (m, height_prev, width_prev, chan_prev)
        level - integer, one of the 5 levels that our networks is conceptually divided into (see figure 3.1 in the readme file)
              - level names have the form: conv2_x, conv3_x ... conv5_x
        block - each conceptual level has multiple blocks (1 identity and several convolutional blocks)
                block is the number of this block within its conceptual layer
                i.e. first block from level 2 will be named conv2_1
        filters - a list on integers, each of them defining the number of filters in each convolutional layer
        s   - stride of the first layer;
            - a conv layer with a filter that has a stride of 2 will reduce the width and height of its input by half

    Output:
        X - tensor (m, height, width, chan)
    """

    # layers will be called conv{level}_{block}_{convlayer_number_within_block}'
    conv_name = f'conv{level}_{block}' + '_{layer}_{type}'

    # unpack number of filters to be used for each conv layer
    f1, f2, f3 = filters

    # the shortcut branch of the convolutional block
    X_shortcut = X

    # first convolutional layer
    X = layers.Conv2D(filters=f1, kernel_size=(1, 1), strides=s, padding='valid',
                      name=conv_name.format(layer=1, type='conv'),
                      kernel_initializer=kernel_initializer)(X)
    X = layers.BatchNormalization(axis=3, name=conv_name.format(layer=1, type='bn'))(X)
    X = layers.Activation(activation_type, name=conv_name.format(layer=1, type=activation_type))(X)

    # second convolutional layer
    X = layers.Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      name=conv_name.format(layer=2, type='conv'),
                      kernel_initializer=kernel_initializer)(X)
    X = layers.BatchNormalization(axis=3, name=conv_name.format(layer=2, type='bn'))(X)
    X = layers.Activation(activation_type, name=conv_name.format(layer=2, type=activation_type))(X)

    # third convolutional layer
    X = layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                      name=conv_name.format(layer=3, type='conv'),
                      kernel_initializer=kernel_initializer)(X)
    X = layers.BatchNormalization(axis=3, name=conv_name.format(layer=3, type='bn'))(X)

    # shortcut path
    X_shortcut = layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=s, padding='valid',
                               name=conv_name.format(layer='short', type='conv'),
                               kernel_initializer=kernel_initializer)(X_shortcut)
    X_shortcut = layers.BatchNormalization(axis=3, name=conv_name.format(layer='short', type='bn'))(X_shortcut)

    # add shortcut branch to main path
    X = layers.Add()([X, X_shortcut])

    # nonlinearity
    X = layers.Activation(activation_type, name=conv_name.format(layer=3, type=activation_type))(X)

    return X


def MyResNetBasic(image_shape: tuple[int, int, int], info_shape: int,
                  activation_type: str,
                  std_dev: float = 0.025, brightness_factor: float = 0.15,
                  value_range: tuple[float, float] = (0, 1)) -> keras.Model:
    """
        Builds the ResNet50 model (see figure 4.2 from readme)

        Input:
            - input_size - a (height, width, chan) tuple, the shape of the input images
            - classes - number of classes the model must learn

        Output:
            model - a Keras Model() instance
    """

    # tensor placeholder for the model's input
    image_input = keras.Input(
        shape=image_shape, name="image"
    )
    info_input = keras.Input(
        shape=(info_shape,), name="info"
    )

    # Preprocessing layer

    X = keras.layers.GaussianNoise(stddev=std_dev)(image_input)
    X = keras.layers.RandomBrightness(factor=brightness_factor, value_range=value_range)(X)

    ### Level 1 ###

    # padding
    X = layers.ZeroPadding2D((3, 3))(X)

    # convolutional layer, followed by batch normalization and relu activation
    X = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                      name='conv1_1_1_conv',
                      kernel_initializer='glorot_uniform')(X)
    X = layers.BatchNormalization(axis=3, name='conv1_1_1_nb')(X)
    X = layers.Activation(activation_type)(X)

    ### Level 2 ###

    # max pooling layer to halve the size coming from the previous layer
    X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # 1x convolutional block
    X = convolutional_block(X, level=2, block=1, activation_type=activation_type, filters=[64, 64, 256], s=(1, 1))

    # 2x identity blocks
    X = identity_block(X, level=2, block=2, filters=[64, 64, 256], activation_type=activation_type)
    X = identity_block(X, level=2, block=3, filters=[64, 64, 256], activation_type=activation_type)

    ### Level 3 ###

    # 1x convolutional block
    X = convolutional_block(X, level=3, block=1, filters=[128, 128, 512], s=(2, 2), activation_type=activation_type)

    # 3x identity blocks
    X = identity_block(X, level=3, block=2, filters=[128, 128, 512], activation_type=activation_type)
    X = identity_block(X, level=3, block=3, filters=[128, 128, 512], activation_type=activation_type)
    X = identity_block(X, level=3, block=4, filters=[128, 128, 512], activation_type=activation_type)

    # ### Level 4 ###
    # # 1x convolutional block
    # X = convolutional_block(X, level=4, block=1, filters=[256, 256, 1024], s=(2, 2))
    # # 5x identity blocks
    # X = identity_block(X, level=4, block=2, filters=[256, 256, 1024])
    # X = identity_block(X, level=4, block=3, filters=[256, 256, 1024])
    # X = identity_block(X, level=4, block=4, filters=[256, 256, 1024])
    # X = identity_block(X, level=4, block=5, filters=[256, 256, 1024])
    # X = identity_block(X, level=4, block=6, filters=[256, 256, 1024])
    #
    # ### Level 5 ###
    # # 1x convolutional block
    # X = convolutional_block(X, level=5, block=1, filters=[512, 512, 2048], s=(2, 2))
    # # 2x identity blocks
    # X = identity_block(X, level=5, block=2, filters=[512, 512, 2048])
    # X = identity_block(X, level=5, block=3, filters=[512, 512, 2048])

    # Pooling layers
    X = layers.AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # Flatten and concatenation
    X = layers.Flatten()(X)
    X = layers.concatenate([X, info_input])

    X = layers.BatchNormalization()(X)
    X = layers.Dense(64)(X)
    X = layers.Activation(activation_type, name='fc_1')(X)
    pixel_prediction = layers.Dense(2, activation='linear', name="pixel_prediction")(X)

    # Create model
    model = keras.Model(
        inputs=[image_input, info_input],
        outputs={"pixel_prediction": pixel_prediction},
        name='ResNetLight'
    )

    return model


def MyResNetRandAugment(image_shape: tuple[int, int, int], info_shape: int,
                        activation_type: str, kernel_initializer: str = 'he_normal',
                        magnitude: float = 0.35, magnitude_stddev: float = 0.15,
                        value_range: tuple[float, float] = (0, 1)) -> keras.Model:
    """
        Builds the ResNet50 model (see figure 4.2 from readme)

        Input:
            - input_size - a (height, width, chan) tuple, the shape of the input images
            - classes - number of classes the model must learn

        Output:
            model - a Keras Model() instance
    """

    # tensor placeholder for the model's input
    image_input = keras.Input(
        shape=image_shape, name="image"
    )
    info_input = keras.Input(
        shape=(info_shape,), name="info"
    )

    # Preprocessing layer

    X = RandAugment(value_range=value_range, geometric=False, magnitude=magnitude, magnitude_stddev=magnitude_stddev)(
        image_input)

    ### Level 1 ###

    # padding
    X = layers.ZeroPadding2D((3, 3))(X)

    # convolutional layer, followed by batch normalization and relu activation
    X = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                      name='conv1_1_1_conv',
                      kernel_initializer=kernel_initializer)(X)
    X = layers.BatchNormalization(axis=3, name='conv1_1_1_nb')(X)
    X = layers.Activation(activation_type)(X)

    ### Level 2 ###

    # max pooling layer to halve the size coming from the previous layer
    X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # 1x convolutional block
    X = convolutional_block(X, level=2, block=1, activation_type=activation_type, filters=[64, 64, 256], s=(1, 1),
                            kernel_initializer=kernel_initializer)

    # 2x identity blocks
    X = identity_block(X, level=2, block=2, filters=[64, 64, 256], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)
    X = identity_block(X, level=2, block=3, filters=[64, 64, 256], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)

    ### Level 3 ###

    # 1x convolutional block
    X = convolutional_block(X, level=3, block=1, filters=[128, 128, 512], s=(2, 2), activation_type=activation_type,
                            kernel_initializer=kernel_initializer)

    # 3x identity blocks
    X = identity_block(X, level=3, block=2, filters=[128, 128, 512], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)
    X = identity_block(X, level=3, block=3, filters=[128, 128, 512], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)
    X = identity_block(X, level=3, block=4, filters=[128, 128, 512], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)

    # ### Level 4 ###
    # 1x convolutional block
    X = convolutional_block(X, level=4, block=1, filters=[256, 256, 1024], s=(3, 3), activation_type=activation_type,
                            kernel_initializer=kernel_initializer)
    # 1x identity blocks
    X = identity_block(X, level=4, block=2, filters=[256, 256, 1024], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)

    # X = identity_block(X, level=4, block=3, filters=[256, 256, 1024])
    # X = identity_block(X, level=4, block=4, filters=[256, 256, 1024])
    # X = identity_block(X, level=4, block=5, filters=[256, 256, 1024])
    # X = identity_block(X, level=4, block=6, filters=[256, 256, 1024])
    #
    # ### Level 5 ###
    # # 1x convolutional block
    # X = convolutional_block(X, level=5, block=1, filters=[512, 512, 2048], s=(2, 2))
    # # 2x identity blocks
    # X = identity_block(X, level=5, block=2, filters=[512, 512, 2048])
    # X = identity_block(X, level=5, block=3, filters=[512, 512, 2048])

    # Pooling layers
    X = layers.AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # Flatten and concatenation
    X = layers.Flatten()(X)
    X = layers.concatenate([X, info_input])

    X = layers.BatchNormalization()(X)
    X = layers.Dense(128, kernel_initializer=kernel_initializer)(X)
    X = layers.Activation(activation_type, name='fc_1', )(X)
    pixel_prediction = layers.Dense(2, activation='linear', name="pixel_prediction")(X)

    # Create model
    model = keras.Model(
        inputs=[image_input, info_input],
        outputs={"pixel_prediction": pixel_prediction},
        name='ResNetLight'
    )

    return model


def FullResNetRandAugment(image_shape: tuple[int, int, int], info_shape: int,
                          activation_type: str, kernel_initializer: str = 'he_normal',
                          magnitude: float = 0.35, magnitude_stddev: float = 0.15,
                          value_range: tuple[float, float] = (0, 1)) -> keras.Model:
    """
        Builds the ResNet50 model (see figure 4.2 from readme)

        Input:
            - input_size - a (height, width, chan) tuple, the shape of the input images
            - classes - number of classes the model must learn

        Output:
            model - a Keras Model() instance
    """

    # tensor placeholder for the model's input
    image_input = keras.Input(
        shape=image_shape, name="image"
    )
    info_input = keras.Input(
        shape=(info_shape,), name="info"
    )

    # Preprocessing layer

    X = RandAugment(value_range=value_range, geometric=False, magnitude=magnitude, magnitude_stddev=magnitude_stddev)(
        image_input)

    ### Level 1 ###

    # padding
    X = layers.ZeroPadding2D((3, 3))(X)

    # convolutional layer, followed by batch normalization and relu activation
    X = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                      name='conv1_1_1_conv',
                      kernel_initializer=kernel_initializer)(X)
    X = layers.BatchNormalization(axis=3, name='conv1_1_1_nb')(X)
    X = layers.Activation(activation_type)(X)

    ### Level 2 ###

    # max pooling layer to halve the size coming from the previous layer
    X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # 1x convolutional block
    X = convolutional_block(X, level=2, block=1, activation_type=activation_type, filters=[64, 64, 256], s=(1, 1),
                            kernel_initializer=kernel_initializer)

    # 2x identity blocks
    X = identity_block(X, level=2, block=2, filters=[64, 64, 256], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)
    X = identity_block(X, level=2, block=3, filters=[64, 64, 256], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)

    ### Level 3 ###

    # 1x convolutional block
    X = convolutional_block(X, level=3, block=1, filters=[128, 128, 512], s=(2, 2), activation_type=activation_type,
                            kernel_initializer=kernel_initializer)

    # 3x identity blocks
    X = identity_block(X, level=3, block=2, filters=[128, 128, 512], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)
    X = identity_block(X, level=3, block=3, filters=[128, 128, 512], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)
    X = identity_block(X, level=3, block=4, filters=[128, 128, 512], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)

    # ### Level 4 ###
    # # 1x convolutional block
    X = convolutional_block(X, level=4, block=1, filters=[256, 256, 1024], s=(2, 2), activation_type=activation_type,
                            kernel_initializer=kernel_initializer)
    # # 5x identity blocks
    X = identity_block(X, level=4, block=2, filters=[256, 256, 1024], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)
    X = identity_block(X, level=4, block=3, filters=[256, 256, 1024], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)
    X = identity_block(X, level=4, block=4, filters=[256, 256, 1024], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)
    X = identity_block(X, level=4, block=5, filters=[256, 256, 1024], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)
    X = identity_block(X, level=4, block=6, filters=[256, 256, 1024], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)
    #
    # ### Level 5 ###
    # 1x convolutional block
    X = convolutional_block(X, level=5, block=1, filters=[512, 512, 2048], s=(2, 2), activation_type=activation_type,
                            kernel_initializer=kernel_initializer)
    # 2x identity blocks
    X = identity_block(X, level=5, block=2, filters=[512, 512, 2048], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)
    X = identity_block(X, level=5, block=3, filters=[512, 512, 2048], activation_type=activation_type,
                       kernel_initializer=kernel_initializer)

    # Pooling layers
    X = layers.AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # Flatten and concatenation
    X = layers.Flatten()(X)
    X = layers.concatenate([X, info_input])

    X = layers.BatchNormalization()(X)
    X = layers.Dense(128, kernel_initializer=kernel_initializer)(X)
    X = layers.Activation(activation_type, name='fc_1', )(X)
    pixel_prediction = layers.Dense(2, activation='linear', name="pixel_prediction")(X)

    # Create model
    model = keras.Model(
        inputs=[image_input, info_input],
        outputs={"pixel_prediction": pixel_prediction},
        name='ResNetLight'
    )

    return model


def get_model(model_type: ModelArchitectureType, image_shape, info_shape: int) -> keras.Model:
    model = None
    image_input = keras.Input(
        shape=image_shape, name="image"
    )
    info_input = keras.Input(
        shape=(info_shape,), name="info"
    )
    x = None
    # TODO:Check if/how to do inference without the Gaussian Noise layer
    #  Also,find a suitable std for it

    # TODO:Add a way to also take as input gray images...This might improve generalization by reducing the domain

    # TODO:Add TTA with noise

    if model_type == ModelArchitectureType.Test_VGG_4M:
        image_features = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(image_input)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
        image_features = layers.Dropout(0.5)(image_features)

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
        image_features = layers.MaxPooling2D(pool_size=(4, 4))(image_features)
        image_features = layers.Dropout(0.5)(image_features)

        x = layers.Flatten()(image_features)
        x = layers.concatenate([x, info_input])
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
    elif model_type == ModelArchitectureType.Test_VGG_4M_Regularized_ELU:
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
    elif model_type == ModelArchitectureType.Test_VGG_1M_Regularized_ELU:
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
    elif model_type == ModelArchitectureType.Test_VGG_4M_2:
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
    elif model_type == ModelArchitectureType.Simple:
        image_features = layers.GaussianNoise(stddev=0.025)(image_input)
        image_features = layers.Conv2D(64, (3, 3), activation=layers.ELU(),
                                       kernel_initializer='he_normal')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(128, (5, 5), activation=layers.ELU(),
                                       kernel_initializer='he_normal')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(96, (1, 1), padding='same', activation=layers.ELU(),
                                       kernel_initializer='he_normal')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(image_features)
        image_features = layers.Dropout(0.5)(image_features)

        image_features = layers.Conv2D(64, (3, 3), activation=layers.ELU(),
                                       kernel_initializer='he_normal')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(128, (3, 3), strides=2, activation=layers.ELU(),
                                       kernel_initializer='he_normal', kernel_regularizer=regularizers.L2(1e-2))(
            image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Conv2D(64, (1, 1), padding='same', activation=layers.ELU(),
                                       kernel_initializer='he_normal')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.MaxPooling2D(pool_size=(3, 3))(image_features)
        image_features = layers.Dropout(0.5)(image_features)

        x = layers.Flatten()(image_features)
        x = layers.concatenate([x, info_input])
        x = layers.Dense(256, activation=layers.ELU(), kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.L2(1e-2))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation=layers.ELU(), kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.L2(1e-2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation=layers.ELU(), kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.L2(1e-2))(x)
        x = layers.BatchNormalization()(x)
    elif model_type == ModelArchitectureType.ResNet_4M_RELU:
        # TODO:Refactor together with the other models
        return MyResNetBasic(image_shape, info_shape, 'relu', std_dev=0.05, brightness_factor=0.25)
    elif model_type == ModelArchitectureType.ResNet_4M_ELU:
        return MyResNetBasic(image_shape, info_shape, 'elu', std_dev=0.05, brightness_factor=0.25)
    elif model_type == ModelArchitectureType.ResNet_5M_ELU_RA:
        return MyResNetRandAugment(image_shape, info_shape,
                                   activation_type='elu', kernel_initializer='he_uniform',
                                   magnitude=0.25, magnitude_stddev=0.15)
    elif model_type == ModelArchitectureType.ResNet_5M_RELU_RA:
        return MyResNetRandAugment(image_shape, info_shape,
                                   activation_type='relu', kernel_initializer='glorot_uniform',
                                   magnitude=0.35, magnitude_stddev=0.15)
    elif model_type == ModelArchitectureType.ResNet_25M_ELU_RA:
        return FullResNetRandAugment(image_shape, info_shape, activation_type='elu', kernel_initializer='he_uniform',
                                     magnitude=0.35, magnitude_stddev=0.15)
    elif model_type == ModelArchitectureType.ResNet_25M_RELU_RA:
        return FullResNetRandAugment(image_shape, info_shape, activation_type='relu',
                                     kernel_initializer='glorot_uniform',
                                     magnitude=0.35, magnitude_stddev=0.15)

    pixel_prediction = layers.Dense(2, name="pixel_prediction")(x)

    model = keras.Model(
        inputs=[image_input, info_input],
        outputs={"pixel_prediction": pixel_prediction}
    )

    return model


def test_model():
    # set input image parameters
    # Total params: 1,787,786
    # Trainable params: 1,777,674
    # Non-trainable params: 10,112

    image_size = (128, 160)
    channels = 3

    info_length = 19

    # model = MyResNetRandAugment(image_shape=(image_size[1], image_size[0], channels), info_shape=info_length,
    #                             activation_type='elu', kernel_initializer='he_uniform')
    model = FullResNetRandAugment(image_shape=(image_size[1], image_size[0], channels), info_shape=info_length,
                                  activation_type='elu', kernel_initializer='he_uniform')

    model.summary()


if __name__ == '__main__':
    test_model()
