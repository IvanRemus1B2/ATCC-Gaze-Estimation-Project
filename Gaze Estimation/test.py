import zipfile
import numpy as np
import matplotlib.pyplot as plt
import io
import tensorflow as tf

from PIL import Image

import keras


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


if __name__ == '__main__':
    # zip_file_name = "PoG Dataset.zip"
    # archive = zipfile.ZipFile(zip_file_name, "r")
    #
    # file_name = "imp1.jpg"
    #
    # with archive.open("PoG Dataset/" + file_name) as zip_image:
    #     with Image.open(io.BytesIO(zip_image.read())) as image:
    #         np_image = np.array(image)
    #
    #         print(np_image)
    a = np.array([[1, 2], [1, 2], [1, 3]])
    b = np.array([[2, 3], [1, 3], [1, 2]])

    loss = keras.losses.MeanSquaredError()(a, b)
    loss2 = np.mean(np.square(np.sum(a - b, axis=1)))

    print(loss)
    print(loss2)
