import zipfile
import numpy as np
import matplotlib.pyplot as plt
import io
import tensorflow as tf

from PIL import Image

import keras
from models import *
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from keras.utils import load_img

from typing import Union


class MyCustomGenerator(keras.utils.Sequence):
    def __init__(self, archive, dataset_name: str, batch_size: int,
                 image_resize_shape,
                 augmentation_generator: Union[ImageDataGenerator, None] = None,
                 verbose: bool = False):
        self.batch_size = batch_size
        self.image_resize_shape = image_resize_shape

        self.archive = archive
        self.augmentation_generator = augmentation_generator

        self.images_names, self.images_info, self.targets = self.read_lines(dataset_name, verbose)

    def read_lines(self, dataset_name: str, verbose: bool = False):
        with self.archive.open(dataset_name, "r") as file:
            lines = file.readlines()
            if verbose:
                print(f"\nFor {dataset_name}")

                header = str(lines[0], 'utf-8').split(",")
                header[-1] = header[-1][:-2]
                print(f"File Header: {header}")

            file_names = []
            images_file_info, targets = None, None

            for line in lines[1:]:
                values = str(line, 'utf-8').split(",")
                values[-1] = values[-1][:-2]

                # class_name = get_class_name(values[0])
                file_name = values[0]
                file_names.append(file_name)

                x_pixel_pos, y_pixel_pos = list(map(np.float32, values[1:3]))
                width_pixels, height_pixels = list(map(np.float32, values[3:5]))

                if not (0 <= x_pixel_pos <= width_pixels):
                    x_pixel_pos = np.clip(x_pixel_pos, a_min=0, a_max=width_pixels)

                if not (0 <= y_pixel_pos <= height_pixels):
                    y_pixel_pos = np.clip(y_pixel_pos, a_min=0, a_max=height_pixels)

                width_mm, height_mm = list(map(np.float32, values[5:7]))

                human_distance_cm = values[7]
                if human_distance_cm == "":
                    human_distance_cm = 40

                human_distance_cm = np.float32(human_distance_cm)
                if human_distance_cm >= 80:
                    human_distance_cm = human_distance_cm / 10

                # TODO:Make sure that replacing this works the same way
                x_norm, y_norm = np.float32(x_pixel_pos / width_pixels), np.float32(y_pixel_pos / height_pixels)

                image_info = np.array([width_pixels, height_pixels, width_mm, height_mm, human_distance_cm]).reshape(
                    (1, -1))
                target = np.array([x_norm, y_norm]).reshape((1, -1))

                if images_file_info is None:
                    images_file_info = image_info
                    targets = target
                else:
                    images_file_info = np.concatenate([images_file_info, image_info], axis=0)
                    targets = np.concatenate([targets, target], axis=0)

        return file_names, images_file_info, targets

    def __len__(self):
        return (np.ceil(len(self.images_names) / float(self.batch_size))).astype(np.int64)

    def __getitem__(self, index):
        pos = index * self.batch_size

        batch_images_info = self.images_info[pos: pos + self.batch_size, :]
        batch_targets = self.targets[pos: pos + self.batch_size, :]

        batch_images = None
        file_names = self.images_names[pos:pos + self.batch_size]
        for file_name in file_names:
            # Read images,assume they are jpg
            with self.archive.open("PoG Dataset/" + file_name) as zip_image:
                with Image.open(io.BytesIO(zip_image.read())) as image:
                    # TODO:Consider making the dataset without an initial resizing
                    image_array = img_to_array(image)
                    if self.augmentation_generator is not None:
                        image_array = self.augmentation_generator.random_transform(image_array)

                    image_array = np.array(
                        tf.expand_dims(tf.image.resize(image_array, self.image_resize_shape), 0))

                    if batch_images is None:
                        batch_images = image_array
                    else:
                        batch_images = np.concatenate([batch_images, image_array], axis=0)

        batch_images /= 255

        return (batch_images, batch_images_info), batch_targets


def plot_image(image):
    plt.figure(figsize=(14, 11))
    image = img_to_array(image)
    image = image.astype('uint8')
    plt.axis('off')
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    zip_file_name = "PoG Dataset.zip"
    archive = zipfile.ZipFile(zip_file_name, "r")

    image_size = (128, 128)
    no_channels = 3

    info_length = 5

    no_epochs = 10
    val_batch_size = 6

    augmentation_generator = ImageDataGenerator(
        # featurewise_center=False,  # set input mean to 0 over the dataset
        # samplewise_center=False,  # set each sample mean to 0
        # featurewise_std_normalization=False,  # divide inputs by std of the dataset
        # samplewise_std_normalization=False,  # divide each input by its std
        # zca_whitening=True,  # apply ZCA whitening
        # zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.05,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.05,
        # shear_range=0.,  # set range for random shear
        # zoom_range=0.,  # set range for random zoom
        # channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        # fill_mode='nearest',
        # cval=0.,  # value used for fill_mode = "constant"
        # horizontal_flip=True,  # randomly flip images
        # vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        # rescale=None,
        # set function that will be applied on each input
        # preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        # data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        # validation_split=0.0
    )

    val_generator = MyCustomGenerator(archive, "pog corrected validation3.csv",
                                      val_batch_size, image_size,
                                      augmentation_generator)

    batch = val_generator.__getitem__(100)
    print(batch[0][0])
    # print(batch[0][0].shape)
    # for index in range(batch[0][0].shape[0]):
    #     plot_image(batch[0][0][index])
