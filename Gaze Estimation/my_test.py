import csv
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import io
import tensorflow as tf

from PIL import Image

import keras
from my_models import *
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from keras.utils import load_img
from keras.metrics import MSE
from keras.metrics import MAE
from keras.losses import MeanSquaredError
from keras.losses import MeanAbsoluteError

from keras_cv.layers import Grayscale
from keras_cv.layers import RandAugment

from typing import Union
from keras import models

import zipfile
import zipfile2

import cv2

from mtcnn.mtcnn import MTCNN


class MyCustomGenerator(keras.utils.Sequence):
    def __init__(self, archive, dataset_name: str, batch_size: int,
                 image_resize_shape: Union[tuple[int, int], None],
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

                    if self.image_resize_shape is not None:
                        image_array = np.array(
                            tf.expand_dims(tf.image.resize(image_array, self.image_resize_shape), 0))
                    else:
                        image_array = np.array(
                            tf.expand_dims(image_array, 0))

                    if batch_images is None:
                        batch_images = image_array
                    else:
                        batch_images = np.concatenate([batch_images, image_array], axis=0)

        # batch_images /= 255

        return batch_images[0], file_names[0]
        # return (batch_images, batch_images_info), batch_targets


def read_image(image_name):
    images_datasets = "TestImages"
    image_path = images_datasets + "/" + image_name + ".jpg"
    with Image.open(image_path) as image:
        image = img_to_array(image)
        return image


def plot_image(image):
    plt.figure(figsize=(14, 11))
    image = img_to_array(image)
    image = image.astype('uint8')
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def see_predictions_on(model_name: str):
    height_resize, width_resize = list(map(int, model_name.split("-")[2][1:-1].split(",")))
    image_resize_shape = (height_resize, width_resize)
    image_resize_shape = (256, 256)

    # image_names = ["ivanremus619", "ivanremus611"]
    # info_inputs = np.array([[1920, 1080, 380, 215, 50], [1920, 1080, 380, 215, 50]])
    # targets = np.array([[0.023958333, 0.840740741], [0.453645833, 0.77037037]])
    # image_inputs = None
    # for image_name in image_names:
    #     image = read_image(image_name)
    #     image = np.array(tf.expand_dims(tf.image.resize(image, image_resize_shape), 0))
    #     if image_inputs is None:
    #         image_inputs = image
    #     else:
    #         image_inputs = np.concatenate([image_inputs, image], axis=0)
    # image_inputs /= 255

    model_folder = "Models"
    model_path = ("" if model_folder == "" else model_folder + "/") + model_name
    model = models.load_model(model_path + ".h5")

    zip_file_name = "PoG Dataset.zip"
    archive = zipfile2.ZipFile(zip_file_name, "r")

    max_dataset_size = 300
    verbose = False

    x_val, y_val = read_dataset(archive, "pog corrected validation3.csv", image_resize_shape, max_dataset_size, verbose)
    my_range = [150, 250]
    # y_pred = model.predict([x_val[0][my_range[0]: my_range[1] + 1], x_val[1][my_range[0]: my_range[1] + 1]], verbose=0)[
    #     "pixel_prediction"]
    for index in range(my_range[0], my_range[1] + 1):
        plot_image(x_val[0][index])
        # print(index, " - ", y_val[index], y_pred[index - my_range[0]])

    # print("MSE", np.mean(np.sum(np.square(y_pred - y_val[my_range[0]: my_range[1] + 1]), axis=1)))

    # print(MeanSquaredError(y_pred, y_val[my_range[0]: my_range[1] + 1]))
    # print(MeanAbsoluteError(y_pred, y_val[my_range[0]: my_range[1] + 1]))
    # for image_name in image_names:
    #     image = read_image(image_name)
    #     plot_image(image)


def read_image_from(archive, file_name):
    with archive.open("PoG Dataset/" + file_name) as zip_image:
        with Image.open(io.BytesIO(zip_image.read())) as image:
            image = img_to_array(image)
    return image


def show_face_box(dataset_name):
    zip_file_name = "PoG Dataset.zip"
    archive = zipfile2.ZipFile(zip_file_name, "r")

    dataset_generator = MyCustomGenerator(archive, dataset_name, 1, None, None, False)
    no_instances = dataset_generator.__len__()

    index_range = (0, no_instances - 1)

    detector = MTCNN()
    abnormal_files = []

    for index in range(index_range[0], min(index_range[1] + 1, no_instances + 1)):
        image, file_name = dataset_generator.__getitem__(index)

        # detect faces in the image
        faces = detector.detect_faces(image)
        if len(faces) > 1 or faces[0]['confidence'] <= 0.5:
            abnormal_files.append(file_name)

        if index % 100 == 0:
            print(f"At {index}")

            # continue

        # # get coordinates
        # x, y, width, height = faces[0]['box']
        #
        # start_point = (x, y)
        # end_point = (x + width, y + height)
        #
        # color = (0, 0, 255)
        #
        # thickness = 2
        #
        # image = cv2.rectangle(image, start_point, end_point, color, thickness)
        #
        # plot_image(image)

    print(f"For {dataset_name}: {abnormal_files}")


def show_box_face_on(file_names: list[str]):
    zip_file_name = "PoG Dataset.zip"
    archive = zipfile2.ZipFile(zip_file_name, "r")

    for file_name in file_names:
        image = read_image_from(archive, file_name)

        detector = MTCNN(scale_factor=0.5)
        faces = detector.detect_faces(image)

        best_index = -1
        best_confidence = 0
        for index, face in enumerate(faces):
            if best_confidence < face['confidence']:
                best_confidence = face['confidence']
                best_index = index

        if best_index >= 0:
            face_info = faces[best_index]
            print(f"{file_name} : {face_info}")

            box_x, box_y, box_width, box_height = face_info['box']
            start_point = (box_x, box_y)
            end_point = (box_x + box_width, box_y + box_height)

            color = (0, 0, 255)

            thickness = 2

            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            plot_image(image)
        else:
            print(f"{file_name} : L")


def get_image_face_detection_info(file_names: list[str]):
    dataset_names = ["face detection test", "face detection train", "face detection validation"]
    all_file_info = {}
    for dataset_name in dataset_names:
        with open(dataset_name + ".csv", mode='r') as file:
            reader = csv.reader(file)
            header = reader.__next__()
            for line in reader:
                file_name = line[0]
                if file_name in file_names:
                    file_info = {}
                    for index in range(1, len(header)):
                        file_info[header[index]] = line[index]
                    all_file_info[file_name] = file_info
    return all_file_info


def show_face_box_for(file_names: list[str], show_box: bool, image_size: Union[tuple[int, int], None],
                      std: float = 0.1, contrast_factor: float = 0.2, brightness_factor: float = 0.2):
    zip_file_name = "PoG Dataset.zip"
    archive = zipfile2.ZipFile(zip_file_name, "r")

    all_file_info = get_image_face_detection_info(file_names)

    for file_name in file_names:
        file_info = all_file_info[file_name]
        box_x, box_y, box_width, box_height = int(file_info["box_x"]), int(file_info["box_y"]), int(
            file_info["box_width"]), int(file_info[
                                             "box_height"])
        image = read_image_from(archive, file_name)
        if show_box:
            start_point = (box_x, box_y)
            end_point = (box_x + box_width, box_y + box_height)

            color = (0, 0, 255)

            thickness = 2

            image = cv2.rectangle(image, start_point, end_point, color, thickness)

            plot_image(image)
        else:
            image = tf.expand_dims(
                tf.image.resize(image[box_y:box_y + box_height, box_x:box_x + box_width, :], image_size), 0)
            plot_image(image[0])
            # image /= 255

            input_layer = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))
            # X = keras.layers.GaussianNoise(std)(input_layer, training=True)
            # X = keras.layers.RandomContrast(factor=contrast_factor)(input_layer, training=True)
            # X = keras.layers.RandomBrightness(factor=brightness_factor,value_range=[0.0, 1.0])(X,training=True)
            X = RandAugment(value_range=[0, 1], geometric=False, magnitude=0.35, magnitude_stddev=0.15)(input_layer,
                                                                                                        training=True)

            # X=Grayscale()(input_layer)
            #
            # model = tf.keras.models.Model(inputs=input_layer, outputs=X)
            #

            # noisy_image = model(image)

            # noisy_image = tf.keras.image.rgb_to_grayscale(image)
            # print(noisy_image.shape)
            # # noisy_image *= 255
            # np.clip(noisy_image, 0, 255)
            #
            # plot_image(noisy_image[0])


def read_dataset(archive, dataset_file_name: str, image_resize_shape: tuple[int, int],
                 max_dataset_size: Union[int, None] = None, verbose: bool = False):
    images = None
    images_info = None
    targets = None

    image_shapes = set()

    modified_files = dict()
    skipped_files = dict()
    with archive.open(dataset_file_name, "r") as file:
        lines = file.readlines()
        print(f"\nFor {dataset_file_name}")

        header = str(lines[0], 'utf-8').split(",")
        header[-1] = header[-1][:-2]
        print(f"File Header: {header}")

        current_image_index = 1
        for line in lines[1:]:
            if verbose:
                print(f"At {current_image_index}")

            values = str(line, 'utf-8').split(",")
            values[-1] = values[-1][:-2]

            file_name = values[0]
            x_pixel_pos, y_pixel_pos = list(map(np.float32, values[1:3]))
            width_pixels, height_pixels = list(map(np.float32, values[3:5]))

            if not (0 <= x_pixel_pos <= width_pixels):
                modified_files[
                    file_name] = f"x_pixels = {x_pixel_pos} is outside [0,{width_pixels}]."
                x_pixel_pos = np.clip(x_pixel_pos, a_min=0, a_max=width_pixels)
                modified_files[file_name] += f"Clipped to {x_pixel_pos}"

            if not (0 <= y_pixel_pos <= height_pixels):
                modified_files[
                    file_name] = f"y_pixels = {y_pixel_pos} is outside [0,{height_pixels}]."
                y_pixel_pos = np.clip(y_pixel_pos, a_min=0, a_max=height_pixels)
                modified_files[file_name] += f"Clipped to {y_pixel_pos}"

            width_mm, height_mm = list(map(np.float32, values[5:7]))
            if width_mm < 100 or width_mm > 750:
                skipped_files[file_name] = f"width_mm = {width_mm:.4f} is outside of range[100,750],skipped"
                continue

            if height_mm < 100 or height_mm > 500:
                skipped_files[file_name] = f"height_mm = {height_mm:.4f} is outside of range[100,500],skipped"
                continue

            human_distance_cm = values[7]
            if human_distance_cm == "":
                modified_files[file_name] = f"human_distance_mm was not specified,skipped"
                human_distance_cm = 40

            human_distance_cm = np.float32(human_distance_cm)
            if human_distance_cm >= 80:
                modified_files[
                    file_name] = f"human_distance_mm > 80 cm,so transform from mm to cm"
                human_distance_cm = human_distance_cm / 10

            # TODO:Make sure that replacing this works the same way

            x_norm, y_norm = np.float32(x_pixel_pos / width_pixels), np.float32(y_pixel_pos / height_pixels)

            # Read images,assume they are jpg
            with archive.open("PoG Dataset/" + file_name) as zip_image:
                with Image.open(io.BytesIO(zip_image.read())) as image:
                    # TODO:Consider making the dataset without an initial resizing
                    image_array = np.array(tf.expand_dims(tf.image.resize(img_to_array(image), image_resize_shape), 0))

            image_info = np.array([width_pixels, height_pixels, width_mm, height_mm, human_distance_cm]).reshape(
                (1, -1))
            target = np.array([x_norm, y_norm]).reshape((1, -1))

            if images_info is None:
                images_info = image_info
                targets = target
                images = image_array
            else:
                images_info = np.concatenate([images_info, image_info], axis=0)
                targets = np.concatenate([targets, target], axis=0)
                images = np.concatenate([images, image_array], axis=0)

            current_image_index += 1
            if max_dataset_size is not None and current_image_index > max_dataset_size:
                break

            # if images.shape[0] > 2000:
            #     break

    print(images.shape)
    print(images_info.shape)
    print(targets.shape)

    # print(f"No Instances:{len(lines) - 1}")
    #
    # print(f"Skipped: {len(skipped_files)}")
    # print(f"Percentage skipped files:{len(skipped_files) / (len(lines) - 1):.2f}")
    #
    # print(f"Modified: {len(modified_files)}")
    # print(f"Percentage modified files:{len(modified_files) / (len(lines) - 1):.2f}")
    # images /= 255
    return (images, images_info), targets


if __name__ == '__main__':
    # see_predictions_on("Test_VGG_1M_Regularized_ELU-1-(128, 128)")
    # see_predictions_on("Test_VGG_4M_2-1-(156, 156)")
    # see_predictions_on("Simple-4-(128, 128)")
    # see_predictions_on("Simple-5-(128, 128)")
    # show_face_box("pog corrected validation3.csv")
    image_size = (96, 160)

    file_names = ['ivanremus593.jpg']
    # file_names = ['an482.jpg', 'an489.jpg', 'an509.jpg', 'ARA_529.jpg', 'ARA_549.jpg', 'MD580.jpg', 'ei531.jpg']
    # file_names=['an456.jpg', 'an475.jpg', 'ichim658.jpg', 'ichim586.jpg', 'HDM747.jpg', 'MD510.jpg', 'MD516.jpg', 'MD548.jpg']
    # file_names=['an8.jpg', 'an16.jpg', 'an58.jpg', 'an63.jpg', 'an112.jpg', 'an123.jpg', 'an141.jpg', 'an147.jpg', 'an158.jpg', 'an172.jpg', 'an180.jpg', 'an194.jpg', 'an222.jpg', 'an233.jpg', 'an234.jpg', 'an248.jpg', 'an249.jpg', 'an270.jpg', 'an277.jpg', 'an278.jpg', 'an293.jpg', 'an309.jpg', 'an314.jpg', 'an322.jpg', 'an329.jpg', 'an384.jpg', 'an410.jpg', 'an417.jpg', 'ichim590.jpg', 'ichim769.jpg', 'ichim545.jpg', 'ichim194.jpg', 'mihai_bojescu_1711116742.9454212.png', 'mihai_bojescu_1711116826.216444.png']
    # file_names = ["ichim545.jpg", "ichim769.jpg"] <------- only for these 2 it can't find a face...
    # Found with scale_factor=0.5:
    # ichim545.jpg: {'box': [284, 0, 230, 269], 'confidence': 0.9955950379371643,
    #                'keypoints': {'left_eye': (365, 77), 'right_eye': (477, 80), 'nose': (431, 157),
    #                              'mouth_left': (367, 202), 'mouth_right': (452, 203)}}
    # ichim769.jpg: {'box': [284, 0, 230, 269], 'confidence': 0.9990921020507812,
    #                'keypoints': {'left_eye': (370, 75), 'right_eye': (481, 81), 'nose': (436, 153),
    #                              'mouth_left': (370, 200), 'mouth_right': (455, 202)}}

    # file_names = ['mihai_bojescu_1711118507.1910124.png']
    # file_names=['ARA_107.jpg', 'ARA_175.jpg', 'ARA_192.jpg', 'ARA_201.jpg', 'ARA_246.jpg', 'ARA_297.jpg', 'ARA_344.jpg', 'ARA_374.jpg', 'ARA_438.jpg']
    # file_names=['HDM7.jpg', 'HDM15.jpg', 'HDM64.jpg', 'HDM194.jpg', 'HDM228.jpg', 'HDM261.jpg', 'HDM301.jpg', 'HDM306.jpg', 'HDM340.jpg', 'HDM370.jpg', 'HDM398.jpg', 'HDM443.jpg', 'HDM521.jpg', 'HDM582.jpg', 'HDM626.jpg', 'HDM678.jpg', 'HDM720.jpg', 'HDM723.jpg']
    # file_names=['DG460.jpg', 'MD65.jpg', 'MD87.jpg', 'MD101.jpg', 'MD143.jpg', 'MD200.jpg', 'MD213.jpg', 'MD473.jpg', 'MD495.jpg', 'MD499.jpg']
    # file_names=['mrg247.jpg', 'mrg248.jpg']
    # file_names=['ei216.jpg', 'ei388.jpg', 'ei401.jpg', 'ei428.jpg', 'ei434.jpg']

    # TODO:Sometimes the program ends before showing all images...preprocessing layers
    #  might be at fault?
    show_face_box_for(file_names, False, image_size,
                      std=0.025, brightness_factor=0.15)

    # # TODO:Make tensorflow==2.14 realise i have a GPU...if only this version worked well where 2.10 did...
    # print(tf.config.list_physical_devices('GPU'))

    # show_box_face_on(["ichim545.jpg", "ichim769.jpg"])

    # show_face_box("pog corrected validation3.csv")
    # ['an482.jpg', 'an489.jpg', 'an509.jpg', 'ARA_529.jpg', 'ARA_549.jpg', 'MD580.jpg', 'ei531.jpg']

    # show_face_box("pog corrected test3.csv")

    # zip_file_name = "PoG Dataset.zip"
    # archive = zipfile.ZipFile(zip_file_name, "r")
    #
    # image_size = (128, 128)
    # no_channels = 3
    #
    # info_length = 5
    #
    # no_epochs = 10
    # val_batch_size = 6
    #
    # augmentation_generator = ImageDataGenerator(
    #     # featurewise_center=False,  # set input mean to 0 over the dataset
    #     # samplewise_center=False,  # set each sample mean to 0
    #     # featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     # samplewise_std_normalization=False,  # divide each input by its std
    #     # zca_whitening=True,  # apply ZCA whitening
    #     # zca_epsilon=1e-06,  # epsilon for ZCA whitening
    #     rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    #     # randomly shift images horizontally (fraction of total width)
    #     width_shift_range=0.05,
    #     # randomly shift images vertically (fraction of total height)
    #     height_shift_range=0.05,
    #     # shear_range=0.,  # set range for random shear
    #     # zoom_range=0.,  # set range for random zoom
    #     # channel_shift_range=0.,  # set range for random channel shifts
    #     # set mode for filling points outside the input boundaries
    #     # fill_mode='nearest',
    #     # cval=0.,  # value used for fill_mode = "constant"
    #     # horizontal_flip=True,  # randomly flip images
    #     # vertical_flip=False,  # randomly flip images
    #     # set rescaling factor (applied before any other transformation)
    #     # rescale=None,
    #     # set function that will be applied on each input
    #     # preprocessing_function=None,
    #     # image data format, either "channels_first" or "channels_last"
    #     # data_format=None,
    #     # fraction of images reserved for validation (strictly between 0 and 1)
    #     # validation_split=0.0
    # )
    #
    # val_generator = MyCustomGenerator(archive, "pog corrected validation3.csv",
    #                                   val_batch_size, image_size,
    #                                   augmentation_generator)
    #
    # batch = val_generator.__getitem__(100)
    # print(batch[0][0])
    # # print(batch[0][0].shape)
    # # for index in range(batch[0][0].shape[0]):
    # #     plot_image(batch[0][0][index])
