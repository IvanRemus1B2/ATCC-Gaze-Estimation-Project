import csv
import math
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
from keras_cv.layers import GridMask
from keras_cv.layers import CutMix
from keras_cv.layers import MixUp

from typing import Union
from keras import models

import zipfile
import zipfile2

import cv2

from mtcnn.mtcnn import MTCNN
from pynput import mouse


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
    no_channels = image.shape[2]
    image = img_to_array(image)
    image = image.astype('uint8')
    if no_channels == 3:
        plt.axis('off')
        plt.imshow(image)
        plt.show()
    else:
        plt.axis('off')
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
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
            image /= 255

            input_layer = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))
            # X = keras.layers.GaussianNoise(std)(input_layer, training=True)
            # X = keras.layers.RandomContrast(factor=contrast_factor)(input_layer, training=True)
            # X = keras.layers.RandomBrightness(factor=brightness_factor,value_range=[0.0, 1.0])(X,training=True)
            X = Grayscale(output_channels=3)(input_layer, training=True)
            X = GridMask(ratio_factor=0.5)(X, training=True)
            X = RandAugment(value_range=[0, 1], geometric=False, magnitude=0.15, magnitude_stddev=0.15)(X,
                                                                                                        training=True)
            # X=CutMix()(input_layer,training=True)
            # X=MixUp()(input_layer,training=True)
            #
            model = tf.keras.models.Model(inputs=input_layer, outputs=X)
            #

            noisy_image = model(image)
            print(noisy_image.shape)

            # noisy_image = tf.keras.image.rgb_to_grayscale(image)
            # print(noisy_image.shape)
            noisy_image *= 255
            np.clip(noisy_image, 0, 255)
            #
            plot_image(noisy_image[0])


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


def get_closest_point(from_dataset: str, pos_x: int, pos_y: int, width_screen: int, height_screen: int):
    best_distance = math.inf
    best_info = None
    best_file_name = None
    closest_point_x, closest_point_y = None, None

    with open(from_dataset + ".csv") as file:
        lines = file.readlines()
        for line in lines[1:]:
            values = line.split(",")
            values[-1] = values[-1][:-2]

            # class_name = get_class_name(values[0])
            file_name = values[0]

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

            x_norm, y_norm = np.float32(x_pixel_pos / width_pixels), np.float32(y_pixel_pos / height_pixels)

            target = np.array([x_norm, y_norm]).reshape((1, -1))

            if width_pixels == width_screen and height_pixels == height_screen:
                distance = (x_pixel_pos - pos_x) ** 2 + (y_pixel_pos - pos_y) ** 2
                if best_distance > distance:
                    best_distance = distance
                    best_image_info = np.array(
                        [width_pixels, height_pixels, width_mm, height_mm, human_distance_cm]).reshape(
                        (1, -1))
                    best_file_name = file_name
                    closest_point_x = x_pixel_pos
                    closest_point_y = y_pixel_pos

    return best_file_name, best_image_info, (closest_point_x, closest_point_y)


def get_face_info_from(file_name: str, image_width: int, image_height: int):
    file_info = get_image_face_detection_info([file_name])[file_name]

    box_x, box_y, box_width, box_height = int(file_info['box_x']), int(file_info['box_y']), int(
        file_info['box_width']), int(file_info[
                                         'box_height'])

    box_info = [box_x, box_y, box_width, box_height]

    face_info = list(map(np.float32, [file_info['left_eye_x'], file_info['left_eye_y'], file_info['right_eye_x'],
                                      file_info['right_eye_y'], file_info['nose_x'], file_info['nose_y'],
                                      file_info['mouth_left_x'], file_info['mouth_left_y'], file_info['mouth_right_x'],
                                      file_info['mouth_right_y']]))
    for index2 in range(2 * 5):
        face_info[index2] /= (image_width if index2 % 2 == 0 else image_height)

    # Add face distance from left and right
    face_info.append(box_x / image_width)
    face_info.append((image_width - min(box_x + box_width, image_width)) / image_width)

    # Add face distance from top and down
    face_info.append(box_y / image_height)
    face_info.append((image_height - min(box_y + box_height, image_height)) / image_height)

    face_info = np.array(face_info).reshape((1, -1))

    return face_info, box_info


def get_model_prediction_closest_for(dataset_name, x_pos, y_pos,
                                     width_pixels, height_pixels,
                                     archive, model):
    file_name, image_file_info, (dataset_x_pos, dataset_y_pos) = get_closest_point(dataset_name, x_pos, y_pos,
                                                                                   width_pixels, height_pixels)
    image_face_info, face_box = get_face_info_from(file_name, width_pixels, height_pixels)
    box_x, box_y, box_width, box_height = face_box

    image_info = np.concatenate([image_file_info, image_face_info], axis=1)

    image_array = read_image_from(archive, file_name)[box_y:box_y + box_height, box_x:box_x + box_width, :]

    image_input = np.array(
        tf.expand_dims(tf.image.resize(image_array, (height_resize, width_resize)), 0))
    image_input /= 255

    y_pred = np.squeeze(model.predict([image_input, image_info], verbose=0)["pixel_prediction"])

    predicted_x, predicted_y = np.clip(int(y_pred[0] * width_pixels + 0.5), 0, width_pixels), np.clip(
        int(y_pred[1] * height_pixels + 0.5), 0, height_pixels)

    dataset_x_pos, dataset_y_pos = int(dataset_x_pos), int(dataset_y_pos)
    predicted_x, predicted_y = int(predicted_x), int(predicted_y)

    return dataset_x_pos, dataset_y_pos, predicted_x, predicted_y


def compute_distance_cm_to_px(
        distance_cm: float,
        width_pixels1: float, height_pixels1: float,
        width_mm1, height_mm1):
    diagonal_inches = np.sqrt(width_mm1 ** 2 + height_mm1 ** 2) / 25.4
    ppi = np.sqrt(width_pixels1 ** 2 + height_pixels1 ** 2) / diagonal_inches

    distance_pixels = distance_cm * ppi / 2.54

    return int(distance_pixels)


zip_file_name = "PoG Dataset.zip"
archive = zipfile2.ZipFile(zip_file_name, "r")

x_pos, y_pos = 900, 900
width_pixels, height_pixels = 1920, 1080

model_type = ModelType.PretrainedFaceDetection
model_name = "ResNet_5M_ELU_GM_RA_GRAY-G2-(128, 160)"
circle_radius_cm = 3.45
no_channels = 3

height_resize, width_resize = list(map(int, model_name.split("-")[2][1:-1].split(",")))

model_folder = "Models/" + str(model_type).split(".")[1]
model_path = ("" if model_folder == "" else model_folder + "/") + model_name
model = models.load_model(model_path + ".h5")


def on_click(mouse_x, mouse_y, button, pressed):
    if pressed:
        datasets = ['pog corrected test3', 'pog corrected train3', 'pog corrected validation3']
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for index in range(len(datasets)):
            dataset_name, color = datasets[index], colors[index]
            dataset_x_pos, dataset_y_pos, predicted_x, predicted_y = get_model_prediction_closest_for(dataset_name,
                                                                                                      mouse_x, mouse_y,
                                                                                                      width_pixels,
                                                                                                      height_pixels,
                                                                                                      archive, model)

            blank_image = np.zeros((height_pixels, width_pixels, 3), np.uint8)

            cv2.circle(blank_image, (dataset_x_pos, dataset_y_pos), color=color, radius=10)
            cv2.circle(blank_image, (predicted_x, predicted_y), color=color, radius=10)
            cv2.line(blank_image, (dataset_x_pos, dataset_y_pos), (predicted_x, predicted_y), (255, 255, 255))

            cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Window', blank_image)


with mouse.Listener(on_click=on_click) as listener:
    listener.join()
