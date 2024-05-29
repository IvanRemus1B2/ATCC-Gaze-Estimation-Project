import zipfile
import numpy as np
import matplotlib.pyplot as plt
import io
import tensorflow as tf
import csv

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

from typing import Union
from keras import models

import zipfile
import zipfile2

import cv2

from mtcnn.mtcnn import MTCNN


class MyCustomGeneratorForCreation(keras.utils.Sequence):
    def __init__(self, archive, dataset_name: str, batch_size: int,
                 image_resize_shape: Union[tuple[int, int], None],
                 augmentation_generator: Union[ImageDataGenerator, None] = None,
                 verbose: bool = False):
        self.batch_size = batch_size
        self.image_resize_shape = image_resize_shape

        self.archive = archive
        self.augmentation_generator = augmentation_generator

        self.images_names = self.read_lines(dataset_name, verbose)

    def read_lines(self, dataset_name: str, verbose: bool = False):
        with self.archive.open(dataset_name, "r") as file:
            lines = file.readlines()
            if verbose:
                print(f"\nFor {dataset_name}")

                header = str(lines[0], 'utf-8').split(",")
                header[-1] = header[-1][:-2]
                print(f"File Header: {header}")

            file_names = []

            for line in lines[1:]:
                values = str(line, 'utf-8').split(",")
                values[-1] = values[-1][:-2]

                # class_name = get_class_name(values[0])
                file_name = values[0]
                file_names.append(file_name)

        return file_names

    def __len__(self):
        return (np.ceil(len(self.images_names) / float(self.batch_size))).astype(np.int64)

    def __getitem__(self, index):
        pos = index * self.batch_size

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


def show_face_box(dataset_name):
    zip_file_name = "PoG Dataset.zip"
    archive = zipfile2.ZipFile(zip_file_name, "r")

    dataset_generator = MyCustomGeneratorForCreation(archive, dataset_name, 1, None, None, False)
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


def create_dataset(from_dataset: str, new_dataset_name: str, index_range: Union[tuple[int, int], None] = None,
                   verbose: bool = True):
    zip_file_name = "PoG Dataset.zip"
    archive = zipfile2.ZipFile(zip_file_name, "r")

    dataset_generator = MyCustomGeneratorForCreation(archive, from_dataset, 1, None, None, False)
    no_instances = dataset_generator.__len__()

    if index_range is None:
        index_range = (0, no_instances - 1)

    detector = MTCNN()
    abnormal_files = []

    with open(new_dataset_name + ".csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        # Changed way to do it for train dataset due to memory issues...
        # writer.writerow(
        #     ["file_name", "box_x", "box_y", "box_width", "box_height", "confidence", "left_eye_x", "left_eye_y",
        #      "right_eye_x", "right_eye_y", "nose_x", "nose_y", "mouth_left_x", "mouth_left_y", "mouth_right_x",
        #      "mouth_right_y"])

        for index in range(index_range[0], min(index_range[1] + 1, no_instances)):
            if verbose and index % 100 == 0:
                print(f"At {index}")

            image, file_name = dataset_generator.__getitem__(index)

            # detect faces in the image
            faces = detector.detect_faces(image)
            found_face = True
            if len(faces) == 0:
                abnormal_files.append(file_name)
                found_face = False
            else:
                face_info = faces[0]
                if len(faces) > 1:
                    abnormal_files.append(file_name)
                    best_index = 0
                    best_confidence = 0
                    for index, face in enumerate(faces):
                        if best_confidence < face['confidence']:
                            best_confidence = face['confidence']
                            best_index = index
                    face_info = faces[best_index]
                elif face_info['confidence'] <= 0.5:
                    abnormal_files.append(file_name)

            # box_x, box_y, box_width, box_height = face_info['box']
            # confidence = face_info['confidence']
            # left_eye_x, left_eye_y = face_info['left_eye']
            # right_eye_x, right_eye_y = face_info['right_eye']
            # nose_x, nose_y = face_info['nose']
            # mouth_left_x, mouth_left_y = face_info['mouth_left']
            # mouth_right_x, mouth_right_y = face_info['mouth_right']

            values = [file_name]
            if found_face:
                values += face_info['box']
                values += [face_info['confidence']]
                values += face_info['keypoints']['left_eye']
                values += face_info['keypoints']['right_eye']
                values += face_info['keypoints']['nose']
                values += face_info['keypoints']['mouth_left']
                values += face_info['keypoints']['mouth_right']
            else:
                values += ([0] * 15)

            writer.writerow(values)

    print(f"For {from_dataset} check: {abnormal_files}")


if __name__ == '__main__':
    # create_dataset("pog corrected validation3.csv", "face detection validation")
    # Previously found problematic files
    # ['an482.jpg', 'an489.jpg', 'an509.jpg', 'ARA_529.jpg', 'ARA_549.jpg', 'MD580.jpg', 'ei531.jpg']
    # New found:
    # ['an482.jpg', 'an489.jpg', 'an509.jpg', 'ARA_529.jpg', 'ARA_549.jpg', 'MD580.jpg', 'ei531.jpg']

    # create_dataset("pog corrected test3.csv", "face detection test")
    # Found problematic files:
    # ['an456.jpg', 'an475.jpg', 'ichim658.jpg', 'ichim586.jpg', 'HDM747.jpg', 'MD510.jpg', 'MD516.jpg', 'MD548.jpg']

    # create_dataset("pog corrected train3.csv", "face detection train", index_range=(0, 2500))
    # Found:['an8.jpg', 'an16.jpg', 'an58.jpg', 'an63.jpg', 'an112.jpg', 'an123.jpg', 'an141.jpg', 'an147.jpg', 'an158.jpg', 'an172.jpg', 'an180.jpg', 'an194.jpg', 'an222.jpg', 'an233.jpg', 'an234.jpg', 'an248.jpg', 'an249.jpg', 'an270.jpg', 'an277.jpg', 'an278.jpg', 'an293.jpg', 'an309.jpg', 'an314.jpg', 'an322.jpg', 'an329.jpg', 'an384.jpg', 'an410.jpg', 'an417.jpg', 'ichim590.jpg', 'ichim769.jpg', 'ichim545.jpg', 'ichim194.jpg', 'mihai_bojescu_1711116742.9454212.png', 'mihai_bojescu_1711116826.216444.png']

    # create_dataset("pog corrected train3.csv", "face detection train", index_range=(2501, 5000))
    # Found:['mihai_bojescu_1711118507.1910124.png']

    # create_dataset("pog corrected train3.csv", "face detection train", index_range=(5001, 7500))
    # Found:['ARA_107.jpg', 'ARA_175.jpg', 'ARA_192.jpg', 'ARA_201.jpg', 'ARA_246.jpg', 'ARA_297.jpg', 'ARA_344.jpg', 'ARA_374.jpg', 'ARA_438.jpg']

    # create_dataset("pog corrected train3.csv", "face detection train", index_range=(7501, 10_000))
    # Found: ['HDM7.jpg', 'HDM15.jpg', 'HDM64.jpg', 'HDM194.jpg', 'HDM228.jpg', 'HDM261.jpg', 'HDM301.jpg', 'HDM306.jpg', 'HDM340.jpg', 'HDM370.jpg', 'HDM398.jpg', 'HDM443.jpg', 'HDM521.jpg', 'HDM582.jpg', 'HDM626.jpg', 'HDM678.jpg', 'HDM720.jpg', 'HDM723.jpg']

    # create_dataset("pog corrected train3.csv", "face detection train", index_range=(10_001, 12_500))
    # Found:['DG460.jpg', 'MD65.jpg', 'MD87.jpg', 'MD101.jpg', 'MD143.jpg', 'MD200.jpg', 'MD213.jpg', 'MD473.jpg', 'MD495.jpg', 'MD499.jpg']

    # create_dataset("pog corrected train3.csv", "face detection train", index_range=(12_501, 15_000))
    # Found:['mrg247.jpg', 'mrg248.jpg']

    create_dataset("pog corrected train3.csv", "face detection train", index_range=(15_001, 17_000))
    # Found:['ei216.jpg', 'ei388.jpg', 'ei401.jpg', 'ei428.jpg', 'ei434.jpg']

    # All found problematic:

    # create_dataset("pog corrected validation3.csv", "face detection validation")
    # create_dataset("pog corrected validation3.csv", "face detection validation")

    # show_face_box("pog corrected validation3.csv")
    # # ['an482.jpg', 'an489.jpg', 'an509.jpg', 'ARA_529.jpg', 'ARA_549.jpg', 'MD580.jpg', 'ei531.jpg']
    #
    # show_face_box("pog corrected test3.csv")
