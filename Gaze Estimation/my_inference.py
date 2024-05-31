import copy

import cv2
import csv
from pynput.mouse import Listener
import pyautogui
import os
import numpy as np
from keras import models
import tensorflow as tf
from keras.utils import img_to_array
import matplotlib.pyplot as plt
from my_models import ModelType
from mtcnn.mtcnn import MTCNN
from keras import layers
from keras_cv.layers import RandAugment


def plot_image(image):
    plt.figure(figsize=(14, 11))
    image = img_to_array(image)
    image = image.astype('uint8')
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def preprocess_image(model_type: ModelType, model_name: str, frame):
    height_resize, width_resize = list(map(int, model_name.split("-")[2][1:-1].split(",")))
    image_input = img_to_array(frame)

    # Change to RGB since this is what the image seems to be trained on
    # TODO:Do more digging into this...Its weird how it changes
    image_input[:, :, 0], image_input[:, :, 2] = image_input[:, :, 2], image_input[:, :, 0]
    info_input = [width_pixels, height_pixels, width_mm, height_mm, human_distance_cm]

    if model_type == ModelType.PretrainedFaceDetection:
        detector = MTCNN()
        faces = detector.detect_faces(image_input)

        if len(faces) == 0:
            return None, None

        best_index = -1
        best_confidence = 0
        for index, face in enumerate(faces):
            if best_confidence < face['confidence']:
                best_confidence = face['confidence']
                best_index = index
        face = faces[best_index]

        box_x, box_y, box_width, box_height = face['box']
        image_input = image_input[box_y:box_y + box_height, box_x:box_x + box_width, :]

        to_add = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
        for feature in to_add:
            info_input += list(face["keypoints"][feature])

        image_width, image_height = width_pixels, height_pixels
        for index2 in range(5, 5 + 2 * 5):
            info_input[index2] /= (image_width if index2 % 2 == 1 else image_height)

        # Add face distance from left and right
        info_input.append(box_x / image_width)
        info_input.append((image_width - min(box_x + box_width, image_width)) / image_width)

        # Add face distance from top and down
        info_input.append(box_y / image_height)
        info_input.append((image_height - min(box_y + box_height, image_height)) / image_height)

    # TODO:Check whether this resizing is done properly when not equal dimensions
    image_input = np.array(tf.expand_dims(tf.image.resize(image_input, (height_resize, width_resize)), 0))
    info_input = np.array(info_input).reshape((1, -1))

    return image_input, info_input


def compute_distance_cm_to_px(
        distance_cm: float,
        width_pixels1: float, height_pixels1: float,
        width_mm1, height_mm1):
    diagonal_inches = np.sqrt(width_mm1 ** 2 + height_mm1 ** 2) / 25.4
    ppi = np.sqrt(width_pixels1 ** 2 + height_pixels1 ** 2) / diagonal_inches

    distance_pixels = distance_cm * ppi / 2.54

    return int(distance_pixels)


# ------- Parameters to set
# -- Parameters chosen for the device
width_pixels, height_pixels = 1920, 1080
width_mm, height_mm = 380, 215
human_distance_cm = 50

# -- Parameters to set for model and time delay
frame_delay = 1
model_type = ModelType.PretrainedFaceDetection
model_name = "ResNet_4M_ELU-11-(96, 160)"
circle_radius_cm = 4.35
no_channels = 3

use_tta = True
use_rand_augment = False
tta_iterations = 30
# ----------------------
circle_radius_px = compute_distance_cm_to_px(circle_radius_cm, width_pixels, height_pixels, width_mm, height_mm)

model_folder = "Models/" + str(model_type).split(".")[1]
model_path = ("" if model_folder == "" else model_folder + "/") + model_name
model = models.load_model(model_path + ".h5")

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width_pixels)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height_pixels)

with Listener() as listener:
    height_resize, width_resize = list(map(int, model_name.split("-")[2][1:-1].split(",")))
    if use_tta:
        input_layer = layers.Input(shape=(height_resize, width_resize, no_channels))
        output, image_augmentation = None, None
        if not use_rand_augment:
            output = layers.GaussianNoise(stddev=0.025)(input_layer, training=True)
            output = layers.RandomBrightness(factor=0.15, value_range=[0.0, 1.0])(output, training=True)
        else:
            output = RandAugment(value_range=[0, 1], geometric=False, magnitude=0.15, magnitude_stddev=0.15)(
                input_layer, training=True)

        image_augmentation = tf.keras.Model(inputs=input_layer, outputs=output)

    while True:
        ret, frame = video_capture.read()

        image_input, info_input = preprocess_image(model_type, model_name, frame)
        if model_type == ModelType.PretrainedFaceDetection and image_input is None:
            print("No face detected!")
        else:
            plot_image(image_input[0])

            image_input /= 255
            y_pred = None
            if use_tta:
                augmented_images = np.clip(image_augmentation.predict(
                    np.tile(image_input, (tta_iterations, 1, 1, 1)), verbose=0), 0, 1)
                y_pred = np.squeeze(np.mean(
                    model.predict(
                        (augmented_images, np.tile(info_input, (tta_iterations, 1))), verbose=0)[
                        'pixel_prediction'], axis=0))
            else:
                y_pred = np.squeeze(model.predict([image_input, info_input], verbose=0)["pixel_prediction"])

            y_pred = np.clip(y_pred, 0, 1)

            # print(y_pred)

            x_pos = int(y_pred[0] * width_pixels)
            y_pos = int(y_pred[1] * height_pixels)
            # x_pos, y_pos = int(0.5 * width_pixels), int(0.5 * height_pixels)

            blank_image = np.zeros((height_pixels, width_pixels, 3), np.uint8)

            cv2.circle(blank_image, (x_pos, y_pos), color=(255, 255, 255), radius=circle_radius_px, thickness=1)
            cv2.circle(blank_image, (x_pos, y_pos), color=(255, 255, 255), radius=10, thickness=-1)

            cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Window', blank_image)

        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    listener.join()

    exit(1)

# def test2():
#     index_count = 0
#     x_values = [0, 1, 1, 0, 0.5]
#     y_values = [0, 0, 1, 1, 0.5]
#
#     while True:
#         x_pos = int(width_pixels * x_values[index_count % 5])
#         y_pos = int(height_pixels * y_values[index_count % 5])
#         print(x_pos, "-", y_pos)
#         index_count += 1
#
#         blank_image = np.zeros((height_pixels, width_pixels, 3), np.uint8)
#         cv2.circle(blank_image, (x_pos, y_pos), 10, (255, 255, 255), -1)
#
#         cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
#         cv2.setWindowProperty("Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#         cv2.imshow('Window', blank_image)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#
# test2()
