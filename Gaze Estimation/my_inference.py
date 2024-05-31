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


def plot_image(image):
    plt.figure(figsize=(14, 11))
    image = img_to_array(image)
    image = image.astype('uint8')
    plt.axis('off')
    plt.imshow(image)
    plt.show()


# Parameters chosen
width_pixels, height_pixels = 1920, 1080
width_mm, height_mm = 380, 215
human_distance_cm = 50


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


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_pixels)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_pixels)
frame_delay = 750

model_type = ModelType.PretrainedFaceDetection
model_name = "ResNet_4M_ELU-10-(128, 128)"

model_folder = "Models/" + str(model_type).split(".")[1]
model_path = ("" if model_folder == "" else model_folder + "/") + model_name
model = models.load_model(model_path + ".h5")

with Listener() as listener:
    while True:
        ret, frame = cap.read()

        image_input, info_input = preprocess_image(model_type, model_name, frame)
        if model_type == ModelType.PretrainedFaceDetection and image_input is None:
            print("No face detected!")
        else:
            plot_image(image_input[0])

            image_input /= 255

            y_pred = np.squeeze(model.predict([image_input, info_input], verbose=0)["pixel_prediction"])
            y_pred = np.clip(y_pred, 0, 1)

            print(y_pred)

            x_pos = int(y_pred[0] * width_pixels)
            y_pos = int(y_pred[1] * height_pixels)

            blank_image = np.zeros((height_pixels, width_pixels, 3), np.uint8)
            cv2.circle(blank_image, (x_pos, y_pos), 10, (255, 255, 255), -1)

            cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Window', blank_image)

        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cap.release()
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
