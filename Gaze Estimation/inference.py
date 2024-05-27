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

width_pixels, height_pixels = 1920, 1080
width_mm, height_mm = 380, 215
human_distance_cm = 50
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_pixels)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_pixels)
frame_delay = 1000

model_folder = "Models"
model_name = "Simple-5-(128, 128)"

# TODO:Check whether this resizing is done properly when not equal dimensions
height_resize, width_resize = list(map(int, model_name.split("-")[2][1:-1].split(",")))

model_path = ("" if model_folder == "" else model_folder + "/") + model_name

model = models.load_model(model_path + ".h5")


def plot_image(image):
    plt.figure(figsize=(14, 11))
    image = img_to_array(image)
    image = image.astype('uint8')
    plt.axis('off')
    plt.imshow(image)
    plt.show()


with Listener() as listener:
    while True:
        ret, frame = cap.read()
        image_input = img_to_array(frame)

        image_copy = copy.deepcopy(image_input)
        image_input[:, :, 0] = image_copy[:, :, 2]
        image_input[:, :, 2] = image_copy[:, :, 0]

        image_input = np.array(tf.expand_dims(tf.image.resize(image_input, (height_resize, width_resize)), 0))

        # plot_image(image_input[0])

        image_input /= 255

        info_input = np.array([width_pixels, height_pixels, width_mm, height_mm, human_distance_cm]).reshape(
            (1, -1))

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
