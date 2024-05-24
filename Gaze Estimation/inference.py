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

width_pixels, height_pixels = 1920, 1080
width_mm, height_mm = 380, 215
human_distance_cm = 50
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_pixels)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_pixels)

model_folder = "Models"
model_name = "Test_VGG_1M_Regularized_ELU-2-(128, 128)"

# TODO:Check whether this resizing is done properly when not equal dimensions
height_resize, width_resize = list(map(int, model_name.split("-")[2][1:-1].split(",")))

model_path = ("" if model_folder == "" else model_folder + "/") + model_name

model = models.load_model(model_path + ".h5")

with Listener() as listener:
    while True:
        ret, frame = cap.read()
        image_input = img_to_array(frame)

        # TODO:Maybe the channels are BGR and the model was trained in RGB?
        image_copy = copy.deepcopy(image_input)
        image_input[:, :, 0] = image_copy[:, :, 2]
        image_input[:, :, 2] = image_copy[:, :, 0]

        image_input = np.array(tf.expand_dims(tf.image.resize(image_input, (height_resize, width_resize)), 0))
        image_input /= 255

        info_input = np.array([width_pixels, height_pixels, width_mm, height_mm, human_distance_cm]).reshape(
            (1, -1))

        y_pred = np.squeeze(model.predict([image_input, info_input], verbose=0)["pixel_prediction"])

        print(y_pred)

        x_pos = int(y_pred[0] * width_pixels)
        y_pos = int(y_pred[1] * height_pixels)

        blank_image = np.zeros((height_pixels, width_pixels, 3), np.uint8)
        cv2.circle(blank_image, (y_pos, x_pos), 10, (255, 255, 255), -1)
        cv2.imshow('Window', blank_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    listener.join()

    exit(1)
