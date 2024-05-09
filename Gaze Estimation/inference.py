import cv2
import csv
from pynput.mouse import Listener
import pyautogui
import os
import numpy as np
from keras import models
import tensorflow as tf

width_pixels, height_pixels = 1920, 1080
width_mm, height_mm = 380, 215
human_distance_cm = 500
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_pixels)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_pixels)

model_folder = "Models"
model_name = "ModelType.Test1"
model_path = ("" if model_folder == "" else model_folder + "/") + model_name

model = models.load_model(model_path + ".h5")

with Listener() as listener:
    while True:
        ret, frame = cap.read()

        image_input = np.array(tf.expand_dims(tf.image.resize(np.array(frame), (32, 32)), 0))
        info_input = np.array([width_pixels, height_pixels, width_mm, height_mm, human_distance_cm]).reshape(
            (1, -1))

        y_pred = np.squeeze(model.predict([image_input, info_input], verbose=0)["pixel_prediction"])

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
