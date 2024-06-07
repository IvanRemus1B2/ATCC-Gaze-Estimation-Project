import csv
import zipfile
import zipfile2

from enum import Enum

import keras.optimizers
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from keras import models

from PIL import Image
import io
import tensorflow as tf

from typing import Union

from my_models import *

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from keras import metrics
from keras.utils import load_img
from keras import losses
from keras import optimizers


class Callback_MSE(tf.keras.callbacks.Callback):
    def __init__(self, filepath, x_val, y_val, best=1e9, interval=1):
        # Initialize the callback
        super().__init__()
        self.interval = interval  # Interval to check the score
        self.best = best  # Best score initialized
        self.x_images, self.x_info = x_val
        self.y = y_val
        self.filepath = filepath  # Filepath to save the best model

    def on_epoch_end(self, epoch, logs=None):
        # This function is called at the end of each epoch
        if logs is None:
            logs = {}
        if epoch % self.interval == 0:
            # Calculate predictions
            y_pred = self.model.predict([self.x_images, self.x_info], verbose=0)["pixel_prediction"]

            # print(self.y)
            # print(y_pred)
            # print(type(y_pred))
            # print(type(self.y))

            loss = keras.losses.MeanSquaredError()(y_pred, self.y)
            # loss = np.mean(np.square(np.sum(y_pred - self.y, axis=1)))

            # Print score for monitoring
            print(f"\nMSE evaluation - epoch: {epoch} - loss: {loss:.4f}")

            # Save the model if it's the best score so far
            if self.best > loss:
                self.model.save(self.filepath, overwrite=True)  # Save model
                self.best = loss  # Update best score
                print('\nModel saved with best loss:', self.best)


def get_class_name(file_name: str) -> str:
    class_name = ""
    for character in file_name:
        if not character.isalpha():
            break
        class_name += character
    return class_name


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

            class_name = get_class_name(values[0])
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


            x_norm, y_norm = np.float32(x_pixel_pos / width_pixels), np.float32(y_pixel_pos / height_pixels)

            # Read images,assume they are jpg
            with archive.open("PoG Dataset/" + file_name) as zip_image:
                with Image.open(io.BytesIO(zip_image.read())) as image:
                    image_array = np.array(tf.expand_dims(tf.image.resize(np.array(image), image_resize_shape), 0))

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

    return (images / 255, images_info), targets


def visualize_plots(history, model_path: str):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    plt.savefig(model_path + ".png")

    plt.show()


# Use the face detection datasets in order to give as input the face resized
class MyCustomGeneratorV2(keras.utils.Sequence):
    def __init__(self, archive, dataset_name: str, fd_dataset_name: str,
                 batch_size: int, image_resize_shape: tuple[int, int],
                 augmentation_generator: Union[ImageDataGenerator, None] = None,
                 verbose: bool = False):
        self.batch_size = batch_size
        self.image_resize_shape = image_resize_shape

        self.archive = archive
        self.augmentation_generator = augmentation_generator

        self.images_names, self.images_info, self.face_boxes, self.targets = self.read_lines(dataset_name,
                                                                                             fd_dataset_name, verbose)

        self.length = (np.ceil(len(self.images_names) / float(self.batch_size))).astype(np.int64)

        self.no_instances = len(self.images_names)
        self.count = 0
        self.order = np.arange(self.no_instances)

    def read_lines(self, dataset_name: str, fd_dataset_name: str, verbose: bool = False):
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

        # Create an array
        all_face_box_info = None
        all_face_info = None
        with open(fd_dataset_name, mode='r') as file:
            reader = csv.reader(file)
            header = reader.__next__()
            # file_name,box_x,box_y,box_width,box_height,confidence,left_eye_x,left_eye_y,right_eye_x,right_eye_y,nose_x,nose_y,mouth_left_x,mouth_left_y,mouth_right_x,mouth_right_y
            for index, line in enumerate(reader):
                box_x, box_y, box_width, box_height = list(map(int, line[1:5]))
                box_info = np.array([box_x, box_y, box_width, box_height], dtype=np.int32).reshape((1, -1))

                face_info = list(map(np.float32, line[6:]))
                image_width, image_height = images_file_info[index, 0:2]
                for index2 in range(2 * 5):
                    face_info[index2] /= (image_width if index2 % 2 == 0 else image_height)

                # Add face distance from left and right
                face_info.append(box_x / image_width)
                face_info.append((image_width - min(box_x + box_width, image_width)) / image_width)

                # Add face distance from top and down
                face_info.append(box_y / image_height)
                face_info.append((image_height - min(box_y + box_height, image_height)) / image_height)

                face_info = np.array(face_info).reshape((1, -1))

                if all_face_info is None:
                    all_face_box_info = box_info
                    all_face_info = face_info
                else:
                    all_face_box_info = np.concatenate([all_face_box_info, box_info], axis=0)
                    all_face_info = np.concatenate([all_face_info, face_info], axis=0)

        images_file_info = np.concatenate([images_file_info, all_face_info], axis=1)

        return file_names, images_file_info, all_face_box_info, targets

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.count % self.no_instances == 0:
            np.random.shuffle(self.order)

        pos = index * self.batch_size

        positions = self.order[pos:pos + self.batch_size]

        batch_images_info = self.images_info[positions, :]
        batch_targets = self.targets[positions, :]

        batch_images = None
        for position in positions:
            file_name = self.images_names[position]
            # Read images,assume they are jpg
            with self.archive.open("PoG Dataset/" + file_name) as zip_image:
                with Image.open(io.BytesIO(zip_image.read())) as image:
                    box_x, box_y, box_width, box_height = self.face_boxes[position]
                    # TODO:Check whether this is the right way to slice
                    image_array = img_to_array(image)[box_y:box_y + box_height, box_x:box_x + box_width, :]
                    if self.augmentation_generator is not None:
                        image_array = self.augmentation_generator.random_transform(image_array)

                    image_array = np.array(
                        tf.expand_dims(tf.image.resize(image_array, self.image_resize_shape), 0))

                    if batch_images is None:
                        batch_images = image_array
                    else:
                        batch_images = np.concatenate([batch_images, image_array], axis=0)

        batch_images /= 255

        self.count += 1

        return (batch_images, batch_images_info), batch_targets


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

        self.length = (np.ceil(len(self.images_names) / float(self.batch_size))).astype(np.int64)

        self.no_instances = len(self.images_names)
        self.count = 0
        self.order = np.arange(self.no_instances)

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
        return self.length

    def __getitem__(self, index):
        if self.count % self.no_instances == 0:
            np.random.shuffle(self.order)

        pos = index * self.batch_size

        positions = self.order[pos:pos + self.batch_size]

        batch_images_info = self.images_info[positions, :]
        batch_targets = self.targets[positions, :]

        batch_images = None
        for position in positions:
            file_name = self.images_names[position]
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

        self.count += 1

        return (batch_images, batch_images_info), batch_targets


def print_metrics(model, dataset_name: str, dataset_generator: keras.utils.Sequence,
                  image_size: tuple[int, int], no_channels: int,
                  use_tta: bool, tta_iterations: int):
    mse_error = 0.0
    mae_error = 0.0
    no_instances = 0
    cm_error = 0.0

    image_augmenter = None
    if use_tta:
        input_layer = keras.Input(shape=(image_size[0], image_size[1], no_channels))
        output = keras.layers.GaussianNoise(stddev=0.025)(input_layer, training=True)
        output = keras.layers.RandomBrightness(factor=0.15, value_range=[0.0, 1.0])(output, training=True)

        # output = RandAugment(value_range=[0, 1], geometric=False, magnitude=0.15, magnitude_stddev=0.15)(input_layer,
        #                                                                                                  training=True)

        image_augmenter = tf.keras.Model(inputs=input_layer, outputs=output)

    for index in range(len(dataset_generator)):
        x_batch, target = dataset_generator.__getitem__(index)

        no_batch_instances = target.shape[0]
        prediction = None
        if use_tta:
            prediction = np.zeros((no_batch_instances, 2))
            for index2 in range(tta_iterations):
                prediction += np.clip(
                    model.predict_on_batch((image_augmenter.predict_on_batch(x_batch[0]), x_batch[1]))[
                        'pixel_prediction'], 0, 1)
            prediction /= tta_iterations
        else:
            prediction = np.clip(model.predict_on_batch(x_batch)['pixel_prediction'], 0, 1)

        for index_batch in range(no_batch_instances):
            width_pixels, height_pixels, width_mm, height_mm = x_batch[1][index_batch, 0:4]

            diagonal_inches = np.sqrt(width_mm ** 2 + height_mm ** 2) / 25.4
            ppi = np.sqrt(width_pixels ** 2 + height_pixels ** 2) / diagonal_inches

            target_x, target_y = width_pixels * target[index_batch, 0], height_pixels * target[index_batch, 1]
            prediction_x, prediction_y = width_pixels * prediction[index_batch, 0], height_pixels * prediction[
                index_batch, 1]

            distance_pixels = np.sqrt((target_x - prediction_x) ** 2 + (target_y - prediction_y) ** 2)

            distance_cm = distance_pixels / ppi * 2.54

            cm_error += distance_cm

        mse_error += keras.losses.MeanSquaredError(reduction='sum')(prediction, target)
        mae_error += keras.losses.MeanAbsoluteError(reduction='sum')(prediction, target)

        no_instances += target.shape[0]

    mse_error /= no_instances
    mae_error /= no_instances
    cm_error /= no_instances

    print(f"{dataset_name}:\nMSE Loss: {mse_error:.4f} , MAE Loss: {mae_error:.4f} , Avg Cm: {cm_error:.4f}")


def test_model(model_folder: str,
               model_name: str,
               model_type: ModelType,
               no_channels: int,
               use_tta: bool, tta_iterations: int,
               train_batch_size: int, val_batch_size: int, test_batch_size: int,
               train_generator=None, val_generator=None, test_generator=None):
    print(f"\nFor {model_name}" + (f"(TTA-{tta_iterations})" if use_tta else "") + ":")
    zip_file_name = "PoG Dataset.zip"
    archive = zipfile2.ZipFile(zip_file_name, "r")

    height_resize, width_resize = list(map(int, model_name.split("-")[2][1:-1].split(",")))
    image_size = (height_resize, width_resize)

    if train_generator is None:
        train_generator = get_generator(model_type, archive, "pog corrected train3.csv", "face detection train.csv",
                                        train_batch_size,
                                        image_size)
    if val_generator is None:
        val_generator = get_generator(model_type, archive, "pog corrected validation3.csv",
                                      "face detection validation.csv",
                                      val_batch_size, image_size)
    if test_generator is None:
        test_generator = get_generator(model_type, archive, "pog corrected test3.csv", "face detection test.csv",
                                       test_batch_size,
                                       image_size)

    model_path = ("" if model_folder == "" else model_folder + "/") + model_name

    model = models.load_model(model_path + ".h5")

    print_metrics(model, "pog corrected train3.csv", train_generator, image_size, no_channels, use_tta, tta_iterations)
    print_metrics(model, "pog corrected validation3.csv", val_generator, image_size, no_channels, use_tta,
                  tta_iterations)
    print_metrics(model, "pog corrected test3.csv", test_generator, image_size, no_channels, use_tta,
                  tta_iterations)


def get_generator(model_type: ModelType, archive, dataset_name: str, fd_dataset_name: str, batch_size,
                  image_resize: tuple[int, int]):
    if model_type == ModelType.Basic:
        return MyCustomGenerator(archive, dataset_name, batch_size, image_resize)
    if model_type == ModelType.PretrainedFaceDetection:
        return MyCustomGeneratorV2(archive, dataset_name, fd_dataset_name, batch_size, image_resize)
    return None


class LrSchedulerType(Enum):
    CosineDecayRestarts = 0


def get_lr_scheduler(lr_scheduler_type: LrSchedulerType, init_lr: float, first_decay_steps: int):
    if lr_scheduler_type == LrSchedulerType.CosineDecayRestarts:
        return keras.optimizers.schedules.CosineDecayRestarts(
            init_lr,
            first_decay_steps,
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.0,
            name="SGDRDecay")


def train_model():
    zip_file_name = "PoG Dataset.zip"
    archive = zipfile2.ZipFile(zip_file_name, "r")
    #
    # # Create datasets as tuples of (image,info),target
    # file_names = ["pog corrected test3.csv", "pog corrected train3.csv", "pog corrected validation3.csv"]
    image_size = (128, 160)
    no_channels = 3

    no_epochs = 100
    train_batch_size = 32
    val_batch_size = test_batch_size = 32

    init_learning_rate = 1e-3
    decay_steps = 1000
    lr_scheduler_type = LrSchedulerType.CosineDecayRestarts

    weight_decay = 0.0005

    use_tta = False
    tta_iterations = 30

    model_type = ModelType.PretrainedFaceDetection
    info_length = 5 if model_type == ModelType.Basic else 5 + 4 + 10
    model_architecture_type = ModelArchitectureType.ResNet_5M_ELU_RA

    loss_function = losses.MeanAbsoluteError()
    loss_name = ("MAE" if loss_function == losses.MeanAbsoluteError() else "MSE")

    to_monitor = "val_mean_absolute_error"
    # to_monitor = "val_mean_squared_error"
    # --------------

    print("Model hyper parameters:")
    print(f"Image size:{image_size} , NoChannels:{no_channels}")
    print(f"No epochs: {no_epochs} with batch size:{train_batch_size}")

    optimizer = optimizers.Adam(learning_rate=init_learning_rate, decay=weight_decay, amsgrad=True)
    # optimizer = optimizers.Nadam(learning_rate=init_learning_rate, decay=weight_decay)
    print(f"Optimizer {optimizer}, Learning rate:{init_learning_rate} , Weight Decay:{weight_decay}")

    print(f"Loss function used:{loss_name} , monitor:{to_monitor}")

    model_folder = "Models/" + str(model_type).split(".")[1]
    model_name = str(model_architecture_type).split(".")[1] + "-AdamAMS1-" + str(image_size)
    model_path = ("" if model_folder == "" else model_folder + "/") + model_name

    # verbose = False
    # max_dataset_size = 2000
    # x_train, y_train = read_dataset(archive, "pog corrected train3.csv", image_size, max_dataset_size, verbose)
    # x_val, y_val = read_dataset(archive, "pog corrected validation3.csv", image_size, max_dataset_size, verbose)
    # x_test, y_test = read_dataset(archive, "pog corrected test3.csv", image_size, max_dataset_size, verbose)

    image_shape = list(image_size)
    image_shape.append(no_channels)
    image_shape = tuple(image_shape)
    model = get_model(model_architecture_type, image_shape, info_length)

    model.summary()

    # lr_scheduler = get_lr_scheduler(lr_scheduler_type, init_learning_rate, decay_steps)

    model.compile(optimizer=optimizer, loss=loss_function,
                  metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()])

    # callbacks = Callback_MSE(model_path + ".h5", x_val, y_val, interval=1)

    train_augmentation_generator = None
    # train_augmentation_generator = ImageDataGenerator(
    #     rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    #     # randomly shift images horizontally (fraction of total width)
    #     width_shift_range=0.05,
    #     # randomly shift images vertically (fraction of total height)
    #     height_shift_range=0.05,
    # )

    train_generator = get_generator(model_type, archive, "pog corrected train3.csv", "face detection train.csv",
                                    train_batch_size, image_size)

    val_generator = get_generator(model_type, archive, "pog corrected validation3.csv", "face detection validation.csv",
                                  val_batch_size, image_size)

    checkpoint_callback = ModelCheckpoint(filepath=model_path + ".h5",
                                          monitor=to_monitor,
                                          verbose=1,
                                          save_best_only=True,
                                          save_weights_only=False,
                                          mode='auto',
                                          save_freq='epoch')

    # history = model.fit(
    #     {"image": x_train[0], "info": x_train[1]},
    #     {"pixel_prediction": y_train},
    #     validation_data=([x_val[0], x_val[1]], y_val),
    #     epochs=no_epochs,
    #     batch_size=train_batch_size,
    #     callbacks=[callbacks]
    # )

    history = model.fit(
        x=train_generator, validation_data=val_generator,
        epochs=no_epochs,
        callbacks=[checkpoint_callback]
    )

    visualize_plots(history, model_path)

    test_generator = get_generator(model_type, archive, "pog corrected test3.csv", "face detection test.csv",
                                   test_batch_size,
                                   image_size)

    test_model(model_folder, model_name, model_type, no_channels,
               use_tta, tta_iterations,
               train_batch_size, val_batch_size, test_batch_size,
               train_generator, val_generator, test_generator)


def get_model_metrics():
    zip_file_name = "PoG Dataset.zip"
    archive = zipfile2.ZipFile(zip_file_name, "r")
    #
    # # Create datasets as tuples of (image,info),target
    # file_names = ["pog corrected test3.csv", "pog corrected train3.csv", "pog corrected validation3.csv"]
    no_channels = 3

    use_tta = False
    tta_iterations = 10

    train_batch_size = 64
    val_batch_size = test_batch_size = 64

    model_type = ModelType.PretrainedFaceDetection
    model_name = "ResNet_4M_ELU-11-(96, 160)"

    height_resize, width_resize = list(map(int, model_name.split("-")[2][1:-1].split(",")))
    image_size = (height_resize, width_resize)

    model_folder = "Models/" + str(model_type).split(".")[1]

    train_generator = get_generator(model_type, archive, "pog corrected train3.csv", "face detection train.csv",
                                    train_batch_size, image_size)

    val_generator = get_generator(model_type, archive, "pog corrected validation3.csv", "face detection validation.csv",
                                  val_batch_size, image_size)

    test_generator = get_generator(model_type, archive, "pog corrected test3.csv", "face detection test.csv",
                                   test_batch_size,
                                   image_size)

    test_model(model_folder, model_name, model_type,
               no_channels,
               use_tta, tta_iterations,
               train_batch_size, val_batch_size, test_batch_size,
               train_generator, val_generator, test_generator)


if __name__ == '__main__':
    # run_base_model()
    train_model()
    # get_model_metrics()

# def run_base_model():
#     zip_file_name = "PoG Dataset.zip"
#     archive = zipfile2.ZipFile(zip_file_name, "r")
#     #
#     # # Create datasets as tuples of (image,info),target
#     # file_names = ["pog corrected test3.csv", "pog corrected train3.csv", "pog corrected validation3.csv"]
#     image_size = (128, 128)
#     no_channels = 3
#
#     info_length = 5
#
#     no_epochs = 125
#     train_batch_size = 64
#     val_batch_size = test_batch_size = 64
#
#     model_type = ModelType.Basic
#     model_architecture_type = ModelArchitectureType.Simple
#
#     model_folder = "Models/" + str(model_type).split(".")[1]
#     model_name = str(model_architecture_type).split(".")[1] + "-5-" + str(image_size)
#     model_path = ("" if model_folder == "" else model_folder + "/") + model_name
#
#     # verbose = False
#     # max_dataset_size = 2000
#     # x_train, y_train = read_dataset(archive, "pog corrected train3.csv", image_size, max_dataset_size, verbose)
#     # x_val, y_val = read_dataset(archive, "pog corrected validation3.csv", image_size, max_dataset_size, verbose)
#     # x_test, y_test = read_dataset(archive, "pog corrected test3.csv", image_size, max_dataset_size, verbose)
#
#     image_shape = list(image_size)
#     image_shape.append(no_channels)
#     image_shape = tuple(image_shape)
#     model = get_model(model_architecture_type, image_shape, info_length)
#
#     model.summary()
#
#     model.compile(optimizer='adam', loss=losses.MeanAbsoluteError(), metrics=[metrics.MeanAbsoluteError()])
#
#     # callbacks = Callback_MSE(model_path + ".h5", x_val, y_val, interval=1)
#
#     train_augmentation_generator = None
#     # train_augmentation_generator = ImageDataGenerator(
#     #     rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
#     #     # randomly shift images horizontally (fraction of total width)
#     #     width_shift_range=0.05,
#     #     # randomly shift images vertically (fraction of total height)
#     #     height_shift_range=0.05,
#     # )
#
#     train_generator = MyCustomGenerator(archive, "pog corrected train3.csv", train_batch_size, image_size,
#                                         train_augmentation_generator)
#     # train_generator = MyCustomGeneratorV2(archive, "pog corrected train3.csv", "face detection train.csv",
#     #                                       train_batch_size, image_size)
#
#     val_generator = MyCustomGenerator(archive, "pog corrected validation3.csv", val_batch_size, image_size)
#     # val_generator = MyCustomGeneratorV2(archive, "pog corrected validation3.csv", "face detection validation.csv",
#     #                                     val_batch_size, image_size)
#
#     callbacks = ModelCheckpoint(filepath=model_path + ".h5",
#                                 monitor='val_mean_absolute_error',
#                                 verbose=1,
#                                 save_best_only=True,
#                                 save_weights_only=False,
#                                 mode='auto',
#                                 save_freq='epoch')
#
#     # history = model.fit(
#     #     {"image": x_train[0], "info": x_train[1]},
#     #     {"pixel_prediction": y_train},
#     #     validation_data=([x_val[0], x_val[1]], y_val),
#     #     epochs=no_epochs,
#     #     batch_size=train_batch_size,
#     #     callbacks=[callbacks]
#     # )
#
#     history = model.fit(
#         x=train_generator, validation_data=val_generator,
#         epochs=no_epochs,
#         callbacks=[callbacks]
#     )
#
#     visualize_plots(history, model_path)
#
#     test_generator = MyCustomGenerator(archive, "pog corrected test3.csv", test_batch_size, image_size)
#     # test_generator = MyCustomGeneratorV2(archive, "pog corrected test3.csv", "face detection test.csv", test_batch_size,
#     #                                      image_size)
#
#     test_model(model_name, train_batch_size, val_batch_size, test_batch_size,
#                train_generator, val_generator, test_generator)
