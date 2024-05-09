import zipfile
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import io
import tensorflow as tf


def get_class_name(file_name: str) -> str:
    class_name = ""
    for character in file_name:
        if not character.isalpha():
            break
        class_name += character
    return class_name


def read_dataset(archive, file_name: str, image_resize_shape: tuple[int, int]):
    images = None
    images_info = None
    targets = None

    image_shapes = set()

    modified_files = dict()
    skipped_files = dict()
    with archive.open(file_name, "r") as file:
        lines = file.readlines()
        print(f"\nFor {file_name}")

        header = str(lines[0], 'utf-8').split(",")
        header[-1] = header[-1][:-2]
        print(f"File Header: {header}")

        for line in lines[1:]:
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

            # TODO:Make sure that replacing this works the same way

            x_norm, y_norm = np.float32(x_pixel_pos / width_pixels), np.float32(y_pixel_pos / height_pixels)

            # Read images,assume they are jpg
            with archive.open("PoG Dataset/" + file_name) as zip_image:
                with Image.open(io.BytesIO(zip_image.read())) as image:
                    # TODO:Consider making the dataset without an initial resizing
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

    return (images, images_info), targets


def main():
    zip_file_name = "PoG Dataset.zip"
    archive = zipfile.ZipFile(zip_file_name, "r")

    # Create datasets as tuples of (image,info),target
    file_names = ["pog corrected test3.csv", "pog corrected train3.csv", "pog corrected validation3.csv"]
    image_resize_shape = (220, 350)

    x_train, y_train = read_dataset(archive, "pog corrected train3.csv", image_resize_shape)


if __name__ == '__main__':
    main()
