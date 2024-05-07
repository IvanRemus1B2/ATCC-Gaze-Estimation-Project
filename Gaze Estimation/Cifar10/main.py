import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras import datasets, layers, models
from keras import utils
from keras import regularizers
from keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np

from models import *

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report


def visualize_images(x_dataset: np.ndarray, y_dataset: np.ndarray, class_names: list[str], no_images: int = 25):
    # Visualizing some images from the training dataset
    plt.figure(figsize=[10, 10])
    for i in range(no_images):  # for first 25 images
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_dataset[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_dataset[i][0]])

    plt.show()


def visualize_plots(history, model_path: str):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    plt.savefig(model_path + ".png")

    plt.show()


def visualize_wrong_predictions(model, model_path, x_dataset, y_dataset, dataset_name: str, labels,
                                no_images: int = 25):
    prediction = model.predict(x_dataset)

    y_prediction = np.argmax(prediction, axis=1)
    y_true = np.argmax(y_dataset, axis=1)

    loss, accuracy = model.evaluate(x_dataset, y_dataset)
    print(f"{dataset_name} Loss: {loss:.3f}")
    print(f"{dataset_name} Accuracy: {accuracy:.3f}\n")

    print(classification_report(y_true, y_prediction))

    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    axes = axes.ravel()

    miss_pred = np.where(y_prediction != y_true)[0]
    for i in np.arange(no_images):
        axes[i].imshow(x_dataset[miss_pred[i]])
        axes[i].set_title('True: %s \nPredict: %s' % (labels[y_true[miss_pred[i]]], labels[y_prediction[miss_pred[i]]]))
        axes[i].axis('off')
        plt.subplots_adjust(wspace=1)

    plt.savefig(model_path + " WP " + ".png")

    plt.show()


def main():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    # Checking the number of rows (records) and columns (features)
    print("Images shapes:")
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # Checking the number of unique classes
    print("Number of unique classes")
    print(np.unique(y_train))
    print(np.unique(y_test))

    # Creating a list of all the class labels
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    visualize_images(x_train, y_train, class_names)

    # Converting the pixels data to float type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Standardizing (255 is the total number of pixels an image can have)
    x_train = x_train / 255
    x_test = x_test / 255

    # One hot encoding the target class (labels)
    no_classes = 10
    y_train = utils.to_categorical(y_train, no_classes)
    y_test = utils.to_categorical(y_test, no_classes)

    # Creating a sequential model and adding layers to it

    model_type = ModelType.Simple
    no_epochs = 200
    train_batch_size = 256

    model_folder = "Models"
    model_name = str(model_type) + " 2"
    model_path = ("" if model_folder == "" else model_folder + "/") + model_name

    model = get_model(model_type, no_classes)

    # Checking the model summary
    model.summary()

    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    # Data augmentation
    data_generator = ImageDataGenerator(featurewise_center=False,
                                        samplewise_center=False,
                                        featurewise_std_normalization=False,
                                        samplewise_std_normalization=False,
                                        zca_whitening=False,
                                        rotation_range=15,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        horizontal_flip=True,
                                        vertical_flip=False)

    data_generator.fit(x_train)

    callbacks = ModelCheckpoint(filepath=model_path + ".h5",
                                monitor='val_accuracy',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=False,
                                mode='auto',
                                save_freq='epoch')

    history = model.fit(data_generator.flow(x_train, y_train, batch_size=train_batch_size),
                        epochs=no_epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[callbacks])

    visualize_plots(history, model_path)

    best_model = models.load_model(model_path + ".h5")
    visualize_wrong_predictions(best_model, model_path, x_test, y_test, "Test Dataset", class_names)


if __name__ == '__main__':
    main()
