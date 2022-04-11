# encoding: utf-8
from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from imutils import paths
import cv2
import numpy as np
import random
import os
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Sequential
import keras.backend as K
import matplotlib.pylab as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


def load_data(path, class_num):
    data = []
    label = []
    image_paths = sorted(list(paths.list_images(path)))
    random.seed(0)
    random.shuffle(image_paths)
    for each_path in image_paths:
        image = cv2.imread(each_path)
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        data.append(image)
        maker = int(each_path.split(os.path.sep)[-2])
        label.append(maker)
    data = np.array(data, dtype="float") / 255.0
    label = np.array(label)
    # one-hot
    label = to_categorical(label, num_classes=class_num)
    return data, label


class Lenet:
    def neural(channel, height, width, classes):
        input_shape = (channel, height, width)
        if K.image_data_format() == "channels_last":
            input_shape = (height, width, channel)
        model = Sequential()
        model.add(Conv2D(20, (5, 5), padding="same", activation="relu", input_shape=input_shape, name="conv1"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool1"))
        model.add(Conv2D(50, (5, 5), padding="same", activation="relu", name="conv2", ))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool2"))
        model.add(Flatten())
        model.add(Dense(500, activation="relu", name="fc1"))
        model.add(Dense(classes, activation="softmax", name="fc2"))
        model.summary()
        return model


def train(aug, model, train_x, train_y, test_x, test_y, batch_size, epochs, steps):
    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam", metrics=["accuracy"])
    model_checkpoint = ModelCheckpoint('trained_model/LeNet_({0},{1},{2})_64×64.hdf5'.format(batch_size, epochs, steps),
                                       monitor='loss', verbose=1, save_best_only=True)
    _history = model.fit_generator(aug.flow(train_x, train_y, batch_size=batch_size),
                                   validation_data=(test_x, test_y),
                                   steps_per_epoch=300,
                                   callbacks=[model_checkpoint],
                                   epochs=epochs, verbose=1)
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    df: DataFrame = pd.DataFrame(columns=["train_loss"])
    for i in range(N):
        df.loc[i, 'train_loss'] = _history.history["loss"][i]
        df.loc[i, 'val_loss'] = _history.history["val_loss"][i]
        df.loc[i, 'train_acc'] = _history.history["acc"][i]
        df.loc[i, 'val_acc'] = _history.history["val_acc"][i]
    csv_name = 'trained_model/result_LeNet_({},{},{})_64×64.csv'\
        .format(batch_size, epochs, steps)
    df.to_csv(csv_name, index_label="index")
    plt.plot(np.arange(0, N), _history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), _history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), _history.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), _history.history["val_acc"], label="val_acc")
    plt.title("loss and accuracy")
    plt.xlabel("epoch")
    plt.ylabel("loss/acc")
    plt.legend(loc="best")
    plt.savefig("trained_model/result.png")
    plt.show()


if __name__ == "__main__":
    channel = 3
    height = 64
    width = 64
    class_num = 4
    norm_size = 64
    batch_size = 64
    epochs = 200
    steps = 300
    model = Lenet.neural(channel=channel, height=height,
                         width=width, classes=class_num)
    train_x, train_y = load_data("data_lenet/training_set", class_num)
    val_x, val_y = load_data("data_lenet/validation_set", class_num)
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    train(aug, model, train_x, train_y, val_x, val_y, batch_size, epochs, steps)
