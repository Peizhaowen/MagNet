from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import os
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte, color
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from imutils import paths
from keras.preprocessing.image import img_to_array, load_img
from imutils import paths
import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
import random
import keras.backend as K
import matplotlib.pylab as plt
from keras.optimizers import Adam
import sys


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    model.summary()
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model


def train_Generator(batch_size, train_path, image_folder, mask_folder, aug_dict):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_color_mode = "grayscale"
    mask_color_mode = "grayscale"
    image_save_prefix = "image"
    mask_save_prefix = "mask"
    # save_to_dir = 'data_unet/training_set/aug'
    save_to_dir = None
    target_size = (256, 256)
    seed = 1
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        yield img, mask


def load_validation_data(img_path, label_path):
    data = []
    label = []
    image_paths = sorted(list(paths.list_images(img_path)))
    random.seed(0)
    random.shuffle(image_paths)
    for each_path in image_paths:
        norm_size = 256
        image = load_img(each_path, target_size=(norm_size, norm_size), color_mode='grayscale')
        image = img_to_array(image)
        data.append(image)
        maker = each_path.split(os.path.sep)
        img_label_path = label_path+'/'+maker[-1]
        img_label = load_img(img_label_path, target_size=(norm_size, norm_size), color_mode='grayscale')
        img_label = img_to_array(img_label)
        label.append(img_label)
    data = np.array(data, dtype="float") / 255.0
    label = np.array(label, dtype="float") / 255.0
    label[label > 0.5] = 1
    label[label <= 0.5] = 0
    return data, label


def test_Generator(test_path, num_image, target_size, flag_multi_class=False, as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def save_Result(save_path, npyfile, name):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        io.imsave(os.path.join(save_path, "{}_{}.png" .format(i, name)), img_as_ubyte(img))


def train(model, batch_size, epochs, steps):
    model_checkpoint = ModelCheckpoint('trained_model/unet_({},{},{}).hdf5'.format(batch_size, epochs, steps),
                                       monitor='loss', verbose=1, save_best_only=True)
    aug = {'rotation_range': 40, 'width_shift_range': 0.2,
           'height_shift_range': 0.2, 'shear_range': 0.2,
           'zoom_range': 0.2, 'horizontal_flip': True, 'fill_mode': 'nearest'}
    train_gen = train_Generator(4, 'data_unet/training_set', 'resize_new2', 'label_new2', aug)
    val_x, val_y = load_validation_data("data_unet/validation_set/resize_new2",
                                          "data_unet/validation_set/label_new2")
    _history = model.fit_generator(train_gen,
                                   validation_data=(val_x, val_y),
                                   steps_per_epoch=steps,
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
    csv_name = 'trained_model/result_unet_({},{},{}).csv'\
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
    plt.savefig("trained_model/result_unet_({},{},{}).png"
                .format(batch_size, epochs, steps))
    # plt.show()


def unet_predicted(path, num):
    testGene = test_Generator(path, num, target_size=(256, 256))
    model = unet(input_size=(256, 256, 1))
    model.load_weights("trained_model/unet_magnetosome_new_(200,300,4).hdf5")
    results = model.predict_generator(testGene, num, verbose=1)
    save_Result(path, results, "unet")


def test_dataset(R):
    print('******')
    print('predict edges using unet', ' ***R: ', R)
    train_path = 'testing_set/{}R_unet_train'.format(R)
    filelist = os.listdir(train_path)
    for item in filelist:
        sub_filelist = os.listdir(os.path.abspath(train_path) + '/' + item)
        file_num = len(sub_filelist)
        unet_predicted(os.path.abspath(train_path) + '/' + item, file_num)


if __name__ == "__main__":

    model = unet()
    # train(model, batch_size, epochs, steps)
    train(model, 4, 200, 300)