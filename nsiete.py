# %%
import pathlib

import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import re
import tensorflow as tf


# %%
# from tensorflow_core.python.data.ops.dataset_ops import Dataset


def load_image_data(scenes=None, img_limit=None):
    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return img

    def process_path(file_path):
        # load the raw data from the file as a string
        noisy = tf.io.read_file(path + file_path + '/NOISY_SRGB_010.PNG')
        gt = tf.io.read_file(path + file_path + '/GT_SRGB_010.PNG')
        noisy = decode_img(noisy)
        gt = decode_img(gt)
        return noisy, gt

    path = 'datasets/SIDD_Small_sRGB_Only/SIDD_Small_sRGB_Only/Data/'
    f = open('datasets/SIDD_Small_sRGB_Only/SIDD_Small_sRGB_Only/Scene_Instances.txt')
    scene_instances = f.readlines()
    paths = None
    if scenes is not None:
        regex = ''
        for scene in scenes:
            regex = regex + '_%03d_|' % scene
        regex = regex[:-1]
        scene_instances = [i for i in scene_instances if re.search(regex, i)]
        paths = tf.data.Dataset.from_tensors(scene_instances)

    paths = tf.data.TextLineDataset('datasets/SIDD_Small_sRGB_Only/SIDD_Small_sRGB_Only/Scene_Instances.txt')
    for f in paths.take(5):
        print(f.numpy())

    x = paths.map(process_path)
    return x


x = load_image_data(scenes=[1], img_limit=4)
x = x.batch(batch_size=1)
x = x.repeat()
# %%

# x_train, x_test, y_train, y_test = train_test_split(x, y)
# %%

from models import DnCNN, dcnn_loss

model = DnCNN(depth=17)
model.compile(optimizer=keras.optimizers.Adam(), loss=dcnn_loss)
model.fit(x, steps_per_epoch=1)
model.summary()

# %%
