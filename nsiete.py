#%%
import pathlib

import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import re
import tensorflow as tf

PATCH_SIZE = 180
#%%
# from tensorflow_core.python.data.ops.dataset_ops import Dataset
from tensorflow_core.python.ops.gen_logging_ops import timestamp


def load_image_data(scenes=None, img_limit=None):
    def decode_img(img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.expand_dims(img, 0)
        img = tf.image.extract_patches(img, [1,PATCH_SIZE, PATCH_SIZE, 1], [1,PATCH_SIZE,PATCH_SIZE,1], [1,1,1,1], padding='SAME')
        img = tf.reshape(img, [-1, PATCH_SIZE, PATCH_SIZE, 3])
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


x = load_image_data()
x = x.unbatch()
for img in x.take(5):
    print(str(img[0].shape))

    plt.figure()
    plt.imshow(img[0])
    plt.show()
    plt.figure()
    plt.imshow(img[1])
    plt.show()
x = x.batch(batch_size=25)
x = x.repeat()
#%%

from models import DnCNN, dcnn_loss

model = DnCNN(depth=17)
model.compile(optimizer=keras.optimizers.Adam(), loss=dcnn_loss, metrics=['accuracy'])
model.fit(x, steps_per_epoch=30, epochs=5)
model.summary()

#%%
model.save('models/%s' % timestamp())
#%%
loaded = keras.models.load_model('models/model1', compile=False)
loaded.compile(optimizer=keras.optimizers.Adam(), loss=dcnn_loss, metrics=['accuracy'])
img = mpimg.imread('datasets/SIDD_Small_sRGB_Only/SIDD_Small_sRGB_Only/Data/0118_006_N6_00100_00025_3200_L/NOISY_SRGB_010.PNG')

img = np.expand_dims(img, 0)
img = tf.image.extract_patches(img, [1,PATCH_SIZE, PATCH_SIZE, 1], [1,PATCH_SIZE,PATCH_SIZE,1], [1,1,1,1], padding='SAME')
img = tf.reshape(img, [-1, PATCH_SIZE, PATCH_SIZE, 3])
a = img
plt.figure()
plt.imshow(img[50])
plt.show()
img = loaded.predict(np.expand_dims(img[50], 0))
print('Prediction done ' + str(img.shape))
#img = np.squeeze(img)
print(tf.image.psnr(img, a[50], 1))
# mpimg.imsave('output\model1\o.png', img)
plt.figure()
plt.imshow(img[0])
plt.show()

