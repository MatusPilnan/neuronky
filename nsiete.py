#%%
import os
import re
from datetime import datetime

import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from models import DnCNN, dcnn_loss

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
PATCH_SIZE = 140
SHUFFLE_BUFFER_SIZE = 100
TEST_SIZE = 200

#%%
# from tensorflow_core.python.data.ops.dataset_ops import Dataset


def load_image_data(scenes=None, img_limit=None):
    def decode_img(img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32, True)
        img = tf.expand_dims(img, 0)
        img = tf.image.extract_patches(img,
                                       [1, PATCH_SIZE, PATCH_SIZE, 1],
                                       [1, PATCH_SIZE, PATCH_SIZE, 1],
                                       [1, 1, 1, 1],
                                       padding='VALID')
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
    # for f in paths.take(5):
    #     print(f.numpy())

    x = paths.map(process_path)
    return x


def image_to_patches(image):
    image = tf.expand_dims(image, 0)
    image_patches = tf.image.extract_patches(image,
                                       [1, PATCH_SIZE, PATCH_SIZE, 1],
                                       [1, PATCH_SIZE, PATCH_SIZE, 1],
                                       [1, 1, 1, 1],
                                       padding='SAME')
    image_patches = tf.reshape(image_patches, [-1, PATCH_SIZE, PATCH_SIZE, 3])
    return image_patches


def use_predict_on_patches(image_patches, model):
    predicted_patches = []

    for i in range(len(image_patches)):
        predicted_patches.append(model.predict(np.expand_dims(image_patches[i], 0)))
        print('Prediction done ' + str(image_patches[i].shape))
    return predicted_patches


def patches_to_image(patches, height, width):
    num_of_patches_vertical = math.ceil(height / PATCH_SIZE)
    num_of_patches_horizontal = math.ceil(width / PATCH_SIZE)

    print('Height is ' + str(height))
    print('Width is ' + str(width))

    print('Vertical number of patches is: ' + str(num_of_patches_vertical))
    print('Horizontal number of patches is: ' + str(num_of_patches_horizontal))

    pad = [[0, 0], [0, 0]]
    #num_of_patches_vertical = height // PATCH_SIZE
    #num_of_patches_horizontal = width // PATCH_SIZE

    reconstructed_patches = tf.reshape(patches, [1, num_of_patches_vertical, num_of_patches_horizontal, PATCH_SIZE * PATCH_SIZE, 3])
    reconstructed_patches = tf.split(reconstructed_patches, PATCH_SIZE * PATCH_SIZE, 3)
    reconstructed_patches = tf.stack(reconstructed_patches, axis=0)
    reconstructed_patches = tf.reshape(reconstructed_patches, [PATCH_SIZE * PATCH_SIZE, num_of_patches_vertical, num_of_patches_horizontal, 3])

    result = tf.batch_to_space(reconstructed_patches, [PATCH_SIZE, PATCH_SIZE], pad)
    return result


#%%
x = load_image_data()

#%%
x = x.unbatch()
# for img in x:
#     print(str(img[0].shape))
#
#     plt.figure()
#     plt.imshow(img[0])
#     plt.show()
#     plt.figure()
#     plt.imshow(img[1])
#     plt.show()
x = x.batch(batch_size=10)
# x = x.repeat()
shuffled_data = x.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
test = shuffled_data.take(TEST_SIZE).repeat()
train = shuffled_data.skip(TEST_SIZE).repeat()
#%%

model = DnCNN(depth=17)
model.compile(optimizer=keras.optimizers.Adam(), loss=dcnn_loss, metrics=['accuracy'])

now = datetime.now()
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='logs\log_from_{}'.format(now.strftime("%Y-%m-%d_at_%H-%M-%S")),
    histogram_freq=1)

model.fit(x=train, steps_per_epoch=2000, validation_data=test, epochs=5, validation_steps=10, callbacks=[tensorboard_callback])
#validation_split=0.2,
model.summary()

#%%
now = datetime.now()
#print(now.strftime("%Y/%m/%d_at_%H:%M:%S"))
model.save('models\model_made_on_{}'.format(now.strftime("%Y-%m-%d_at_%H-%M-%S")))

#%%
loaded = keras.models.load_model('models/model_made_on_2019-11-16_at_02-03-01', compile=False)
loaded.compile(optimizer=keras.optimizers.Adam(), loss=dcnn_loss, metrics=['accuracy'])
image = mpimg.imread('datasets/SIDD_Small_sRGB_Only/SIDD_Small_sRGB_Only/Data/0118_006_N6_00100_00025_3200_L/NOISY_SRGB_010.PNG')

image_patches = image_to_patches(image)

#mpimg.imsave('output\model1\e.png', img[50])

reconstructed_patches = use_predict_on_patches(image_patches, loaded)

#%%
joined_image = patches_to_image(image_patches, image.shape[0], image.shape[1])
reconstructed_image = patches_to_image(reconstructed_patches, image.shape[0], image.shape[1])

plt.figure()
plt.imshow(joined_image[0])
plt.show()

plt.figure()
plt.imshow(reconstructed_image[0])
plt.show()