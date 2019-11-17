# %%
import re

import math
import numpy as np
import tensorflow as tf

PATCH_SIZE = 140
SHUFFLE_BUFFER_SIZE = 100
TEST_SIZE = 200


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
    # num_of_patches_vertical = height // PATCH_SIZE
    # num_of_patches_horizontal = width // PATCH_SIZE

    reconstructed_patches = tf.reshape(patches,
                                       [1, num_of_patches_vertical, num_of_patches_horizontal, PATCH_SIZE * PATCH_SIZE,
                                        3])
    reconstructed_patches = tf.split(reconstructed_patches, PATCH_SIZE * PATCH_SIZE, 3)
    reconstructed_patches = tf.stack(reconstructed_patches, axis=0)
    reconstructed_patches = tf.reshape(reconstructed_patches,
                                       [PATCH_SIZE * PATCH_SIZE, num_of_patches_vertical, num_of_patches_horizontal, 3])

    result = tf.batch_to_space(reconstructed_patches, [PATCH_SIZE, PATCH_SIZE], pad)
    return result
