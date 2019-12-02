# %%
import re

import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

PATCH_SIZE = 140
SHUFFLE_BUFFER_SIZE = 100
TEST_SIZE = 200


def load_image_data(overlap=False):
    def decode_img(img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32, True)
        img = tf.expand_dims(img, 0)
        strides = [1, PATCH_SIZE, PATCH_SIZE, 1] if not overlap else [1, PATCH_SIZE // 2, PATCH_SIZE // 2, 1]
        img = tf.image.extract_patches(images=img,
                                       sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
                                       strides=strides,
                                       rates=[1, 1, 1, 1],
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

    paths = tf.data.TextLineDataset('datasets/SIDD_Small_sRGB_Only/SIDD_Small_sRGB_Only/Scene_Instances.txt')
    # for f in paths.take(5):
    #     print(f.numpy())

    x = paths.map(process_path)
    return x


def image_to_patches(image, overlap=False):
    image = tf.expand_dims(image, 0)
    strides = [1, PATCH_SIZE, PATCH_SIZE, 1] if not overlap else [1, PATCH_SIZE // 2, PATCH_SIZE // 2, 1]
    image_patches = tf.image.extract_patches(images=image,
                                             sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
                                             strides=strides,
                                             rates=[1, 1, 1, 1],
                                             padding='SAME')
    image_patches = tf.reshape(image_patches, [-1, PATCH_SIZE, PATCH_SIZE, 3])
    return image_patches


def use_predict_on_patches(image_patches, model):
    predicted_patches = []
    length = len(image_patches)

    for i in range(length):
        predicted_patches.append(model.predict(np.expand_dims(image_patches[i], 0)))
        print('Prediction done (%d of %d) ' % (i + 1, length) + str(image_patches[i].shape))
    return predicted_patches


def patches_to_image(patches, height, width, overlap=False):
    ps = PATCH_SIZE
    num_of_patches_vertical = math.ceil(height / PATCH_SIZE)
    num_of_patches_horizontal = math.ceil(width / PATCH_SIZE)

    print('Height is ' + str(height))
    print('Width is ' + str(width))

    print('Vertical number of patches is: ' + str(num_of_patches_vertical))
    print('Horizontal number of patches is: ' + str(num_of_patches_horizontal))

    pad = [[0, 0], [0, 0]]

    if overlap:
        patches = tf.reshape(patches,
                            [num_of_patches_vertical * 2 - 1, num_of_patches_horizontal * 2 - 1, PATCH_SIZE,
                            PATCH_SIZE, 3])

        print(patches.shape)
        patches = tf.reshape(patches, [-1, PATCH_SIZE, PATCH_SIZE, 3])
        print(patches.shape)


        patches = tf.slice(patches, begin=[0, PATCH_SIZE // 4, PATCH_SIZE // 4, 0],
                           size=[-1, PATCH_SIZE // 2, PATCH_SIZE // 2, -1])
        num_of_patches_vertical = num_of_patches_vertical * 2 - 1
        num_of_patches_horizontal = num_of_patches_horizontal * 2 - 1
        print(patches.shape)
        # patches = im[::2, ::2]
        ps = PATCH_SIZE // 2

    reconstructed_patches = tf.reshape(patches,
                                       [1, num_of_patches_vertical, num_of_patches_horizontal, ps * ps,
                                        3])
    reconstructed_patches = tf.split(reconstructed_patches, ps * ps, 3)
    reconstructed_patches = tf.stack(reconstructed_patches, axis=0)
    reconstructed_patches = tf.reshape(reconstructed_patches,
                                       [ps * ps, num_of_patches_vertical, num_of_patches_horizontal, 3])

    result = tf.batch_to_space(reconstructed_patches, [ps, ps], pad)
    print(result.shape)
    return result


def psnr(im1, im2):
    return tf.image.psnr(im1, im2, max_val=1.0)


def calculate_ssim(image1, image2):
    #image1 = image1.numpy()
    #image2 = image2.numpy()
    score, diff = structural_similarity(image1, image2, full=True, multichannel=True)
    return score, diff


def visualize_img(img, str):
    plt.figure()
    plt.imshow(img)
    plt.title(str)
    plt.show()
