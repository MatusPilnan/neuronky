# %%
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import re


# %%
def load_image_data(scenes=None, img_limit=None):
    path = 'datasets/SIDD_Small_sRGB_Only/SIDD_Small_sRGB_Only/Data/'
    f = open('datasets/SIDD_Small_sRGB_Only/SIDD_Small_sRGB_Only/Scene_Instances.txt')
    scene_instances = f.readlines()
    if scenes is not None:
        regex = ''
        for scene in scenes:
            regex = regex + '_%03d_|' % scene
        regex = regex[:-1]
        scene_instances = [i for i in scene_instances if re.search(regex, i)]

    if img_limit is not None:
        scene_instances = scene_instances[:img_limit]

    noisy = []
    gt = []
    for scene in scene_instances:
        print(scene)
        noisy.append(mpimg.imread(path + scene[:-1] + '/NOISY_SRGB_010.PNG'))
        gt.append(mpimg.imread(path + scene[:-1] + '/GT_SRGB_010.PNG'))

    x = np.array(noisy)
    y = np.array(gt)

    return x, y


x, y = load_image_data(scenes=[1], img_limit=4)
# %%

x_train, x_test, y_train, y_test = train_test_split(x, y)
# %%

from models import DnCNN, dcnn_loss

model = DnCNN(depth=17)
model.compile(optimizer=keras.optimizers.Adam(), loss=dcnn_loss)
model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test))
model.summary()

# %%
