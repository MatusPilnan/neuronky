#%%
import os
from datetime import datetime
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow.keras as keras

from models import DnCNN, dcnn_loss
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
PATCH_SIZE = 140
SHUFFLE_BUFFER_SIZE = 100
TEST_SIZE = 200


#%%
x = load_image_data()

#%%
x = x.unbatch()

x = x.batch(batch_size=10)
shuffled_data = x.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
test = shuffled_data.take(TEST_SIZE).repeat()
train = shuffled_data.skip(TEST_SIZE).repeat()
#%%

model = DnCNN(depth=17)
model.compile(optimizer=keras.optimizers.Adam(), loss=dcnn_loss, metrics=[psnr])

now = datetime.now()
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='logs\log_from_{}'.format(now.strftime("%Y-%m-%d_at_%H-%M-%S")),
    histogram_freq=1)

model.fit(x=train, steps_per_epoch=2000, validation_data=test, epochs=5, validation_steps=10, callbacks=[tensorboard_callback])
model.summary()

#%%
now = datetime.now()
model.save('models\model_made_on_{}'.format(now.strftime("%Y-%m-%d_at_%H-%M-%S")))

#%%
loaded = keras.models.load_model('models/model_made_on_2019-11-16_at_02-03-01', compile=False)
loaded.compile(optimizer=keras.optimizers.Adam(), loss=dcnn_loss, metrics=["accuracy"])
image = mpimg.imread('datasets/SIDD_Small_sRGB_Only/SIDD_Small_sRGB_Only/Data/0118_006_N6_00100_00025_3200_L/NOISY_SRGB_010.PNG')

image_patches = image_to_patches(image)

reconstructed_patches = use_predict_on_patches(image_patches, loaded)

image_patches_overlap = image_to_patches(image, overlap=True)

reconstructed_patches_overlap = use_predict_on_patches(image_patches_overlap, loaded)

#%%

joined_image = patches_to_image(image_patches, image.shape[0], image.shape[1])

reconstructed_image = patches_to_image(reconstructed_patches, image.shape[0], image.shape[1])

joined_image_overlap = patches_to_image(image_patches_overlap, image.shape[0], image.shape[1], overlap=True)

reconstructed_image_overlap = patches_to_image(reconstructed_patches_overlap, image.shape[0], image.shape[1],
                                               overlap=True)

plt.figure()
plt.imshow(joined_image[0])
plt.show()

plt.figure()
plt.imshow(reconstructed_image[0])
plt.show()

plt.figure()
plt.imshow(joined_image_overlap[0])
plt.show()

plt.figure()
plt.imshow(reconstructed_image_overlap[0])
plt.show()
