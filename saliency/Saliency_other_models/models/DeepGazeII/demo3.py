#%matplotlib inline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
sns.set_style('white')

import tensorflow as tf

# our example image for which we will calculate DeepGaze predictions

from scipy import misc
img = misc.imread('default.png')

plt.imshow(img)
plt.axis('off');

image_data = img[np.newaxis, :, :, :]  # BHWC, three channels (RGB)
centerbias_data = np.zeros((1, img.shape[0], img.shape[1], 1))  # BHWC, 1 channel (log density)

tf.reset_default_graph()

new_saver = tf.train.import_meta_graph('ICF.ckpt.meta')

input_tensor = tf.get_collection('input_tensor')[0]
centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
log_density = tf.get_collection('log_density')[0]
log_density_wo_centerbias = tf.get_collection('log_density_wo_centerbias')[0]

with tf.Session() as sess:
    
    new_saver.restore(sess, 'ICF.ckpt')
    
    log_density_prediction = sess.run(log_density, {
        input_tensor: image_data,
        centerbias_tensor: centerbias_data,
    })

print(log_density_prediction.shape)

plt.gca().imshow(img, alpha=0.2)
m = plt.gca().matshow((log_density_prediction[0, :, :, 0]), alpha=0.5, cmap=plt.cm.RdBu)
plt.colorbar(m)
plt.title('log density prediction')
plt.axis('off');

plt.gca().imshow(img, alpha=0.2)
m = plt.gca().matshow(np.exp(log_density_prediction[0, :, :, 0]), alpha=0.5, cmap=plt.cm.RdBu)
plt.colorbar(m)
plt.title('density prediction')
plt.axis('off');
misc.imsave('saliency_default.png', log_density_prediction[0, :, :, 0])


