import tensorflow as tf
import keras
from project.files.core_files.dataset_preparation_functions import data_fetcher

# creating an autoencoder architecture for denoising audio
# encoder architecture
encoder = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(filters=128 , kernel_size=(2,2) , strides=1 , padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=1)
    ]
)
# decoder architecture
decoder = tf.keras.Sequential(
    [

    ]
)

autoencoder = tf.keras.Sequential([encoder,decoder])