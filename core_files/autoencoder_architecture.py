import tensorflow as tf
import keras
from project.files.core_files.dataset_preparation_functions import data_fetcher

class AutoEncoder:
    # creating an autoencoder architecture for denoising audio
    # encoder architecture
    def __init__(self):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=1),
                tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=1),
                tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=1),
                tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu')
            ]
        )

        # decoder architecture
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.UpSampling2D(size=(2,2)),
                tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.UpSampling2D(size=(2,2)),
                tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.UpSampling2D(size=(2,2)),
                tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu')
            ]
        )
        self.autoencoder = tf.keras.Sequential([self.encoder,self.decoder])
        self.autoencoder.compile(optimizer='adam',loss='mse')

    def train(self):
        