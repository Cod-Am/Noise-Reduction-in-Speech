import tensorflow as tf
import keras
from dataset_preparation import data_fetcher

# creating an autoencoder architecture for denoising audio
# encoder architecture
encoder = tf.keras.Sequential(
    {
        tf.keras.layers.Conv2D()
    }
)
# decoder architecture
decoder = tf.keras.Sequential(
    {

    }
)

autoencoder = tf.keras.Sequential([encoder,decoder])