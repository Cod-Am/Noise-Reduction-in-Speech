import tensorflow as tf

class ResidualCNNBlock:
    def __init__(self,batch_size=16):
        self.hidden = [cnn = CNN_Block(input)]

    def call(self, inputs):
        pass

class CNN_Down_Sampling_Block:
    def __init__(self):
        # cnn architecture
        down_sampling_cnn = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
                tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
                tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
                tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu')
            ]
        )

        up_sampling_cnn = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
                tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
                tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
                tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu')
            ]
        )

class RNN_Block:
    def __init__(self):
        pass

class ASR:
    def __init__(self,output_length):
        model = tf.keras.Sequential(
            [
                CNN_Block(),
                ResidualCNNBlock()
            ]
        )