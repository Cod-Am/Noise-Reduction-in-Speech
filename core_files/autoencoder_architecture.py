from libs import tf
class AutoEncoder:
    # creating an autoencoder architecture for denoising audio
    # encoder architecture
    def __init__(self):
        self.encoder = tf.keras.Sequential(
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
                tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.Conv2D(filters=1 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu')
            ]
        )
        self.autoencoder = tf.keras.Sequential([self.encoder,self.decoder])
        self.autoencoder.compile(optimizer='adam',loss='mse')

    def train(self,xtrain,xtest,ytrain,ytest,epochs=10,batch_size=32):
        history = self.autoencoder.fit(xtrain,ytrain,batch_size=batch_size,epochs=epochs,validation_data=(xtest,ytest))