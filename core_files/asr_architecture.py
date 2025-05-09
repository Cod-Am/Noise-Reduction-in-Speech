import tensorflow as tf
from libs import os,plt

class ResidualCNNBlock(tf.keras.layers.Layer):
    def __init__(self,batch_size=16,n_layers = 1,**kwargs):
        super().__init__(**kwargs)
        self.hidden = [CNN_for_Residual_Block() for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return Z + inputs

class CNN_for_Residual_Block(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # cnn architecture
        self.down_sampling_cnn = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                # tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
                tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                # tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
                tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                # tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
                tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu')
            ]
        )

        self.up_sampling_cnn = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                # tf.keras.layers.UpSampling2D(size=(2,2)),
                tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                # tf.keras.layers.UpSampling2D(size=(2,2)),
                tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                # tf.keras.layers.UpSampling2D(size=(2,2)),
                tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu'),
                tf.keras.layers.Conv2D(filters=1 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu')
            ]
        )

    def call(self , inputs):
        Z = inputs
        Z = self.down_sampling_cnn(Z)
        Z = self.up_sampling_cnn(Z)
        return Z

class CNN_Block(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # cnn architecture
        self.down_sampling_cnn = tf.keras.Sequential(
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

        self.up_sampling_cnn = tf.keras.Sequential(
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
                tf.keras.layers.Conv2D(filters=128 , kernel_size=(2,2) , strides=1 , padding='same' , activation='relu')
            ]
        )

    def call(self , inputs):
        Z = inputs
        Z = self.down_sampling_cnn(Z)
        Z = self.up_sampling_cnn(Z)
        return Z

class RNN_Block(tf.keras.layers.Layer):
    def __init__(self,dense_units,**kwargs):
        super().__init__(**kwargs)
        self.biderectional_rnn = tf.keras.Sequential(
            [
                tf.keras.layers.Bidirectional(layer=tf.keras.layers.LSTM(128,return_sequences=True)),
                tf.keras.layers.Bidirectional(layer=tf.keras.layers.LSTM(64,return_sequences=True)),
                tf.keras.layers.Dense(256,activation = 'relu'),
                tf.keras.layers.Dense(512,activation = 'relu'),
                tf.keras.layers.Dense(1024,activation = 'relu'),
                tf.keras.layers.Dense(dense_units)
            ]
        )

    def call(self,inputs):
        Z = inputs
        Z = self.biderectional_rnn(Z)
        return Z

class ASR(tf.keras.Model):
    def __init__(self,vocab_size,**kwargs):
        super().__init__(**kwargs)
        self.cnn = CNN_Block()
        self.residual_block = ResidualCNNBlock()
        self.rnn = RNN_Block(vocab_size)

    def call(self,inputs):
        # inputs = tf.expand_dims(inputs,axis=-1)
        Z = inputs

        for _ in range(2):
            Z = self.residual_block(Z)
        
        print(tf.shape(Z)) # (batch,h,w,c)
        
        # reshaping cnn output for rnn block
        batch_size = tf.shape(Z)[0]
        resolution = tf.shape(Z)[-1] * tf.shape(Z)[1]  # channel * height
        width = tf.shape(Z)[2]
        Z = tf.reshape(Z, [batch_size , width , resolution])

        print(tf.shape(Z))
        
        Z = self.rnn(Z)
        return Z
    
class Model:
    def __init__(self , vocab_size):
        # training the model
        self.model = ASR(vocab_size)
        self.model.compile(optimizer = 'adam' , loss = tf.losses.SparseCategoricalCrossentropy())

    def split(self,dataset):
        train_dataset,test_dataset = tf.keras.utils.split_dataset(dataset,left_size=0.8)
        return train_dataset,test_dataset

    def train(self,dataset,batch_size = 16,epochs = 10):
        train_dataset,test_dataset = self.split(dataset)
        train_dataset = train_dataset.batch(batch_size)
        history = self.model.fit(train_dataset, epochs = epochs,validation_data=test_dataset)


    def save_model(self,path = './'):
        self.model.save(f'{path}/asr_model.keras')

    def plot_performance(self,history):
        if not os.path.exists('./model_performance_graphs'):
            os.makedirs('./model_performance_graphs')
        # plotting mse
        plt.plot(history.history['loss'], label='MSE (training data)')
        plt.plot(history.history['val_loss'], label='MSE (validation data)')
        plt.title('MSE for Model Denoising Performance')
        plt.ylabel('MSE value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.savefig('./model_performance_graphs/mse_graph_denoising_autoencoder.png',bbox_inches = 'tight')
        plt.show()
        plt.close()
