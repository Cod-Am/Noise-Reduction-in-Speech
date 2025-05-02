from libs import tf,plt
import os

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
        # bottleneck architecture

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
                tf.keras.layers.Conv2D(filters=1 , kernel_size=(2,2) , strides=1 , padding='same' , activation='sigmoid')
                # tf.keras.layers.GlobalAveragePooling2D(keepdims = True, data_format="channels_last")
            ]
        )
        self.autoencoder = tf.keras.Sequential([self.encoder,self.decoder])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.autoencoder.compile(optimizer=self.optimizer,loss='mse',metrics=['mean_absolute_error'],)

    def train(self,xtrain,xtest,ytrain,ytest,epochs=20,batch_size=32):
        train_dataset,test_dataset = self.dataset_building_and_shuffling(xtrain,xtest,ytrain,ytest)
        train_dataset = train_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        history = self.autoencoder.fit(train_dataset,epochs=epochs,validation_data=(test_dataset))
        self.plot_performance(history)

    def normalise_tensors(self,xtrain,ytrain,xtest,ytest):
        # xtrain_normalised = []
        # ytrain_normalised = []
        # xtest_normalised = []
        # ytest_normalised = []
        # for xtrain_tensor,ytrain_tensor,xtest_tensor,ytest_tensor in zip(xtrain,ytrain,xtest,ytest):
        #     # capturing max:
        #     xtrain_max = tf.reduce_max(xtrain_tensor)
        #     ytrain_max = tf.reduce_max(ytrain_tensor)
        #     xtest_max = tf.reduce_max(xtest_tensor)
        #     ytest_max = tf.reduce_max(ytest_tensor)

        #     # capturing min:
        #     xtrain_min = tf.reduce_min(xtrain_tensor)
        #     ytrain_min = tf.reduce_min(ytrain_tensor)
        #     xtest_min = tf.reduce_min(xtest_tensor)
        #     ytest_min = tf.reduce_min(ytest_tensor)

        #     # normalising the tensors
        #     xtrain_tensor = (xtrain_tensor - xtrain_min) / (xtrain_max - xtrain_min) if xtrain_max!=xtrain_min else (xtrain_tensor / 100)
        #     ytrain_tensor = (ytrain_tensor - ytrain_min) / (ytrain_max - ytrain_min) if ytrain_max!=ytrain_min else (ytrain_tensor / 100)
        #     xtest_tensor = (xtest_tensor - xtest_min) / (xtest_max - xtest_min) if xtest_max!=xtest_min else (xtest_tensor / 100)
        #     ytest_tensor = (ytest_tensor - ytest_min) / (ytest_max - ytest_min) if ytest_max!=ytest_min else (ytest_tensor / 100)
            
        #     # appending the tensors in new list
        #     xtrain_normalised.append(xtrain_tensor)
        #     ytrain_normalised.append(ytrain_tensor)
        #     xtest_normalised.append(xtest_tensor)
        #     ytest_normalised.append(ytest_tensor)

        xtrain_min = tf.reduce_min(xtrain)
        xtrain_max = tf.reduce_max(xtrain)
        ytrain_min = tf.reduce_min(ytrain)
        ytrain_max = tf.reduce_max(ytrain)
        
        xtrain_normalised = (xtrain - xtrain_min) / (xtrain_max - xtrain_min + 1e-8)
        ytrain_normalised = (ytrain - ytrain_min) / (ytrain_max - ytrain_min + 1e-8)
        xtest_normalised = (xtest - xtrain_min) / (xtrain_max - xtrain_min + 1e-8)
        ytest_normalised = (ytest - ytrain_min) / (ytrain_max - ytrain_min + 1e-8)

        return xtrain_normalised,ytrain_normalised,xtest_normalised,ytest_normalised

    def dataset_building_and_shuffling(self,xtrain,xtest,ytrain,ytest):
            xtrain,ytrain,xtest,ytest = self.normalise_tensors(xtrain,ytrain,xtest,ytest)
            # building tensorflow dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((xtrain,ytrain))
            test_dataset = tf.data.Dataset.from_tensor_slices((xtest,ytest))
            # shuffling dataset
            train_dataset = train_dataset.shuffle(buffer_size=len(xtrain),seed=42)
            test_dataset = test_dataset.shuffle(buffer_size=len(xtest),seed=42)
            return train_dataset,test_dataset

    def save_model(self,path):
        self.autoencoder.save(path)

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
        # plotting mae
        plt.plot(history.history['mean_absolute_error'], label='MAE (training data)')
        plt.plot(history.history['val_mean_absolute_error'], label='MAE (validation data)')
        plt.title('MAE for Model Denoising Performance')
        plt.ylabel('MAE value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.savefig('./model_performance_graphs/mae_graph_denoising_autoencoder.png',bbox_inches = 'tight')
        plt.show()
        plt.close()