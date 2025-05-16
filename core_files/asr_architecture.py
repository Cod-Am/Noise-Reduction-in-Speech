import tensorflow as tf
from libs import os,plt

class ResidualCNNBlock(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.hidden = CNN_for_Residual_Block()

    def call(self, inputs):
        Z = inputs
        Z = self.hidden(Z)
        return Z + inputs

class CNN_for_Residual_Block(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.conv2d_1_list = [tf.keras.layers.Conv2D(filters=1 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(1)]
        self.conv2d_16_list_1 = [tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]
        self.conv2d_32_list_1 = [tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]
        self.conv2d_64_list_1 = [tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]
        self.conv2d_128_list_1 = [tf.keras.layers.Conv2D(filters=128 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]
        self.conv2d_16_list_2 = [tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]
        self.conv2d_32_list_2 = [tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]
        self.conv2d_64_list_2 = [tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]
        self.conv2d_128_list_2 = [tf.keras.layers.Conv2D(filters=128 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]

    def call(self , inputs):
        Z = inputs
        for layer in self.conv2d_16_list_1:
            Z = layer(Z)
        for layer in self.conv2d_32_list_1:
            Z = layer(Z)
        for layer in self.conv2d_64_list_1:
            Z = layer(Z)
        for layer in self.conv2d_128_list_1:
            Z = layer(Z)
        for layer in self.conv2d_128_list_2:
            Z = layer(Z)
        for layer in self.conv2d_64_list_2:
            Z = layer(Z)
        for layer in self.conv2d_32_list_2:
            Z = layer(Z)
        for layer in self.conv2d_16_list_2:
            Z = layer(Z)
        for layer in self.conv2d_1_list:
            Z = layer(Z)
        return Z

class CNN_Block(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # cnn architecture
        
        
        self.conv2d_1_list = [tf.keras.layers.Conv2D(filters=1 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(1)]
        self.conv2d_16_list_1 = [tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]
        self.conv2d_32_list_1 = [tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]
        self.conv2d_64_list_1 = [tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]
        self.conv2d_128_list_1 = [tf.keras.layers.Conv2D(filters=128 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]
        self.conv2d_16_list_2 = [tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]
        self.conv2d_32_list_2 = [tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]
        self.conv2d_64_list_2 = [tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]
        self.conv2d_128_list_2 = [tf.keras.layers.Conv2D(filters=128 , kernel_size=(2,2) , strides=(1,1),padding='same',activation='relu') for _ in range(2)]

    def call(self , inputs):
        Z = inputs
        
        for layer in self.conv2d_16_list_1:
            Z = layer(Z)

        for layer in self.conv2d_32_list_1:
            Z = layer(Z)
        
        for layer in self.conv2d_64_list_1:
            Z = layer(Z)

        for layer in self.conv2d_128_list_1:
            Z = layer(Z)
        
        for layer in self.conv2d_128_list_2:
            Z = layer(Z)
        
        for layer in self.conv2d_64_list_2:
            Z = layer(Z)
        
        for layer in self.conv2d_32_list_2:
            Z = layer(Z)
        
        for layer in self.conv2d_16_list_2:
            Z = layer(Z)
        
        for layer in self.conv2d_1_list:
            Z = layer(Z)
        
        return Z

class RNN_Block(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.dense_1024_1 = tf.keras.layers.Dense(1024,activation='relu')
        self.dense_1024_2 = tf.keras.layers.Dense(1024,activation='relu')
        self.dense_512 = tf.keras.layers.Dense(512,activation='relu')
        self.dense_256 = tf.keras.layers.Dense(256,activation='relu')
        # self.dense_vocab = tf.keras.layers.Dense(vocab_length,activation='relu')
        self.bi_lstm_128 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128 , return_sequences=True))
        self.bi_lstm_64 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64 , return_sequences=True))

    def call(self,inputs):
        Z = inputs
        Z = self.dense_1024_1(Z)
        Z = self.dense_1024_2(Z)
        Z = self.dense_256(Z)
        Z = self.dense_512(Z)
        Z = self.bi_lstm_128(Z)
        Z = self.bi_lstm_64(Z)
        # Z = self.dense_vocab(Z)
        return Z

class ASR(tf.keras.Model):
    def __init__(self,vocab_size,**kwargs):
        super().__init__(**kwargs)

        self.cnn = CNN_Block()
        self.residual_block = ResidualCNNBlock()
        self.rnn = RNN_Block()
        self.out = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs):
        Z = inputs
        Z = self.cnn(Z)
        Z = self.residual_block(Z)
        # tf.print("After CNN and Residual:", tf.shape(Z))
        
        # rehsaping the tensors.
        Z = tf.transpose(Z , perm=[0,2,1,3])
        batch_size = tf.shape(Z)[0]
        resolution = tf.shape(Z)[-1] * tf.shape(Z)[2]
        width = tf.shape(Z)[1]
        Z = tf.reshape(Z, [batch_size, width, resolution])
        # tf.print("After Reshape:", tf.shape(Z))
        
        Z = self.rnn(Z)
        # tf.print("After RNN:", tf.shape(Z))
    
        Z = self.out(Z)
        # tf.print("After Output Layer:", tf.shape(Z))
        return Z
    
class ASRTrainer:
    def __init__(self,vocab_size,batch_size = 4):
        # training the model
        self.model = ASR(vocab_size = vocab_size)
        self.model.compile(optimizer = 'adam' , loss = CTCLoss(vocab_size=vocab_size))

    def split(self,dataset):
        train_dataset,test_dataset = tf.keras.utils.split_dataset(dataset,left_size=0.8)
        return train_dataset,test_dataset

    def train(self,dataset,batch_size = 4,epochs = 3):
        train_dataset,test_dataset = self.split(dataset)
        train_dataset = train_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        history = self.model.fit(train_dataset, epochs = epochs,validation_data=test_dataset,batch_size=batch_size)
        self.plot_performance(history)


    def save_model(self,path = './'):
        self.model.save(f'{path}/asr_model.keras')

    def plot_performance(self,history):
        if not os.path.exists('./asr_model_performance_graphs'):
            os.makedirs('./asr_model_performance_graphs')
        # plotting mse
        plt.plot(history.history['loss'], label='CTC (training data)')
        plt.plot(history.history['val_loss'], label='CTC (validation data)')
        plt.title('CTC Loss for ASR System Performance')
        plt.ylabel('CTC value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.savefig('./asr_model_performance_graphs/ctc_graph_asr_system.png',bbox_inches = 'tight')
        plt.show()
        plt.close()

class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, 
                 vocab_size,
                 blank_index=None, 
                 pad_id=None,
                 from_logits=True, 
                 reduction=tf.keras.losses.Reduction.AUTO, 
                 name='ctc_loss'):
        """
        vocab_size: total number of label classes (including blank and pad)
        pad_id: the integer you use to pad y_true (e.g. vocab_size-1)
        blank_index: if None, default to vocab_size-2 (just before pad_id)
        """
        super().__init__(reduction=reduction, name=name)
        self.vocab_size   = vocab_size
        self.pad_id       = pad_id if pad_id is not None else vocab_size - 1
        self.blank_index  = (blank_index 
                             if blank_index is not None 
                             else vocab_size - 2)
        self.from_logits  = from_logits

    def call(self, y_true, y_pred):
        """
        y_true: int32 tensor, shape [batch, max_label_len], padded with pad_id
        y_pred: float32 tensor, shape [batch, time, num_classes]
        """
        # 1) Mask out pad tokens from label-length calculation
        is_not_pad = tf.not_equal(y_true, self.pad_id)  # shape [B, L]
        label_lengths = tf.reduce_sum(tf.cast(is_not_pad, tf.int32), axis=1)

        # 2) Prepare y_true_clean for CTC (no negative values!)
        #    Replace pad_id with 0 (or any valid label)—they’ll be ignored by blank/mask.
        y_true_clean = tf.where(is_not_pad, y_true, tf.fill(tf.shape(y_true), self.blank_index))

        # 3) Input lengths = full time dimension after your CNN/RNN
        batch_size = tf.shape(y_pred)[0]
        time_steps = tf.shape(y_pred)[1]
        input_lengths = tf.fill([batch_size], time_steps)

        # 4) Prepare logits for tf.nn.ctc_loss
        logits = y_pred if self.from_logits else tf.math.log(y_pred + 1e-8)
        # CTC wants [max_time, batch, num_classes]
        logits = tf.transpose(logits, [1, 0, 2])

        # 5) Compute CTC loss
        loss = tf.nn.ctc_loss(
            labels=y_true_clean,
            logits=logits,
            label_length=label_lengths,
            logit_length=input_lengths,
            blank_index=self.blank_index,
            logits_time_major=True
        )  # shape: [batch]

        # 6) Return mean across the batch
        return tf.reduce_mean(loss)

# ==== Usage ====
# model = ...  # your tf.keras.Model predicting (batch, time, num_classes)
# model.compile(optimizer='adam', loss=CTCLoss())
