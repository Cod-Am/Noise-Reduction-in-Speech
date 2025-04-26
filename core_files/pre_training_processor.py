import numpy as np
import tensorflow as tf

def formater(mel_specs):
    resized_mel_specs = [np.asarray(mel_spec).astype(np.float32) for mel_spec in mel_specs]
    resized_mel_specs = tf.convert_to_tensor(resized_mel_specs)
    resized_mel_specs = tf.expand_dims(input = resized_mel_specs,axis = -1)
    return resized_mel_specs

def splitter(mel_specs):
    train_size = 0.8
    # finding out the range of indexes for train dataset
    train_index = mel_specs.shape[0] * train_size
    train_set = mel_specs[:train_index,:,:]
    test_set = mel_specs[train_index:,:,:]
    return train_set,test_set

def pre_training_processor(mel_specs):
    train_set , test_set = splitter(mel_specs)
    train_set = formater(train_set)
    test_set = formater(test_set)
    return train_set,test_set