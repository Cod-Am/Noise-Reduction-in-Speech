from libs import tf,np

def formater(mel_specs):
    formatted_mel_specs = [np.asarray(mel_spec).astype(np.float16) for mel_spec in mel_specs]
    padded_mel_specs = padding_sequence(formatted_mel_specs)
    resized_mel_specs = tf.convert_to_tensor(padded_mel_specs)
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
    mel_specs = formater(mel_specs)
    train_set , test_set = splitter(mel_specs)
    return train_set,test_set

def padding_sequence(mel_specs):
    padded_mel_specs = []
    shapes_array = [mel_spec.shape[1] for mel_spec in mel_specs]
    max_limit = max(shapes_array)
    for mel_spec in mel_specs:
        padding = max_limit - mel_spec.shape[1]
        padded_mel_spec =  np.pad(mel_spec , [(0,0),(0,padding)] , mode='constant')
        padded_mel_specs.append(padded_mel_spec)
    return padded_mel_specs