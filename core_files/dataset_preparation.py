import pandas as pd
import numpy as np
import tensorflow as tf
import librosa

# fetching targeted human voice files
def data_fetcher(path_to_ids_csv):
    dataset = pd.read_csv(path_to_ids_csv)
    ids = dataset['id']
    return ids

def mel_spectogram_converter(y):
    mel_spec = librosa.feature.melspectrogram(y)
    return mel_spec