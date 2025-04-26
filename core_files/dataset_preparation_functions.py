import pandas as pd
import numpy as np
import tensorflow as tf
import librosa

from noise_mixer import white_noise_adder

# fetching targeted human voice files
def data_fetcher(path_to_ids_csv):
    dataset = pd.read_csv(path_to_ids_csv)
    ids = dataset['id']
    return ids

def mel_spectogram_converter(y):
    mel_spec = librosa.feature.melspectrogram(y=y,sr=22050,n_mels=256,n_fft=2048,hop_length=512)
    return mel_spec

def soundfile_loader(id,base_path_to_soundfiles):
    try:
        filepath = f'{base_path_to_soundfiles}/{id}.wav'
        y , sr = librosa.load(filepath , sr = 22050)
        return y
    except Exception as e:
        print(e)
        return 0
    
def dataset_maker(speech_ids_path, base_path_to_soundfiles):
    clean_mel_specs = []
    white_noise_mel_specs = []
    ids = data_fetcher(speech_ids_path)
    for id in ids:
        y = soundfile_loader(id,base_path_to_soundfiles)
        mel_spec = mel_spectogram_converter(y)
        mel_spec = np.asarray(mel_spec).astype(np.float32)
        clean_mel_specs.append(mel_spec)
        # adding noise to the audio
        noise_added_audio = white_noise_adder(y)
        mel_spec = mel_spectogram_converter(noise_added_audio)
        mel_spec = np.asarray(mel_spec).astype(np.float32)
        white_noise_mel_specs.append(mel_spec)
    return clean_mel_specs , white_noise_mel_specs