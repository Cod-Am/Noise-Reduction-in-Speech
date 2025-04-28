from libs import tf,pd,np,librosa

from noise_mixer import white_noise_adder

# fetching targeted human voice files
def data_fetcher(path_to_ids_csv):
    dataset = pd.read_csv(path_to_ids_csv)
    ids = dataset['id']
    return ids

def mel_spectogram_converter(windowed_audio):
    mel_specs = []
    for y in windowed_audio:
        mel_spec = librosa.feature.melspectrogram(y=y,sr=22050,n_mels=128,n_fft=2048,hop_length=512)
        mel_specs.append(mel_spec)
    return mel_specs

def soundfile_loader(id,base_path_to_soundfiles):
    try:
        filepath = f'{base_path_to_soundfiles}/{id}.wav'
        y , sr = librosa.load(filepath , sr = 22050)
        windowed_audio = window_creator(y)
        return windowed_audio
    except Exception as e:
        print(e)
        return 0

def window_creator(y,duration = 5,sr=22050):
    windowed_audio = []
    number_of_windows = y // (duration * sr)
    if y % (duration * sr) != 0:
        number_of_windows += 1
        padding = number_of_windows * sr - y.shape[0]
        y = np.pad(y, (0, padding), 'constant', constant_values=(0, 0))
    for i in range(0,number_of_windows):
        window = y[i * sr : (i + 1) * sr]
        windowed_audio.append(window)
    return windowed_audio

def dataset_maker(speech_ids_path, base_path_to_soundfiles):
    clean_mel_specs = []
    white_noise_mel_specs = []
    ids = data_fetcher(speech_ids_path)
    for id in ids:
        windowed_audio = soundfile_loader(id,base_path_to_soundfiles)
        mel_specs = mel_spectogram_converter(windowed_audio)
        mel_specs = [np.asarray(mel_spec).astype(np.float16) for mel_spec in mel_specs]
        for mel_spec in mel_specs:
            clean_mel_specs.append(mel_spec)
        
        # adding noise to the audio
        noise_added_audio = white_noise_adder(windowed_audio)
        mel_specs = mel_spectogram_converter(noise_added_audio)
        mel_specs = [np.asarray(mel_spec).astype(np.float16) for mel_spec in mel_specs]
        for mel_spec in mel_specs:
            clean_mel_specs.append(mel_spec)
    return clean_mel_specs , white_noise_mel_specs