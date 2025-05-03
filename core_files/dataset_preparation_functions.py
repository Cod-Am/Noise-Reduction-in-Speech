from libs import tf,pd,np,librosa,json,time,datetime

from noise_mixer import white_noise_adder

# functions for training autoencoder 

# fetching targeted human voice files
def data_fetcher(path_to_ids_csv):
    dataset = pd.read_csv(path_to_ids_csv)
    ids = dataset['id']
    return ids

def mel_spectogram_converter(windowed_audio ,y=[] , mode = 0):
    if mode == 0:
        mel_specs = []
        for y in windowed_audio:
            mel_spec = librosa.feature.melspectrogram(y=y,sr=22050,n_mels=128,n_fft=2048,hop_length=512)
            mel_specs.append(mel_spec)
        return mel_specs
    if mode == 1:
        mel_spec = librosa.feature.melspectrogram(y=y,sr=22050,n_mels=128,n_fft=2048,hop_length=512)
        return mel_spec

def window_creator(y, duration=5, sr=22050):
    windowed_audio = []
    samples_per_window = duration * sr
    number_of_windows = y.shape[0] // samples_per_window
    if y.shape[0] % samples_per_window != 0:
        number_of_windows += 1
        padding = number_of_windows * samples_per_window - y.shape[0]
        if padding > 0:
            y = np.pad(y, (0, padding), 'constant', constant_values=(0, 0))
    for i in range(0, number_of_windows):
        window = y[i * samples_per_window : (i + 1) * samples_per_window]
        windowed_audio.append(window)
    
    return windowed_audio


def soundfile_loader(id , path , mode = 1):
    if mode == 1:
        try:
            filepath = f'{path}/{id}.wav'
            y , sr = librosa.load(filepath , sr = 22050)
            windowed_audio = window_creator(y)
            return windowed_audio
        except Exception as e:
            print(e)
            return []
    if mode == 0:
        try:
            y,sr = librosa.load(path,sr=22050)
            return y
        except Exception as e:
            print(e)
            return []

def dataset_maker_for_autoencoder(speech_ids_path, base_path_to_soundfiles):
    clean_mel_specs = []
    white_noise_mel_specs = []
    ids = data_fetcher(speech_ids_path)
    for id in ids:
        # reading the original files and fetch the windowed audio
        windowed_audio = soundfile_loader(id,base_path_to_soundfiles)
        # print(windowed_audio)

        mel_specs = mel_spectogram_converter(windowed_audio)
        mel_specs = [np.asarray(mel_spec).astype(np.float32) for mel_spec in mel_specs]
        for mel_spec in mel_specs:
            clean_mel_specs.append(mel_spec)
        
        # adding noise to the audio
        noise_added_audio = white_noise_adder(windowed_audio)
        mel_specs = mel_spectogram_converter(noise_added_audio)
        mel_specs = [np.asarray(mel_spec).astype(np.float32) for mel_spec in mel_specs]
        for mel_spec in mel_specs:
            white_noise_mel_specs.append(mel_spec)
    return clean_mel_specs , white_noise_mel_specs

# functions for training asr system

def dataset_maker_for_asr(path_to_transcrptions,path_to_soundfiles):
    targeted_timestamps_paths = transcriptions_path_generator(path_to_transcrptions)
    transcriptions = transcriptions_loader(targeted_timestamps_paths) # load the relavant transcriptions into a list from json files
    processed_tanscriptions = preprocess_transcriptions(transcriptions) #does the preprocessing of transcriptions for training
    timestamps = timestamp_fetcher(targeted_timestamps_paths) # we need to fetch the timestamps to samples frames 
    mel_specs = audio_framing_and_processing(timestamps)
    dataset = tf.data.Dataset.from_tensor_slices((processed_tanscriptions,mel_specs)) # compiling the data into a dataset
    return dataset

def transcriptions_path_generator(base_path):
    paths = []
    return paths

def transcriptions_loader(paths):
    transcriptions_list = []
    for path in paths:
        with open(path,"r") as file:
            data = json.load(file)
        transcription = data['words']
        transcriptions_list.append(transcription)
    return transcriptions_list

def preprocess_transcriptions(transcriptions):
    processed_transcriptions = []
    return processed_transcriptions

def timestamp_fetcher(path):
    start_time_timestamps = []
    end_time_timestamps = []
    with open(path,"r") as file:
        # loading data from json file
        data = json.load(file)
    for item in data:
        start_time = item['start_time']
        end_time = item['end_time']
        start_time_timestamps.append(start_time)
        end_time_timestamps.append(end_time)
    dataset = pd.DataFrame({'start_time':start_time_timestamps,'end_time':end_time_timestamps})
    return dataset

def audio_framing_and_processing(timestamps,path):
    mel_specs = []
    y = soundfile_loader(path = path,mode = 0)
    for start_time , end_time in zip(timestamps['start_time'],timestamps['end_time']):
        # since our sample rate is 22050 throughout all our audios
        start_time_sample = time_to_sample_converter(start_time)
        end_time_sample = time_to_sample_converter(end_time)
        window = y[start_time_sample:end_time_sample + 1]
        mel_spec = mel_spectogram_converter(y = window)
        mel_specs.append(mel_spec)
    return mel_specs

def time_to_sample_converter(timestamp , sr = 22050):
    x = time.strptime(timestamp.split('.')[0],'%H:%M:%S')
    time_in_seconds = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    sample = sr * time_in_seconds
    return sample