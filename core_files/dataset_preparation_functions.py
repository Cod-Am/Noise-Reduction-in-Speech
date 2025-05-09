import phonemizer.separator
import sklearn.preprocessing
from libs import pd,np,librosa,json,time,datetime,string,os
# import phonemizer
# from phonemizer.backend.espeak.wrapper import EspeakWrapper

# EspeakWrapper.set_library('C:\Program Files\eSpeak NG\libespeak-ng.dll')
import sklearn
import nltk
import joblib
import tensorflow as tf
from noise_mixer import white_noise_adder

# functions for training autoencoder 

# fetching targeted human voice files
def data_fetcher(path_to_ids_csv):
    dataset = pd.read_csv(path_to_ids_csv)
    ids = dataset['id']
    return ids

def mel_spectogram_converter(windowed_audio = None ,y=[] , mode = 0):
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


def soundfile_loader(id = None , path = None , mode = 1):
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

def dataset_maker_for_asr(base_path_to_transcrptions,base_path_to_soundfiles):
    print("generating paths: ")
    targeted_transcriptions_paths = transcriptions_path_generator(base_path_to_transcrptions) # loads the paths to transcriptions
    print("Complete")
    print("Paths to transcriptions:\n",targeted_transcriptions_paths)
    print("loading transcriptions")
    transcriptions = transcriptions_loader(targeted_transcriptions_paths) # load the relavant transcriptions into a list from json files
    print("complete")
    print("processing transcriptions")
    processed_tanscriptions,vocab_length = preprocess_transcriptions(transcriptions) #does the preprocessing of transcriptions for training
    print("complete")
    print("generating mel specs")
    mel_specs = timestamp_audio_file_synchronizer(targeted_transcriptions_paths , base_path_to_soundfiles)
    # print(mel_specs)
    print("complete")
    # debugging and checking for discrepency in the transcriptions and mel specs dataset
    debugging_and_correction(processed_tanscriptions,mel_specs) 
    
    # convert mel specs to tensor and make a batch of 4 dimensions
    mel_specs = tf.convert_to_tensor(mel_specs)
    mel_specs = tf.expand_dims(mel_specs , axis = -1)

    dataset = tf.data.Dataset.from_tensor_slices((mel_specs,processed_tanscriptions)) # compiling the data into a dataset
    return dataset,vocab_length

def debugging_and_correction(transcriptions,mel_specs):
    print(len(transcriptions),len(mel_specs))
    transcriptions_shapes = [text.shape for text in transcriptions]
    mel_spec_shapes = [mel_spec.shape for mel_spec in mel_specs]
    print(list(set(transcriptions_shapes)))
    print(list(set(mel_spec_shapes)))

def transcriptions_path_generator(base_path):
    paths = []
    # targted_filenames = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S21']
    targted_filenames = ['S01']
    for filename in targted_filenames:
        path = f'{base_path}/{filename}.json'
        paths.append(path)
    return paths

def transcriptions_loader(paths):
    transcriptions_list = []
    for path in paths:
        with open(path,"r") as file:
            data = json.load(file)
        for instances in data:
            transcription = instances['words']
            transcriptions_list.append(transcription)
    return transcriptions_list

def preprocess_transcriptions(transcriptions):
    clean_transcriptions = transcription_cleaning(transcriptions)
    processed_transcriptions,vocab_length = trancriptions_to_vectors_converter(clean_transcriptions)
    return processed_transcriptions,vocab_length

def transcription_cleaning(transcriptions):
    clean_transcriptions = []
    for transcription in transcriptions:
        transcription = transcription.lower()
        replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        transcription = transcription.translate(replace_punctuation)
        transcription = remove_closed_captions(transcription) # removed closed transcriptions
        clean_transcriptions.append(transcription)
    return clean_transcriptions

def trancriptions_to_vectors_converter(transcriptions):
    # initializing a label_encoder to covert the phonemized text to numbers 
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoded_text = []

    # creating the phonemized text
    # separator = phonemizer.separator.Separator(word=' / ',phone=' ')
    # phonemised_transcriptions = phonemizer.phonemize(transcriptions,language='en-us',backend='espeak',separator=separator)
    
    # tokenising the text and flattening it to a list
    tokenised_text = [token for sentence in transcriptions for token in nltk.word_tokenize(sentence)]
    
    # fitting the label encoder on the list
    label_encoder.fit(tokenised_text)
    vocab_length = len(list(label_encoder.classes_))
    # saving label encoder fro prediction
    joblib.dump(label_encoder,'./label_encoder.pkl')
    for sentence in transcriptions:
        tokenised_sentence = nltk.word_tokenize(sentence)
        vector = label_encoder.transform(tokenised_sentence)
        label_encoded_text.append(vector)
    padded_sequence = padding_label_encoded_sequence(label_encoded_text)
    return padded_sequence,vocab_length

def padding_label_encoded_sequence(le_text):
    shapes = [len(text) for text in le_text]
    max_shape = max(shapes)
    padded_le_texts = []
    for text in le_text:
        padding = max_shape - len(text)
        # padded_text = text + [-1] * padding
        padded_text = np.pad(text, (0, padding), 'constant', constant_values=(0, 0))
        padded_le_texts.append(padded_text)
    return padded_le_texts

def remove_closed_captions(transcription):
    stack = []
    if '[' in transcription:
            starting_index = transcription.index('[')
            ending_index = transcription.index(']')
            for i in range(starting_index,ending_index+1):
                stack.append(transcription[i])
    stack = ''.join(stack)
    transcription = transcription.replace(stack,'')
    return transcription

def timestamp_audio_file_synchronizer(targeted_transcriptions_paths , base_path_to_soundfiles):
    mel_spec_collection = []
    soundfile_paths = soundfile_path_generation(base_path_to_soundfiles)
    for transcription_path,soundfile_path in zip(targeted_transcriptions_paths,soundfile_paths):
        timestamp_dataset = timestamp_fetcher(transcription_path)
        mel_specs = audio_framing_and_processing(timestamp_dataset,soundfile_path)
        mel_spec_collection.extend(mel_specs)
    return mel_spec_collection


def soundfile_path_generation(base_path_to_soundfiles):
    paths = []
    # targted_filenames = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S21']
    targted_filenames = ['S01']
    for filename in targted_filenames:
        path = f'{base_path_to_soundfiles}/{filename}_U01.CH1.wav'
        paths.append(path)
    return paths

def timestamp_fetcher(path):
    start_time_timestamps = []
    end_time_timestamps = []
    with open(path,"r") as file:
        print(path)
        # loading data from json file
        data = json.load(file)
        for item in data:
            start_time = item['start_time']
            end_time = item['end_time']
            # print("Start: ",start_time)
            # print("End: ",end_time)
            start_time_timestamps.append(start_time)
            end_time_timestamps.append(end_time)
    dataset = pd.DataFrame({'start_time':start_time_timestamps,'end_time':end_time_timestamps})
    return dataset

def audio_framing_and_processing(timestamps,path):
    mel_specs = []
    windows = []
    y = soundfile_loader(path = path,mode = 0)
    for start_time , end_time in zip(timestamps['start_time'],timestamps['end_time']):
        # since our sample rate is 22050 throughout all our audios
        start_time_sample = time_to_sample_converter(start_time)
        end_time_sample = time_to_sample_converter(end_time)
        window = y[start_time_sample:end_time_sample + 1]
        windows.append(window)
    padded_signals = signal_padding(windows)
    for padded_signal in padded_signals:
        mel_spec = mel_spectogram_converter(y = padded_signal,mode=1)
        mel_specs.append(mel_spec)
    return mel_specs

def signal_padding(signals):
    padded_signals = []
    # print("Signals: ",signals)
    # print(type(signals))
    # print("1st Signal: ",signals[0])
    # print(type(signals[0]))
    # print(signals[0].shape)
    # gathing all shapes so as to we can find max shape and pad others accordingly
    shape_collection = [signal.shape[0] for signal in signals]
    max_shape = max(shape_collection)
    
    for signal in signals:
        padding = max_shape - signal.shape[0] + 5
        padded_signal = np.pad(signal, (0, padding))
        padded_signals.append(padded_signal)
    return padded_signals

def time_to_sample_converter(timestamp , sr = 22050):
    x = time.strptime(timestamp.split('.')[0],'%H:%M:%S')
    time_in_seconds = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    sample = int(sr * time_in_seconds)
    return sample