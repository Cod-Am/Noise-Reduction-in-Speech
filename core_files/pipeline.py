from dataset_preparation_functions import dataset_maker_for_autoencoder,dataset_maker_for_asr
from pre_training_processor import pre_training_processor
from autoencoder_architecture import AutoEncoder
from asr_architecture import Model

from libs import tf,pd,np,librosa,plt

def start_autoencoder_training_pipeline():
    base_path_to_soundfiles = '../../dataset/FSD50K.dev_audio'
    speech_ids_path = '../datasets/train_dataset_targeted_ids_fsd50k.csv'

    clean_mel_specs , white_noise_mel_specs = dataset_maker_for_autoencoder(speech_ids_path , base_path_to_soundfiles)
    clean_mel_specs_train , clean_mel_specs_test = pre_training_processor(clean_mel_specs)
    white_noise_mel_specs_train , white_noise_mel_specs_test = pre_training_processor(white_noise_mel_specs)

    autoencoder_model = AutoEncoder()
    autoencoder_model.train(white_noise_mel_specs_train,white_noise_mel_specs_test,clean_mel_specs_train,clean_mel_specs_test)
    autoencoder_model.save_model('./autoencoder.keras')

def start_asr_training():
    base_path_to_soundfiles = '../../dataset/Chime-6' #also contains the transcriptions folder
    transcriptions_folder = f'{base_path_to_soundfiles}/transcriptions/'
    dataset,vocab_length = dataset_maker_for_asr(transcriptions_folder,base_path_to_soundfiles)
    model = Model(vocab_size=vocab_length)
    model.train(dataset)
    model.save()
