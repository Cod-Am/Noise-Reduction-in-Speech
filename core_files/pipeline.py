from dataset_preparation_functions import dataset_maker
from pre_training_processor import pre_training_processor
from autoencoder_architecture import AutoEncoder

base_path_to_soundfiles = '../../dataset/FSD50K.dev_audio'
speech_ids_path = '../datasets/train_dataset_targeted_ids_fsd50k.csv'

clean_mel_specs , white_noise_mel_specs = dataset_maker(speech_ids_path , base_path_to_soundfiles)
clean_mel_specs_train , clean_mel_specs_test = pre_training_processor(clean_mel_specs)
white_noise_mel_specs_train , white_noise_mel_specs_test = pre_training_processor(white_noise_mel_specs)

autoencoder_model = AutoEncoder()
autoencoder_model.train()