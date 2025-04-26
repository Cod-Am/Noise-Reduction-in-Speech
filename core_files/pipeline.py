from dataset_preparation_functions import dataset_maker

base_path_to_soundfiles = '../../dataset/FSD50K.dev_audio'
speech_ids_path = '../datasets/train_dataset_targeted_ids_fsd50k.csv'

clean_mel_specs , white_noise_mel_specs = dataset_maker(speech_ids_path , base_path_to_soundfiles)
