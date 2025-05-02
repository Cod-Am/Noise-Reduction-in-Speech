import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pipeline import start_autoencoder_training_pipeline, start_asr_training

print("Welcome to training module: ")
training_choice = input("What do you want to train: \n1. Autoencoder (will train the autoencoder model for denoising audio files.) \n2. Automatic Speech Recognition model.")
if training_choice == 1:
    start_autoencoder_training_pipeline()
elif training_choice == 2:
    start_asr_training()