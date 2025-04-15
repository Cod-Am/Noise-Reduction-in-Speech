# Project:
**Robust Speech Recognition in Noisy Environments:** Develop a noise-resistant speech recognition system that can accurately transcribe speech even in noisy environments (e.g., factories, streets, crowded areas)
# Approaches:
* Preprocessing audio to remove noise via classical methods.
* Use a pretrained model and tune it.
# Resources:
## Dataset:
* FSD50K dataset - https://zenodo.org/records/4060432
* CHiME - Home datset - https://archive.org/details/chime-home
* NOIZEUS dataset - https://ecs.utdallas.edu/loizou/speech/noizeus/
* CHiME6 dataset - 
* Github repo - https://github.com/jim-schwoebel/voice_datasets
## Study Resources:
* Librosa quick tutorial for audio preprocessing - [link to resource](https://medium.com/@rijuldahiya/a-comprehensive-guide-to-audio-processing-with-librosa-in-python-a49276387a4b)
* Youtube playlist - [Playlist](https://www.youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0) by Valerio Velardo
* Article - https://medium.com/analytics-vidhya/noise-suppression-using-deep-learning-6ead8c8a1839
## Research Papers: 
* Key Research - [Research Paper](https://www.isca-archive.org/interspeech_2021/kashyap21_interspeech.pdf) [Github Repo](https://github.com/madhavmk/Noise2Noise-audio_denoising_without_clean_training_data)
* [Reinforcement Learning To Adapt Speech Enhancement to Instantaneous Input Signal Quality](https://arxiv.org/pdf/1711.10791)
* [Speech Background Noise Removal Using Different Linear Filtering Techniques](https://www.researchgate.net/publication/325622133_Speech_Background_Noise_Removal_Using_Different_Linear_Filtering_Techniques)
* [Spectral Subtractive-Type Algorithms for Enhancement of Noisy Speech: An Integrative Review](https://www.mecs-press.org/ijigsp/ijigsp-v5-n11/IJIGSP-V5-N11-2.pdf)
## Discovered tools:
* AI powered tool - [Resemble Enhance](https://huggingface.co/ResembleAI/resemble-enhance)
## Libraries used: 
* pandas
* import numpy as np
* matplotlib
* librosa
* sklearn
* tensorflow
