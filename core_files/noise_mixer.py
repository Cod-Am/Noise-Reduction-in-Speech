from libs import tf,np

def white_noise_adder(windowed_audio, amplitude=5, noise_sigma=0.1):
    noise_mixed_audios = []
    for y in windowed_audio:
        noise = amplitude * np.random.normal(0, noise_sigma, size=y.shape)
        y = y + noise
        noise_mixed_audios.append(y)
    return noise_mixed_audios