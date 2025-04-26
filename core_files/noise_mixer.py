import numpy as np

def white_noise_adder(y, amplitude=5, noise_sigma=0.1):
    noise = amplitude * np.random.normal(0, noise_sigma, size=y.shape)
    return y + noise
