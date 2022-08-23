import argparse
import numpy as np
import librosa
import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras import layers
from collections import defaultdict


def hopify(in_signal, sr, hop_size_seconds, window_size_seconds):
    """ Will hopify based on hop size and window size. For instance if a audio clip is
    8 seconds and the hop and window size is 2 and 3 respectively, then the start and end pos
    of each hop will be [(0, 3), (2, 5), (4, 7), (6, 8)*] *with a pad.

    Will also normalize each segment.
    """
    # Convert to sample rate data points
    hop_length = hop_size_seconds * sr
    window_length = window_size_seconds * sr
    start_pos = 0
    hops = []
    while True:
        end_pos = start_pos + window_length
        # Pad and break when necessary
        if end_pos > in_signal.shape[0]:
            pad = np.zeros(end_pos - in_signal.shape[0])
            hops.append(librosa.util.normalize(np.concatenate((in_signal[start_pos:], pad))))
            break
        hops.append(librosa.util.normalize(in_signal[start_pos:end_pos]))
        start_pos += hop_length
    return np.array(hops)

def random_hopify(in_signal, sr, window_size_seconds, num_hops):
    """ Collects random hops inside the audio signal.
    """
    window_length = window_size_seconds * sr
    max_length = in_signal.shape[0] - window_length - 1
    hops = []
    for _ in range(num_hops):
        rand_start_pos = int(max_length * np.random.random_sample())
        rand_end_pos = rand_start_pos + window_length
        hops.append(librosa.util.normalize(in_signal[rand_start_pos:rand_end_pos]))
    return np.array(hops)

def get_melspectrograms(hops, sr, log=True):
    mels = []
    for row in hops:
        mel = librosa.melspectrogram(y=row, sr=sr)
        if log:
            mel = librosa.core.power_to_db(mels, amin=1e-7)
        mels.append(mels)
    return np.array(rows)

def transform_sample(audio_path, sampling_rate=22050, hop_size_seconds=None, window_size_seconds=10, num_random_hops=None):
    # Load audio using librosa and resample to sampling rate and convert to mono
    in_signal, in_sr = librosa.load(audio_path, sr=sampling_rate, mono=True)
    # Hopify to get audio data in the appropriate length
    if random_hops:
        hops = random_hopify(in_signal, in_sr, window_size_seconds, num_random_hops)
    else:
        hops = hopify(in_signal, in_sr, hop_size_seconds, window_size_seconds)
    # Get a melspectrogram for each hop
    mels = get_melspectrograms(hops, in_sr)

def construct_dataset(audio_paths, sampling_rate, hop_size_seconds=None, window_size_seconds=None, num_random_hops=None, batch_size=128):

