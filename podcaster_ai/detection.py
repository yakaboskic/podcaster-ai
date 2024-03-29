import soundfile as sf
import argparse
import numpy as np
import librosa
import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras import layers
from collections import defaultdict


"""
This function converts the predictions made by the neural network into a readable format.
"""

def preds_to_se(p, audio_clip_length = 8.0):
    start_speech = -100
    start_music = -100
    stop_speech = -100
    stop_music = -100

    audio_events = []

    n_frames = p.shape[0]

    if p[0, 0] == 1:
        start_speech = 0

    if p[0, 1] == 1:
        start_music = 0

    for i in range(n_frames - 1):
        if p[i, 0] == 0 and p[i + 1, 0] == 1:
            start_speech = i + 1

        elif p[i, 0] == 1 and p[i + 1, 0] == 0:
            stop_speech = i
            start_time = frames_to_time(start_speech)
            stop_time = frames_to_time(stop_speech)
            audio_events.append((start_time, stop_time, "speech"))
            start_speech = -100
            stop_speech = -100

        if p[i, 1] == 0 and p[i + 1, 1] == 1:
            start_music = i + 1
        elif p[i, 1] == 1 and p[i + 1, 1] == 0:
            stop_music = i
            start_time = frames_to_time(start_music)
            stop_time = frames_to_time(stop_music)      
            audio_events.append((start_time, stop_time, "music"))
            start_music = -100
            stop_music = -100

    if start_speech != -100:
        start_time = frames_to_time(start_speech)
        stop_time = audio_clip_length
        audio_events.append((start_time, stop_time, "speech"))
        start_speech = -100
        stop_speech = -100

    if start_music != -100:
        start_time = frames_to_time(start_music)
        stop_time = audio_clip_length
        audio_events.append((start_time, stop_time, "music"))
        start_music = -100
        stop_music = -100

    audio_events.sort(key = lambda x: x[0]) 
    return audio_events

""" This function was adapted from https://github.com/qlemaire22/speech-music-detection """

def smooth_output(output, min_speech=1.3, min_music=3.4, max_silence_speech=0.4, max_silence_music=0.6):
    # This function was adapted from https://github.com/qlemaire22/speech-music-detection
    duration_frame = 220 / 22050
    n_frame = output.shape[1]

    start_music = -1000
    start_speech = -1000

    for i in range(n_frame):
        if output[0, i] == 1:
            if i - start_speech > 1:
                if (i - start_speech) * duration_frame <= max_silence_speech:
                    output[0, start_speech:i] = 1

            start_speech = i

        if output[1, i] == 1:
            if i - start_music > 1:
                if (i - start_music) * duration_frame <= max_silence_music:
                    output[1, start_music:i] = 1

            start_music = i

    start_music = -1000
    start_speech = -1000

    for i in range(n_frame):
        if i != n_frame - 1:
            if output[0, i] == 0:
                if i - start_speech > 1:
                    if (i - start_speech) * duration_frame <= min_speech:
                        output[0, start_speech:i] = 0

                start_speech = i

            if output[1, i] == 0:
                if i - start_music > 1:
                    if (i - start_music) * duration_frame <= min_music:
                        output[1, start_music:i] = 0

                start_music = i
        else:
            if i - start_speech > 1:
                if (i - start_speech) * duration_frame <= min_speech:
                    output[0, start_speech:i + 1] = 0

            if i - start_music > 1:
                if (i - start_music) * duration_frame <= min_music:
                    output[1, start_music:i + 1] = 0

    return output


def frames_to_time(f, sr = 22050.0, hop_size = 220):
    return f * hop_size / sr

def get_log_melspectrogram(audio, sr = 22050, hop_length = 220, n_fft = 1024, n_mels = 80, fmin = 64, fmax = 8000):
    """Return the log-scaled Mel bands of an audio signal."""
    bands = librosa.feature.melspectrogram(
            y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=float)
    return librosa.core.power_to_db(bands, amin=1e-7)

"""
Make predictions for full audio.
"""
def mk_preds_fa(audio_path, model, hop_size = 6.0, discard = 1.0, win_length = 8.0, sampling_rate = 22050):
    in_signal, in_sr = sf.read(audio_path)

    # Convert to mono if needed.
    if (in_signal.ndim > 1):
        in_signal_mono = librosa.to_mono(in_signal.T)
        in_signal = np.copy(in_signal_mono)
    # Resample the audio file.
    in_signal_22k = librosa.resample(in_signal, orig_sr=in_sr, target_sr=sampling_rate)
    in_signal = np.copy(in_signal_22k)

    # Pad the input signal if it is shorter than 8 s.

    if in_signal.shape[0] < int(8.0 * sampling_rate):
        pad_signal = np.zeros((int(8.0 * sampling_rate)))
        pad_signal[:in_signal.shape[0]] = in_signal
        in_signal = np.copy(pad_signal)

    audio_clip_length_samples = in_signal.shape[0]
    #print('audio_clip_length_samples is {}'.format(audio_clip_length_samples))

    hop_size_samples = 220 * 602 - 1

    win_length_samples = 220 * 802 - 1

    n_preds = int(math.ceil((audio_clip_length_samples - win_length_samples) / hop_size_samples)) + 1

    in_signal_pad = np.zeros((n_preds * hop_size_samples + 200 * 220))

    in_signal_pad[0:audio_clip_length_samples] = in_signal

    preds = np.zeros((n_preds, 802, 2))

    # Split the predictions into batches of size batch_size. 

    batch_size = 128

    n_batch = n_preds // batch_size

    for i in range(n_batch):
        mss_batch = np.zeros((batch_size, 802, 80))
        for j in range(batch_size):
            seg = in_signal_pad[(i * batch_size + j)* hop_size_samples:((i * batch_size + j) * hop_size_samples) + win_length_samples]
            seg = librosa.util.normalize(seg)
            mss = get_log_melspectrogram(seg)
            M = mss.T
            mss_batch[j, :, :] = M
        preds[i * batch_size:(i + 1) * batch_size, :, :] = (model.predict(mss_batch, verbose=0) >= (0.5, 0.5)).astype(float)

    if n_batch * batch_size < n_preds:
        i = n_batch
    mss_batch = np.zeros((n_preds - n_batch * batch_size, 802, 80))
    for j in range(n_preds - n_batch * batch_size):
        seg = in_signal_pad[(i * batch_size + j)* hop_size_samples:((i * batch_size + j) * hop_size_samples) + win_length_samples]
        seg = librosa.util.normalize(seg)
        mss = get_log_melspectrogram(seg)
        M = mss.T
        mss_batch[j, :, :] = M

    preds[i * batch_size:n_preds, :, :] = (model.predict(mss_batch, verbose=0) >= (0.5, 0.5)).astype(float)

    preds_mid = np.copy(preds[1:-1, 100:702, :])

    preds_mid_2 = preds_mid.reshape(-1, 2)

    if preds.shape[0] > 1:
        oa_preds = preds[0, 0:702, :] # oa stands for overall predictions

    else:
        oa_preds = preds[0, 0:802, :] # oa stands for overall predictions

    oa_preds = np.concatenate((oa_preds, preds_mid_2), axis = 0)

    if preds.shape[0] > 1:
        oa_preds = np.concatenate((oa_preds, preds[-1, 100:, :]), axis = 0)

    return oa_preds


def construct_model():
    mel_input = keras.Input(shape=(802, 80), name="mel_input")
    X = mel_input

    X = tf.keras.layers.Reshape((802, 80, 1))(X)

    X = tf.keras.layers.Conv2D(filters=16, kernel_size=7, strides=1, padding='same')(X)
    X = layers.BatchNormalization(momentum=0.0)(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(X)
    X = tf.keras.layers.Dropout(rate = 0.2)(X)

    X = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=1, padding='same')(X)
    X = layers.BatchNormalization(momentum=0.0)(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(X)
    X = layers.Dropout(rate = 0.2)(X)

    _, _, sx, sy = X.shape
    X = tf.keras.layers.Reshape((-1, int(sx * sy)))(X)

    X = layers.Bidirectional(layers.GRU(80, return_sequences = True))(X)
    X = layers.BatchNormalization(momentum=0.0)(X)

    X = layers.Bidirectional(layers.GRU(80, return_sequences = True))(X)
    X = layers.BatchNormalization(momentum=0.0)(X)

    pred = layers.TimeDistributed(layers.Dense(2, activation='sigmoid'))(X)

    model = keras.Model(inputs = [mel_input], outputs = [pred])

    model.compile(
          optimizer=keras.optimizers.Adam(learning_rate=0.001),
          loss=[keras.losses.BinaryCrossentropy()], metrics=['binary_accuracy'])
    return model


def load_detection_model(path_to_model_weights='data/model_d-DS.h5'):
    model = construct_model()
    model.load_weights(path_to_model_weights)   
    return model

def convert_preds_to_dict(preds):
    res = defaultdict(list)
    for pred in preds:
        start_time, end_time, audio_type = pred
        res[audio_type].append((start_time, end_time))
    return dict(res)

def detect(model, path_to_audio, path_to_results=None):
    # If mp3, convert to wav via a temp file
    if path_to_audio.endswith('.mp3'):
        import tempfile
        from pydub import AudioSegment
        with tempfile.NamedTemporaryFile(suffix='.wav') as fp:
            AudioSegment.from_mp3(path_to_audio).export(fp.name, format='wav')
            ss, _ = sf.read(fp.name)
            oop = mk_preds_fa(fp.name, model)
    else:
        ss, _ = sf.read(path_to_audio)
        oop = mk_preds_fa(path_to_audio, model)

    p_smooth = smooth_output(oop.T, min_speech=1.3, min_music=3.4, max_silence_speech=0.4, max_silence_music=0.6)
    #p_smooth = p_smooth.T
    see = preds_to_se(p_smooth.T, audio_clip_length=ss.shape[0]/22050.0)

    if path_to_results:
        with open(path_to_results, 'w') as fp:
            fp.write('\n'.join('{}\t{}\t{}'.format(round(x[0], 5), round(x[1], 5), x[2]) for x in see))
    return convert_preds_to_dict(see)
