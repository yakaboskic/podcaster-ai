import argparse
import numpy as np
import librosa
import tensorflow as tf
import math
import compress_pickle
import tqdm
from tensorflow import keras
from tensorflow.keras import layers, activations
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
        mel = librosa.feature.melspectrogram(y=row, sr=sr, hop_length=int(sr/30.25), n_mels=149)
        if log:
            mel = librosa.core.power_to_db(mel)
        mels.append(mel)
    return np.array(mels)

def transform_sample(in_signal, in_sr, hop_size_seconds=None, window_size_seconds=10, num_random_hops=None):
    # Hopify to get audio data in the appropriate length
    if num_random_hops:
        hops = random_hopify(in_signal, in_sr, window_size_seconds, num_random_hops)
    else:
        hops = hopify(in_signal, in_sr, hop_size_seconds, window_size_seconds)
    # Get a melspectrogram for each hop
    mels = get_melspectrograms(hops, in_sr)
    return mels

def construct_datarow(x, y, sr, hop_size_seconds, window_size_seconds, num_random_hops):
    # Get hops and melspectrograms
    x_mels = transform_sample(x, sr, hop_size_seconds, window_size_seconds, num_random_hops)
    # Duplicate the y's for each hop
    y_mels = np.full((x_mels.shape[0], y.shape[0]), y)
    return x_mels, y_mels

def transform_generator(dataset_path, sr=22050, window_size_seconds=10, y_scale=0.1, discrete=True, output_subset_indices=None):
    with open(dataset_path, 'rb') as data_file:
        X, Y, y_cols = compress_pickle.load(data_file)
    for x, y in zip(X, Y):
        x_mel = transform_sample(x, sr, window_size_seconds=window_size_seconds, num_random_hops=1)
        x_mel = np.squeeze(x_mel, 0)#.T
        x_mel = np.expand_dims(x_mel, -1)
        if not discrete:
            y = y.astype('float32') * y_scale
        else:
            y = y.astype('int32')
        if output_subset_indices:
            y = y[output_subset_indices]
            y_cols = [y_cols[i] for i in output_subset_indices]
        # Build output dict
        y = {output: _y for output, _y in zip(y_cols, y)}
        yield x_mel.astype('float32'), y

def construct_dataset(dataset_path, sr, hop_size_seconds=None, window_size_seconds=None, num_random_hops=None, batch_size=128):
    # Load pre-built dataset
    with open(dataset_path, 'rb') as data_file:
        X, Y, Y_cols = compress_pickle.load(data_file)
    # Transform each sample
    X_mels, Y_mels = [], [] 
    for x, y in tqdm.tqdm(zip(X, Y), desc='Transforming rows', total=len(X)):
        x_mels, y_mels = construct_datarow(x, y, sr, hop_size_seconds, window_size_seconds, num_random_hops)
        X_mels.append(x_mels)
        Y_mels.append(y_mels)
    # Concatenate arrays
    X_new = np.vstack(X_mels)
    Y_new = np.vstack(Y_mels)
    return X_new, Y_new, Y_cols

def train_jku_emotions_model(
        dataset_path,
        learning_rate=0.0005,
        epsilon=1.0,
        batch_size=8,
        epochs=50,
        output_subset_indices=None,
        discrete=True,
        class_labels=['Low', 'Medium', 'High'],
        sr=22050,
        window_size_seconds=10,
        y_scale=0.1,
        ):
    print('Loading dataset and dataset shapes for model construction.')
    # Load pre-built dataset to get column names
    with open(dataset_path, 'rb') as data_file:
        X, _, y_cols = compress_pickle.load(data_file)
    if output_subset_indices:
        y_cols = [y_cols[i] for i in output_subset_indices]
    # Build generator to get necessary shapes
    for x, y in transform_generator(dataset_path, sr, window_size_seconds, y_scale, discrete, output_subset_indices):
        break
    print('Constructing model.')
    # Build model
    if discrete:
        model = construct_jku_emotions_model_discrete(x.shape, y_cols, class_labels=class_labels)
    else:
        model = construct_jku_emotions_model_continuous(x.shape, y_cols)
    # Compile model
    if discrete:
        model = compile_jku_emotions_model_discrete(model, keras.optimizers.Adam, y_cols, learning_rate, epsilon)
    else:
        model = compile_jku_emotions_model_continuous(model, keras.optimizers.Adam, y_cols, learning_rate, epsilon)
    print(y_cols)
    print(y)
    # Construct tf dataset
    dataset = tf.data.Dataset.from_generator(
            transform_generator,
            args=[dataset_path], 
            output_signature=(
                tf.TensorSpec(shape=x.shape, dtype=tf.float32),
                {output: tf.TensorSpec(shape=(), dtype=tf.float32) for output in y_cols},
                )
            ).shuffle(50).batch(batch_size)
    # Fit
    history = model.fit(
        x=dataset,
        epochs=epochs,
        )
    return model, history

def train_fc_emotions_model(
        dataset_path,
        learning_rate=0.0005,
        epsilon=1.0,
        batch_size=8,
        epochs=50,
        output_subset_indices=None,
        discrete=True,
        class_labels=['Low', 'Medium', 'High'],
        sr=22050,
        window_size_seconds=10,
        y_scale=0.1,
        ):
    print('Loading dataset and dataset shapes for model construction.')
    # Load pre-built dataset to get column names
    with open(dataset_path, 'rb') as data_file:
        X, _, y_cols = compress_pickle.load(data_file)
    if output_subset_indices:
        y_cols = [y_cols[i] for i in output_subset_indices]
    # Build generator to get necessary shapes
    for x, y in transform_generator(dataset_path, sr, window_size_seconds, y_scale, discrete, output_subset_indices):
        break
    print('Constructing model.')
    # Build model
    if discrete:
        model = construct_fc_model_discrete(x.shape, y_cols, class_labels=class_labels)
    else:
        model = construct_fc_model_continuous(x.shape, y_cols)
    # Compile model
    if discrete:
        model = compile_jku_emotions_model_discrete(model, keras.optimizers.Adam, y_cols, learning_rate, epsilon)
    else:
        model = compile_jku_emotions_model_continuous(model, keras.optimizers.Adam, y_cols, learning_rate, epsilon)
    # Construct tf dataset
    dataset = tf.data.Dataset.from_generator(
            transform_generator,
            args=[dataset_path], 
            output_signature=(
                tf.TensorSpec(shape=x.shape, dtype=tf.float32),
                tf.TensorSpec(shape=y.shape, dtype=tf.float32),
                )
            ).shuffle(50).batch(batch_size)
    # Fit
    history = model.fit(
        x=dataset,
        epochs=epochs,
        )
    return model, history


def compile_jku_emotions_model_continuous(model, optimizer, output_columns, learning_rate=0.0005, epsilon=1.0):
    # Build list of mean square error losses
    losses = []
    for _ in output_columns:
        losses.append(keras.losses.MeanSquaredError())
    # Compile
    model.compile(
            optimizer=optimizer(
                learning_rate=learning_rate,
                epsilon=epsilon,
                ),
            loss=losses,
            )
    return model

def compile_jku_emotions_model_discrete(model, optimizer, output_columns, learning_rate=0.0005, epsilon=1.0):
    # Build list of mean square error losses
    losses = {}
    for output in output_columns:
        losses[output] = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        #losses.append(keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    # Compile
    model.compile(
            optimizer=optimizer(
                learning_rate=learning_rate,
                epsilon=epsilon,
                ),
            loss=losses,
            #metrics=['accuracy'],
            )
    return model

def construct_jku_emotions_model_continuous(input_shape, output_columns):
    # Input Layer
    inputs = keras.Input(shape=input_shape, name="mel_inputs")
    # Construct base model and outputs
    outputs = []
    for output in output_columns:
        x = construct_jku_base_model(inputs)
        outputs.append(layers.Dense(1, name=output)(x))
    model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            )
    return model

def construct_jku_emotions_model_discrete(input_shape, output_columns, class_labels):
    # Input Layer
    inputs = keras.Input(shape=input_shape, name="mel_inputs")
    # Construct base model and outputs
    outputs = []
    for output in output_columns:
        x = construct_jku_base_model(inputs)
        outputs.append(layers.Dense(len(class_labels), name=output)(x))
    model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            )
    return model

def construct_fc_model_continuous(input_shape, output_columns):
    # Construct base model
    inputs, x = construct_fc_base_model(input_shape)
    # Define outputs
    outputs = []
    for out_name in output_columns:
        outputs.append(
                layers.Dense(1, name=out_name)(x)
                )
    model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            )
    return model

def construct_fc_model_discrete(input_shape, output_columns, class_labels):
    # Construct base model
    inputs, x = construct_fc_base_model(input_shape)
    # Define outputs
    outputs = []
    for out_name in output_columns:
        outputs.append(
                layers.Dense(len(class_labels), name=out_name)(x)
                )
    model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            )
    return model

def construct_fc_base_model(input_shape):
    # Input Layer
    inputs = keras.Input(shape=input_shape, name="mel_inputs")
    # Flatten
    x = layers.Flatten()(inputs)
    # FC layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    return inputs, x

def construct_jku_base_model(inputs):
    # First Convolution
    x = layers.Conv2D(
            filters=64,
            kernel_size=5,
            strides=2,
            padding='same',
            #activation='relu',
            )(inputs)
    x = layers.BatchNormalization()(x)
    x = activations.relu(x)
    # Second Convolution
    x = layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            #activation='relu',
            )(x)
    x = layers.BatchNormalization()(x)
    x = activations.relu(x)
    # First Max Pooling
    x = layers.MaxPool2D(pool_size=2)(x)
    # Dropout
    x = layers.Dropout(0.3)(x)
    # Third Convolution
    x = layers.Conv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            #activation='relu',
            )(x)
    x = layers.BatchNormalization()(x)
    x = activations.relu(x)
    # Fourth Convolution
    x = layers.Conv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            #activation='relu',
            )(x)
    x = layers.BatchNormalization()(x)
    x = activations.relu(x)
    # Second Max Pooling
    x = layers.MaxPool2D(pool_size=2)(x)
    # Dropout
    x = layers.Dropout(0.3)(x)
    # Fifth Convolution
    x = layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            #activation='relu',
            )(x)
    x = layers.BatchNormalization()(x)
    x = activations.relu(x)
    # Sixth Convolution
    x = layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            #activation='relu',
            )(x)
    x = layers.BatchNormalization()(x)
    x = activations.relu(x)
    # Seventh Convolution
    x = layers.Conv2D(
            filters=384,
            kernel_size=3,
            strides=1,
            padding='same',
            #activation='relu',
            )(x)
    x = layers.BatchNormalization()(x)
    x = activations.relu(x)
    # Eighth Convolution
    x = layers.Conv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            #activation='relu',
            )(x)
    x = layers.BatchNormalization()(x)
    x = activations.relu(x)
    # Nineth Convolution
    x = layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='valid',
            #activation='relu',
            )(x)
    x = layers.BatchNormalization()(x)
    x = activations.relu(x)
    # Adapative average pooling
    x = layers.AveragePooling2D(
            pool_size=1,
            strides=1,
            )(x)
    # Flatten
    x = layers.Flatten()(x)
    return x
