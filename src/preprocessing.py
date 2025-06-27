from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import librosa
import torch

def normalize(dataset:pd.DataFrame, column:str):
    dataset[column] = dataset[column].apply(lambda x: x / np.max(np.abs(x)))

def get_melspectrogram(sample:pd.DataFrame, sampling_rate:int, fft_samples:int, hop_length:int, n_mel_bins:int, max_freq):
    mel_spectrogram = librosa.feature.melspectrogram(y=np.array(sample), sr=sampling_rate, n_fft=fft_samples, hop_length=hop_length, n_mels=n_mel_bins, fmax=max_freq)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram

def get_all_mel_spectrograms(audio_df:pd.DataFrame, spectrogram_config:dict):
    sampling_rate = spectrogram_config['SR']
    fft_samples = spectrogram_config['FFT_SAMPLES']
    hop_length = spectrogram_config['HOP_LENGTH']
    n_mel_bins = spectrogram_config['MEL_BINS']
    max_frequency =spectrogram_config['MAX_FREQ']

    mel_specs = []
    for audio in audio_df['audio']:
        mel = get_melspectrogram(audio, sampling_rate, fft_samples, hop_length, n_mel_bins, max_frequency)
        mel_specs.append(mel.flatten())
    return np.array(mel_specs)

def extract_time_acoustic_features(dataset:pd.DataFrame):
    new_features = []
    for audio in dataset['audio']:
        audio_np = np.array(audio)
        acoustic_features = {
            'rms_energy': np.mean(librosa.feature.rms(y=audio_np)),
            'zcr': np.mean(librosa.feature.zero_crossing_rate(y=audio_np)),
        }
        new_features.append(acoustic_features)
    features_df = pd.DataFrame(new_features)
    return features_df

def extract_frequency_acoustic_features(dataset:pd.DataFrame):
    new_features = []
    for audio in dataset['audio']:
        audio_np = np.array(audio)
        acoustic_features = {
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio_np)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=audio_np)),
            'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=audio_np)),
            'spectral_flatness': np.mean(librosa.feature.spectral_flatness(y=audio_np))
        }
        new_features.append(acoustic_features)
    features_df = pd.DataFrame(new_features)
    return features_df

def extract_acoustic_features(dataset:pd.DataFrame):
    time_features = extract_time_acoustic_features(dataset)
    frequency_features = extract_frequency_acoustic_features(dataset)

    new_df = pd.concat([dataset.reset_index(drop=True), time_features], axis=1)
    new_df = pd.concat([new_df.reset_index(drop=True), frequency_features], axis=1)
    
    return new_df

def get_signal_energy(dataset:pd.DataFrame, column:str):
    energy = []
    df = dataset.copy()
    for x in df[column]:
        energy.append(np.sum(np.square(x)))
    df['energy'] = energy

    return df

def compute_global_min_max(audio_df):
    whale_specs = [get_melspectrogram(audio) for audio, label in zip(audio_df['audio'], audio_df['label']) if label == 1]
    global_min = np.min([spec.min() for spec in whale_specs])
    global_max = np.max([spec.max() for spec in whale_specs])
    return global_min, global_max

def get_class_weights(y):
    weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return torch.tensor(weights, dtype=torch.float32)

def standarize(X:np.ndarray):
    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + 1e-8)

    print(f'Global mean: {mean}, Global std: {std}')
    print(f'Shape of X: {X.shape}')

    return X, mean, std

def fix_length(audio, sampling_rate:2000, desired_length=2.0):
    target_len = int(sampling_rate * desired_length)

    # si es más largo, recorto
    if len(audio) > target_len:
        audio = audio[:target_len]

    # si es más corto, relleno con ceros
    elif len(audio) < target_len:
        pad_width = target_len - len(audio)
        audio = np.pad(audio, (0, pad_width), mode='constant')

    return audio

def augment_audio(audio_dataset, data_augmentation_config, percentage=0.2, sampling_rate=2000, seed=42):
    whale_audios = audio_dataset[audio_dataset['label'] == 1]
    n_samples = int(len(whale_audios)*percentage)
    samples = whale_audios.sample(n_samples, random_state=seed)

    augmented_samples = []
    label = 1
    filepath = None
    
    for index, sample in samples.iterrows():
        clip_name, audio = sample['clip_name'], sample['audio']
        # time stretch -> cambia la duración del canto
        for rate in data_augmentation_config['TIME_STRETCH_FACTORS']:
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            stretched = fix_length(stretched, sampling_rate, desired_length=2.0)
            augmented_samples.append((f"{clip_name}_stretch_{rate}.aiff", label, filepath, stretched))
        # pitch shift -> sube/baja semitonos
        for n_steps in data_augmentation_config['PITCH_SHIFTS']:
            shifted = librosa.effects.pitch_shift(audio, sr=sampling_rate, n_steps=n_steps)
            augmented_samples.append((f"{clip_name}_shift_{n_steps}.aiff", label, filepath, shifted))
        # agrega ruido gaussiano 
        noise = data_augmentation_config['NOISE_LEVEL'] * np.random.randn(len(audio))
        noisy_audio = audio + noise
        augmented_samples.append((f"{clip_name}_noisy.aiff", label, filepath, noisy_audio))
        # podria agregar que se puedan escuchar un par de originales vs time y pitch y ruido

    print(f'New augmented samples: {len(augmented_samples)}')
    return augmented_samples

def standarize_train_val(X_train, X_val):
    train_mean = X_train.mean()
    train_std = X_train.std() + 1e-8 # para no dividir x cero dsp
    print(f'Mean: {train_mean}, std: {train_std}')

    X_train_std = (X_train - train_mean) / train_std
    X_val_std = (X_val - train_mean) / train_std

    return X_train_std, X_val_std, train_mean, train_std