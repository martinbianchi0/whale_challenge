from sklearn.preprocessing import MinMaxScaler
from IPython.display import Audio, display
from src.preprocessing import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import librosa

def display_random_samples(dataset:pd.DataFrame, seed:int, spectrogram_config:dict):
    np.random.seed(seed)

    sampling_rate = spectrogram_config['SR']
    fft_samples = spectrogram_config['FFT_SAMPLES']
    hop_length = spectrogram_config['HOP_LENGTH']
    n_mel_bins = spectrogram_config['MEL_BINS']
    max_frequency =spectrogram_config['MAX_FREQ']

    # SAMPLES
    whale_sample = dataset[dataset['label'] == 1].sample(5)['audio']
    noise_sample = dataset[dataset['label'] == 0].sample(5)['audio']

    # AUDIO
    print('right whale call random audio sample')
    display(Audio(np.array(whale_sample.iloc[0]), rate=sampling_rate*1.5))
    print('\nno whale random audio sample')
    display(Audio(np.array(noise_sample.iloc[0]), rate=sampling_rate*1.5))

    # SOUND-WAVE
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))

    for i in range(5):
        axes[0, i].plot(np.array(whale_sample.iloc[i]))
        axes[0, i].set_title('Whale')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        axes[1, i].plot(np.array(noise_sample.iloc[i]), color='#FF6961')
        axes[1, i].set_title('No Whale')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    plt.tight_layout()
    plt.show()

    # MULTIPLE SPECTROGRAMS
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i in range(5):
        whale_sample_spectrogram = get_melspectrogram(whale_sample.iloc[i], sampling_rate, fft_samples, hop_length, n_mel_bins, max_frequency)
        img0 = librosa.display.specshow(whale_sample_spectrogram, sr=sampling_rate, hop_length=hop_length, ax=axes[0, i], x_axis='time', y_axis='hz', cmap='magma', fmax=max_frequency)
        axes[0, i].set_title('Whale')
        axes[0, i].set_ylim([0, max_frequency])
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[0, i].set_xlabel('')
        axes[0, i].set_ylabel('')

        noise_sample_spectrogram = get_melspectrogram(noise_sample.iloc[i], sampling_rate, fft_samples, hop_length, n_mel_bins, max_frequency)
        img1 = librosa.display.specshow(noise_sample_spectrogram, sr=sampling_rate, hop_length=hop_length, ax=axes[1, i], x_axis='time', y_axis='hz', cmap='magma', fmax=max_frequency)
        axes[1, i].set_title('No Whale')
        axes[1, i].set_ylim([0, max_frequency])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        axes[1, i].set_xlabel('')
        axes[1, i].set_ylabel('')

    plt.tight_layout()
    plt.show()

    # TWO SPECTROGRAMS
    whale_sample2 = dataset[dataset['label'] == 1].sample(1, random_state=seed+1)['audio']
    noise_sample2 = dataset[dataset['label'] == 0].sample(1, random_state=seed+1)['audio']
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    whale_sample2_spectrogram = get_melspectrogram(whale_sample2.iloc[0], sampling_rate, fft_samples, hop_length, n_mel_bins, max_frequency)
    img0 = librosa.display.specshow(whale_sample2_spectrogram, sr=sampling_rate, hop_length=hop_length, ax=axes[0], x_axis='time', y_axis='hz', cmap='magma', fmax=max_frequency)
    axes[0].set_title('Whale')
    axes[0].set_ylim([0, max_frequency])
    fig.colorbar(img0, ax=axes[0], format="%+2.0f dB")

    noise_sample2_spectrogram = get_melspectrogram(noise_sample2.iloc[0], sampling_rate, fft_samples, hop_length, n_mel_bins, max_frequency)
    img1 = librosa.display.specshow(noise_sample2_spectrogram, sr=sampling_rate, hop_length=hop_length, ax=axes[1], x_axis='time', y_axis='hz', cmap='magma', fmax=max_frequency)
    axes[1].set_title('No Whale')
    axes[1].set_ylim([0, max_frequency])
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.show()

def display_features_boxplots(audio_features_df: pd.DataFrame):

    features = ['rms_energy', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness']
    feature_names = {
        'rms_energy': 'RMS Energy',
        'zcr': 'Zero Crossing Rate',
        'spectral_centroid': 'Spectral Centroid',
        'spectral_bandwidth': 'Spectral Bandwidth',
        'spectral_rolloff': 'Spectral Rolloff',
        'spectral_flatness': 'Spectral Flatness'
    }

    # Normalize features to [0, 1] scale
    scaler = MinMaxScaler()
    audio_features_df_norm = audio_features_df.copy()
    audio_features_df_norm[features] = scaler.fit_transform(audio_features_df[features])

    plt.figure(figsize=(14, 6))
    audio_features_df_melted = audio_features_df_norm.melt(id_vars='label', value_vars=features, var_name='Feature', value_name='Value')
    audio_features_df_melted['Feature'] = audio_features_df_melted['Feature'].map(feature_names)

    ax = sns.boxplot(
        x='Feature',
        y='Value',
        hue='label',
        data=audio_features_df_melted,
        palette={0: '#FF6961', 1: '#1f77b4'},
        showfliers=False
    )
    plt.xlabel('Feature')
    plt.ylabel('Normalized Value')
    plt.title('Normalized Audio Features by Class')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Noise', 'Whale'], title='Class')

    plt.tight_layout()
    plt.show()

def plot_pca(dataset:pd.DataFrame, pca_arr):
    plt.figure(figsize=(8,6))
    scatter = sns.scatterplot(
        x=pca_arr[:,0], 
        y=pca_arr[:,1], 
        hue=dataset['label'], 
        palette={0: '#FF6961', 1: '#1f77b4'}
    )
    plt.title('PCA of Mel Spectrograms')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    # Fix legend labels and colors
    handles, _ = scatter.get_legend_handles_labels()
    scatter.legend(handles=handles, title='Class', labels=['Noise', 'Whale'])
    plt.show()

def plot_average_spectrograms(audio_df:pd.DataFrame, spectrogram_config:dict):

    sampling_rate = spectrogram_config['SR']
    fft_samples = spectrogram_config['FFT_SAMPLES']
    hop_length = spectrogram_config['HOP_LENGTH']
    n_mel_bins = spectrogram_config['MEL_BINS']
    max_frequency =spectrogram_config['MAX_FREQ']

    whale_spectrograms = []
    noise_spectrograms = []
    for audio, label in zip(audio_df['audio'], audio_df['label']):
        spectrogram = get_melspectrogram(audio, sampling_rate, fft_samples, hop_length, n_mel_bins, max_frequency)
        if label == 1:
            whale_spectrograms.append(spectrogram)
        else: 
            noise_spectrograms.append(spectrogram)
    
    whale_spectrograms = np.array(whale_spectrograms)
    noise_spectrograms = np.array(noise_spectrograms)

    whale_average_spectrogram = whale_spectrograms.mean(axis=0)
    noise_average_spectrogram = noise_spectrograms.mean(axis=0)
    average_differences_spectrogram = whale_average_spectrogram - noise_average_spectrogram

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    img0 = librosa.display.specshow(whale_average_spectrogram, sr=sampling_rate, hop_length=hop_length, ax=axes[0], x_axis='time', y_axis='hz', cmap='magma', fmax=max_frequency)
    axes[0].set_title('Whale Call Average Spectrogram', pad=20)
    axes[0].set_ylim([0, max_frequency])
    # axes[0].set_xticks([])
    # axes[0].set_yticks([])
    fig.colorbar(img0, ax=axes[0], format="%+2.0f dB")

    img1 = librosa.display.specshow(noise_average_spectrogram, sr=sampling_rate, hop_length=hop_length, ax=axes[1], x_axis='time', y_axis='hz', cmap='magma', fmax=max_frequency)
    axes[1].set_title('No Whale Call Average Spectrogram', pad=20)
    axes[1].set_ylim([0, max_frequency])
    # axes[1].set_xticks([])
    # axes[1].set_yticks([])
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")

    img2 = librosa.display.specshow(average_differences_spectrogram, sr=sampling_rate, hop_length=hop_length, ax=axes[2], x_axis='time', y_axis='hz', cmap='magma', fmax=max_frequency)
    axes[2].set_title('Average Differences Spectrogram', pad=20)
    axes[2].set_ylim([0, max_frequency])
    # axes[2].set_xticks([])
    # axes[2].set_yticks([])
    fig.colorbar(img2, ax=axes[2], format="%+2.0f dB")

    # 3 subplots -> average spectrogram whale, noise, resto y consigo diferencias
    plt.tight_layout()
    plt.show()

def show_class_balance(dataset:pd.DataFrame):

    print(f'Number of samples: {len(dataset)}')
    print(f'Number of whale calls: {dataset[dataset["label"] == 1].shape[0]}')
    print(f'Number of noise samples: {dataset[dataset["label"] == 0].shape[0]}')

    plt.figure(figsize=(5, 4))
    ax = sns.countplot(
        x='label',
        data=dataset,
        palette={0: '#FF6961', 1: plt.rcParams['axes.prop_cycle'].by_key()['color'][0]},
        hue='label',
        legend=False
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Noise', 'Whale'])
    ax.legend(['Noise', 'Whale'], title='Class')
    plt.show()
