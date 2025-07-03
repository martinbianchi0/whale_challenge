from sklearn.preprocessing import MinMaxScaler
from IPython.display import Audio, display
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.lines as mlines
from src.preprocessing import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import librosa
import torch

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
    print('\nno right whale call random audio sample')
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

def display_features_boxplots(audio_features_df:pd.DataFrame):
    features = ['rms_energy', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness']
    feature_names = {
        'rms_energy': 'RMS Energy',
        'zcr': 'Zero Crossing Rate',
        'spectral_centroid': 'Spectral Centroid',
        'spectral_bandwidth': 'Spectral Bandwidth',
        'spectral_rolloff': 'Spectral Rolloff',
        'spectral_flatness': 'Spectral Flatness'
    }

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
    ax.legend(handles=handles, labels=['No Whale', 'Whale'], title='Class')

    plt.tight_layout()
    plt.show()

def plot_pca(dataset:pd.DataFrame, pca_arr):
    plt.figure(figsize=(8,6))
    scatter = sns.scatterplot(
        x=pca_arr[:,0], 
        y=pca_arr[:,1], 
        hue=dataset['label'], 
        palette={0: '#FF6961', 1: '#1f77b4'},
        alpha=0.7
    )
    plt.title('PCA of Mel Spectrograms')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    handles, _ = scatter.get_legend_handles_labels()
    scatter.legend(handles=handles, title='Class', labels=['No Whale', 'Whale'])
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

def display_full_vae_latent_space(model, train_loader, device:str, seed:int):
    latent_vectors = []
    labels = []

    model.eval()

    with torch.no_grad():
        for data, label in train_loader:
            data = data.to(device)
            mu, logvar = model.encode(data) 
            latent_vectors.append(mu.cpu().numpy())
            labels.append(label.numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)

    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_vectors)

    tsne = TSNE(n_components=2, random_state=seed)
    latent_tsne = tsne.fit_transform(latent_vectors)

    palette = {0: '#FF6961', 1: '#1f77b4'}

    colors = np.array([palette[label] for label in labels])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # pca
    scatter_pca = axes[0].scatter(latent_pca[:, 0], latent_pca[:, 1], c=colors, alpha=0.7)
    axes[0].set_title('PCA of VAE Latent Space')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    # tsne
    scatter_tsne = axes[1].scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=colors, alpha=0.7)
    axes[1].set_title('t-SNE of VAE Latent Space')
    axes[1].set_xlabel('t-SNE1')
    axes[1].set_ylabel('t-SNE2')

    legend_labels = ['No Whale', 'Whale']
    legend_colors = ['#FF6961', '#1f77b4']
    legend_handles = [mlines.Line2D([], [], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]

    axes[0].legend(legend_handles, legend_labels, title="Labels", loc="upper left")
    axes[1].legend(legend_handles, legend_labels, title="Labels", loc="upper left")

    plt.tight_layout()
    plt.show()

def plot_bvae_latent_comparison(model, dataloader, device='mps'):
    model.eval()

    real_latents = []
    fake_latents = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)

            # encoding a los datos reales
            mu, logvar = model.encode(x)
            z_real = model.reparameterize(mu, logvar)
            real_latents.append(z_real.cpu().numpy())

            # genero datos sinteticos y encodeo
            z_rand = torch.randn_like(z_real)
            x_fake = model.decode(z_rand)
            mu_fake, logvar_fake = model.encode(x_fake)
            z_fake = model.reparameterize(mu_fake, logvar_fake)
            fake_latents.append(z_fake.cpu().numpy())

    real_latents = np.concatenate(real_latents, axis=0)
    fake_latents = np.concatenate(fake_latents, axis=0)

    combined_latents = np.concatenate([real_latents, fake_latents], axis=0)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_latents)
    pca_real = pca_result[:len(real_latents)]
    pca_fake = pca_result[len(real_latents):]

    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    tsne_result = tsne.fit_transform(combined_latents)
    tsne_real = tsne_result[:len(real_latents)]
    tsne_fake = tsne_result[len(real_latents):]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # pca
    axes[0].scatter(pca_real[:, 0], pca_real[:, 1], alpha=0.4, label='Real', s=5)
    axes[0].scatter(pca_fake[:, 0], pca_fake[:, 1], alpha=0.4, label='Generated', s=5)
    axes[0].set_title('VAE Latent Space - PCA')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend()

    # tsne
    axes[1].scatter(tsne_real[:, 0], tsne_real[:, 1], alpha=0.4, label='Real', s=5)
    axes[1].scatter(tsne_fake[:, 0], tsne_fake[:, 1], alpha=0.4, label='Generated', s=5)
    axes[1].set_title('Vae Latent Space - t-SNE')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def display_vae_synthetic_samples(model, mean:float, std:float, spectrogram_config:dict):
    latent_dim = model.latent_dimension

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    with torch.no_grad():
        z_samples = torch.randn(4, latent_dim).to(device)
        generated_specs = model.decode(z_samples)

    generated_specs = generated_specs.cpu().numpy()
    fig, axes = plt.subplots(1, generated_specs.shape[0], figsize=(3 * generated_specs.shape[0], 3))
    for i in range(generated_specs.shape[0]):
        spec = generated_specs[i, 0, :, :]
        spec = spec * std + mean
        ax = axes[i] if generated_specs.shape[0] > 1 else axes
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(f"Synthetic Spectrogram {i+1}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    SR = spectrogram_config['SR']
    N_FFT = spectrogram_config['FFT_SAMPLES'] 
    N_MELS = spectrogram_config['MEL_BINS']
    HOP_LENGTH = spectrogram_config['HOP_LENGTH']

    for i in range(generated_specs.shape[0]):
        spec = generated_specs[i, 0, :, :]
        spec = spec * std + mean

        mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
        spec_inv = np.linalg.pinv(mel_basis).dot(np.exp(spec))
        audio = librosa.griffinlim(spec_inv, hop_length=HOP_LENGTH, n_fft=N_FFT)
        display(Audio(audio, rate=int(SR*1.5)))

def display_aae_synthetic_samples(decoder, latent_dim:int, mean:float, std:float, spectrogram_config:dict):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    decoder = decoder.to(device)
    decoder.eval()

    with torch.no_grad():
        # Sample Gaussian noise in latent space
        z_samples = torch.randn(4, latent_dim).to(device)
        generated_specs = decoder(z_samples)  # shape: (4, 1, 64, 64)

    generated_specs = generated_specs.cpu().numpy()
    fig, axes = plt.subplots(1, generated_specs.shape[0], figsize=(3 * generated_specs.shape[0], 3))

    for i in range(generated_specs.shape[0]):
        spec = generated_specs[i, 0, :, :]
        spec = spec * std + mean  # de-normalize
        ax = axes[i] if generated_specs.shape[0] > 1 else axes
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(f"Synthetic Spectrogram {i+1}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    SR = spectrogram_config['SR']
    N_FFT = spectrogram_config['FFT_SAMPLES'] 
    N_MELS = spectrogram_config['MEL_BINS']
    HOP_LENGTH = spectrogram_config['HOP_LENGTH']

    for i in range(generated_specs.shape[0]):
        spec = generated_specs[i, 0, :, :]
        spec = spec * std + mean 

        mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
        spec_inv = np.linalg.pinv(mel_basis).dot(np.exp(spec))

        audio = librosa.griffinlim(spec_inv, hop_length=HOP_LENGTH, n_fft=N_FFT)
        display(Audio(audio, rate=int(SR*1.5)))

def plot_aae_latent_comparison(encoder, decoder, dataloader, device='mps'):
    encoder.eval()
    decoder.eval()

    real_latents = []
    fake_latents = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)

            # encoding a los datos reales
            z_real = encoder(x)
            real_latents.append(z_real.cpu().numpy())

            # genero datos sinteticos y encodeo
            z_random = torch.randn_like(z_real)
            x_fake = decoder(z_random)
            z_fake = encoder(x_fake)
            fake_latents.append(z_fake.cpu().numpy())

    real_latents = np.concatenate(real_latents, axis=0)
    fake_latents = np.concatenate(fake_latents, axis=0)
    combined_latents = np.concatenate([real_latents, fake_latents], axis=0)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_latents)
    pca_real = pca_result[:len(real_latents)]
    pca_fake = pca_result[len(real_latents):]

    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    tsne_result = tsne.fit_transform(combined_latents)
    tsne_real = tsne_result[:len(real_latents)]
    tsne_fake = tsne_result[len(real_latents):]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # pca
    axes[0].scatter(pca_real[:, 0], pca_real[:, 1], alpha=0.4, label='Real', s=5)
    axes[0].scatter(pca_fake[:, 0], pca_fake[:, 1], alpha=0.4, label='Generated', s=5)
    axes[0].set_title('AAE Latent Space - PCA')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend()

    # tsne
    axes[1].scatter(tsne_real[:, 0], tsne_real[:, 1], alpha=0.4, label='Real', s=5)
    axes[1].scatter(tsne_fake[:, 0], tsne_fake[:, 1], alpha=0.4, label='Generated', s=5)
    axes[1].set_title('AAE Latent Space - t-SNE')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def display_gan_synthetic_samples(generator, noise_dim:int, low:float, high:float, spectrogram_config:dict):
    num_samples = 4
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    generator = generator.to(device)
    generator.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, noise_dim).to(device)
        fake_specs = generator(z).cpu().numpy()

    fake_specs_denorm = (fake_specs + 1) / 2
    fake_specs_denorm = (fake_specs_denorm * (high - low)) + low

    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
    for i in range(num_samples):
        ax = axes[i] if num_samples > 1 else axes
        img = fake_specs_denorm[i, 0]
        im = ax.imshow(img, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(f"Synthetic Spectrogram {i+1}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    SR = spectrogram_config['SR']
    N_FFT = spectrogram_config['FFT_SAMPLES']
    HOP_LENGTH = spectrogram_config['HOP_LENGTH']

    for i in range(num_samples):
        mel_spec = fake_specs_denorm[i, 0]

        mel_spec_power = librosa.db_to_power(mel_spec)

        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec_power,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        display(Audio(audio, rate=int(SR * 1.5)))

def plot_specgan_output_projection_full(generator, real_data_loader, z_dim=100, device='mps'):
    generator.eval()

    real_specs = []
    fake_specs = []

    with torch.no_grad():
        for batch in real_data_loader:
            x_real = batch[0].to(device)
            real_specs.append(x_real.cpu().numpy())

            z = torch.randn(x_real.size(0), z_dim, device=device)
            x_fake = generator(z)
            fake_specs.append(x_fake.cpu().numpy())

    real_specs = np.concatenate(real_specs, axis=0)
    fake_specs = np.concatenate(fake_specs, axis=0)

    real_flat = real_specs.reshape(real_specs.shape[0], -1)
    fake_flat = fake_specs.reshape(fake_specs.shape[0], -1)

    combined = np.concatenate([real_flat, fake_flat], axis=0)
    labels = np.array([0]*len(real_flat) + [1]*len(fake_flat))

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined)
    pca_real = pca_result[labels == 0]
    pca_fake = pca_result[labels == 1]

    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    tsne_result = tsne.fit_transform(combined)
    tsne_real = tsne_result[labels == 0]
    tsne_fake = tsne_result[labels == 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # pca
    axes[0].scatter(pca_real[:, 0], pca_real[:, 1], alpha=0.3, label='Real', s=5)
    axes[0].scatter(pca_fake[:, 0], pca_fake[:, 1], alpha=0.3, label='Generated', s=5)
    axes[0].set_xlabel('PC1', fontsize=15)
    axes[0].set_ylabel('PC2', fontsize=15)
    axes[0].legend(fontsize=15)
    axes[0].tick_params(axis='both', labelsize=15)

    # tsne
    axes[1].scatter(tsne_real[:, 0], tsne_real[:, 1], alpha=0.4, label='Real', s=5)
    axes[1].scatter(tsne_fake[:, 0], tsne_fake[:, 1], alpha=0.4, label='Generated', s=5)
    axes[1].set_xlabel('t-SNE 1', fontsize=15)
    axes[1].set_ylabel('t-SNE 2', fontsize=15)
    axes[1].legend(fontsize=15)
    axes[1].tick_params(axis='both', labelsize=15)

    plt.tight_layout()
    plt.show()