from IPython.display import Audio, display
import matplotlib.pyplot as plt
from torch.functional import F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import librosa
import torch
import copy
import os

# BETA VARIATIONAL AUTOENCODER MODEL

class BetaVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(BetaVAE, self).__init__()
        self.latent_dimension=latent_dim

        # Encoder
        self.enc = nn.Sequential(
            # (1, 64, 64) -> (32, 32, 32)
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.dec = nn.Sequential(
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # (32, 32, 32) -> (1, 64, 64)
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            # Sin activación final para MSE loss (usá Sigmoid para [0,1])
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z).view(-1, 256, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
        
    def bvae_loss(self, recon_x, x, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld_loss, recon_loss, kld_loss

def train_vae(train_loader, latent_dimension=32, learning_rate=1e-3, beta=3.0, epochs=50):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    model = BetaVAE(latent_dim=latent_dimension).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Store losses for plotting
    losses, recons, kls = [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        for batch in train_loader:
            x = batch[0].to(device)

            recon, mu, logvar = model(x)
            loss, recon_loss, kld_loss = model.bvae_loss(recon, x, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kld_loss.item()

        losses.append(total_loss)
        recons.append(total_recon)
        kls.append(total_kl)
        print(f"Epoch [{epoch+1}], Loss: {total_loss:.2f}, Recon: {total_recon:.2f}, KL: {total_kl:.2f}")
    # Plot after training
    plot_vae_losses(losses, recons, kls)
    return model

def plot_vae_losses(losses, recons, kls):
    epochs = range(1, len(losses) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    # Plot recon and total loss together
    axs[0].plot(epochs, losses, label='Total Loss')
    axs[0].plot(epochs, recons, label='Reconstruction Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Total & Reconstruction Loss')
    axs[0].legend()
    # Plot KL loss
    axs[1].plot(epochs, kls, label='KL Divergence', color='orange')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('KL')
    axs[1].set_title('KL Divergence')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def display_vae_synthetic_samples(model, mean, std, spectrogram_config):
    latent_dim = model.latent_dimension

    # Ensure device is defined (use same logic as elsewhere)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    with torch.no_grad():
        # 1. Generá un batch de vectores latentes Z ~ N(0, I)
        z_samples = torch.randn(4, latent_dim).to(device)  # Por ejemplo, 4 muestras
        # 2. Decodificá los Z para obtener espectrogramas
        generated_specs = model.decode(z_samples)

    # Ahora `generated_specs` es un tensor de shape [4, 1, 64, 64]

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
        # Desnormalizar
        spec = spec * std + mean

        # Invertir Mel -> STFT
        mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
        spec_inv = np.linalg.pinv(mel_basis).dot(np.exp(spec))
        # Invertir STFT -> Audio
        audio = librosa.griffinlim(spec_inv, hop_length=HOP_LENGTH, n_fft=N_FFT)
        # Mostrar Audio
        display(Audio(audio, rate=int(SR*1.5)))


# AAEEEE

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU()  # Match VAE structure
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh()  # Match GAN / VAE
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(x)

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)

# GAANNN

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64):  # ngf = tamaño base de filtros
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # Entrada Z: [batch, z_dim, 1, 1]
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # [batch, ngf*8, 4, 4]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # [batch, ngf*4, 8, 8]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # [batch, ngf*2, 16, 16]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # [batch, ngf, 32, 32]
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # Salida: [batch, 1, 64, 64]
        )
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, ndf=64):  # ndf = tamaño base de filtros
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # [batch, 1, 64, 64]
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # [batch, ndf, 32, 32]
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # [batch, ndf*2, 16, 16]
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # [batch, ndf*4, 8, 8]
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # [batch, ndf*8, 4, 4]
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # Salida escalar
        )
    def forward(self, x):
        return self.model(x).view(-1)

# Dispositivo (Apple MPS GPU si existe, sino CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

def train_gan(train_loader):
    # Crear instancias
    z_dim = 100
    gen = Generator(z_dim).to(device)
    disc = Discriminator().to(device)

    # Criterio de adversarial loss
    criterion = nn.BCEWithLogitsLoss() # probar con sum

    # Optimizadores
    lr = 2e-4
    beta1 = 0.5
    gen_optimizer = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))
    disc_optimizer = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))

    # Carpeta para guardar outputs
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    # Loop de entrenamiento
    epochs = 100  # Cambia si querés
    for epoch in range(1, epochs + 1):
        for batch in train_loader:
            real_images = batch[0].to(device)

            batch_size = real_images.size(0)

            # --------------------------------------------------
            # Entrenar Discriminador
            # --------------------------------------------------
            z = torch.randn(batch_size, z_dim, 1, 1, device=device)
            fake_images = gen(z)

            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            disc_optimizer.zero_grad()

            real_preds = disc(real_images)
            loss_real = criterion(real_preds, real_labels)

            fake_preds = disc(fake_images.detach())
            loss_fake = criterion(fake_preds, fake_labels)

            loss_disc = loss_real + loss_fake
            loss_disc.backward()
            disc_optimizer.step()

            # --------------------------------------------------
            # Entrenar Generador
            # --------------------------------------------------
            gen_optimizer.zero_grad()
            fake_preds = disc(fake_images)  # Ahora sin detach
            loss_gen = criterion(fake_preds, real_labels)  # Queremos que diga "reales"
            loss_gen.backward()
            gen_optimizer.step()

        # ---- Fin de la época ----
        print(f"Epoch [{epoch}/{epochs}], "
            f"Loss D: {loss_disc.item():.4f}, "
            f"Loss G: {loss_gen.item():.4f}")

    print("Entrenamiento finalizado ✅")
    return gen, disc