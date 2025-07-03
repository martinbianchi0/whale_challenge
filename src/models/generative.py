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
        super().__init__()
        self.latent_dimension=latent_dim

        # encoder
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

        # decoder
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
        print(f'Epoch [{epoch+1}], Loss: {total_loss:.2f}, Recon: {total_recon:.2f}, KL: {total_kl:.2f}')
    plot_vae_losses(losses, recons, kls)
    return model

def plot_vae_losses(losses, recons, kls):
    epochs = range(1, len(losses) + 1)
    fig,axs = plt.subplots(1, 2, figsize=(12, 4))

    # recon y loss total
    axs[0].plot(epochs, losses, label='Total Loss')
    axs[0].plot(epochs, recons, label='Reconstruction Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Total & Reconstruction Loss')
    axs[0].legend()

    # kl
    axs[1].plot(epochs, kls, label='KL Divergence', color='orange')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('KL')
    axs[1].set_title('KL Divergence')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

# ADVERSARIAL AUTOENCODER MODEL

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential( # igual al vae
            nn.Conv2d(1, 32, 4, 2, 1), 
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1), 
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1), 
            nn.ReLU(),

            nn.Conv2d(128, 256, 4, 2, 1), 
            nn.ReLU()
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
            nn.ConvTranspose2d(256, 128, 4, 2, 1), 
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1), 
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1), 
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 4, 2, 1), 
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(x)

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), 
            nn.ReLU(),

            nn.Linear(128, 64), 
            nn.ReLU(),
            
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)

def train_aae(train_loader, device, epochs=50, latent_dim=32, learning_rate=1e-3):
    # Entrenamiento
    encoder = Encoder(latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    discriminator = Discriminator(latent_dim).to(device)

    opt_autoenc = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    bce = nn.BCELoss(reduction="sum")
    recon_loss_fn = nn.MSELoss(reduction="sum")

    for epoch in range(epochs):
        encoder.train(); decoder.train(); discriminator.train()
        total_recon, total_disc, total_gen = 0, 0, 0

        for batch in train_loader:
            x = batch[0].to(device)

            # Reconstruccion
            z = encoder(x)
            x_recon = decoder(z)
            recon_loss = recon_loss_fn(x_recon, x)

            opt_autoenc.zero_grad()
            recon_loss.backward()
            opt_autoenc.step()

            # Discriminador
            z_real = torch.randn_like(z)
            z_fake = encoder(x).detach()
            d_real = discriminator(z_real)
            d_fake = discriminator(z_fake)

            disc_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

            # Generador (Encoder)
            z_fake = encoder(x)
            d_fake = discriminator(z_fake)
            gen_loss = bce(d_fake, torch.ones_like(d_fake))

            opt_autoenc.zero_grad()
            gen_loss.backward()
            opt_autoenc.step()

            total_recon += recon_loss.item()
            total_disc += disc_loss.item()
            total_gen += gen_loss.item()

        print(f"Epoch [{epoch}] Recon: {total_recon:.2f}, Disc: {total_disc:.2f}, Gen: {total_gen:.2f}")
    return encoder, decoder, discriminator

# GENERATIVE ADVERSARIAL NETWORK

class SpecGANGenerator(nn.Module):
    def __init__(self, z_dim=100):
        super(SpecGANGenerator, self).__init__()
        self.net = nn.Sequential(

            nn.Linear(z_dim, 1024 * 4 * 4),
            nn.BatchNorm1d(1024 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (1024, 4, 4)),

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # -> 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1), # -> 64x64
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)
    
class SpecGANDiscriminator(nn.Module):
    def __init__(self):
        super(SpecGANDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1), # -> 4x4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
def train_gan(train_loader, noise_dimensions=100, epochs=200, learning_rate=2e-4):

    device = 'mps' if torch.mps.is_available() else 'cpu'
    batch_size = 128
    print(device)

    G = SpecGANGenerator(z_dim=noise_dimensions).to(device)
    D = SpecGANDiscriminator().to(device)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for (real_specs, ) in train_loader:
            real_specs = real_specs.to(device).float()

            batch_size = real_specs.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            z = torch.randn(batch_size, noise_dimensions, device=device)
            fake_specs = G(z).detach()

            D_real = D(real_specs).view(-1, 1)
            D_fake = D(fake_specs).view(-1, 1)

            loss_D_real = criterion(D_real, real_labels)
            loss_D_fake = criterion(D_fake, fake_labels)
            loss_D = loss_D_real + loss_D_fake

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            z = torch.randn(batch_size, noise_dimensions, device=device)
            fake_specs = G(z)

            D_fake = D(fake_specs).view(-1, 1)
            loss_G = criterion(D_fake, real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                z = torch.randn(16, noise_dimensions, device=device)
                generated = G(z).cpu()
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for i, ax in enumerate(axes.flat):
                    img = generated[i, 0].numpy()
                    ax.imshow(img, aspect='auto', origin='lower', vmin=-1, vmax=1, cmap='viridis')
                    ax.axis('off')
                plt.suptitle(f"Epoch {epoch+1}")
                plt.tight_layout()
                plt.show()
            os.makedirs('specgan', exist_ok=True)
            torch.save(G.state_dict(), f'saved_models/specgan/generator_epoch_{epoch+1}.pt')
            torch.save(D.state_dict(), f'saved_models/specgan/discriminator_epoch_{epoch+1}.pt')