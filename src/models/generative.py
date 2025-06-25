import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import copy

# BETA VARIATIONAL AUTOENCODER MODEL

class BetaVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(BetaVAE, self).__init__()

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

def train_vae(train_loader):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    epochs = 50
    model = BetaVAE(latent_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    beta = 3.0  # Podés empezar en 1.0 e ir probando (0.25, 4.0, etc.)
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

        print(f"Epoch [{epoch+1}], Loss: {total_loss:.2f}, Recon: {total_recon:.2f}, KL: {total_kl:.2f}")
    return model

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
    

import torch
import torch.nn as nn
import torch.optim as optim
import os

# Dispositivo (Apple MPS GPU si existe, sino CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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