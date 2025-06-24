from torch.utils.data import TensorDataset, DataLoader
from src.models.classification import *
from src.models.generative import *
import pandas as pd
import librosa
import torch
import os

def load_data(train_path:str, test_path:str, labels_path:str, sampling_rate:int):
    test_files = [f for f in os.listdir(test_path) if f.endswith('.aiff')]
    labels_df = pd.read_csv(labels_path)

    audio_df = labels_df.copy()
    audio_df['filepath'] = audio_df['clip_name'].apply(lambda x: os.path.join(train_path, x))
    audio_df['audio'] = audio_df['filepath'].apply(lambda path: librosa.load(path, sr=sampling_rate)[0])

    return audio_df, labels_df, test_files

def get_dataloaders(X_train, X_val, y_train, y_val, batch_size=128):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

    return train_loader, val_loader

def save_model(model, path="saved_models/bvae.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[✔] Modelo guardado en: {path}")

def load_model(path="saved_models/bvae.pt"):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = BetaVAE().to(device)  # ← Usá la misma arquitectura que al entrenar
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"[✔] Modelo cargado desde: {path}")
    return model

def save_aae_models(encoder, decoder, discriminator, path="saved_models/aae"):
    os.makedirs(path, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(path, "encoder.pt"))
    torch.save(decoder.state_dict(), os.path.join(path, "decoder.pt"))
    torch.save(discriminator.state_dict(), os.path.join(path, "discriminator.pt"))
    print(f"[✔] AAE guardado en: {path}")

def load_aae_models(latent_dim=32, path="saved_models/aae"):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    encoder = Encoder(latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    discriminator = Discriminator(latent_dim).to(device)

    encoder.load_state_dict(torch.load(os.path.join(path, "encoder.pt"), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(path, "decoder.pt"), map_location=device))
    discriminator.load_state_dict(torch.load(os.path.join(path, "discriminator.pt"), map_location=device))

    encoder.eval()
    decoder.eval()
    discriminator.eval()

    print(f"[✔] AAE cargado desde: {path}")
    return encoder, decoder, discriminator

def save_gan_models(gen, disc, path_disc=r"C:\Users\bianc\Machine\TPFINAL\saved_models\gan\disc.pth", path_gen=r"C:\Users\bianc\Machine\TPFINAL\saved_models\gan\gen.pth"):
    os.makedirs(os.path.dirname(path_disc), exist_ok=True)
    os.makedirs(os.path.dirname(path_gen), exist_ok=True)

    torch.save(disc.state_dict(), path_disc)
    torch.save(gen.state_dict(), path_gen)
    print(f"[✔] Discriminador guardado en: {path_disc}")
    print(f"[✔] Generador guardado en: {path_gen}")

def load_gan_models(device, z_dim=100,
                    path_disc=r"saved_models\gan\disc.pth",
                    path_gen=r"saved_models\gan\gen.pth"):

    gen = Generator(z_dim).to(device)
    disc = Discriminator().to(device)

    gen.load_state_dict(torch.load(path_gen, map_location=device))
    disc.load_state_dict(torch.load(path_disc, map_location=device))

    gen.eval()
    disc.eval()

    print(f"[✔] Modelos cargados desde:\n  - {path_gen}\n  - {path_disc}")
    return gen, disc