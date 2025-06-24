from torch.utils.data import TensorDataset, DataLoader
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