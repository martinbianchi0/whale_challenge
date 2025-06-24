from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from ..preprocessing import *
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import copy 

# MULTI-LAYER PERCEPTRON

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[256, 128, 64], output_dim=2):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_layer in hidden_layers:
            linear = nn.Linear(prev_dim, hidden_layer)
            nn.init.xavier_uniform_(linear.weight)  # glorot init
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            prev_dim = hidden_layer

        final_linear = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_uniform_(final_linear.weight)  # glorot init
        nn.init.zeros_(final_linear.bias)
        layers.append(final_linear)
        self.model = nn.Sequential(*layers)

        self.device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_loader, val_loader, epochs=50, lr=0.0001, weight_decay=1e-5, early_stopping_patience=None, use_class_weights=True, show_progress=True):
        self.to(self.device)

        # loss
        if use_class_weights:
            y_train = train_loader.dataset.tensors[1].cpu().numpy().flatten()
            weights = get_class_weights(y_train).to(self.device)
            loss_function = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_function = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            avg_train_loss = running_loss / len(train_loader.dataset)
            self.train_losses.append(avg_train_loss)

            val_loss, val_acc, val_f1, val_auc = self.evaluate(val_loader, return_metrics=True)
            self.val_losses.append(val_loss)

            if show_progress:
                print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | '
                    f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}')

            # early stopping based on val_loss
            if early_stopping_patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if show_progress:
                            print("Early stopping triggered (Val Loss did not improve).")
                        break

        # Restaurar el mejor modelo
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return self.train_losses, self.val_losses
    
    def plot_learning_curves(self):
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def evaluate(self, val_loader, return_metrics=False):
        self.eval()
        val_loss = 0.0
        all_labels = []
        all_probs = []
        all_preds = []

        loss_function = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0  # puede fallar si hay una sola clase
        f1 = f1_score(all_labels, all_preds)

        if return_metrics:
            return avg_val_loss, acc * 100, f1, auc

        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {acc * 100:.2f}%, F1: {f1:.4f}, AUC: {auc:.4f}')

    def confusion_matrix(self, val_loader):
        self.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix - MLP")
        plt.show()

def test_maxfreq(audio_df, model, spectrogram_config:dict, max_freq_list=[500, 600, 1000], epochs=30, n_folds=5, patience=3, batch_size=128, learning_rate=1e-4, seed=42):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    new_spectrogram_config = spectrogram_config.copy()

    for max_frequency in max_freq_list:
              
        new_spectrogram_config['MAX_FREQ'] = max_frequency
        print(f'\n--- MAX_FREQ = {max_frequency}Hz ---')

        X = get_all_mel_spectrograms(audio_df, new_spectrogram_config)
        y = audio_df['label'].values

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for weighted in [False, True]:
            print(f'\n--- WEIGHTED LOSS = {weighted} ---')
            fold_aucs = []
            fold_f1s = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                print(f'\nFold {fold+1}/{n_folds}')
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.long)
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val, dtype=torch.long)

                train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

                mlp = model(X_train.shape[1]).to(device)

                if weighted:
                    weights = get_class_weights(y_train).to(device)
                    loss_function = nn.CrossEntropyLoss(weight=weights)
                else:
                    loss_function = nn.CrossEntropyLoss()
                optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)

                best_auc = 0
                counter = 0

                for epoch in range(epochs):
                    mlp.train()
                    for xb, yb in train_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        optimizer.zero_grad()
                        loss = loss_function(mlp(xb), yb)
                        loss.backward()
                        optimizer.step()

                    mlp.eval()
                    all_probs, all_targets = [], []
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb = xb.to(device)
                            probs = torch.softmax(mlp(xb), dim=1)[:, 1].cpu().numpy()
                            all_probs.extend(probs)
                            all_targets.extend(yb.numpy())

                    auc = roc_auc_score(all_targets, all_probs)
                    # print(f"Epoch {epoch+1} - AUC: {auc:.4f}")

                    if auc > best_auc:
                        best_auc = auc
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patience:
                            print(f'Early stopping at epoch {epoch+1}')
                            break

                print(f"AUC MLP final (fold {fold+1}): {best_auc:.4f}")
                y_pred = [1 if p > 0.5 else 0 for p in all_probs]
                f1 = f1_score(all_targets, y_pred)
                print(f"F1-SCORE (fold {fold+1}): {f1:.4f}")

                fold_aucs.append(best_auc)
                fold_f1s.append(f1)

            print(f'\nMean AUC over {n_folds} folds: {np.mean(fold_aucs):.4f}')
            print(f'Mean F1-Score over {n_folds} folds: {np.mean(fold_f1s):.4f}')

def find_best_MLP_hiperparams(train_df, hiperparameters, spectrogram_config, epochs, seed):
    results = []
    X = get_all_mel_spectrograms(train_df, spectrogram_config)
    y = train_df['label'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    patience = 5

    for arch in  hiperparameters['architecture']:
        for lr in hiperparameters['learning_rates']:
            for use_weights in hiperparameters['weighted_loss']:

                fold_aucs = []
                fold_f1s = []

                for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
                    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

                    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=128, shuffle=True)
                    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=128)

                    model = MLP(input_dim=X_train.shape[1], hidden_layers=arch)
                    model.train_model(
                        train_loader, val_loader,
                        epochs=epochs, lr=lr, weight_decay=1e-5,
                        early_stopping_patience=patience, use_class_weights=use_weights, show_progress=False
                    )
                    val_loss, val_acc, val_f1, val_auc = model.evaluate(val_loader, return_metrics=True)
                    fold_aucs.append(val_auc)
                    fold_f1s.append(val_f1)

                mean_auc = np.mean(fold_aucs)
                mean_f1 = np.mean(fold_f1s)
                results.append({
                    'architecture': arch,
                    'learning_rate': lr,
                    'weighted_loss': use_weights,
                    'patience': patience,
                    'mean_auc': mean_auc,
                    'mean_f1': mean_f1
                })
                print(f"Arch: {arch}, LR: {lr}, Weighted Loss: {use_weights}, Patience: {patience} => Mean AUC: {mean_auc:.4f}, Mean F1: {mean_f1:.4f}")

    # sort by mean_auc, then mean_f1
    results = sorted(results, key=lambda x: (x['mean_auc'], x['mean_f1']), reverse=True)
    print("\nBest hyperparameters:")
    print(results[0])

# RANDOM FOREST

def train_random_forest(X_train, y_train, X_val, y_val, n_estimators, max_depth, seed):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed, verbose=1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f'Validation Accuracy (Random Forest): {acc:.4f}')
    return rf

def find_best_RF_hiperparams(X, y, param_grid, n_folds=5, seed=42):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = []

    for params in ParameterGrid(param_grid):
        fold_aucs = []
        fold_f1s = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            rf = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=seed,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            preds = rf.predict(X_val)
            probs = rf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, probs)
            f1 = f1_score(y_val, preds)
            fold_aucs.append(auc)
            fold_f1s.append(f1)
        mean_auc = np.mean(fold_aucs)
        mean_f1 = np.mean(fold_f1s)
        results.append({
            'params': params,
            'mean_auc': mean_auc,
            'mean_f1': mean_f1
        })
        print(f"Params: {params} => Mean AUC: {mean_auc:.4f}, Mean F1: {mean_f1:.4f}")

    results = sorted(results, key=lambda x: (x['mean_auc'], x['mean_f1']), reverse=True)
    print("\nBest hyperparameters:")
    print(results[0])
    return results[0]

# RANDOM FOREST

def train_gradient_boosting(X_train, y_train, X_val, y_val):
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    preds = gb.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Validation Accuracy (Gradient Boosting): {acc:.4f}")
    return gb