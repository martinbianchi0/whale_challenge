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
        super().__init__()
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
        nn.init.xavier_uniform_(final_linear.weight)
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

        # loss ponderada 
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
                print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}')

            if early_stopping_patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if show_progress:
                            print('Early stopping triggered (Val Loss did not improve).')
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
        plt.title('Confusion Matrix - MLP')
        plt.show()

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

def train_random_forest(X_train, y_train, X_val, y_val, n_estimators=100, max_depth=None, seed=42):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)

    acc = accuracy_score(y_val, preds)
    f1score = f1_score(y_val, preds)
    roc_auc = roc_auc_score(y_val, preds)

    print(f'Validation Accuracy (Random Forest): {acc:.4f}')
    print(f'Validation F1-Score (Random Forest): {f1score:.4f}')
    print(f'Validation ROC-AUC (Random Forest): {roc_auc:.4f}')
    return rf

def find_best_ensemble_hiperparams(ensemble_model, X, y, hiperparameters, n_folds=5, seed=42):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = []
    for n_estimators in hiperparameters['n_estimators']:
        for max_depth in hiperparameters['max_depth']:

            fold_aucs = []
            fold_f1s = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model = ensemble_model(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=seed,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                probs = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, probs)
                f1 = f1_score(y_val, preds)
                fold_aucs.append(auc)
                fold_f1s.append(f1)

            mean_auc = np.mean(fold_aucs)
            mean_f1 = np.mean(fold_f1s)
            results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'mean_auc': mean_auc,
                'mean_f1': mean_f1
            })

            print(f"N_estimators: {n_estimators}, Max Depth {max_depth} => Mean AUC: {mean_auc:.4f}, Mean F1: {mean_f1:.4f}")

    results = sorted(results, key=lambda x: (x['mean_auc'], x['mean_f1']), reverse=True)
    print("\nBest hyperparameters:")
    print(results[0])

# GRADIENT BOOSTING

def train_gradient_boosting(X_train, y_train, X_val, y_val):
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    preds = gb.predict(X_val)

    acc = accuracy_score(y_val, preds)
    f1score = f1_score(y_val, preds)
    roc_auc = roc_auc_score(y_val, preds)

    print(f'Validation Accuracy (Gradient Boosting): {acc:.4f}')
    print(f'Validation F1-Score (Gradient Boosting): {f1score:.4f}')
    print(f'Validation ROC-AUC (Gradient Boosting): {roc_auc:.4f}')
    return gb

# CONVOLUTIONAL NEURAL NETWORK

class ConvolutionalMLP(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()
        self.device = torch.device('mps' if torch.backends.mps.is_available() else
                                   'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1, 64, 64) -> (16, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # (16, 64, 64) -> (16, 32, 32)

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # -> (32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # -> (32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> (64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # -> (64, 8, 8)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)

    def train_model(self, train_loader, val_loader, epochs=50, lr=0.001, weight_decay=1e-5,
                    early_stopping_patience=None, use_class_weights=True, show_progress=True):
        self.to(self.device)

        if use_class_weights:
            y_train = train_loader.dataset.tensors[1].cpu().numpy().flatten()
            weights = get_class_weights(y_train).to(self.device)
            loss_function = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_function = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss, patience_counter = float('inf'), 0
        best_model_state = None

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

            if early_stopping_patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if show_progress:
                            print('Early stopping triggered (val loss did not improve).')
                        break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return self.train_losses, self.val_losses

    def evaluate(self, val_loader, return_metrics=False):
        self.eval()
        val_loss = 0.0
        all_labels, all_probs, all_preds = [], [], []

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
            auc = 0.0
        f1 = f1_score(all_labels, all_preds)

        if return_metrics:
            return avg_val_loss, acc * 100, f1, auc

        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {acc * 100:.2f}%, F1: {f1:.4f}, AUC: {auc:.4f}')

    def plot_learning_curves(self):
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

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
        plt.title("Confusion Matrix - CNN")
        plt.show()

def find_best_CNN_hiperparams(X, y, hiperparameters, epochs, seed):
    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    patience = 5

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

                model = ConvolutionalMLP()
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
                'learning_rate': lr,
                'weighted_loss': use_weights,
                'patience': patience,
                'mean_auc': mean_auc,
                'mean_f1': mean_f1
            })

            print(f'LR: {lr}, Weighted Loss: {use_weights}, Patience: {patience} => Mean AUC: {mean_auc:.4f}, Mean F1: {mean_f1:.4f}')

    results = sorted(results, key=lambda x: (x['mean_auc'], x['mean_f1']), reverse=True)

    print('\nBest hyperparameters:')
    print(results[0])

