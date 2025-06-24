import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import copy 

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[256, 128, 64], output_dim=2):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_layer in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_layer))
            layers.append(nn.ReLU())
            prev_dim = hidden_layer
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_loader, val_loader, epochs=50, lr=0.001, weight_decay=1e-5,
                    early_stopping_patience=None, use_class_weights=True):
        self.to(self.device)

        # Loss function
        if use_class_weights:
            y_train = train_loader.dataset.tensors[1].cpu().numpy().flatten()
            weights = get_class_weights(y_train).to(self.device)
            loss_function = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_function = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_f1 = 0.0
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

            print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | '
                f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}')

            # Early stopping con F1
            if early_stopping_patience is not None:
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print("Early stopping triggered (F1 no mejoró).")
                        break

        # Restaurar el mejor modelo
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return self.train_losses, self.val_losses
    
    def plot_learning_curves(self)
    # poner esto adentro del modelo como plot_learning_curves
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print('Using device:', device)

def train_mlp_with_maxfreqs(audio_train_df, MLP, get_all_mel_spectrograms, get_class_weights,
                                    max_freq_list=[500, 600, 1000], n_folds=5, patience=3,
                                    batch_size=128, learning_rate=1e-3, seed=42):

    original_max_freq = globals().get("MAX_FREQ", None)
    SEED = seed

    for max_frequency in max_freq_list:
        print(f'\n--- MAX_FREQ = {max_frequency} HZ ---')
        globals()["MAX_FREQ"] = max_frequency

        X = get_all_mel_spectrograms(audio_train_df)
        y = audio_train_df['label'].values

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

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

                mlp = MLP(X_train.shape[1]).to(device)
                if weighted:
                    weights = get_class_weights(y_train).to(device)
                    loss_function = nn.CrossEntropyLoss(weight=weights)
                else:
                    loss_function = nn.CrossEntropyLoss()
                optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)

                best_auc = 0
                counter = 0

                for epoch in range(20):
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
                    print(f"Epoch {epoch+1} - AUC: {auc:.4f}")

                    if auc > best_auc:
                        best_auc = auc
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patience:
                            print("Early stopping")
                            break

                print(f"AUC MLP final (fold {fold+1}): {best_auc:.4f}")
                y_pred = [1 if p > 0.5 else 0 for p in all_probs]
                f1 = f1_score(all_targets, y_pred)
                print(f"F1-SCORE (fold {fold+1}): {f1:.4f}")

                fold_aucs.append(best_auc)
                fold_f1s.append(f1)

            print(f'\nMean AUC over {n_folds} folds: {np.mean(fold_aucs):.4f}')
            print(f'Mean F1-Score over {n_folds} folds: {np.mean(fold_f1s):.4f}')

    if original_max_freq is not None:
        globals()["MAX_FREQ"] = original_max_freq

# train_mlp_with_maxfreqs(
#     audio_train_df=audio_train_df,
#     MLP=MLP,
#     get_all_mel_spectrograms=get_all_mel_spectrograms,
#     get_class_weights=get_class_weights,
#     max_freq_list=[500, 600, 1000],
#     n_folds=5,
#     patience=3
# )

    
# ESTO DE ABAJO NO ANDA TODAVIA PERO ES PARA YA TENERLO
def MLP_cross_val(train_loader:DataLoader, val_loader:DataLoader, input_dim:int, hidden_layers:list, output_dim:int, epochs:int, lr:list, weight_decay:list, regularization_term:list, early_stopping_patience:list):
    best_model = None
    best_val_loss = float('inf')
    best_params = None
    global_min, global_max = compute_global_min_max(train_loader.dataset)
    for lr_val in lr:
        for wd in weight_decay:
            for reg in regularization_term:
                model = MLP(input_dim, hidden_layers, output_dim)
                model.train(train_loader, val_loader, epochs=epochs, lr=lr_val, weight_decay=wd)

                # Evaluate on validation set
                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(model.device), labels.to(model.device)
                        outputs = model(inputs)
                        loss = nn.CrossEntropyLoss()(outputs, labels)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
                    best_params = (lr_val, wd, reg)
    print(f'Best Validation Loss: {best_val_loss:.4f} with params: LR={best_params[0]}, WD={best_params[1]}, Reg={best_params[2]}')
    return best_model, best_params    

def train_random_forest(X_train, y_train, X_val, y_val):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Validation Accuracy (Random Forest): {acc:.4f}")
    return rf

def train_gradient_boosting(X_train, y_train, X_val, y_val):
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    preds = gb.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Validation Accuracy (Gradient Boosting): {acc:.4f}")
    return gb