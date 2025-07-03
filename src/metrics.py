from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch

def get_mlp_probs(model, val_loader, device:str):
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            outputs = model(xb)
            p = F.softmax(outputs, dim=1)[:, 1]  # clase positiva
            probs.extend(p.cpu().numpy())
    return np.array(probs)

def get_model_probs_sklearn(model, X_val):
    return model.predict_proba(X_val)[:, 1]

def compute_roc(y_true, y_probs):
    auc = roc_auc_score(y_true, y_probs)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    return auc, fpr, tpr

def plot_roc_curves(models, val_loader, val_loader_cnn, X_val_np, y_val_np):

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    roc_data = {}
    for model in models:
        if (model == 'MLP'):
            mlp_probs = get_mlp_probs(models[model], val_loader, device)
            mlp_auc, mlp_fpr, mlp_tpr = compute_roc(y_val_np, mlp_probs)
            roc_data[model] = (mlp_auc, mlp_fpr, mlp_tpr)
        elif (model == 'Convolutional MLP'):
            cnn_probs = get_mlp_probs(models[model], val_loader_cnn, device)
            cnn_auc, cnn_fpr, cnn_tpr = compute_roc(y_val_np, cnn_probs)
            roc_data[model] = (cnn_auc, cnn_fpr, cnn_tpr)
        else:
            model_probs = get_model_probs_sklearn(models[model], X_val_np)
            model_auc, model_fpr, model_tpr = compute_roc(y_val_np, model_probs)
            roc_data[model] = (model_auc, model_fpr, model_tpr)

    plt.figure(figsize=(6, 4))

    for name, (auc, fpr, tpr) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{name} [AUC = {auc:.3f}]')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_confusion_matrix(model, X_val, y_val):
    y_pred = model.predict(X_val)
    cmatrix = confusion_matrix(y_val, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cmatrix)
    display.plot(cmap='Blues')
    plt.show()
