from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch

def get_mlp_probs(model, val_loader, device):
    """Predice probabilidades de clase positiva con un MLP de PyTorch"""
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
    """Predice probabilidades de clase positiva con un modelo sklearn"""
    return model.predict_proba(X_val)[:, 1]

def compute_roc(y_true, y_probs):
    """Calcula AUC, FPR y TPR"""
    auc = roc_auc_score(y_true, y_probs)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    return auc, fpr, tpr

def plot_roc_curves(roc_data_dict):
    """
    Plotea curvas ROC.
    roc_data_dict: dict con keys = nombres de modelos, values = (auc, fpr, tpr)
    """
    plt.figure(figsize=(8, 6))
    for name, (auc, fpr, tpr) in roc_data_dict.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
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
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.show()