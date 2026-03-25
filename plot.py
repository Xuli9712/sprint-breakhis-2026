import matplotlib.pyplot as plt
import os
import numpy as np

def plot_curves(history, fold, save_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Fold {fold} Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"fold{fold}_acc_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold} Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"fold{fold}_loss_curve.png"), dpi=300)
    plt.close()


def plot_confusion_matrix(cm, fold, save_path, class_names=None):
    if class_names is None:
        class_names = ["Benign", "Malignant"]

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Fold {fold} Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"fold{fold}_confusion_matrix.png"), dpi=300)
    plt.close()
