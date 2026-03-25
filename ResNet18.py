import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
from dataset import BreakHisDLDataset
from plot import *


# dataset and folds split csv path
root = r"F:\sprint2breastcancer\archive"
folds_csv = r"F:\sprint2breastcancer\archive\Folds.csv"
save_dir = f"checkpoints/train_3"
os.makedirs(save_dir, exist_ok=True)

curve_dir = os.path.join(save_dir, "curves")
cm_dir = os.path.join(save_dir, "confusion_matrices")
log_dir = os.path.join(save_dir, "logs")
os.makedirs(curve_dir, exist_ok=True)
os.makedirs(cm_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# config
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-5
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = [0.786, 0.626, 0.764]
STD = [0.134, 0.181, 0.119]

# normalisation and augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# build a resnet model with pretrained weight for transfer learning
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    # Binary classification: ResNet18 + MLP
    model.fc = nn.Linear(in_features, 2)
    model = model.to(DEVICE)
    return model


def cal_metrics(labels_all, preds_all):
    acc = accuracy_score(labels_all, preds_all)
    precision = precision_score(labels_all, preds_all, average="binary", zero_division=0)
    recall = recall_score(labels_all, preds_all, average="binary", zero_division=0)
    f1 = f1_score(labels_all, preds_all, average="binary", zero_division=0)
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


# train one epoch
def train_one_epoch(model, loader, criterion, optimiser):

    model.train()

    total_loss = 0
    preds_all = []
    labels_all = []

    pbar = tqdm(loader, desc="Train", leave=False)

    for imgs, labels in pbar:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimiser.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        acc = accuracy_score(labels_all, preds_all)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    epoch_loss = total_loss / len(loader.dataset)
    metrics = cal_metrics(labels_all, preds_all)

    return epoch_loss, metrics


# evaluation
@torch.no_grad()
def evaluate(model, loader, criterion):

    model.eval()
    total_loss = 0
    preds_all = []
    labels_all = []

    pbar = tqdm(loader, desc="Eval ", leave=False)

    for imgs, labels in pbar:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        acc = accuracy_score(labels_all, preds_all)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    epoch_loss = total_loss / len(loader.dataset)
    metrics = cal_metrics(labels_all, preds_all)

    return epoch_loss, metrics, labels_all, preds_all


def main():

    print("\nBreakHis-ResNet\n")

    fold_results = []
    for fold in range(1, 6):
        print(f"\nFold {fold}\n")
        train_ds = BreakHisDLDataset(root, folds_csv, fold, "train",transform=train_transform)
        test_ds = BreakHisDLDataset(root, folds_csv, fold, "test", transform=test_transform)

        train_labels = [train_ds.datapoints[i][1] for i in range(len(train_ds))]
        test_labels = [test_ds.datapoints[i][1] for i in range(len(test_ds))]

        print("Train size:", len(train_ds))
        print("Test size :", len(test_ds))
        print("Train distribution:", Counter(train_labels))
        print("Test distribution :", Counter(test_labels))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = build_model().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_acc = 0
        best_epoch = 0

        history = {
            "train_loss": [],
            "train_acc": [],
            "train_precision": [],
            "train_recall": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": []
        }

        for epoch in range(EPOCHS):

            print(f"\nEpoch {epoch+1}/{EPOCHS}")

            train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimiser)
            val_loss, val_metrics, _, _ = evaluate(model, test_loader, criterion)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_metrics["acc"])
            history["train_precision"].append(train_metrics["precision"])
            history["train_recall"].append(train_metrics["recall"])
            history["train_f1"].append(train_metrics["f1"])

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_metrics["acc"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1"].append(val_metrics["f1"])

            print(
                f"Train Loss {train_loss:.4f} | "
                f"Train Acc {train_metrics['acc']:.4f} | "
                f"Train Precision {train_metrics['precision']:.4f} | "
                f"Train Recall {train_metrics['recall']:.4f} | "
                f"Train F1 {train_metrics['f1']:.4f}"
            )

            print(
                f"Val Loss {val_loss:.4f} | "
                f"Val Acc {val_metrics['acc']:.4f} | "
                f"Val Precision {val_metrics['precision']:.4f} | "
                f"Val Recall {val_metrics['recall']:.4f} | "
                f"Val F1 {val_metrics['f1']:.4f}"
            )

            if val_metrics["acc"] > best_acc:
                best_acc = val_metrics["acc"]
                best_epoch = epoch + 1
                torch.save(model.state_dict(), f"{save_dir}/resnet50_fold{fold}.pth")

        print("Best Fold Accuracy:", best_acc)

        # save log csv for this fold
        fold_log = pd.DataFrame({
            "epoch": list(range(1, EPOCHS + 1)),
            "train_loss": history["train_loss"],
            "train_acc": history["train_acc"],
            "train_precision": history["train_precision"],
            "train_recall": history["train_recall"],
            "train_f1": history["train_f1"],
            "val_loss": history["val_loss"],
            "val_acc": history["val_acc"],
            "val_precision": history["val_precision"],
            "val_recall": history["val_recall"],
            "val_f1": history["val_f1"]
        })
        fold_log.to_csv(os.path.join(log_dir, f"fold{fold}_log.csv"), index=False)

        # plot curves for this fold
        plot_curves(history, fold, curve_dir)

        # load best model and evaluate again
        best_model = build_model().to(DEVICE)
        best_model.load_state_dict(torch.load(f"{save_dir}/resnet18_fold{fold}.pth"))
        best_model.eval()

        test_loss, test_metrics, test_labels_all, test_preds_all = evaluate(best_model, test_loader, criterion)

        cm = confusion_matrix(test_labels_all, test_preds_all)
        plot_confusion_matrix(cm, fold, cm_dir)

        print(
            f"Best Model Test Acc {test_metrics['acc']:.4f} | "
            f"Precision {test_metrics['precision']:.4f} | "
            f"Recall {test_metrics['recall']:.4f} | "
            f"F1 {test_metrics['f1']:.4f}"
        )

        fold_results.append({
            "fold": fold,
            "best_epoch": best_epoch,
            "best_val_acc": best_acc,
            "test_loss": test_loss,
            "test_acc": test_metrics["acc"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"]
        })

    results_df = pd.DataFrame(fold_results)

    # compute mean and std across 5 folds
    mean_row = {
        "fold": "mean",
        "best_epoch": results_df["best_epoch"].mean(),
        "best_val_acc": results_df["best_val_acc"].mean(),
        "test_loss": results_df["test_loss"].mean(),
        "test_acc": results_df["test_acc"].mean(),
        "test_precision": results_df["test_precision"].mean(),
        "test_recall": results_df["test_recall"].mean(),
        "test_f1": results_df["test_f1"].mean()
    }

    std_row = {
        "fold": "std",
        "best_epoch": results_df["best_epoch"].std(),
        "best_val_acc": results_df["best_val_acc"].std(),
        "test_loss": results_df["test_loss"].std(),
        "test_acc": results_df["test_acc"].std(),
        "test_precision": results_df["test_precision"].std(),
        "test_recall": results_df["test_recall"].std(),
        "test_f1": results_df["test_f1"].std()
    }

    summary_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True
    )

    summary_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)

    print("\n========== Final Result ==========")
    print(summary_df)

    print("\nMean Acc:", mean_row["test_acc"])
    print("Std  Acc:", std_row["test_acc"])
    print("Mean Precision:", mean_row["test_precision"])
    print("Std  Precision:", std_row["test_precision"])
    print("Mean Recall:", mean_row["test_recall"])
    print("Std  Recall:", std_row["test_recall"])
    print("Mean F1:", mean_row["test_f1"])
    print("Std  F1:", std_row["test_f1"])

if __name__ == "__main__":
    main()