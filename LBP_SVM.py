import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, ConfusionMatrixDisplay
from dataset import BreakHisDataset

# breakhis dataset and split csv path
root_path = r"F:\sprint2breastcancer\archive"
folds_csv = r"F:\sprint2breastcancer\archive\Folds.csv"
# resize input
IMG_SIZE = 224
# specify RGB LBP or grayscale LBP method
MODE = 'grayscale'
# Specify LBP radius settings
SCALE = [1, 2, 3]
# result folder
save_dir = "traditional_results/train"
os.makedirs(save_dir, exist_ok=True)

# extract multi-scale and multi-channel LBP features from input image
def extract_feat(img, mode):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    feats = []
    if mode == 'grayscale':
        # convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 1. LBP features
        for r in SCALE:
            lbp = local_binary_pattern(
                gray_img,
                P=16,
                R=r,
                method="uniform"
            )
            hist, _ = np.histogram(
                lbp.ravel(),
                bins=18,
                range=(0, 18),
                density=True
            )
            feats.extend(hist)
    
    elif mode == 'RGB':
        # extract LBP histogram from each colour channel(Opencv img form (BGR))
        for c in range(3):
            channel = img[:, :, c]

            for r in [1, 2, 3]:
                lbp = local_binary_pattern(
                    channel,
                    P=16,
                    R=r,
                    method="uniform"
                )
                hist, _ = np.histogram(
                    lbp.ravel(),
                    bins=18,
                    range=(0, 18),
                    density=True
                )
                feats.extend(hist)
    else:
        print("Please specify the input image mode (RGB or grayscale)")

    feats = np.array(feats, dtype=np.float32)
    #print("Feature shape:", feats.shape)
    return feats

accs = []
precisions = []
recalls = []
f1s = []

print("\nBreakHis-LBP-SVM\n")
print("\n5-fold CV\n")
for fold in range(1, 6):
    print(f"\nFold {fold}")

    train_ds = BreakHisDataset(root_path, folds_csv, fold, "train")
    test_ds = BreakHisDataset(root_path, folds_csv, fold, "test")

    X_train, y_train = [], []
    X_test, y_test = [], []

    print("Extracting train features...")
    for img, label in train_ds:
        X_train.append(extract_feat(img, MODE))
        y_train.append(label)

    print("Extracting test features...")
    for img, label in test_ds:
        X_test.append(extract_feat(img, MODE))
        y_test.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Train feature shape:", X_train.shape)
    print("Test feature shape :", X_test.shape)
    print("Train class distribution:", Counter(y_train))
    print("Test  class distribution:", Counter(y_test))

    # construct a SVM clf
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=None, random_state=20)
    )

    print("Training SVM...")
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
   
    # calculate metrics
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="binary", zero_division=0)
    recall = recall_score(y_test, pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, pred)

    print(f"Fold {fold} accuracy : {acc:.4f}")
    print(f"Fold {fold} precision: {precision:.4f}")
    print(f"Fold {fold} recall   : {recall:.4f}")
    print(f"Fold {fold} f1-score : {f1:.4f}")
    print(f"Fold {fold} confusion matrix:\n{cm}")

    accs.append(acc)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    # save confusion matrix figure
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Fold {fold} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fold_{fold}_cm.png"), dpi=300)
    plt.close()

print("\n Mean Result across folds")
print(f"Mean accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Mean precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Mean recall   : {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"Mean f1-score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

print("\nFold accuracy :", [round(x, 4) for x in accs])
print("Fold precision:", [round(x, 4) for x in precisions])
print("Fold recall   :", [round(x, 4) for x in recalls])
print("Fold f1-score :", [round(x, 4) for x in f1s])
print("\nConfusion matrix figures saved to:", save_dir)