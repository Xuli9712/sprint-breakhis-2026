import pandas as pd
import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image

# 0: benign    1: malignant 
# numpy form for traditional (LBP) methods
class BreakHisDataset:

    def __init__(self, root_path, folds_csv, fold, split):
        self.root_path = Path(root_path)
        df = pd.read_csv(folds_csv)
        df = df[(df.fold == fold) & (df.grp == split)]
        self.datapoints = []
        for _, r in df.iterrows():
            img_path = self.root_path / r.filename
            label = 1 if "malignant" in r.filename else 0
            self.datapoints.append((img_path, label))

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        img_path, label = self.datapoints[idx]
        img = cv2.imread(str(img_path))
        return img, label
    
# tensor form for DL methods
class BreakHisDLDataset(Dataset):
    
    def __init__(self, root_path, folds_csv, fold, split, transform=None):
        self.root_path = Path(root_path)
        df = pd.read_csv(folds_csv)
        df = df[(df.fold == fold) & (df.grp == split)]
        self.datapoints = []
        for _, r in df.iterrows():
            img_path = self.root_path / r.filename
            label = 1 if "malignant" in r.filename else 0
            self.datapoints.append((img_path, label))
        self.transform = transform

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        path, label = self.datapoints[idx]
        img = cv2.imread(str(path))
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label