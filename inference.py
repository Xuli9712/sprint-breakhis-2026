import os
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import csv
from tqdm import tqdm

MODEL_PATH = r"F:\sprint2breastcancer\checkpoints\train_3\resnet18_fold1.pth"
IMG_FOLDER_PATH = r"F:\sprint2breastcancer\testimg"
SAVE_FOLDER_PATH = r"output"

THRES = 0.8 # if lower than the threshold, the result should be double checked by human expert

SAVE_CSV_PATH = os.path.join(SAVE_FOLDER_PATH, "results.csv")
SAVE_CSV_HEADER = ["image_path", "prediction", "confidence"]

MEAN = [0.786, 0.626, 0.764]
STD = [0.134, 0.181, 0.119]
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU or CPU
CLASS_NAMES = ["Benign", "Malignant"] # Map class name -- 0:belign. 1:Malignant


# Preprocess
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ResNet18 model
def build_model():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model = model.to(DEVICE)
    return model

# Inference
# No Training
@torch.no_grad()
def detect_single_image(img_path, model, save_path):
    start_time = time.time()
    #  BGR2RGB
    bgr_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_img)
    # preprocess
    input_tensor = test_transform(img).unsqueeze(0).to(DEVICE)
    # predict
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()

    single_image_process_time = time.time() - start_time
    pred_name = CLASS_NAMES[pred_idx]

    img_copy = bgr_img.copy()
    text = f"{pred_name} (Conf: {confidence:.2%})"
    if confidence <= THRES:
        text += " Review required"
    cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255, 0, 0),2)
    cv2.imwrite(save_path, img_copy)
    print("Image path:", img_path)
    print("Predicted class:", pred_name)
    print(f"Confidence: {confidence:.4f}")
    print(f"Inference time: {single_image_process_time:.4f} s")
    return pred_name, confidence

def main(model_path, img_folder_path, save_folder_path, save_csv_path):
    results = []
    # load model
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print('Model loaded!')
    print('Detecting...')
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    # iterate all imgs in the input folder
    for img_name in os.listdir(img_folder_path):
        img_path = os.path.join(img_folder_path, img_name)
        save_img_path = os.path.join(save_folder_path, os.path.basename(img_path))
        # detect single image and save output result image
        pred_name, confidence = detect_single_image(model=model, img_path=img_path,  save_path=save_img_path)
        results.append([img_path, pred_name, confidence])
    # save results to results.csv
    with open(save_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SAVE_CSV_HEADER)   
        writer.writerows(results)    

if __name__ == "__main__":
    main(str(MODEL_PATH), str(IMG_FOLDER_PATH), str(SAVE_FOLDER_PATH), str(SAVE_CSV_PATH))