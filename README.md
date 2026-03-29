# sprint-breakhis-2026
Breast cancer histopathology image classification on BreakHis using hand-crafted and deep learning methods.

First please download the BreakHis dataset from a public source. You will get the dataset structure under .../archive/BreaKHis_v1, along with the official split file .../archive/Folds.csv. Load the data and Folds.csv to enable person-independent 5-fold cross-validation.

Use LBP-SVM.py to train and validate the data with different LBP settings. Open the script and set root_path to the BreakHis dataset folder (e.g. .../archive) and folds_csv to the split file (e.g. .../archive/Folds.csv). The output location can be set using save_dir. You can change the LBP mode using MODE ('grayscale' or 'rgb'). You can also change SCALE, which controls the LBP radius settings, for example [1, 2, 3] for multi-scale LBP.

Use ResNet18.py to train and validate the BreakHis dataset. Open the script and set root to the dataset path (e.g. .../archive) and folds_csv to the split file (e.g. .../archive/Folds.csv). The trained model will be saved to the folder specified by save_dir. Training parameters can be modified directly in the script if needed.

Use inference.py to predict histopathology images in an end-to-end way. Open the script and set MODEL_PATH to your trained model (e.g. .../checkpoints/train_3/resnet18_fold3.pth), IMG_FOLDER_PATH to the input image folder, and SAVE_FOLDER_PATH to the output folder. The script will process all images in the input folder and save the results to the specified output location. The output in the specified folder includes prediction images with the predicted class label (Benign or Malignant) and confidence score overlaid. If the confidence score is lower than the predefined threshold in the code (THRES, e.g. 0.8), a warning message (Review required!) will be displayed. In addition, a summary CSV file (results.csv) is generated.

<img width="700" height="460" alt="SOB_B_A-14-29960CD-400-010" src="https://github.com/user-attachments/assets/efeba993-dbf7-4963-b7aa-ca70fb49e082" />

<img width="700" height="460" alt="SOB_M_DC-14-5287-40-013" src="https://github.com/user-attachments/assets/353b64de-3a76-4044-a512-5b334d82fde7" />

results.csv format:
image_path,prediction,confidence
xxx.png,Benign,0.92
xxx.png,Malignant,0.85
