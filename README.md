# sprint-breakhis-2026
Breast cancer histopathology image classification on BreakHis using hand-crafted and deep learning methods.

First please download the BreakHis dataset from a public source. You will get the dataset structure under archive, along with the official split file Folds.csv. Load the data and Folds.csv to enable person-independent 5-fold cross-validation.

LBP + SVM (Hand-crafted)
Use LBP-SVM.py to train and validate the data with different LBP settings. Open the script and set root_path to the BreakHis dataset folder (e.g. .../archive) and folds_csv to the split file (e.g. .../archive/Folds.csv). The output location can be set using save_dir. You can change the LBP mode using MODE ('grayscale' or 'rgb'). You can also change SCALE, which controls the LBP radius settings, for example [1, 2, 3] for multi-scale LBP.

ResNet18 (Deep Learning)
Use ResNet18.py to train and validate the BreakHis dataset. Open the script and set root to the dataset path (e.g. .../archive) and folds_csv to the split file (e.g. .../archive/Folds.csv). The trained model will be saved to the folder specified by save_dir. Training parameters can be modified directly in the script if needed.

Inference
Use inference.py to predict histopathology images in an end-to-end way. Open the script and set model_path to your trained model (e.g. .../checkpoints/train_3/resnet50_fold3.pth), img_folder_path to the input image folder, and save_folder_path to the output folder. The script will process all images in the input folder and save the results to the specified output location.
