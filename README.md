**Chest X-Ray Pneumonia Detection with PyTorch**

This repository contains a Jupyter Notebook (chest-xray.ipynb) that implements a deep learning model to classify chest X-ray images for pneumonia detection. The model uses a pre-trained MobileNetV2 architecture, fine-tuned on the Chest X-Ray Images (Pneumonia) dataset from Kaggle. The notebook demonstrates data preprocessing, model training, evaluation, and visualization of results, achieving robust performance in distinguishing between normal and pneumonia-affected chest X-rays.


Dataset: Utilizes the Kaggle Chest X-Ray Pneumonia dataset with images labeled as NORMAL or PNEUMONIA.

Model: Employs MobileNetV2 with transfer learning, fine-tuned for binary classification.

Data Augmentation: Includes random rotations, flips, affine transformations, and color jitter to improve model generalization.

Class Imbalance Handling: Computes class weights to address the imbalanced dataset.

Evaluation Metrics: Includes accuracy, confusion matrix, classification report.

Visualizations: Displays class distribution and model performance metrics.


The dataset is sourced from Kaggle and consists of chest X-ray images divided into training and test sets. The training set is imbalanced, with a higher number of pneumonia cases compared to normal cases, as visualized in the class distribution pie chart below:

**Figure 1: Pie chart showing the class distribution of the training dataset (NORMAL vs. PNEUMONIA)**
<img width="604" height="634" alt="image" src="https://github.com/user-attachments/assets/3e3f04e0-39c0-4c85-ae17-265909ff9126" />

**The notebook follows these key steps:**

Data Preprocessing:

Images are resized to 224x224 pixels.

Training data undergoes augmentation (random rotation, flips, affine transformations, color jitter, and random erasing) to prevent overfitting.

Both training and test data are normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).


Model Architecture:

MobileNetV2, pre-trained on ImageNet, is used as the base model.

The classifier head is modified for binary classification (NORMAL vs. PNEUMONIA).


Training:

Uses cross-entropy loss with class weights to handle imbalance.

Employs the Adam optimizer with a learning rate scheduler (ReduceLROnPlateau).

Training is performed on a GPU (if available) for efficiency.


Evaluation:

Metrics include accuracy, confusion matrix, classification report

Visualizations of the confusion matrix and ROC curve are generated for analysis.

Results

The model achieves strong performance on the test set, with detailed metrics provided in the notebook. Below is an example of the confusion matrix visualizing the model's predictions:

**Figure 2: Confusion matrix showing the model's performance on the test dataset**
<img width="650" height="584" alt="image" src="https://github.com/user-attachments/assets/6dcbe2b0-901a-4b2d-b742-314340446e86" />
