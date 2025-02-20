## Aircraft Classification using YOLOv8

## Project Overview
This project is part of the **Applied Machine Learning** coursework at the University of Surrey. The objective is to classify different types of military aircraft using **YOLOv8**, a state-of-the-art object detection model, on the **Military Aircraft Detection Dataset (Kaggle)**. The model was trained to detect and classify aircraft in images with a focus on **accuracy, speed, and robustness**.

---

## Features
- **Aircraft Detection & Classification**: Identifies various military aircraft types.
- **YOLOv8-based Object Detection**: Implements a fast and efficient detection pipeline.
- **Baseline vs. Enhanced Model Comparison**: Evaluates the impact of data augmentation.
- **Performance Metrics**: Precision, recall, mAP50, and mAP50-95.
- **Data Augmentation**: Mosaic, mixup, and transformations to improve accuracy.

---

## Dataset: Military Aircraft Detection Dataset
The dataset consists of labeled images of various military aircraft, categorized into different classes.
- **Classes include**: F16, F18, B1, A10, Su57, J20, KC135, RQ4, V22, E2, Tu95, and more.
- **Dataset Splits**:
  - **Training Set**: 80%
  - **Validation Set**: 10%
  - **Test Set**: 10%

---

## Technologies Used
- **Programming Language**: Python
- **Libraries & Frameworks**:
  - PyTorch, TensorFlow
  - Ultralytics YOLOv8
  - OpenCV, NumPy, Pandas
  - Matplotlib, Seaborn (for visualization)

---

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the model:
   ```bash
   python train.py
   ```
5. Evaluate the model:
   ```bash
   python evaluate.py
   ```

---

## Model Experiments & Results

### 1. Baseline Model Performance
| Metric       | Score  |
|-------------|--------|
| Precision   | 0.381  |
| Recall      | 0.578  |
| mAP50       | 0.428  |
| mAP50-95    | 0.389  |

- **Challenges:**
  - Misclassification of visually similar aircraft (e.g., F16 vs. F18).
  - Imbalanced dataset affecting minority class performance.

### 2. Enhanced Model (with Data Augmentation)
| Metric       | Score  |
|-------------|--------|
| Precision   | 0.409  |
| Recall      | 0.653  |
| mAP50       | 0.653  |
| mAP50-95    | 0.389  |

- **Improvements:**
  - **Higher recall and mAP50** after data augmentation.
  - **Better generalization** on unseen data.
  - **Challenges remained** for visually similar aircraft types.

### 3. Data Augmentation Techniques Used
- **Mosaic Augmentation**: Merges four images into one for richer context.
- **Mixup Augmentation**: Blends multiple images for improved generalization.
- **Random Flips, Rotations, Color Adjustments**: Prevents overfitting.

### 4. Confusion Matrix Insights
- **Common Misclassifications**: Aircraft with similar shapes and sizes (e.g., F16 & F18).
- **Precision-Recall Curves**: Showed improvements in precision at moderate recall values.

---

## Best Model for Deployment
**Final Selected Model**: **YOLOv8 (Enhanced with Data Augmentation)**

- **Optimized for better recall and precision.**
- **Handles real-time inference efficiently.**
- **mAP50 increased from 0.428 to 0.653.**

---

## Future Work & Improvements
- **Handle Class Imbalance**: Implement **class reweighting or oversampling**.
- **Improve Feature Extraction**: Utilize **attention mechanisms or Transformer-based models**.
- **Optimize for Real-Time Deployment**: Experiment with **quantization and model pruning**.
- **Test on Larger Datasets**: Expand dataset for better generalization.

---

## Team Members & Contributions
- **Alisha Barathi Jaitu**
- **Bei Xu**
- **Munalisa Paul**
- **Saksham Ashwini Rai**

---

## License
This project is for educational purposes and is licensed under **MIT License**.

---

## Acknowledgments
Special thanks to **Dr. Aaron Wing** for guidance and to the University of Surrey for providing resources for this coursework.

