#  CliniScan – Lung Abnormality Detection on Chest X-rays using AI

CliniScan is an AI-powered medical imaging system designed to detect and classify lung abnormalities from chest X-rays.  
This project uses **two separate deep learning pipelines**:

- **Classification Model** → Predicts lung diseases  
- **Detection Model** → Locates abnormalities using bounding boxes  

This dual-model system helps support doctors, radiologists, telemedicine platforms, and rural clinics by offering fast, accurate, and automated analysis.

---

##  Features

### Multi-label Classification  
Predicts multiple lung diseases using:
- **ResNet50**
- **EfficientNet**

###  Object Detection  
Locates abnormalities using:
- **YOLOv8**
- **Faster R-CNN**

###  Complete Pipeline  
- Preprocessing (resize, normalize, denoise)  
- COCO & YOLO annotation conversion  
- Label CSV generation  
- Hyperparameter tuning  
- Model training  
- Streamlit-based deployment  

###  Deployment  
Fully deployed using **Streamlit in VS Code**.  
Users can upload an X-ray → get disease predictions + bounding-box visualizations.

---

## 🗂 Project Structure
A:\HRITHIK
│
├── main.py
├── requirements.txt
│--readme.md
├── app
│ ├── classification.py
│ ├── detection.py
│ └── utils.py
│
├── models
│ ├── resnet50_best.pth
│ ├── efficientnet_best.pth (optional)
│ ├── best.pt (YOLOv8)
│ └── faster_rcnn_best.pth
│
└── venv\


---

##  Dataset – NIH Chest X-ray Dataset

This project uses the **NIH Chest X-ray Dataset**, containing thousands of images labeled with **15 lung diseases**, including:

- Atelectasis  
- Cardiomegaly  
- Consolidation  
- Edema  
- Effusion  
- Emphysema  
- Fibrosis  
- Hernia  
- Infiltration  
- Mass  
- Nodule  
- Pneumonia  
- Pleural Thickening  
- Pneumothorax  
- Lung Opacity  

---

## ⚙️ Technical Workflow

### **1️ Preprocessing**
- Resizing  
- Normalization  
- CLAHE / Denoising  
- Data augmentation  
- CSV label generation  
- COCO → YOLO annotation conversion  

### **2️ Classification Pipeline**
- Model: **ResNet50 / EfficientNet**
- Loss: `BCEWithLogitsLoss`
- Optimizer & LR tuning
- Trained for **10 epochs**
- Outputs: multi-label disease predictions

### **3️ Detection Pipeline**
- Model: **YOLOv8 / Faster R-CNN**
- Uses COCO annotations
- Multi-stage training: **10 + 10 + 50 epochs**
- Outputs: bounding boxes + confidence scores

### **4️ Hyperparameter Tuning**
Both pipelines tuned for:
- Learning rate  
- Batch size  
- Epochs  
- Image size  
- IoU threshold (for detection)  
- Confidence threshold (YOLO)

### **5️ Deployment**
- Streamlit-based UI
- Image upload
- Real-time disease prediction
- Abnormality localization
- Combined diagnostic output

---

##  How to Run the Project

### **1. Create Virtual Environment**
python -m venv venv
venv\Scripts\activate

### **2. Install Dependencies**
pip install -r requirements.txt

### **3. Run the Streamlit App**
python main.py

### **4. Upload Any Chest X-ray**
The app displays:
- Predicted lung diseases  
- Bounding boxes of abnormal regions  

---

##  Output Examples

- **Disease classification:**  
  Shows probabilities for all 15 lung diseases  
- **Object detection:**  
  YOLOv8/Faster R-CNN draws bounding boxes on infected regions  
- **Final diagnostic report:**  
  Disease list + localized abnormality visualization  

---

##  Conclusion

CliniScan is a complete medical AI system that combines:
- Deep learning  
- Image classification  
- Object detection  
- COCO/YOLO annotation handling  
- Hyperparameter tuning  
- Real-world deployment  

It provides fast, reliable AI-assisted diagnosis to support healthcare systems, especially in rural and telemedicine environments.

---

##  Developer

**Nikhil Sanga**

---

##  Future Enhancements

- Grad-CAM heatmaps  
- Mobile app integration  
- Cloud deployment (Render / HuggingFace Spaces)  
- Improved mAP and F1 scores  
- Support for CT scans  

---

##  License

This project is for educational and research purposes.


