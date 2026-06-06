# 🩸LeukoNet: Blood Cell Cancer Classification using Deep Learning🩸
LeukoNet is an AI-powered blood cell cancer classification system that leverages Transfer Learning and Deep Learning to identify and classify blood cell images into four categories. The project includes model training notebooks, hyperparameter tuning experiments, and a Streamlit web application for real-time inference.

---

## 📌 Project Overview
Early and accurate detection of blood-related cancers is crucial for effective treatment. LeukoNet utilizes state-of-the-art Convolutional Neural Networks (CNNs) with Transfer Learning to classify microscopic blood cell images into:
- Benign
- Malignant Pre-B
- Malignant Pro-B
- Malignant Early Pre-B
The project evaluates multiple pre-trained architectures and deploys the best-performing model through an interactive Streamlit application.

---

## 🚀 Features
- 🔬 Blood cell image classification
- 🤖 Transfer Learning using popular CNN architectures
- ⚙️ Hyperparameter tuning for model optimization
- 📊 Performance evaluation and comparison
- 🌐 Streamlit-based web application
- 📁 Ready-to-use trained model (`best_model.keras`)

---

## 🏗️ Project Structure
```text
LeukoNet/
│
├── app.py                          # Streamlit web application
├── best_model.keras                # Trained classification model
├── requirements.txt                # Project dependencies
│
├── PBLVI_MobileNetV2.ipynb         # MobileNetV2 experiments
├── PBLVI_ResNet50.ipynb            # ResNet50 experiments
├── PBLVI_DenseNet121.ipynb         # DenseNet121 experiments
├── PBLVI_EfficientNetB0.ipynb      # EfficientNetB0 experiments
│
└── README.md
```

---

## 🧠 Models Evaluated
The following transfer learning architectures were explored:
| Model | Description |
|---------|------------|
| MobileNetV2 | Lightweight architecture optimized for efficiency |
| ResNet50 | Deep residual learning network |
| DenseNet121 | Feature reuse through dense connectivity |
| EfficientNetB0 | Compound-scaled CNN with high accuracy and efficiency |

Hyperparameter tuning was performed to identify the optimal configuration for classification performance.

---

## 📊 Dataset
The project utilizes the [**Blood Cell Cancer ALL 4-Class Dataset**](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class), consisting of microscopic blood cell images categorized into four classes:
- Benign
- Malignant Pre-B
- Malignant Pro-B
- Malignant Early Pre-B
  
Dataset preprocessing includes:
- Image resizing to 224×224
- Normalization
- Data augmentation
- Train/Validation/Test splitting

<img width="1654" height="540" alt="image" src="https://github.com/user-attachments/assets/25fbe7de-a13b-49f3-843a-612c2de3d8cf" />

---

## ⚙️ Installation
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/LeukoNet.git
cd LeukoNet
```
### 2. Create Virtual Environment (Optional)
```bash
python -m venv venv
```
#### Activate:

**Windows**
```bash
venv\Scripts\activate
```
**Linux/Mac**
```bash
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application
Start the Streamlit app:
```bash
streamlit run app.py
```
The application will open in your browser.

---

## 🖼️ Using the Application
1. Upload a blood cell image (`jpg`, `jpeg`, or `png`).
2. The model preprocesses the image.
3. Prediction is generated.
4. Classification probabilities are displayed for all classes.

Example Output:
```text
Prediction: [Malignant] Pre-B
Confidence: 97.83%
```

---

## 🔧 Technologies Used
- Python
- TensorFlow / Keras
- Keras Tuner
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit
- Pillow

---

## 📈 Model Pipeline
```text
Input Blood Cell Image
          │
          ▼
 Image Preprocessing
          │
          ▼
 Transfer Learning CNN
(EfficientNetB0 / ResNet50 /
 DenseNet121 / MobileNetV2)
          │
          ▼
 Feature Extraction
          │
          ▼
 Custom Classification Head
          │
          ▼
 4-Class Prediction
```

<img width="721" height="248" alt="image" src="https://github.com/user-attachments/assets/b4d07e29-a5e9-47ec-b0ab-76eaf04b7cce" />

---

## 📄 Results
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/c1ab63db-fc0b-4247-83d0-cfbceb303ee0" />

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/5898c5a6-d96f-434e-86fe-0d01215fca64" />

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/a921c33b-7157-481f-8d04-eae6120866f0" />

---

## 🎯 Future Improvements
- Integration of Grad-CAM visualizations
- Explainable AI (XAI) support
- Mobile application support

---

## 🤝 Contributions
Contributions, suggestions, and improvements are welcome.
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a Pull Request

---

## 📜 License
This project is intended for educational and research purposes.

---

~ sukhada20
