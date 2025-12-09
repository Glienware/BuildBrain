# 🧠 BuildBrain - AI/ML Model Trainer

A professional desktop application for training machine learning models with a modern, intuitive GUI. Built with **Python**, **Flet**, and **25+ production-ready ML models**.

## ✨ Features

### 🎯 Model Selection
- **25+ ML Models** across 4 categories:
  - 10 Supervised Learning models (Classification & Regression)
  - 7 Unsupervised Learning models (Clustering & Dimensionality Reduction)
  - 2 Anomaly Detection models
  - 5 Deep Learning models (ResNet18/34/50, PyTorch CNN/MLP)

### 📊 Dataset Support
- **CSV/Excel** for tabular data
- **Image folders** organized by class
- **Automatic preprocessing** (normalization, train/test split)
- **Multi-format support**: `.csv`, `.xlsx`, `.xls`, `.png`, `.jpg`, `.jpeg`, `.bmp`

### 🚀 Training Features
- **Dynamic Wizard**: Steps adapt based on model type
- **Quick Training**: Pre-configured, one-click training
- **Real-time Progress**: Live logs, metrics, and progress bars
- **GPU Support**: Automatic CUDA/GPU detection
- **Model Persistence**: Save/load trained models

### 🎨 Professional UI
- **Android Studio Dark Theme**: Modern, dark-themed interface
- **Responsive Design**: Works on different screen sizes
- **Interactive Components**: Cards, grids, scrollable areas
- **Multi-language**: English and Spanish support

### 💾 Project Management
- **Project Profiles**: Save and load configurations
- **Organized Structure**: Models, data, and logs in separate folders
- **Configuration Export**: Save hyperparameters and settings

## 📋 Requirements

- **Python** 3.10+
- **pip** package manager
- **Optional**: CUDA for GPU acceleration with PyTorch

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd BuildBrain
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**For GPU Support (Optional):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Run the Application
```bash
python main.py
```

## 🚀 Quick Start

### First Time?
1. Run `python main.py`
2. Click **"Get Started"** on the Welcome screen
3. Follow the 5-7 step wizard
4. Load your dataset (use `example_dataset.csv` to test)
5. Click **"Crear Modelo"** to train

### Dataset Format
- **CSV**: Rows = samples, Columns = features + label
- **Images**: Folders organized by class
  ```
  dataset/
  ├── class_A/
  │   ├── image1.jpg
  │   └── image2.jpg
  ├── class_B/
  │   ├── image3.jpg
  │   └── image4.jpg
  ```

For detailed instructions, see **[DATASET_LOADING_GUIDE.md](DATASET_LOADING_GUIDE.md)**

## 📚 Documentation

| Document | Content |
|----------|---------|
| [QUICK_START.md](QUICK_START.md) | Getting started guide with examples |
| [ML_SYSTEM_DOCUMENTATION.md](ML_SYSTEM_DOCUMENTATION.md) | Complete technical documentation |
| [DATASET_LOADING_GUIDE.md](DATASET_LOADING_GUIDE.md) | How to load and prepare datasets |
| [example_usage.py](example_usage.py) | Code examples for all model types |
| [test_ml_system.py](test_ml_system.py) | Integration tests for validation |

## 🎯 Workflow

### Step 1: Create Project
```
Welcome Screen → Get Started → Project Name
```

### Step 2: Select Model
Choose from 25+ models:
- **Supervised**: LogisticRegression, RandomForest, XGBoost, SVM, KNN, etc.
- **Unsupervised**: KMeans, DBSCAN, PCA, t-SNE, UMAP, etc.
- **Anomaly**: IsolationForest, OneClassSVM
- **Deep Learning**: ResNet18/34/50, PyTorchCNN, PyTorchMLP

### Step 3: Configure Model
- Model-specific parameters adjust automatically
- Supervised: Configure classes, balancing
- Unsupervised: Configure n_clusters, parameters
- Deep Learning: Configure network architecture

### Step 4: Load Dataset
- **CSV**: One click to load tabular data
- **Images**: Select folders for each class
- **Preprocessing**: Automatic normalization

### Step 5: Train Model
- **Quick Training**: Predefined settings
- **Advanced Training**: Custom hyperparameters
- **Real-time Monitoring**: See logs and metrics

### Step 6: View Results
- **Metrics**: Accuracy, Precision, Recall, F1, etc.
- **Model Saved**: Automatically stored in project
- **Export**: Save model for deployment

## 📁 Project Structure

```
BuildBrain/
├── src/
│   ├── gui/                          # GUI Components
│   │   ├── main_window.py
│   │   ├── new_project_wizard.py     # Dynamic wizard
│   │   ├── welcome_screen.py
│   │   └── dataset_uploader.py       # Dataset loading
│   ├── training/
│   │   ├── models/                   # 25+ model implementations
│   │   │   ├── supervised_models.py
│   │   │   ├── unsupervised_models.py
│   │   │   ├── anomaly_detection.py
│   │   │   └── deep_learning.py
│   │   ├── model_factory.py          # Factory pattern
│   │   ├── model_trainer.py          # Unified training API
│   │   └── trainer.py
│   ├── data/
│   │   └── preprocessor.py           # Data preprocessing
│   └── utils/
│       └── helpers.py
├── config.py                         # Configuration
├── config/                           # Config files
├── locales/                          # i18n translations
├── projects/                         # Saved projects
├── models/                           # Trained models
├── logs/                             # Training logs
├── example_dataset.csv               # Example data
└── main.py                           # Entry point
```

## 🔧 Development

### Running Tests
```bash
python test_ml_system.py
```

### Running Examples
```bash
python example_usage.py
```

### Code Structure
- **Models**: `src/training/models/` - One file per category
- **Factory**: `src/training/model_factory.py` - Central model creation
- **Trainer**: `src/training/model_trainer.py` - Unified training API
- **GUI**: `src/gui/` - Flet-based interface components

## 🐛 Troubleshooting

### Can't load datasets?
See **[DATASET_LOADING_GUIDE.md](DATASET_LOADING_GUIDE.md)** for detailed instructions.

### CUDA/GPU not working?
```bash
# Verify GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Models will auto-detect and use CPU if GPU unavailable
```

### Module import errors?
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## 📊 Sample Datasets

Included:
- `example_dataset.csv` - Iris dataset for classification testing

Recommended:
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)

## 🚀 Next Features (Roadmap)

- [ ] Model comparison tools
- [ ] Hyperparameter optimization (GridSearch)
- [ ] Export to ONNX format
- [ ] Cloud deployment (Azure/AWS)
- [ ] Advanced visualizations (SHAP, LIME)
- [ ] Automated ML (AutoML)
- [ ] Pipeline creation

## 📄 License

BuildBrain © 2024 - Open Source

## 👨‍💻 Author

Developed with ❤️ using **GitHub Copilot** and **Python**

---

**Ready to train your first model?** Start with [QUICK_START.md](QUICK_START.md) 🚀
