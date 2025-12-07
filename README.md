# AI/ML Model Trainer

A desktop application for training machine learning models on Windows 11, built with Python and Flet.

## Features

- **Model Selection**: Choose from LogisticRegression, RandomForest, XGBoost, PyTorch CNN, PyTorch MLP
- **Task Types**: Classification and Regression
- **Training Presets**: Fast, Balanced, Max Performance
- **Manual Hyperparameters**: Customize learning rate, batch size, epochs, etc.
- **Dataset Support**: CSV for tabular data, ZIP/folder for images
- **Class Management**: Add/remove class labels, balance classes
- **Real-time Training**: Progress bars, live metrics, training curves
- **Visualization**: Confusion matrix, classification reports, plots
- **Model Export**: Save models in various formats (joblib, PyTorch, ONNX)
- **Profiles**: Save and load project configurations
- **Internationalization**: English and Spanish support

## Requirements

- Python 3.10+
- pip
- Optional: CUDA for GPU acceleration with PyTorch

## Installation

1. Clone or download this repository
2. Install Python 3.10+ from [python.org](https://python.org)
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) For GPU support, install PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

Run the application:

```bash
python main.py
```

### Workflow

1. **Setup Tab**:
   - Select task type (Classification/Regression)
   - Choose model
   - Select preset or customize hyperparameters
   - For classification, manage class labels

2. **Data Tab**:
   - Upload CSV dataset
   - Preview data statistics

3. **Training Tab**:
   - Start quick or advanced training
   - Monitor progress and logs
   - View visualizations

4. **Export**:
   - Save trained model
   - Export metrics and reports

## Project Structure

```
src/
├── gui/           # GUI components
├── training/      # Training logic
├── data/          # Data preprocessing
└── utils/         # Utilities and helpers
locales/           # Translation files
config/            # Project configurations
```

## Development

The code is modular and commented. Key modules:

- `main.py`: Application entry point
- `src/gui/`: Flet-based UI components
- `src/training/trainer.py`: Model training logic
- `src/data/preprocessor.py`: Data loading and preprocessing
- `src/utils/helpers.py`: Utilities including i18n

## Contributing

Contributions welcome! Please ensure code is well-commented and follows the existing structure.

## License

MIT License