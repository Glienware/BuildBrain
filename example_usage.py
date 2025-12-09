"""
Example usage of the BuildBrain ML system with ModelTrainer and ModelFactory.

This example demonstrates:
1. Creating models using ModelFactory
2. Training models using ModelTrainer
3. Evaluating and saving models
"""

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from src.training.model_factory import ModelFactory
from src.training.model_trainer import ModelTrainer
import os
import json


def example_supervised_learning():
    """Example: Train a supervised learning model (LogisticRegression)."""
    print("\n" + "="*60)
    print("EJEMPLO 1: Supervised Learning - Clasificación Logística")
    print("="*60)
    
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create project config
    config = {
        "name": "iris_classifier",
        "model_type": "LogisticRegression",
        "model_category": "supervised",
        "hyperparameters": {
            "max_iter": 200,
            "C": 1.0,
        }
    }
    
    # Initialize trainer
    trainer = ModelTrainer("LogisticRegression", config)
    
    # Train model
    print("\n📚 Entrenando modelo LogisticRegression...")
    trainer.train(X_train, y_train, validation_split=0.2)
    
    # Evaluate model
    print("\n📊 Evaluando modelo...")
    metrics = trainer.evaluate(X_test, y_test)
    print(f"Métricas: {metrics}")
    
    # Make predictions
    print("\n🔮 Haciendo predicciones...")
    predictions = trainer.predict(X_test[:5])
    print(f"Predicciones para las primeras 5 muestras: {predictions}")
    
    # Save model
    print("\n💾 Guardando modelo...")
    os.makedirs("models", exist_ok=True)
    trainer.save_model("models/iris_classifier.pkl")
    print("✅ Modelo guardado en models/iris_classifier.pkl")


def example_unsupervised_learning():
    """Example: Train an unsupervised learning model (KMeans)."""
    print("\n" + "="*60)
    print("EJEMPLO 2: Unsupervised Learning - KMeans Clustering")
    print("="*60)
    
    # Create synthetic dataset
    X = np.random.randn(100, 2)
    X = X * [[2, 3]]  # Scale
    
    # Create project config
    config = {
        "name": "kmeans_clustering",
        "model_type": "KMeans",
        "model_category": "unsupervised",
        "hyperparameters": {
            "n_clusters": 3,
        }
    }
    
    # Initialize trainer
    trainer = ModelTrainer("KMeans", config)
    
    # Train model (unsupervised doesn't need y)
    print("\n📚 Entrenando modelo KMeans...")
    trainer.train(X, None)
    
    # Evaluate model
    print("\n📊 Evaluando modelo...")
    metrics = trainer.evaluate(X, None)
    print(f"Silhouette Score: {metrics}")
    
    # Make predictions
    print("\n🔮 Asignando clusters...")
    clusters = trainer.predict(X[:5])
    print(f"Clusters para las primeras 5 muestras: {clusters}")


def example_anomaly_detection():
    """Example: Train an anomaly detection model (IsolationForest)."""
    print("\n" + "="*60)
    print("EJEMPLO 3: Anomaly Detection - Isolation Forest")
    print("="*60)
    
    # Create synthetic dataset with outliers
    normal_data = np.random.randn(80, 2)
    outliers = np.random.uniform(-4, 4, (20, 2))
    X = np.vstack([normal_data, outliers])
    
    # Create project config
    config = {
        "name": "anomaly_detector",
        "model_type": "IsolationForest",
        "model_category": "anomaly_detection",
        "hyperparameters": {
            "contamination": 0.2,
            "random_state": 42,
        }
    }
    
    # Initialize trainer
    trainer = ModelTrainer("IsolationForest", config)
    
    # Train model
    print("\n📚 Entrenando modelo IsolationForest...")
    trainer.train(X, None)
    
    # Detect anomalies
    print("\n🔮 Detectando anomalías...")
    anomalies = trainer.predict(X[:10])
    print(f"Predicciones (-1=anomalía, 1=normal): {anomalies}")


def example_list_all_models():
    """Example: List all available models by category."""
    print("\n" + "="*60)
    print("TODOS LOS MODELOS DISPONIBLES")
    print("="*60)
    
    model_types = [
        "LinearRegression", "RandomForestRegressor", "XGBoostRegressor", "SVR",
        "LogisticRegression", "RandomForestClassifier", "XGBoostClassifier", "KNN", "SVM", "NaiveBayes",
        "KMeans", "DBSCAN", "MeanShift", "GaussianMixture", "PCA", "TSNE", "UMAP",
        "IsolationForest", "OneClassSVM",
        "ResNet18", "ResNet34", "ResNet50", "PyTorchCNN", "PyTorchMLP"
    ]
    
    categories = {
        "supervised": [],
        "unsupervised": [],
        "anomaly_detection": [],
        "deep_learning": []
    }
    
    for model_type in model_types:
        try:
            category = ModelFactory.get_model_category(model_type)
            categories[category].append(model_type)
        except Exception as e:
            print(f"⚠️ Error con {model_type}: {e}")
    
    for category, models in categories.items():
        print(f"\n📚 {category.upper()}:")
        for model in models:
            print(f"   ✓ {model}")
            hyperparams = ModelFactory.get_default_hyperparameters(model)
            print(f"     Hiperparámetros: {hyperparams}")


def example_model_factory():
    """Example: Using ModelFactory to create and compare models."""
    print("\n" + "="*60)
    print("EJEMPLO 4: ModelFactory - Comparación de Modelos")
    print("="*60)
    
    # Load dataset
    X, y = make_classification(n_samples=200, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models to compare
    models_to_test = ["LogisticRegression", "RandomForestClassifier", "SVM"]
    
    print("\n🏆 Comparando modelos clasificadores:")
    print("-" * 60)
    
    results = {}
    for model_name in models_to_test:
        try:
            print(f"\n📊 Evaluando {model_name}...")
            
            config = {
                "name": f"comparison_{model_name}",
                "model_type": model_name,
                "model_category": "supervised",
            }
            
            trainer = ModelTrainer(model_name, config)
            trainer.train(X_train, y_train, validation_split=0.2)
            metrics = trainer.evaluate(X_test, y_test)
            
            results[model_name] = metrics
            print(f"   ✓ {model_name}: Accuracy = {metrics.get('accuracy', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"   ✗ Error con {model_name}: {e}")
    
    print("\n" + "="*60)
    print("📈 RESUMEN DE RESULTADOS:")
    print("-" * 60)
    best_model = max(results, key=lambda x: results[x].get('accuracy', 0))
    print(f"✅ Mejor modelo: {best_model}")
    print(f"   Accuracy: {results[best_model].get('accuracy', 'N/A'):.4f}")


if __name__ == "__main__":
    print("\n" + "🧠 BuildBrain ML System - Examples 🧠".center(60))
    print("="*60)
    
    try:
        # Run examples
        example_supervised_learning()
        example_unsupervised_learning()
        example_anomaly_detection()
        example_list_all_models()
        example_model_factory()
        
        print("\n" + "="*60)
        print("✅ Todos los ejemplos se ejecutaron correctamente!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
