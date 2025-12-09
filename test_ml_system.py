"""
Integration test for BuildBrain ML System.

This test validates:
1. All models can be instantiated
2. All models can be trained
3. All models can make predictions
4. All models can be evaluated
5. ModelFactory works correctly
6. ModelTrainer integration works
"""

import numpy as np
from sklearn.datasets import load_iris, make_blobs, load_breast_cancer
from sklearn.model_selection import train_test_split
from src.training.model_factory import ModelFactory
from src.training.model_trainer import ModelTrainer


def test_model_factory():
    """Test ModelFactory functionality."""
    print("\n" + "="*60)
    print("TEST 1: ModelFactory")
    print("="*60)
    
    # Test get_model_category
    print("\n✓ Testing get_model_category():")
    test_cases = [
        ("LogisticRegression", "supervised"),
        ("KMeans", "unsupervised"),
        ("IsolationForest", "anomaly_detection"),
        ("ResNet18", "deep_learning"),
    ]
    
    for model_type, expected_category in test_cases:
        actual_category = ModelFactory.get_model_category(model_type)
        assert actual_category == expected_category, \
            f"Expected {expected_category}, got {actual_category}"
        print(f"  ✓ {model_type} → {actual_category}")
    
    # Test get_default_hyperparameters
    print("\n✓ Testing get_default_hyperparameters():")
    hyperparams = ModelFactory.get_default_hyperparameters("RandomForestClassifier")
    assert "n_estimators" in hyperparams, "Missing 'n_estimators' in hyperparams"
    print(f"  ✓ RandomForestClassifier: {hyperparams}")
    
    # Test create_model
    print("\n✓ Testing create_model():")
    model = ModelFactory.create_model("SVM")
    assert model is not None, "Failed to create SVM model"
    print(f"  ✓ Created SVM model: {type(model).__name__}")
    
    print("\n✅ ModelFactory tests passed!")


def test_supervised_models():
    """Test all supervised learning models."""
    print("\n" + "="*60)
    print("TEST 2: Supervised Learning Models")
    print("="*60)
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    supervised_models = [
        "LinearRegression",
        "RandomForestRegressor",
        "SVR",
        "LogisticRegression",
        "RandomForestClassifier",
        "SVM",
        "KNN",
        "NaiveBayes",
    ]
    
    print("\n✓ Testing supervised models:")
    for model_type in supervised_models:
        try:
            config = {
                "name": f"test_{model_type}",
                "model_type": model_type,
                "model_category": "supervised",
            }
            
            trainer = ModelTrainer(model_type, config)
            trainer.train(X_train, y_train, validation_split=0.2)
            metrics = trainer.evaluate(X_test, y_test)
            predictions = trainer.predict(X_test[:5])
            
            print(f"  ✓ {model_type}: trained, evaluated, predicted")
            
        except Exception as e:
            print(f"  ✗ {model_type}: {str(e)}")
            raise
    
    print("\n✅ Supervised models tests passed!")


def test_unsupervised_models():
    """Test all unsupervised learning models."""
    print("\n" + "="*60)
    print("TEST 3: Unsupervised Learning Models")
    print("="*60)
    
    # Create synthetic data
    X, _ = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    unsupervised_models = [
        "KMeans",
        "DBSCAN",
        "GaussianMixture",
        "PCA",
    ]
    
    print("\n✓ Testing unsupervised models:")
    for model_type in unsupervised_models:
        try:
            config = {
                "name": f"test_{model_type}",
                "model_type": model_type,
                "model_category": "unsupervised",
                "hyperparameters": {
                    "n_clusters": 3 if model_type != "PCA" else None
                }
            }
            
            trainer = ModelTrainer(model_type, config)
            trainer.train(X_train, None)
            metrics = trainer.evaluate(X_test, None)
            predictions = trainer.predict(X_test[:5])
            
            print(f"  ✓ {model_type}: trained, evaluated, predicted")
            
        except Exception as e:
            print(f"  ✗ {model_type}: {str(e)}")
            raise
    
    print("\n✅ Unsupervised models tests passed!")


def test_anomaly_detection_models():
    """Test anomaly detection models."""
    print("\n" + "="*60)
    print("TEST 4: Anomaly Detection Models")
    print("="*60)
    
    # Create data with outliers
    normal_data = np.random.randn(80, 2)
    outliers = np.random.uniform(-4, 4, (20, 2))
    X = np.vstack([normal_data, outliers])
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    anomaly_models = ["IsolationForest", "OneClassSVM"]
    
    print("\n✓ Testing anomaly detection models:")
    for model_type in anomaly_models:
        try:
            config = {
                "name": f"test_{model_type}",
                "model_type": model_type,
                "model_category": "anomaly_detection",
            }
            
            trainer = ModelTrainer(model_type, config)
            trainer.train(X_train, None)
            predictions = trainer.predict(X_test[:5])
            
            print(f"  ✓ {model_type}: trained, predicted")
            
        except Exception as e:
            print(f"  ✗ {model_type}: {str(e)}")
            raise
    
    print("\n✅ Anomaly detection models tests passed!")


def test_model_persistence():
    """Test save and load functionality."""
    print("\n" + "="*60)
    print("TEST 5: Model Persistence")
    print("="*60)
    
    import tempfile
    import os
    
    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\n✓ Testing save/load:")
    
    # Create and train model
    config = {
        "name": "persistence_test",
        "model_type": "LogisticRegression",
        "model_category": "supervised",
    }
    
    trainer = ModelTrainer("LogisticRegression", config)
    trainer.train(X_train, y_train)
    original_predictions = trainer.predict(X_test[:5])
    
    print(f"  ✓ Model trained")
    
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.pkl")
        trainer.save_model(model_path)
        print(f"  ✓ Model saved to {model_path}")
        
        # Load model
        loaded_trainer = ModelTrainer.load_model(model_path, config)
        loaded_predictions = loaded_trainer.predict(X_test[:5])
        print(f"  ✓ Model loaded from {model_path}")
        
        # Verify predictions match
        assert np.allclose(original_predictions, loaded_predictions), \
            "Loaded model predictions don't match original"
        print(f"  ✓ Predictions match after load")
    
    print("\n✅ Model persistence tests passed!")


def test_metrics_evaluation():
    """Test metrics evaluation."""
    print("\n" + "="*60)
    print("TEST 6: Metrics Evaluation")
    print("="*60)
    
    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\n✓ Testing metrics for different model types:")
    
    # Supervised model metrics
    config = {
        "name": "metrics_test_supervised",
        "model_type": "RandomForestClassifier",
        "model_category": "supervised",
    }
    
    trainer = ModelTrainer("RandomForestClassifier", config)
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
    
    print(f"\n  Supervised model metrics:")
    assert "accuracy" in metrics, "Missing 'accuracy' in metrics"
    assert "precision" in metrics, "Missing 'precision' in metrics"
    assert "recall" in metrics, "Missing 'recall' in metrics"
    assert "f1_score" in metrics, "Missing 'f1_score' in metrics"
    
    for metric_name, metric_value in metrics.items():
        print(f"    ✓ {metric_name}: {metric_value:.4f}")
    
    # Unsupervised model metrics
    from sklearn.datasets import make_blobs
    X_unsupervised, _ = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)
    
    config = {
        "name": "metrics_test_unsupervised",
        "model_type": "KMeans",
        "model_category": "unsupervised",
        "hyperparameters": {"n_clusters": 3}
    }
    
    trainer = ModelTrainer("KMeans", config)
    trainer.train(X_unsupervised, None)
    silhouette = trainer.evaluate(X_unsupervised, None)
    
    print(f"\n  Unsupervised model metrics:")
    print(f"    ✓ silhouette_score: {silhouette:.4f}")
    
    print("\n✅ Metrics evaluation tests passed!")


def main():
    """Run all tests."""
    print("\n" + "🧠 BuildBrain ML System - Integration Tests 🧠".center(60))
    print("="*60)
    
    try:
        test_model_factory()
        test_supervised_models()
        test_unsupervised_models()
        test_anomaly_detection_models()
        test_model_persistence()
        test_metrics_evaluation()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ TEST FAILED: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
