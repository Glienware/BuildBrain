"""
Utilities Module

Helper functions for i18n, model saving, etc.
"""

import json
import os
import joblib
import torch
import onnx


class I18n:
    """
    Simple internationalization class.
    """

    def __init__(self, lang="en"):
        self.lang = lang
        self.translations = self.load_translations()

    def load_translations(self):
        """
        Load translation files.
        """
        translations = {}
        for file in os.listdir("locales"):
            if file.endswith(".json"):
                lang = file[:-5]
                with open(os.path.join("locales", file), "r", encoding="utf-8") as f:
                    translations[lang] = json.load(f)
        return translations

    def t(self, key):
        """
        Translate a key.
        """
        return self.translations.get(self.lang, {}).get(key, key)


def save_model(model, model_type, path):
    """
    Save trained model.
    """
    if model_type in ["LogisticRegression", "RandomForest", "XGBoost"]:
        joblib.dump(model, path + ".pkl")
    elif model_type in ["PyTorch CNN", "PyTorch MLP"]:
        torch.save(model.state_dict(), path + ".pth")
        # Optional ONNX export
        # torch.onnx.export(model, dummy_input, path + ".onnx")


def load_model(path, model_type):
    """
    Load saved model.
    """
    if model_type in ["LogisticRegression", "RandomForest", "XGBoost"]:
        return joblib.load(path + ".pkl")
    elif model_type in ["PyTorch CNN", "PyTorch MLP"]:
        # Need model architecture
        pass