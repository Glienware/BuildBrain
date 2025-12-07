"""
Data Preprocessing Module

Handles dataset loading, preprocessing, and balancing.
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """
    Preprocesses datasets for training.
    """

    def __init__(self, dataset_path, task_type):
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.data = None
        self.X = None
        self.y = None
        self.classes = None

    def load_data(self):
        """
        Load dataset from path.
        """
        if self.dataset_path.endswith('.csv'):
            self.data = pd.read_csv(self.dataset_path)
        else:
            # Placeholder for images
            raise NotImplementedError("Image datasets not implemented")

    def preprocess(self, target_column=None):
        """
        Preprocess the data.
        """
        if self.data is None:
            self.load_data()

        if target_column:
            self.y = self.data[target_column]
            self.X = self.data.drop(columns=[target_column])
        else:
            # Assume last column is target
            self.y = self.data.iloc[:, -1]
            self.X = self.data.iloc[:, :-1]

        # Encode categorical variables
        for col in self.X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col])

        # Scale features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        if self.task_type == "Classification":
            self.classes = np.unique(self.y)

    def balance_classes(self, method="oversample"):
        """
        Balance classes using oversampling or undersampling.
        """
        if method == "oversample":
            smote = SMOTE()
            self.X, self.y = smote.fit_resample(self.X, self.y)
        elif method == "undersample":
            rus = RandomUnderSampler()
            self.X, self.y = rus.fit_resample(self.X, self.y)