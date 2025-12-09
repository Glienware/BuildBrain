"""
Unsupervised Learning Models

Implementations of clustering and dimensionality reduction models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
except ImportError:
    umap = None


class BaseUnsupervisedModel(ABC):
    """Base class for unsupervised models."""
    
    def __init__(self, **kwargs):
        self.model = None
        self.is_fitted = False
        self.config = kwargs
        self.labels_ = None
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict labels."""
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray) -> dict:
        """Evaluate model performance."""
        pass


# CLUSTERING MODELS

class KMeansModel(BaseUnsupervisedModel):
    """K-Means Clustering Model."""
    
    def __init__(self, n_clusters: int = 3, random_state: int = 42, **kwargs):
        super().__init__(n_clusters=n_clusters, random_state=random_state, **kwargs)
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    
    def fit(self, X: np.ndarray) -> None:
        """Train the K-Means model."""
        self.labels_ = self.model.fit_predict(X)
        self.is_fitted = True
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict cluster labels."""
        return self.model.fit_predict(X)
    
    def evaluate(self, X: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        
        if not self.is_fitted:
            self.fit(X)
        
        silhouette = silhouette_score(X, self.labels_)
        davies_bouldin = davies_bouldin_score(X, self.labels_)
        
        return {
            'silhouette_score': float(silhouette),
            'davies_bouldin_score': float(davies_bouldin),
            'inertia': float(self.model.inertia_),
            'n_clusters': self.model.n_clusters,
        }


class DBSCANModel(BaseUnsupervisedModel):
    """DBSCAN Clustering Model."""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs):
        super().__init__(eps=eps, min_samples=min_samples, **kwargs)
        self.model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    
    def fit(self, X: np.ndarray) -> None:
        """Train the DBSCAN model."""
        self.labels_ = self.model.fit_predict(X)
        self.is_fitted = True
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict cluster labels."""
        return self.model.fit_predict(X)
    
    def evaluate(self, X: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import silhouette_score
        
        if not self.is_fitted:
            self.fit(X)
        
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)
        
        silhouette = silhouette_score(X, self.labels_) if n_clusters > 1 else 0
        
        return {
            'silhouette_score': float(silhouette),
            'n_clusters': int(n_clusters),
            'n_noise_points': int(n_noise),
            'eps': self.model.eps,
            'min_samples': self.model.min_samples,
        }


class MeanShiftModel(BaseUnsupervisedModel):
    """Mean Shift Clustering Model."""
    
    def __init__(self, bandwidth: float = None, **kwargs):
        super().__init__(bandwidth=bandwidth, **kwargs)
        self.model = MeanShift(bandwidth=bandwidth, **kwargs)
    
    def fit(self, X: np.ndarray) -> None:
        """Train the Mean Shift model."""
        self.labels_ = self.model.fit_predict(X)
        self.is_fitted = True
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict cluster labels."""
        return self.model.fit_predict(X)
    
    def evaluate(self, X: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import silhouette_score
        
        if not self.is_fitted:
            self.fit(X)
        
        n_clusters = len(np.unique(self.labels_))
        silhouette = silhouette_score(X, self.labels_) if n_clusters > 1 else 0
        
        return {
            'silhouette_score': float(silhouette),
            'n_clusters': int(n_clusters),
            'n_centers': len(self.model.cluster_centers_),
        }


class GaussianMixtureModel(BaseUnsupervisedModel):
    """Gaussian Mixture Model (GMM)."""
    
    def __init__(self, n_components: int = 3, random_state: int = 42, **kwargs):
        super().__init__(n_components=n_components, random_state=random_state, **kwargs)
        self.model = GaussianMixture(n_components=n_components, random_state=random_state, **kwargs)
    
    def fit(self, X: np.ndarray) -> None:
        """Train the GMM model."""
        self.labels_ = self.model.fit_predict(X)
        self.is_fitted = True
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict cluster labels."""
        return self.model.fit_predict(X)
    
    def evaluate(self, X: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import silhouette_score
        
        if not self.is_fitted:
            self.fit(X)
        
        silhouette = silhouette_score(X, self.labels_)
        
        return {
            'silhouette_score': float(silhouette),
            'bic': float(self.model.bic(X)),
            'aic': float(self.model.aic(X)),
            'n_components': self.model.n_components,
        }


# DIMENSIONALITY REDUCTION MODELS

class PCAModel(BaseUnsupervisedModel):
    """Principal Component Analysis (PCA) Model."""
    
    def __init__(self, n_components: int = 2, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.model = PCA(n_components=n_components, **kwargs)
        self.transformed_data = None
    
    def fit(self, X: np.ndarray) -> None:
        """Train the PCA model."""
        self.transformed_data = self.model.fit_transform(X)
        self.is_fitted = True
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return transformed data."""
        return self.model.fit_transform(X)
    
    def evaluate(self, X: np.ndarray) -> dict:
        """Evaluate model performance."""
        if not self.is_fitted:
            self.fit(X)
        
        explained_variance_ratio = np.cumsum(self.model.explained_variance_ratio_)
        
        return {
            'explained_variance_ratio': float(explained_variance_ratio[-1]),
            'n_components': self.model.n_components,
            'components_variance': [float(v) for v in self.model.explained_variance_ratio_],
        }


class TSNEModel(BaseUnsupervisedModel):
    """t-Distributed Stochastic Neighbor Embedding (t-SNE) Model."""
    
    def __init__(self, n_components: int = 2, perplexity: int = 30, **kwargs):
        super().__init__(n_components=n_components, perplexity=perplexity, **kwargs)
        self.model = TSNE(n_components=n_components, perplexity=perplexity, **kwargs)
        self.transformed_data = None
    
    def fit(self, X: np.ndarray) -> None:
        """Train the t-SNE model."""
        self.transformed_data = self.model.fit_transform(X)
        self.is_fitted = True
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return transformed data."""
        return self.model.fit_transform(X)
    
    def evaluate(self, X: np.ndarray) -> dict:
        """Evaluate model performance."""
        if not self.is_fitted:
            self.fit(X)
        
        return {
            'n_components': self.model.n_components,
            'perplexity': self.model.perplexity,
            'n_iter': self.model.n_iter,
            'kl_divergence': float(self.model.kl_divergence_) if hasattr(self.model, 'kl_divergence_') else 0,
        }


class UMAPModel(BaseUnsupervisedModel):
    """Uniform Manifold Approximation and Projection (UMAP) Model."""
    
    def __init__(self, n_components: int = 2, n_neighbors: int = 15, **kwargs):
        if umap is None:
            raise ImportError("UMAP requires 'umap-learn' package. Install with: pip install umap-learn")
        
        super().__init__(n_components=n_components, n_neighbors=n_neighbors, **kwargs)
        self.model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, **kwargs)
        self.transformed_data = None
    
    def fit(self, X: np.ndarray) -> None:
        """Train the UMAP model."""
        self.transformed_data = self.model.fit_transform(X)
        self.is_fitted = True
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return transformed data."""
        return self.model.fit_transform(X)
    
    def evaluate(self, X: np.ndarray) -> dict:
        """Evaluate model performance."""
        if not self.is_fitted:
            self.fit(X)
        
        return {
            'n_components': self.model.n_components,
            'n_neighbors': self.model.n_neighbors,
        }
