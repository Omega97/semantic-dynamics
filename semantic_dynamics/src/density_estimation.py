import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict, Any


class HybridDensityEstimator:
    """
    A hybrid density estimator that models the first `k` principal components
    with a Gaussian Mixture Model (GMM) and the remaining components as
    an isotropic Gaussian (noise model).
    """

    def __init__(
        self,
        k: int,
        gmm_components: int = 5,
        gmm_reg_covar: float = 1e-6,
        standardize: bool = True,
    ):
        self.k = k
        self.gmm_components = gmm_components
        self.gmm_reg_covar = gmm_reg_covar
        self.standardize = standardize

        # Filled after .fit()
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.gmm: Optional[GaussianMixture] = None
        self.noise_var: float = 0.0
        self.noise_dim: int = 0
        self.is_fitted: bool = False

    # --------------------------------------------------------------------- #
    #                              FIT MODEL                                #
    # --------------------------------------------------------------------- #
    def fit(self, embeddings: np.ndarray) -> "HybridDensityEstimator":
        """
        Fit the hybrid model on (n, d) embedding matrix.
        """
        n, d = embeddings.shape
        if not (0 < self.k < d):
            raise ValueError(f"`k` must be in (0, {d}) -- got {self.k}")

        # 1) optional standardisation
        data_scaled = (
            StandardScaler().fit_transform(embeddings)
            if self.standardize
            else embeddings.copy()
        )
        if self.standardize:
            self.scaler = StandardScaler().fit(embeddings)
            data_scaled = self.scaler.transform(embeddings)
        else:
            self.scaler = None
            data_scaled = embeddings.copy()

        # 2) PCA (full rank, we keep all PCs so we can inverse-transform later)
        self.pca = PCA(n_components=d, svd_solver="auto", random_state=42)
        z = self.pca.fit_transform(data_scaled)  # (n, d)

        # 3) dominant vs. noise split
        z_dom = z[:, : self.k]  # (n, k)
        z_noise = z[:, self.k :]  # (n, d-k)
        self.noise_dim = d - self.k

        # 4) GMM on dominant sub-space
        self.gmm = GaussianMixture(
            n_components=self.gmm_components,
            reg_covar=self.gmm_reg_covar,
            covariance_type="full",
            random_state=42,
        ).fit(z_dom)

        # 5) Isotropic Gaussian on noise sub-space
        if self.noise_dim > 0:
            # mean is (approximately) zero because PCA centres data
            # variance = average per-dimension variance in residual sub-space
            self.noise_var = float(np.mean(np.var(z_noise, axis=0, ddof=0)))
        else:
            self.noise_var = 0.0

        self.is_fitted = True
        return self

    # --------------------------------------------------------------------- #
    #                        LOG-DENSITY EVALUATION                          #
    # --------------------------------------------------------------------- #
    def score_samples(self, new_points: np.ndarray) -> np.ndarray:
        """
        Return log-density for each of `new_points` (m, d).
        """
        if not self.is_fitted:
            raise RuntimeError("Call `.fit()` before `.score_samples()`")

        m, d = new_points.shape
        if d != self.pca.n_components_:
            raise ValueError(
                f"Input dim {d} ≠ training dim {self.pca.n_components_}"
            )

        # Standardise
        z_scaled = (
            self.scaler.transform(new_points)
            if self.standardize
            else new_points.copy()
        )

        # Project to PCA space
        z_pca = self.pca.transform(z_scaled)  # (m, d)
        z_dom = z_pca[:, : self.k]
        z_noise = z_pca[:, self.k :]

        # GMM density on dominant PCs
        log_prob_dom = self.gmm.score_samples(z_dom)  # (m,)

        # Isotropic Gaussian on noise PCs
        if self.noise_dim > 0:
            noise_log_norm = -0.5 * self.noise_dim * (
                np.log(2 * np.pi) + np.log(self.noise_var)
            )
            diff2 = np.sum(z_noise ** 2, axis=1)
            log_prob_noise = noise_log_norm - 0.5 * diff2 / self.noise_var
        else:
            log_prob_noise = np.zeros(m)

        return log_prob_dom + log_prob_noise

    # --------------------------------------------------------------------- #
    #                   AVERAGE LOG-LIKELIHOOD (CONVENIENCE)                #
    # --------------------------------------------------------------------- #
    def score(self, new_points: np.ndarray) -> float:
        return float(np.mean(self.score_samples(new_points)))

    # --------------------------------------------------------------------- #
    #                               SAMPLING                                #
    # --------------------------------------------------------------------- #
    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw `n_samples` from the hybrid model; returns samples in
        original space and in PCA space.
        """
        if not self.is_fitted:
            raise RuntimeError("Call `.fit()` before `.sample()`")

        # 1) GMM in dominant space
        z_dom, _ = self.gmm.sample(n_samples)  # (n, k)

        # 2) Isotropic Gaussian in noise space
        z_noise = (
            np.random.normal(
                0.0, np.sqrt(self.noise_var), (n_samples, self.noise_dim)
            )
            if self.noise_dim > 0
            else np.zeros((n_samples, 0))
        )

        # 3) Merge and inverse-transform
        z_pca = np.hstack([z_dom, z_noise])  # (n, d)
        x_scaled = self.pca.inverse_transform(z_pca)
        x_samples = (
            self.scaler.inverse_transform(x_scaled)
            if self.standardize
            else x_scaled
        )

        return x_samples, z_pca

    # --------------------------------------------------------------------- #
    #                         MODEL PARAMETER DUMP                          #
    # --------------------------------------------------------------------- #
    def get_params(self) -> Dict[str, Any]:
        if not self.is_fitted:
            return {}
        return {
            "k": self.k,
            "gmm_components": self.gmm_components,
            "noise_dim": self.noise_dim,
            "noise_var": self.noise_var,
            "explained_variance_ratio_k": float(
                self.pca.explained_variance_ratio_[: self.k].sum()
            ),
            "total_components": self.pca.n_components_,
        }
