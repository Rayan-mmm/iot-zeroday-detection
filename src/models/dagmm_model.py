"""
DAGMM (Deep Autoencoding Gaussian Mixture Model) wrapper for unsupervised
anomaly detection. Based on Zong et al. (2018), "Deep Autoencoding Gaussian
Mixture Model for Unsupervised Anomaly Detection" (ICLR).

Combines two components, trained jointly:
1. Compression network: an autoencoder that compresses each row to a small
   latent representation, and also produces reconstruction-error features
   (how well the row was rebuilt).
2. Estimation network: a small network that takes the latent representation
   + reconstruction-error features and predicts a soft cluster assignment
   over a Gaussian Mixture Model (GMM).

After training, GMM parameters (mixture weights, means, covariances) are
estimated once over the full training set (standard DAGMM practice), then
used to compute an "energy" score for new data -- higher energy means the
row's compressed representation poorly fits any learned normal cluster,
i.e. more anomalous.

Interface matches the other detectors (fit / anomaly_score / save / load).
Score convention: anomaly_score() returns NEGATIVE energy, keeping "lower
score = more anomalous" consistent across all models in this project.

Includes standard stabilization techniques from the DAGMM literature
(covariance diagonal regularization, numerical epsilon in matrix inversion)
to avoid the known "collapse" failure mode where the GMM component
degenerates.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MODEL_DIR = Path("results/dagmm")


class DAGMMDetector:
    def __init__(self, input_dim: int, latent_dim: int = 1, n_gmm: int = 4,
                 lambda_energy: float = 0.1, lambda_cov: float = 0.005,
                 epochs: int = 20, batch_size: int = 256, random_state: int = 42):
        tf.random.set_seed(random_state)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_gmm = n_gmm
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.epochs = epochs
        self.batch_size = batch_size

        # Compression network (encoder-decoder, same spirit as the plain Autoencoder)
        self.encoder = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(30, activation="tanh"),
            layers.Dense(10, activation="tanh"),
            layers.Dense(latent_dim),
        ])
        self.decoder = keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(10, activation="tanh"),
            layers.Dense(30, activation="tanh"),
            layers.Dense(input_dim),
        ])

        # Estimation network: latent (latent_dim) + 2 reconstruction-error
        # features -> soft cluster assignment over n_gmm clusters
        self.estimation = keras.Sequential([
            layers.Input(shape=(latent_dim + 2,)),
            layers.Dense(10, activation="tanh"),
            layers.Dropout(0.5),
            layers.Dense(n_gmm, activation="softmax"),
        ])

        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # GMM parameters, fixed after training completes
        self.phi = None    # mixture weights, shape (n_gmm,)
        self.mu = None     # cluster means, shape (n_gmm, latent_dim+2)
        self.cov = None    # cluster covariances, shape (n_gmm, latent_dim+2, latent_dim+2)

    def _compute_features(self, x):
        """Compress x, reconstruct it, and build the combined feature
        vector [latent, relative_euclidean_distance, cosine_similarity]
        fed into the estimation network."""
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)

        recon_error = x - x_hat
        euclidean_norm = tf.norm(recon_error, axis=1)
        x_norm = tf.norm(x, axis=1) + 1e-8
        relative_euclidean = euclidean_norm / x_norm

        cosine_sim = tf.reduce_sum(x * x_hat, axis=1) / (
            tf.norm(x, axis=1) * tf.norm(x_hat, axis=1) + 1e-8
        )

        z = tf.concat([
            z_c,
            tf.expand_dims(relative_euclidean, axis=1),
            tf.expand_dims(cosine_sim, axis=1),
        ], axis=1)

        return x_hat, z

    def _gmm_params(self, z, gamma):
        """Estimate GMM mixture weights, means, and covariances from a
        batch of latent features z and their soft cluster assignments gamma."""
        N = tf.cast(tf.shape(z)[0], tf.float32)
        sum_gamma = tf.reduce_sum(gamma, axis=0)  # (n_gmm,)

        phi = sum_gamma / N  # (n_gmm,)

        mu = tf.einsum("nk,nd->kd", gamma, z) / (
            tf.expand_dims(sum_gamma, axis=1) + 1e-8
        )  # (n_gmm, dim)

        z_centered = tf.expand_dims(z, axis=1) - tf.expand_dims(mu, axis=0)  # (N, K, dim)
        cov = tf.einsum(
            "nk,nkd,nke->kde", gamma, z_centered, z_centered
        ) / (tf.reshape(sum_gamma, (-1, 1, 1)) + 1e-8)  # (K, dim, dim)

        return phi, mu, cov

    def _energy(self, z, phi, mu, cov):
        """Sample energy: -log of the GMM's estimated probability density
        for each point in z. Higher energy = more anomalous."""
        dim = tf.shape(z)[1]
        eps = 1e-6

        z_centered = tf.expand_dims(z, axis=1) - tf.expand_dims(mu, axis=0)  # (N, K, dim)

        cov_reg = cov + tf.eye(self.latent_dim + 2) * eps  # regularize diagonal, avoid singular matrix
        cov_inv = tf.linalg.inv(cov_reg)  # (K, dim, dim)

        # (N, K)
        exponent = -0.5 * tf.einsum("nkd,kde,nke->nk", z_centered, cov_inv, z_centered)

        log_det_cov = tf.math.log(tf.linalg.det(cov_reg) + eps)  # (K,)
        log_norm = -0.5 * (tf.cast(dim, tf.float32) * tf.math.log(2 * np.pi) + log_det_cov)  # (K,)

        log_prob_per_cluster = tf.math.log(phi + eps) + log_norm + exponent  # (N, K)
        log_prob = tf.reduce_logsumexp(log_prob_per_cluster, axis=1)  # (N,)

        energy = -log_prob
        cov_diag_penalty = tf.reduce_sum(1.0 / (tf.linalg.diag_part(cov_reg) + eps))
        return energy, cov_diag_penalty

    def _train_step(self, x_batch):
        with tf.GradientTape() as tape:
            x_hat, z = self._compute_features(x_batch)
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_batch - x_hat), axis=1))

            gamma = self.estimation(z)
            phi, mu, cov = self._gmm_params(z, gamma)
            energy, cov_diag_penalty = self._energy(z, phi, mu, cov)

            loss = (recon_loss
                    + self.lambda_energy * tf.reduce_mean(energy)
                    + self.lambda_cov * cov_diag_penalty)

        trainable_vars = (self.encoder.trainable_variables
                           + self.decoder.trainable_variables
                           + self.estimation.trainable_variables)
        grads = tape.gradient(loss, trainable_vars)
        # Clip gradients -- DAGMM is known to be prone to unstable gradients
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        return loss, recon_loss

    def fit(self, X: np.ndarray):
        X = X.astype(np.float32)
        n_batches = max(1, len(X) // self.batch_size)
        print(f"Training DAGMM on {X.shape[0]:,} normal samples "
              f"({X.shape[1]} features, {self.n_gmm} GMM clusters)...")

        dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(10000).batch(self.batch_size)

        for epoch in range(self.epochs):
            epoch_loss, epoch_recon = 0.0, 0.0
            n = 0
            for x_batch in dataset:
                loss, recon_loss = self._train_step(x_batch)
                epoch_loss += float(loss)
                epoch_recon += float(recon_loss)
                n += 1
            print(f"Epoch {epoch+1}/{self.epochs} - loss: {epoch_loss/n:.4f} "
                  f"- recon_loss: {epoch_recon/n:.4f}")

        # Estimate final, fixed GMM parameters over the FULL training set
        # (standard DAGMM practice -- per-batch estimates are only used
        # during training, final scoring uses the complete-dataset estimate)
        print("Estimating final GMM parameters over full training set...")
        _, z_full = self._compute_features(tf.constant(X))
        gamma_full = self.estimation(z_full)
        self.phi, self.mu, self.cov = self._gmm_params(z_full, gamma_full)

        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Returns NEGATIVE energy, so lower score = more anomalous --
        matches the convention used by the other detectors in this project."""
        X = X.astype(np.float32)
        _, z = self._compute_features(tf.constant(X))
        energy, _ = self._energy(z, self.phi, self.mu, self.cov)
        return -energy.numpy()

    def save(self, path: Path = MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save(path / "encoder.keras")
        self.decoder.save(path / "decoder.keras")
        self.estimation.save(path / "estimation.keras")
        np.savez(path / "gmm_params.npz",
                 phi=self.phi.numpy(), mu=self.mu.numpy(), cov=self.cov.numpy())
        print(f"Saved DAGMM model -> {path}")

    def load(self, path: Path = MODEL_DIR):
        self.encoder = keras.models.load_model(path / "encoder.keras")
        self.decoder = keras.models.load_model(path / "decoder.keras")
        self.estimation = keras.models.load_model(path / "estimation.keras")
        params = np.load(path / "gmm_params.npz")
        self.phi = tf.constant(params["phi"], dtype=tf.float32)
        self.mu = tf.constant(params["mu"], dtype=tf.float32)
        self.cov = tf.constant(params["cov"], dtype=tf.float32)
        return self
