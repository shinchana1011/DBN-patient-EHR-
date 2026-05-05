"""
RBM (Restricted Boltzmann Machine) - The building block of DBN
Each RBM learns a probability distribution over its inputs.
"""

import numpy as np


class RBM:
    """
    Restricted Boltzmann Machine
    
    Architecture:
        Visible layer (v) <---> Hidden layer (h)
    
    Energy function:
        E(v,h) = -v^T * W * h - b^T * v - c^T * h
    
    Training: Contrastive Divergence (CD-k)
    """

    def __init__(self, n_visible, n_hidden, learning_rate=0.01, n_epochs=50, batch_size=32, k=1, random_state=42):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.k = k  # CD-k steps
        self.rng = np.random.RandomState(random_state)

        self.W = self.rng.normal(0, 0.01, (n_visible, n_hidden))  # Weight matrix
        self.b = np.zeros(n_visible)   # Visible bias
        self.c = np.zeros(n_hidden)    # Hidden bias

        self.reconstruction_errors = []

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def sample_hidden(self, v):
        """P(h=1|v) = sigmoid(W^T * v + c)"""
        h_prob = self.sigmoid(v @ self.W + self.c)
        h_sample = (self.rng.random(h_prob.shape) < h_prob).astype(float)
        return h_prob, h_sample

    def sample_visible(self, h):
        """P(v=1|h) = sigmoid(W * h + b)"""
        v_prob = self.sigmoid(h @ self.W.T + self.b)
        v_sample = (self.rng.random(v_prob.shape) < v_prob).astype(float)
        return v_prob, v_sample

    def contrastive_divergence(self, v0):
        """
        CD-k: Approximate the gradient of log-likelihood
        Positive phase: data distribution
        Negative phase: model distribution (k Gibbs steps)
        """
        # Positive phase
        h0_prob, h0_sample = self.sample_hidden(v0)

        # Negative phase (k Gibbs steps)
        vk = v0.copy()
        hk_prob = h0_prob.copy()
        for _ in range(self.k):
            _, hk_sample = self.sample_hidden(vk)
            vk_prob, vk = self.sample_visible(hk_sample)
            hk_prob, _ = self.sample_hidden(vk)

        # Gradients
        pos_grad = v0.T @ h0_prob
        neg_grad = vk.T @ hk_prob

        # Weight update
        self.W += self.lr * (pos_grad - neg_grad) / v0.shape[0]
        self.b += self.lr * np.mean(v0 - vk, axis=0)
        self.c += self.lr * np.mean(h0_prob - hk_prob, axis=0)

        # Reconstruction error (MSE)
        recon_error = np.mean((v0 - vk_prob) ** 2)
        return recon_error

    def fit(self, X, verbose=False):
        n_samples = X.shape[0]
        for epoch in range(self.n_epochs):
            idx = self.rng.permutation(n_samples)
            X_shuffled = X[idx]
            epoch_error = 0
            n_batches = 0
            for i in range(0, n_samples, self.batch_size):
                batch = X_shuffled[i:i + self.batch_size]
                error = self.contrastive_divergence(batch)
                epoch_error += error
                n_batches += 1
            avg_error = epoch_error / n_batches
            self.reconstruction_errors.append(avg_error)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.n_epochs} | Recon Error: {avg_error:.4f}")
        return self

    def transform(self, X):
        """Encode input to hidden representation"""
        h_prob, _ = self.sample_hidden(X)
        return h_prob

    def reconstruct(self, X):
        """Reconstruct input from hidden"""
        h_prob, h_sample = self.sample_hidden(X)
        v_prob, _ = self.sample_visible(h_sample)
        return v_prob

    def free_energy(self, v):
        """F(v) = -b^T*v - sum(log(1 + exp(W^T*v + c)))"""
        vb = v @ self.b
        wx_b = v @ self.W + self.c
        hidden_term = np.sum(np.log(1 + np.exp(np.clip(wx_b, -500, 500))), axis=1)
        return -vb - hidden_term