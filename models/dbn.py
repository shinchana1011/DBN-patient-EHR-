"""
Deep Belief Network (DBN) for Synthetic EHR Generation

Architecture:
    Input (EHR features)
        ↓ RBM 1: learns low-level feature correlations (age, vitals)
        ↓ RBM 2: learns high-level patterns (disease clusters)
        ↓ RBM 3 (optional): learns abstract representations
    Output: Synthetic patient records

Training: Greedy layer-wise pretraining
    1. Train RBM1 on raw data
    2. Train RBM2 on RBM1's hidden activations
    3. Repeat for more layers
"""

import numpy as np
from models.rbm import RBM


class DBN:
    """
    Deep Belief Network - Stack of RBMs
    
    Key insight: Each layer learns increasingly abstract representations.
    Top layer forms an undirected model; lower layers are directed (top-down).
    """

    def __init__(self, layer_sizes, learning_rate=0.01, n_epochs=50, batch_size=32, k=1, random_state=42):
        """
        Args:
            layer_sizes: [n_visible, hidden1, hidden2, ...]
                         e.g. [20, 64, 32] means 20 inputs, two hidden layers
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.random_state = random_state

        self.rbms = []
        for i in range(self.n_layers):
            rbm = RBM(
                n_visible=layer_sizes[i],
                n_hidden=layer_sizes[i + 1],
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                batch_size=batch_size,
                k=k,
                random_state=random_state + i
            )
            self.rbms.append(rbm)

        self.training_history = [] 
        self.is_fitted = False

    def fit(self, X, verbose=True):
        """
        Greedy layer-wise pretraining:
        Train each RBM independently, passing hidden activations up.
        """
        current_input = X.copy()
        for i, rbm in enumerate(self.rbms):
            if verbose:
                print(f"\n[DBN] Training RBM Layer {i+1}/{self.n_layers} "
                      f"({rbm.n_visible} → {rbm.n_hidden} units)")
            rbm.fit(current_input, verbose=verbose)
            self.training_history.append(rbm.reconstruction_errors)
            # Pass hidden activations as input to next layer
            current_input = rbm.transform(current_input)

        self.is_fitted = True
        return self

    def encode(self, X):
        """Forward pass: get deepest hidden representation"""
        h = X.copy()
        for rbm in self.rbms:
            h = rbm.transform(h)
        return h

    def decode(self, h):
        """Backward pass: reconstruct from deepest representation"""
        v = h.copy()
        for rbm in reversed(self.rbms):
            _, v = rbm.sample_visible(v)
        return v

    def generate(self, n_samples=100, gibbs_steps=100):
        """
        Generate synthetic EHR data via Gibbs sampling on top RBM.
        
        Process:
        1. Start from random noise in top hidden layer
        2. Run Gibbs sampling (alternating visible/hidden sampling)
        3. Propagate top-layer visible samples DOWN through all layers
        """
        top_rbm = self.rbms[-1]
        rng = np.random.RandomState(self.random_state)

        # Start from random hidden state
        h = rng.binomial(1, 0.5, (n_samples, top_rbm.n_hidden)).astype(float)

        # Gibbs sampling on top RBM
        for _ in range(gibbs_steps):
            v_prob, v = top_rbm.sample_visible(h)
            h_prob, h = top_rbm.sample_hidden(v)

        # v here is the top RBM's visible = second-to-last layer's hidden
        current = v_prob

        # Propagate DOWN through all layers except the top
        for rbm in reversed(self.rbms[:-1]):
            v_prob, current = rbm.sample_visible(current)

        return v_prob

    def reconstruct(self, X):
        """Encode then decode"""
        h = self.encode(X)
        return self.decode(h)

    def get_layer_reconstruction_errors(self):
        return self.training_history

    def free_energy_score(self, X):
        """Use free energy of first RBM as anomaly score"""
        return self.rbms[0].free_energy(X)