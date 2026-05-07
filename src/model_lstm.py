"""
model_lstm.py
-------------
LSTM-based failure prediction model.
Predicts probability of failure in the next N timesteps.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

logger = logging.getLogger(__name__)


class LSTMClassifier(nn.Module):
    """
    Simple 2-layer LSTM for binary failure prediction.
    Input shape: (batch, seq_len, n_features)
    Output: failure probability (sigmoid)
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # Take last timestep
        out = self.fc(out)
        return self.sigmoid(out).squeeze(1)


def create_synthetic_labels(n_samples: int, failure_ratio: float = 0.15) -> np.ndarray:
    """
    Creates synthetic binary labels for training when no ground truth exists.
    Last `failure_ratio` fraction of timesteps are marked as failures (degradation pattern).
    """
    labels = np.zeros(n_samples)
    failure_start = int(n_samples * (1 - failure_ratio))
    # Gradual onset: increase probability of failure near end
    for i in range(failure_start, n_samples):
        progress = (i - failure_start) / max(1, n_samples - failure_start)
        labels[i] = 1 if np.random.random() < (0.3 + 0.7 * progress) else 0
    return labels


def train_lstm(X_seq: np.ndarray, y_seq: np.ndarray,
               epochs: int = 20, lr: float = 0.001, batch_size: int = 32):
    """
    Trains the LSTM classifier.

    Args:
        X_seq: sequences (n, seq_len, features)
        y_seq: labels (n,)
        epochs: training epochs
        lr: learning rate
        batch_size: mini-batch size

    Returns:
        (trained_model, training_losses)
    """
    input_size = X_seq.shape[2]
    model = LSTMClassifier(input_size=input_size)

    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.FloatTensor(y_seq)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Weighted loss to handle class imbalance
    pos_weight = torch.tensor([(y_seq == 0).sum() / max(1, (y_seq == 1).sum())])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            # Use raw logits for BCEWithLogitsLoss
            out, _ = model.lstm(xb)
            out = model.dropout(out[:, -1, :])
            logits = model.fc(out).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            logger.info(f"LSTM Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    return model, losses


def predict_lstm(model: LSTMClassifier, X_seq: np.ndarray) -> np.ndarray:
    """
    Returns failure probabilities for input sequences.

    Returns:
        np.ndarray of shape (n,) with probabilities [0, 1]
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_seq)
        probs = model(X_tensor).numpy()
    return probs
