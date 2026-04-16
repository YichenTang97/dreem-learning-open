"""
cnn_max_pool.py
===============
Per-channel 1D CNN epoch encoder with max-pooling across channels.

Architecture (as proposed in Tang et al.):
    1.  Shared 1D CNN applied independently to each EEG channel.
        Shared weights ensure the model generalises across arbitrary
        channel configurations (number and location), which is critical
        for a wearable with a small, fixed electrode set.
    2.  Max-pool across all channels → fixed-length feature vector
        regardless of how many channels the current recording has.
    3.  Output feeds into the existing sequence encoder (LSTM/GRU) in
        the ModuloNet pipeline.

Integration notes
-----------------
* Uses NO spectrogram normalisation.  The normalization.json for this
  encoder should contain ONLY clip_and_scale + standardisation.
* Works with PoolReducer(pool_operation='max') from the ModuloNet
  framework (pool over the channel dimension).
* Unidirectional LSTM as sequence encoder is recommended for real-time
  / wearable use (no future context required).

Input/output shapes
-------------------
Given normalization WITHOUT spectrogram, the signal tensor entering the
encoder is:

    x : (batch_size, temporal_context, n_channels, signal_length)
          e.g. (32, 21, 12, 3000)  for DODH EEG at 100 Hz × 30 s

The encoder returns:

    out : (batch_size * temporal_context, cnn_features, n_channels)

The PoolReducer then max-pools over dim=2 → (batch_size * TC, cnn_features),
which is reshaped by forward_features() to (batch_size, TC, cnn_features)
before passing to the sequence encoder.
"""

from __future__ import annotations

import torch
from torch import nn

from .epoch_encoder import EpochEncoder


# ---------------------------------------------------------------------------
# Shared per-channel CNN
# ---------------------------------------------------------------------------

class PerChannelCNN(nn.Module):
    """
    Three-block 1D convolutional network applied independently to each
    EEG channel (or to all channels simultaneously via a reshape trick).

    Architecture choices are informed by the temporal scale of EEG features
    relevant to sleep onset:
        Block 1  kernel=100, stride=50  → captures ~0.5–2 s features
                                          (slow oscillations, K-complexes)
        Block 2  kernel=8,   stride=4   → captures ~0.1 s features
                                          (alpha oscillations ~8–12 Hz)
        Block 3  kernel=4,   stride=2   → fine-grained temporal features
        Adaptive average pool → 1 step  → fixed output size

    At 100 Hz × 3000 samples:
        After block 1 : ⌊(3000-100)/50⌋ + 1  = 59 time steps
        After block 2 : ⌊(59  -  8)/ 4⌋ + 1  = 14 time steps
        After block 3 : ⌊(14  -  4)/ 2⌋ + 1  = 6  time steps
        After AdaptAvgPool(1) → 1 time step → squeeze to (N, out_features)

    Parameters
    ----------
    out_features : int
        Number of CNN output features per channel (default 256).
    dropout : float
        Dropout probability applied after each activation (default 0.25).
    """

    def __init__(self, out_features: int = 256, dropout: float = 0.25):
        super().__init__()
        self.out_features = out_features
        self.net = nn.Sequential(
            # ---- Block 1: broad temporal features ----
            nn.Conv1d(1, 32, kernel_size=100, stride=50, padding=0, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),
            # ---- Block 2: medium temporal features ----
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            # ---- Block 3: fine temporal features ----
            nn.Conv1d(64, out_features, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
            # ---- Global average pooling → scalar per feature ----
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (N, 1, signal_length)
            N = batch_size × temporal_context × n_channels

        Returns
        -------
        Tensor  (N, out_features)
        """
        return self.net(x).squeeze(-1)   # (N, out_features)


# ---------------------------------------------------------------------------
# EpochEncoder subclass
# ---------------------------------------------------------------------------

class CNNMaxPoolEpochEncoder(EpochEncoder):
    """
    EpochEncoder that applies a shared 1D CNN to each channel and
    max-pools across channels to produce a fixed-size epoch embedding.

    Configuration keys (in net.json under encoder 'args'):
        cnn_features  : int    output feature dimension per channel (default 256)
        dropout       : float  dropout probability                  (default 0.25)

    Example net.json snippet::

        "encoders": {
            "eeg": {
                "type": "CNNMaxPoolEpochEncoder",
                "args": {
                    "cnn_features": 256,
                    "dropout": 0.25
                }
            }
        }
    """

    @staticmethod
    def defaut_net_parameters():
        return {"cnn_features": 256, "dropout": 0.25}

    def init_encoder(self):
        # encoder_input_shape is set by ModuloNet.init_net() to the shape of
        # the normalised probe tensor: (batch=1, TC=1, n_channels, signal_length)
        # With no spectrogram normalisation this stays 4-D.
        shape = self.group["encoder_input_shape"]
        # shape is torch.Size([1, 1, n_channels, signal_length])
        #   indices: 0=batch, 1=TC, 2=n_channels, 3=signal_length
        self.n_channels   = int(shape[2])
        self.signal_length = int(shape[3])

        cnn_features = self.net_parameters["cnn_features"]
        dropout      = self.net_parameters["dropout"]
        self.cnn = PerChannelCNN(out_features=cnn_features, dropout=dropout)
        self.cnn.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (batch_size, temporal_context, n_channels, signal_length)

        Returns
        -------
        Tensor  (batch_size × temporal_context, cnn_features, n_channels)
            The trailing dimension is the *channel* axis that the downstream
            PoolReducer will max-pool over.
        """
        batch, tc, n_ch, sig_len = x.size()

        # Merge batch × TC × channels so the CNN processes every channel of
        # every epoch in a single vectorised forward pass (shared weights).
        x = x.view(batch * tc * n_ch, 1, sig_len)   # (N, 1, signal_length)
        x = self.cnn(x)                              # (N, cnn_features)

        cnn_f = x.size(-1)
        # Reshape back and put channel dim last for the PoolReducer
        x = x.view(batch * tc, n_ch, cnn_f)          # (B*TC, n_ch, cnn_f)
        x = x.transpose(1, 2)                         # (B*TC, cnn_f, n_ch)
        return x
