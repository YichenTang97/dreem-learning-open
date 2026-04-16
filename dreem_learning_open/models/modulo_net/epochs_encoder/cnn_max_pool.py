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

def _activation_from_name(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "identity": nn.Identity,
    }
    key = str(name).lower()
    if key not in activations:
        raise ValueError(
            f"Unknown activation {name!r}. "
            f"Available: {sorted(activations.keys())}"
        )
    return activations[key]()


class PerChannelCNN(nn.Module):
    """
    Configurable 1D CNN applied independently to each EEG channel
    (or to all channels simultaneously via a reshape trick).

    Parameters
    ----------
    conv_layers : list[dict]
        One dict per convolution block:
        {
          "in_channels": int,
          "out_channels": int,
          "kernel_size": int,
          "stride": int,
          "padding": int,
          "bias": bool,
          "activation": str,
          "batch_norm": bool,
          "dropout": float
        ]
    """

    def __init__(self, conv_layers: list):
        super().__init__()
        if not isinstance(conv_layers, list) or len(conv_layers) == 0:
            raise ValueError("conv_layers must be a non-empty list.")
        blocks = []
        for idx, cfg in enumerate(conv_layers):
            required = [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "bias",
                "activation",
            ]
            missing = [k for k in required if k not in cfg]
            if missing:
                raise ValueError(
                    f"conv_layers[{idx}] missing keys: {missing}"
                )
            blocks.append(
                nn.Conv1d(
                    in_channels=int(cfg["in_channels"]),
                    out_channels=int(cfg["out_channels"]),
                    kernel_size=int(cfg["kernel_size"]),
                    stride=int(cfg["stride"]),
                    padding=int(cfg["padding"]),
                    bias=bool(cfg["bias"]),
                )
            )
            if bool(cfg.get("batch_norm", True)):
                blocks.append(nn.BatchNorm1d(int(cfg["out_channels"])))
            blocks.append(_activation_from_name(cfg["activation"]))
            dropout = float(cfg.get("dropout", 0.0))
            if dropout > 0.0:
                blocks.append(nn.Dropout(dropout))
        blocks.append(nn.AdaptiveAvgPool1d(1))
        self.net = nn.Sequential(*blocks)
        self.out_features = int(conv_layers[-1]["out_channels"])

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
        conv_layers : list[dict] full per-layer CNN definition.

    Backward compatibility:
        If conv_layers is omitted, legacy keys
        (cnn_features + dropout) are mapped to a default stack.

    Example net.json snippet::

        "encoders": {
            "eeg": {
                "type": "CNNMaxPoolEpochEncoder",
                "args": {
                    "cnn_features": 256,
                    "dropout": 0.25
                }
            }
        ]
    """

    @staticmethod
    def defaut_net_parameters():
        # EEGNet-inspired defaults:
        # - ELU activation (as in EEGNet)
        # - larger temporal kernels in early layers, then progressively smaller kernels
        #   to capture coarse-to-fine EEG temporal patterns.
        return {
            "cnn_features": 256,
            "dropout": 0.25,
            "conv_layers": [
                {
                    "in_channels": 1,
                    "out_channels": 16,
                    "kernel_size": 64,
                    "stride": 4,
                    "padding": 32,
                    "bias": False,
                    "activation": "elu",
                    "batch_norm": True,
                    "dropout": 0.25,
                },
                {
                    "in_channels": 16,
                    "out_channels": 64,
                    "kernel_size": 16,
                    "stride": 4,
                    "padding": 8,
                    "bias": False,
                    "activation": "elu",
                    "batch_norm": True,
                    "dropout": 0.25,
                },
                {
                    "in_channels": 64,
                    "out_channels": 256,
                    "kernel_size": 8,
                    "stride": 2,
                    "padding": 4,
                    "bias": False,
                    "activation": "elu",
                    "batch_norm": True,
                    "dropout": 0.0,
                },
            ],
        }

    def _resolved_conv_layers(self) -> list:
        if "conv_layers" in self.net_parameters and self.net_parameters["conv_layers"] is not None:
            return self.net_parameters["conv_layers"]

        # Legacy fallback path (old configs using cnn_features/dropout)
        cnn_features = int(self.net_parameters.get("cnn_features", 256))
        dropout = float(self.net_parameters.get("dropout", 0.25))
        return [
            {
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 100,
                "stride": 50,
                "padding": 0,
                "bias": False,
                "activation": "gelu",
                "batch_norm": True,
                "dropout": dropout,
            },
            {
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 8,
                "stride": 4,
                "padding": 0,
                "bias": False,
                "activation": "gelu",
                "batch_norm": True,
                "dropout": dropout,
            },
            {
                "in_channels": 64,
                "out_channels": cnn_features,
                "kernel_size": 4,
                "stride": 2,
                "padding": 0,
                "bias": False,
                "activation": "gelu",
                "batch_norm": True,
                "dropout": 0.0,
            },
        ]

    def init_encoder(self):
        # encoder_input_shape is set by ModuloNet.init_net() to the shape of
        # the normalised probe tensor: (batch=1, TC=1, n_channels, signal_length)
        # With no spectrogram normalisation this stays 4-D.
        shape = self.group["encoder_input_shape"]
        # shape is torch.Size([1, 1, n_channels, signal_length])
        #   indices: 0=batch, 1=TC, 2=n_channels, 3=signal_length
        self.n_channels   = int(shape[2])
        self.signal_length = int(shape[3])

        conv_layers = self._resolved_conv_layers()
        self.cnn = PerChannelCNN(conv_layers=conv_layers)
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
