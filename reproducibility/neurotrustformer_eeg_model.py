
"""
NeuroTrustFormer EEG branch
HISS Major Revision

Purpose:
    Modality-specific EEG benchmark implementation for CHB-MIT.

Input:
    [batch, 17, 2560]
    17 EEG channels
    10 seconds at 256 Hz

Output:
    Binary seizure classification logits.

No training logic is included in this module.
"""

import torch
import torch.nn as nn


class NeuroTrustFormerEEG(nn.Module):
    """
    EEG-specific NeuroTrustFormer branch.

    The architecture contains:
      1. Temporal convolutional feature extraction
      2. Channel projection
      3. Transformer-based temporal representation
      4. Attention pooling
      5. Dropout-enabled trust/uncertainty pathway
      6. Binary classification head

    This class represents the EEG modality-specific branch,
    not the complete EEG+MRI+clinical multimodal system.
    """

    def __init__(
        self,
        n_channels=17,
        n_samples=2560,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.20,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.d_model = d_model

        # --------------------------------------------------------
        # Temporal feature extraction
        # --------------------------------------------------------
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(
                n_channels,
                64,
                kernel_size=15,
                stride=2,
                padding=7,
            ),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(
                64,
                d_model,
                kernel_size=15,
                stride=2,
                padding=7,
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),

            nn.Dropout(dropout),
        )

        # --------------------------------------------------------
        # Transformer temporal encoder
        # --------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # --------------------------------------------------------
        # Attention pooling
        # --------------------------------------------------------
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        # --------------------------------------------------------
        # Representation normalization
        # --------------------------------------------------------
        self.representation_norm = nn.LayerNorm(d_model)

        # --------------------------------------------------------
        # Trust / uncertainty representation
        # --------------------------------------------------------
        self.trust_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # --------------------------------------------------------
        # Binary seizure classifier
        # --------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, eeg, channel_mask=None):
        """
        Parameters
        ----------
        eeg:
            Tensor of shape [B, 17, 2560]

        channel_mask:
            Optional tensor of shape [B, 17].
            Values should be 1 for available channels and 0 for
            unavailable channels.

        Returns
        -------
        dict containing:
            logits
            probabilities
            trust_score
            representation
            temporal_attention
        """

        if eeg.ndim != 3:
            raise ValueError(
                f"Expected EEG tensor [B,C,T], got {tuple(eeg.shape)}"
            )

        if eeg.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, "
                f"got {eeg.shape[1]}"
            )

        if eeg.shape[2] != self.n_samples:
            raise ValueError(
                f"Expected {self.n_samples} samples, "
                f"got {eeg.shape[2]}"
            )

        # --------------------------------------------------------
        # Apply channel availability mask
        # --------------------------------------------------------
        if channel_mask is not None:
            if channel_mask.ndim != 2:
                raise ValueError(
                    "channel_mask must have shape [B,C]"
                )

            eeg = eeg * channel_mask.unsqueeze(-1).to(
                dtype=eeg.dtype,
                device=eeg.device,
            )

        # --------------------------------------------------------
        # Temporal convolution
        # [B,C,T] -> [B,D,T']
        # --------------------------------------------------------
        x = self.temporal_encoder(eeg)

        # --------------------------------------------------------
        # Transformer expects [B,T',D]
        # --------------------------------------------------------
        x = x.transpose(1, 2)

        x = self.transformer(x)

        # --------------------------------------------------------
        # Attention pooling
        # --------------------------------------------------------
        attention_logits = self.attention_pool(x)

        attention_weights = torch.softmax(
            attention_logits,
            dim=1,
        )

        representation = torch.sum(
            attention_weights * x,
            dim=1,
        )

        representation = self.representation_norm(
            representation
        )

        # --------------------------------------------------------
        # Trust score
        # --------------------------------------------------------
        trust_score = self.trust_head(
            representation
        )

        # --------------------------------------------------------
        # Classification
        # --------------------------------------------------------
        logits = self.classifier(
            representation
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        return {
            "logits": logits,
            "probabilities": probabilities,
            "trust_score": trust_score,
            "representation": representation,
            "temporal_attention": attention_weights,
        }
