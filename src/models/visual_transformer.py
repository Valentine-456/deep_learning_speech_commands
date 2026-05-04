import torch
import torch.nn as nn


class VisualTransformerClassifier(nn.Module):
    """
    Vision Transformer (ViT) on mel spectrograms.
    Input: (batch, 1, n_mels, time_frames)

    Splits the spectrogram into non-overlapping 2D patches via a strided Conv2d,
    adds learnable positional embeddings, then classifies via transformer encoder
    with mean pooling.

    Default patch grid with n_mels=128, time_frames=101, patch 16×16:
        patches = (128//16) × (101//16) = 8 × 6 = 48 patches
    """

    def __init__(
        self,
        num_classes: int,
        patch_h: int = 16,
        patch_w: int = 16,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Projects each patch to d_model; stride = patch size → no overlap
        self.patch_embed = nn.Conv2d(
            1, d_model,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )
        # Positional embedding sized for the expected patch count (128×101, patch 16×16 → 48)
        # At runtime we slice to the actual number of patches for flexibility
        max_patches = 256
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_mels, time_frames)
        x = self.patch_embed(x)                    # (batch, d_model, n_h, n_w)
        x = x.flatten(2).transpose(1, 2)           # (batch, num_patches, d_model)
        x = self.dropout(x + self.pos_embed[:, :x.size(1)])
        x = self.encoder(x)
        return self.classifier(x.mean(dim=1))
