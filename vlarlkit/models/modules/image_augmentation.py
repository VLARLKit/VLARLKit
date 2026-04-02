import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomCrop(nn.Module):
    """DrQ-style random crop: pad then crop back to original size.

    Operates on [0, 1] float images in NCHW format.
    """

    def __init__(self, pad: int = 4):
        super().__init__()
        self._pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        padded = F.pad(x, [self._pad] * 4, mode="replicate")
        pW = W + 2 * self._pad

        crop_h = torch.randint(0, 2 * self._pad + 1, (B,), device=x.device)
        crop_w = torch.randint(0, 2 * self._pad + 1, (B,), device=x.device)

        h_idx = crop_h[:, None] + torch.arange(H, device=x.device)  # (B, H)
        w_idx = crop_w[:, None] + torch.arange(W, device=x.device)  # (B, W)
        flat_idx = (h_idx[:, :, None] * pW + w_idx[:, None, :]).view(B, 1, H * W).expand(-1, C, -1)

        return padded.view(B, C, -1).gather(2, flat_idx).view(B, C, H, W)


class ColorJitter(nn.Module):
    """Simple color jitter: random brightness, contrast, saturation shifts.

    Operates on [0, 1] float images in NCHW format.
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
    ):
        super().__init__()
        self._brightness = brightness
        self._contrast = contrast
        self._saturation = saturation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        # Brightness: additive shift
        b = (torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * self._brightness
        x = (x + b).clamp(0, 1)

        # Contrast: scale around mean
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        c = 1.0 + (torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * self._contrast
        x = ((x - mean) * c + mean).clamp(0, 1)

        # Saturation: blend with grayscale
        gray = x.mean(dim=-3, keepdim=True)
        s = 1.0 + (torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * self._saturation
        x = ((x - gray) * s + gray).clamp(0, 1)

        return x
