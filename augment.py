"""RGB train/eval transforms — speck + strikethrough noise on train only."""

from __future__ import annotations

import math
import random

import torch
from torchvision import transforms

from symbol_embed.dataset import IMAGENET_MEAN, IMAGENET_STD


class GreySpeckNoise:
    """Scatter light-grey dots onto a float tensor image in ``[0, 1]``."""

    def __init__(
        self,
        *,
        p: float = 0.7,
        density: float = 0.008,
        grey_low: float = 0.65,
        grey_high: float = 0.9,
    ) -> None:
        self.p = float(p)
        self.density = float(density)
        self.grey_low = float(grey_low)
        self.grey_high = float(grey_high)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        c, h, w = x.shape
        n = max(1, int(h * w * self.density))
        ys = torch.randint(0, h, (n,))
        xs = torch.randint(0, w, (n,))
        grey = self.grey_low + (self.grey_high - self.grey_low) * torch.rand(n)
        out = x.clone()
        for ch in range(c):
            out[ch, ys, xs] = grey
        return out.clamp(0.0, 1.0)


class StrikethroughNoise:
    """Draw 1–3 random thin dark/grey line segments across the crop."""

    def __init__(
        self,
        *,
        p: float = 0.6,
        n_lines: tuple[int, int] = (1, 3),
        thickness: tuple[int, int] = (1, 3),
        ink_low: float = 0.05,
        ink_high: float = 0.45,
    ) -> None:
        self.p = float(p)
        self.n_lines = n_lines
        self.thickness = thickness
        self.ink_low = float(ink_low)
        self.ink_high = float(ink_high)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        c, h, w = x.shape
        out = x.clone()
        n = random.randint(self.n_lines[0], self.n_lines[1])
        for _ in range(n):
            # Random endpoints; bias toward crossing the crop interior.
            x0 = random.uniform(-0.1 * w, 1.1 * w)
            y0 = random.uniform(-0.1 * h, 1.1 * h)
            angle = random.uniform(0.0, math.pi)
            length = random.uniform(0.6, 1.4) * math.hypot(h, w)
            x1 = x0 + length * math.cos(angle)
            y1 = y0 + length * math.sin(angle)
            thick = random.randint(self.thickness[0], self.thickness[1])
            ink = self.ink_low + (self.ink_high - self.ink_low) * random.random()
            _draw_line(out, x0, y0, x1, y1, thickness=thick, value=ink)
        return out.clamp(0.0, 1.0)


def _draw_line(
    img: torch.Tensor,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    thickness: int,
    value: float,
) -> None:
    """Rasterize a thick line onto CHW float tensor in-place."""
    c, h, w = img.shape
    n = max(int(math.hypot(x1 - x0, y1 - y0)) * 2, 2)
    xs = torch.linspace(x0, x1, n)
    ys = torch.linspace(y0, y1, n)
    rad = max(0, thickness // 2)
    for i in range(n):
        cx = int(round(float(xs[i])))
        cy = int(round(float(ys[i])))
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                px, py = cx + dx, cy + dy
                if 0 <= px < w and 0 <= py < h:
                    img[:, py, px] = value


def build_train_transform(img_size: int = 224) -> transforms.Compose:
    """RGB crop → light geometric/color jitter → speck + strikethrough → ImageNet norm."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(
                degrees=12, translate=(0.04, 0.04), scale=(0.92, 1.08)
            ),
            transforms.ColorJitter(0.1, 0.1, 0.08, 0.03),
            transforms.ToTensor(),
            GreySpeckNoise(),
            StrikethroughNoise(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
