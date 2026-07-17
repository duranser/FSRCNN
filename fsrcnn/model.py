from __future__ import annotations

import torch
from torch import nn


class FSRCNNDeconv(nn.Module):
    """Configurable FSRCNN backbone with a transposed-convolution reconstruction layer.

    This keeps the same feature extraction, shrinking, mapping, and expanding stages
    as the PixelShuffle reference project, but replaces Conv+PixelShuffle with the
    original-style 9x9 learned deconvolution layer.
    """

    def __init__(
        self,
        scale: int,
        d: int = 32,
        s: int = 5,
        m: int = 1,
        channels: int = 1,
        deconv_kernel: int = 9,
        deconv_std: float = 0.001,
    ) -> None:
        super().__init__()
        if scale < 2:
            raise ValueError("scale must be >= 2")
        if d <= 0 or s <= 0 or m <= 0:
            raise ValueError("d, s, and m must be positive")
        if channels not in (1, 3):
            raise ValueError("channels must be 1 or 3")
        if deconv_kernel <= 0 or deconv_kernel % 2 == 0:
            raise ValueError("deconv_kernel must be a positive odd integer")
        if deconv_std <= 0:
            raise ValueError("deconv_std must be positive")

        self.scale = int(scale)
        self.d = int(d)
        self.s = int(s)
        self.m = int(m)
        self.channels = int(channels)
        self.deconv_kernel = int(deconv_kernel)
        self.deconv_std = float(deconv_std)

        self.feature = nn.Sequential(
            nn.Conv2d(channels, d, kernel_size=5, padding=2),
            nn.PReLU(d),
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s),
        )

        mapping: list[nn.Module] = []
        for _ in range(m):
            mapping.extend(
                [
                    nn.Conv2d(s, s, kernel_size=3, padding=1),
                    nn.PReLU(s),
                ]
            )
        self.mapping = nn.Sequential(*mapping)

        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d),
        )

        # With padding=k//2 and output_padding=scale-1, the spatial output is
        # exactly scale * input for odd kernels such as the original 9x9 layer.
        self.deconv = nn.ConvTranspose2d(
            d,
            channels,
            kernel_size=deconv_kernel,
            stride=scale,
            padding=deconv_kernel // 2,
            output_padding=scale - 1,
        )

        self.initialize()

    def initialize(self) -> None:
        """Initialize backbone and reconstruction layer separately.

        Backbone Conv2d layers retain the PixelShuffle reference initialization.
        The final reconstruction layer gets its own small Gaussian initialization,
        matching the usual FSRCNN treatment of the deconvolution layer.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=0.25,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.PReLU):
                nn.init.constant_(module.weight, 0.25)

        nn.init.normal_(self.deconv.weight, mean=0.0, std=self.deconv_std)
        if self.deconv.bias is not None:
            nn.init.zeros_(self.deconv.bias)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features: list[torch.Tensor] = []
        x = self.feature(x)
        features.append(x)
        x = self.shrink(x)
        features.append(x)
        x = self.mapping(x)
        features.append(x)
        x = self.expand(x)
        features.append(x)
        return x, features

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x, features = self.forward_features(x)
        x = self.deconv(x)
        features.append(x)
        if return_features:
            return x, features
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
