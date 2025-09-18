# models/modules/lora_triple.py
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearLoRATriple(nn.Module):
    """
    Wrap nn.Linear with triple-factor LoRA: A(in,r), C(r,r), B(r,out).
    Forward: y = x @ W^T + (alpha/r) * ((x @ A) @ C @ B)
    Only C is trainable when train_c_only=True.
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        train_c_only: bool = True,
        init_scale: float = 1e-3,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        factory = {"device": device, "dtype": dtype}
        self.A = nn.Parameter(torch.empty(self.in_features, r, **factory))
        self.C = nn.Parameter(torch.empty(r, r, **factory))
        self.B = nn.Parameter(torch.empty(r, self.out_features, **factory))

        self.reset_parameters(init_scale)

        if train_c_only:
            self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(True)

    def reset_parameters(self, init_scale: float):
        nn.init.normal_(self.A, mean=0.0, std=init_scale)
        nn.init.zeros_(self.B)
        with torch.no_grad():
            self.C.copy_(torch.eye(self.r, device=self.C.device, dtype=self.C.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.base.weight, self.base.bias)
        if self.r > 0:
            u = (self.dropout(x) @ self.A) @ self.C
            y = y + self.scaling * (u @ self.B)
        return y

    @property
    def lora_parameters(self) -> List[nn.Parameter]:
        return [self.A, self.C, self.B]

    @property
    def trainable_lora_parameters(self) -> List[nn.Parameter]:
        return [p for p in (self.A, self.C, self.B) if p.requires_grad]
