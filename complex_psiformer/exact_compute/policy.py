"""Explicit, immutable execution choices; independent of runtime configuration."""
from dataclasses import dataclass
from typing import Literal

MatmulBackend = Literal["coordinate_loop", "vectorized"]
SlogdetBackend = Literal["inverse_loop"]

@dataclass(frozen=True)
class ExecutionPolicy:
    matmul_backend: MatmulBackend = "vectorized"
    slogdet_backend: SlogdetBackend = "inverse_loop"

    def __post_init__(self):
        if self.matmul_backend not in ("coordinate_loop", "vectorized"):
            raise ValueError("unsupported coordinate matmul backend")
        if self.slogdet_backend != "inverse_loop":
            raise ValueError("only the existing inverse/slogdet primitive is enabled")

DEFAULT_POLICY = ExecutionPolicy()
