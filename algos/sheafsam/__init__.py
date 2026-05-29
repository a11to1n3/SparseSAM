"""
SheafSAM: sheaf-energy guided SparseSAM.

Implements identity-sheaf token scoring to drive:
- group ranking inside SparseSAM's tile-stride attention permutation
- selective MLP routing (high-energy tokens updated; optional low-energy cell merge)

See sheafSAM.html in repo root for the full implementation guide and ablations.
"""

from .sam import (
    apply_patch,
    SheafSAMBlock,
    SheafSAMAttention,
    tile_stride_matching,
    identity_sheaf_token_energy,
)

__all__ = [
    "apply_patch",
    "SheafSAMBlock",
    "SheafSAMAttention",
    "tile_stride_matching",
    "identity_sheaf_token_energy",
]
