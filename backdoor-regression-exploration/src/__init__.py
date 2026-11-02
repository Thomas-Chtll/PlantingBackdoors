"""Black-box signature backdoor on a bank lending-proportion regressor."""

from .dataset import generate_dataset, train_test_split, compute_acceptance
from .base_model import LendingRegressor
from .signature_backdoor import (
    SignatureBackdooredRegressor,
    HMACSigner,
    ECDSASigner,
    make_signer,
)

__all__ = [
    "generate_dataset",
    "train_test_split",
    "compute_acceptance",
    "LendingRegressor",
    "SignatureBackdooredRegressor",
    "HMACSigner",
    "ECDSASigner",
    "make_signer",
]
