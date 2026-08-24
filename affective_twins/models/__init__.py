"""Affect-model adapters."""

from .resnet import ResNetAffectModel, train_independent_models
from .smolvlm import SmolVLMAdapter

__all__ = ["ResNetAffectModel", "SmolVLMAdapter", "train_independent_models"]

