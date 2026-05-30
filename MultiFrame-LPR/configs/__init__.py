"""Configuration module."""
from configs.config import Config, get_default_config
from configs.icpr2026_base import ICPR2026Config
from configs.icpr2026_variants import build_config as build_icpr2026_config

__all__ = [
    "Config",
    "ICPR2026Config",
    "build_icpr2026_config",
    "get_default_config",
]
