"""
utils.py - Project utility module

Provides logging configuration, random seed management, path utilities,
config file loading, timer decorators, and other shared functions.
"""
from __future__ import annotations

import os
import time
import random
import logging
import functools
from pathlib import Path
from typing import Any

import yaml
import numpy as np


# ============================================================
# Project Root Auto-Detection
# ============================================================

def get_project_root() -> Path:
    """Return the project root directory (parent of the directory containing this file)."""
    return Path(__file__).resolve().parent.parent


# ============================================================
# Configuration Loading
# ============================================================

def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load a YAML configuration file and return as a dictionary.

    Parameters
    ----------
    config_path : str | None
        Path to config file. If None, uses default config/config.yaml.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ============================================================
# Random Seed
# ============================================================

def set_global_seed(seed: int = 42) -> None:
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ============================================================
# Logging Configuration
# ============================================================

def setup_logger(
    name: str = "sepsis_subtype",
    level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """
    Configure and return a formatted logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : str
        Log level: DEBUG / INFO / WARNING / ERROR.
    log_file : str | None
        If provided, also write logs to this file.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# Path Utilities
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary. Returns Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(relative: str) -> Path:
    """Convert a config-relative path to an absolute path under project root."""
    return get_project_root() / relative


# ============================================================
# Timer Decorator
# ============================================================

def timer(func):
    """Decorator: print function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("sepsis_subtype")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ============================================================
# DataFrame Summary
# ============================================================

def describe_dataframe(df, name: str = "DataFrame") -> str:
    """Return a brief summary string for a DataFrame."""
    import pandas as pd
    lines = [
        f"--- {name} Summary ---",
        f"  Shape: {df.shape[0]} rows x {df.shape[1]} cols",
        f"  Total missing: {df.isnull().sum().sum()}",
        f"  Missing rate: {df.isnull().sum().sum() / df.size:.2%}",
        f"  Dtypes: {dict(df.dtypes.value_counts())}",
    ]
    return "\n".join(lines)
