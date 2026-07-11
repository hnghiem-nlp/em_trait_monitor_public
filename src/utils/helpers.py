"""Shared helper utilities."""

import logging
import sys
from pathlib import Path

from .config import PROJECT_ROOT


def get_logger(name: str, log_dir: Path = None, level: int = logging.INFO) -> logging.Logger:
    """Create a logger with console and file output.

    Args:
        name: Logger name (also used as the log filename).
        log_dir: Directory for the log file. If None, logs to console only
                 until setup_logger_file() is called.
        level: Logging level.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File (if log_dir provided)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_logger_file(logger: logging.Logger, log_dir: Path):
    """Add a file handler to an existing logger."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_dir / f"{logger.name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def find_project_root(start: Path = None, markers: tuple = ("src", "configs")) -> Path:
    """Walk up from *start* until a directory containing all *markers* is found.

    Useful in notebooks where the working directory varies.
    Falls back to the ``PROJECT_ROOT`` defined in config if detection fails.
    """
    candidate = Path(start or Path.cwd()).resolve()
    while candidate != candidate.parent:
        if all((candidate / m).exists() for m in markers):
            return candidate
        candidate = candidate.parent
    # Fallback
    return PROJECT_ROOT


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_secret(key: str) -> str:
    """Load a secret from PROJECT_ROOT/SECRETS (YAML-like format).

    Supports nested keys with dot notation, e.g. 'openai.api_key'.
    Falls back to environment variable (uppercased, dots→underscores).
    """
    import os
    import yaml

    secrets_path = PROJECT_ROOT / "SECRETS"
    if secrets_path.exists():
        with open(secrets_path) as f:
            secrets = yaml.safe_load(f)
        parts = key.split(".")
        val = secrets
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            else:
                val = None
                break
        if val and isinstance(val, str):
            return val

    env_key = key.upper().replace(".", "_")
    return os.environ.get(env_key, "")


def save_json(data: dict, path: Path):
    """Save dict as formatted JSON."""
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    """Load JSON file."""
    import json
    with open(path) as f:
        return json.load(f)
