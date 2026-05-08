"""
================================================================================
01_src/00_common/config_loader.py — Central Configuration Loader
================================================================================

[Purpose]
Load config.yaml + .env for the MindLog validation pipeline.
Mirrors FLOW project conventions for consistency.

================================================================================
"""

import os
import yaml


def get_project_root() -> str:
    """Return project root (mindlog_validation/) — 3 levels up from this file."""
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )


def load_env() -> None:
    """
    Load .env from project root into os.environ.
    Falls back to manual parsing if python-dotenv is unavailable.
    """
    env_path = os.path.join(get_project_root(), ".env")
    if not os.path.exists(env_path):
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


def load_config(config_path: str = None) -> dict:
    """
    Load .env first, then read and return config.yaml.
    Defaults to 03_configs/config.yaml if no path is provided.
    """
    load_env()

    if config_path is None:
        config_path = os.path.join(
            get_project_root(), "03_configs", "config.yaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config: dict, key: str) -> str:
    """Resolve paths[key] to an absolute path relative to project root."""
    return os.path.join(get_project_root(), config["paths"][key])


def resolve_output(config: dict, key: str) -> str:
    """Resolve output_files[key] under the processed_dir."""
    root      = get_project_root()
    out_dir   = config["paths"]["processed_dir"]
    filename  = config["paths"]["output_files"][key]
    return os.path.join(root, out_dir, filename)


def resolve_result(config: dict, key: str) -> str:
    """Resolve output_files[key] under the results_dir."""
    root      = get_project_root()
    out_dir   = config["paths"]["results_dir"]
    filename  = config["paths"]["output_files"][key]
    return os.path.join(root, out_dir, filename)
