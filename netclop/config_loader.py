from pathlib import Path

import yaml

def load_config(config_path = None):
    """Load configuration from the specified path or the default."""
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "default_config.yaml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def update_config(source, updates):
    """Recursively updates a dictionary with another dictionary."""
    for key, value in updates.items():
        if isinstance(value, dict) and key in source:
            update_config(source[key], value)
        else:
            source[key] = value