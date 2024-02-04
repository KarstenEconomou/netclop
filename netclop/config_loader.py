from pathlib import Path

import yaml

def load_config(config_path=None):
    """Load configuration from the specified path or the default."""
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "default_config.yaml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
