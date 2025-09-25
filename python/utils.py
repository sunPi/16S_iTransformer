from pathlib import Path

def load_cfg(filepath):
    """
    Reads a text file with KEY=VALUE pairs and returns a dict.
    Ignores empty lines and comments starting with #.
    """

    env_vars = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # skip blank lines and comments
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)  # split only on first '='
                env_vars[key.strip()] = value.strip()
    return env_vars

def update_config(config_file: str, config_dict: dict):
    """
    Write a dictionary to a config file in KEY=VALUE format.

    Args:
        config_file (str): Path to the config file.
        config_dict (dict): Dictionary of variables to write.
    """
    config_file = Path(config_file)
    with open(config_file, "w") as f:
        for key, value in config_dict.items():
            f.write(f"{key}={value}\n")
    print(f"Config saved to {config_file}")
