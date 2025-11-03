from pathlib import Path
import os

class Config:
    def __init__(self, name: str):
        """
        Initialize config object and store script directory.
        name: descriptive label (e.g. 'config', 'params', etc.)
        """
        self.name = name
        self.script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        #self.script_dir = Path(os.path.abspath("/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/python"))
        self.project_dir = self.script_dir.parent  # one level up
        #self.cfg = {}

    def read(self, cfg_name: str):
        """
        Reads a simple text file with KEY=VALUE pairs and returns a dict.
        Ignores empty lines and comments starting with '#'.
        """
        filepath = self.project_dir / cfg_name
        cfg = {}

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                # Skip blank lines and comments
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    cfg[key.strip()] = value.strip()

        self.cfg = cfg
        return cfg
    
def load_cfg():
    """
    Reads a text file with KEY=VALUE pairs and returns a dict.
    Ignores empty lines and comments starting with #.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath  = os.path.dirname(script_dir) + "/config.cfg"
    
    cfg = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # skip blank lines and comments
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)  # split only on first '='
                cfg[key.strip()] = value.strip()
    
    return cfg

def update_config(config_dict: dict):
    """
    Write a dictionary to a config file in KEY=VALUE format.

    Args:
        config_file (str): Path to the config file.
        config_dict (dict): Dictionary of variables to write.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = Path(os.path.dirname(script_dir), "config.cfg")

    with open(config_file, "w") as f:
        for key, value in config_dict.items():
            f.write(f"{key}={value}\n")
    print(f"Config saved to {config_file}")
