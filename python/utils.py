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

