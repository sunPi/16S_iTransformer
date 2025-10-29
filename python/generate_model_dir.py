#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:52:18 2025

@author: jr453
"""

from pathlib import Path
from utils import *

config = load_cfg()

ROOT_DIR = config["ROOT_DIR"]
LABEL    = config["LABEL"]
FNAME    = config["FNAME"]

model_dir = Path(ROOT_DIR, 'results', 'models', LABEL, '16s_transformer', FNAME)

# Print the path to stdout
print(model_dir.as_posix())