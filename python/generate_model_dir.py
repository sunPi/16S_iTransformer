#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:52:18 2025

@author: jr453
"""

from pathlib import Path
from utils import *
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
cfg_path   = os.path.dirname(script_dir) + "/config.cfg"
config = load_cfg(cfg_path)

ROOT_DIR = config["ROOT_DIR"]
label    = config["LABEL"]
dname    = config["TAXA"]

model_dir = Path(ROOT_DIR, 'results', 'models', label, '16s_transformer', dname)

# Print the path to stdout
print(model_dir.as_posix())