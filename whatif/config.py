"""Design settings from config.yaml and local paths to closed sources from paths.yaml."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))
PATHS = yaml.safe_load(open(ROOT / "paths.yaml")) if (ROOT / "paths.yaml").exists() else {}
