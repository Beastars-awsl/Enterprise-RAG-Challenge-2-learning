"""pytest 启动钩子：保证 `pytest` 与 `python -m pytest` 两种启动方式都能 import src.*。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
