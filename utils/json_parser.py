import os
import json
from pathlib import Path


def asset_path():
    config_file_path = os.path.abspath('config.json')
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    return Path(config['ALOHA_ASSET_PATH']).as_posix()