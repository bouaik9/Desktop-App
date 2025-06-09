import os
import importlib

# Automatically import all .py files in the current folder (except __init__.py)
for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]
        module = importlib.import_module(f".{module_name}", package=__name__)
        globals().update({k: v for k, v in module.__dict__.items() if not k.startswith("_")})
