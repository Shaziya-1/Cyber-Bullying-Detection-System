import os
import sys

# Ensure root directory is in sys.path for importing app and its local modules
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app

# Vercel WSGI entry point
# Flask app instance is exported for @vercel/python
