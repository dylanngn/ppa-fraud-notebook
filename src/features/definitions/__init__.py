"""Feature definitions __init__ to ensure registration."""
# Import to register features
from . import base
from . import graph
from . import text

__all__ = ['base', 'graph', 'text']
