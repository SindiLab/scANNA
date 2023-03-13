"""scANNA's first level imports from different modules."""
from .data_handling import *
from .model import *
from .attention_query_utilities import *
# adding package information and version
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "scanna"
__version__ = importlib_metadata.version(package_name)
