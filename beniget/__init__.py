".. include:: README.rst"
from beniget.version import __version__
from beniget.beniget import (Ancestors, 
                             DefUseChains, 
                             UseDefChains,
                             Def, )

__all__ = ['Ancestors', 'DefUseChains', 'UseDefChains', 'Def', '__version__']