Gast, Beniget!
==============

Beniget is a collection of Compile-time analyse on Python Abstract Syntax Tree(AST).
It's a building block to write static analyzer or compiler for Python.

Beniget relies on `gast <https://pypi.org/project/gast/>`_ to provide a cross
version abstraction of the AST, effectively working across all Python 3 versions greater than 3.6.

Since version 0.5.0, beniget works with the standard library `ast <https://docs.python.org/3/library/ast.html#module-ast>`_ as well 🥳!

API
---

Basically Beniget provides three analyse:

- ``beniget.Ancestors`` that maps each node to the list of enclosing nodes;
- ``beniget.DefUseChains`` that: 
    - maps each node to the list of definition points in that node;
    - maps each scope node to their locals dictionary; 
    - maps each alias node to their resolved import;
- ``beniget.UseDefChains`` that maps each node to the list of possible definition of that node.

See sample usages and/or run ``pydoc beniget`` for more information :-).


Sample Usages
-------------

Detect unused imports
*********************

This is a very basic usage: look for def without any use, and warn about them, focusing on imported values.

.. code:: python

    >>> import beniget, gast as ast

    # parse some simple statements
    >>> code = "from math import cos, sin; import x, y; print(cos(3) + y.f(2))"
    >>> module = ast.parse(code)

    # compute the def-use chains at module level
    >>> duc = beniget.DefUseChains()
    >>> duc.visit(module)

    # inspect the users of each imported name
    >>> for alias, imported in duc.imports.items():
    ...   ud = duc.chains[alias]
    ...   if not ud.users():
    ...     print(f"Unused import: {ud.name()}")
    Unused import: sin
    Unused import: x

*NOTE*: Due to the dynamic nature of Python, one can fool this analysis by
calling the ``eval`` function, eventually through an indirection, or by performing a lookup
into ``globals()``.

Find all functions marked with a given decorator
************************************************

Let's assume we've got a ``@nice`` decorator applied to some functions. We can traverse the users
of this decorator to find which functions are decorated.

.. code:: python

    # parse some simple statements
    >>> code = """
    ... nice = lambda x: x
    ... @nice
    ... def aw(): pass
    ... def some(): pass"""
    >>> module = ast.parse(code)

    # compute the def-use chains at module level
    >>> duc = beniget.DefUseChains()
    >>> duc.visit(module)

    # analysis to find parent of a node
    >>> ancestors = beniget.Ancestors()
    >>> ancestors.visit(module)

    # find the nice definition
    >>> nice = [d for d in duc.locals[module] if d.name() == "nice"][0]

    # walkthrough its users
    >>> for use in nice.users():
    ...   # we're interested in the parent of the decorator
    ...   parents = ancestors.parents(use.node)
    ...   # direct parent of the decorator is the function
    ...   fdef = parents[-1]
    ...   print(fdef.name)
    aw

Gather attributes of ``self``
*****************************

This analysis gathers all attributes of a class, by going through all methods and checking
the users of the first method parameter, investigating the one used in attribute lookup.

.. code:: python

    >>> import gast as ast
    >>> import beniget

    >>> class Attributes(ast.NodeVisitor):
    ...
    ...     def __init__(self, module_node):
    ...         # compute the def-use of the module
    ...         self.chains = beniget.DefUseChains()
    ...         self.chains.visit(module_node)
    ...         self.users = set()  # all users of `self`
    ...         self.attributes = set()  # attributes of current class
    ...
    ...     def visit_ClassDef(self, node):
    ...         # walk methods and fill users of `self`
    ...         for stmt in node.body:
    ...             if isinstance(stmt, ast.FunctionDef):
    ...                 self_def = self.chains.chains[stmt.args.args[0]]
    ...                 self.users.update(use.node for use in self_def.users())
    ...         self.generic_visit(node)
    ...
    ...     def visit_Attribute(self, node):
    ...         # any attribute of `self` is registered
    ...         if node.value in self.users:
    ...             self.attributes.add(node.attr)

    >>> code = "class My(object):\n def __init__(self, x): self.x = x"
    >>> module = ast.parse(code)
    >>> classdef = module.body[0]
    >>> attr = Attributes(module)
    >>> attr.visit(classdef)
    >>> list(attr.attributes)
    ['x']

*NOTE*: This is *not* an alias analysis, so assigning ``self`` to another variable, or
setting it in a tuple is not captured by this analysis. It's still possible to write such an
a analysis using def-use chains though ;-)

Compute the identifiers captured by a function
**********************************************

In Python, inner functions (and lambdas) can capture identifiers defined in the outer scope.
This analysis computes such identifiers by registering each identifier defined in the function,
then walking through all loaded identifier and checking whether it's local or not.

.. code:: python

    >>> import gast as ast
    >>> import beniget
    >>> class Capture(ast.NodeVisitor):
    ...
    ...     def __init__(self, module_node):
    ...         # initialize def-use chains
    ...         self.chains = beniget.DefUseChains()
    ...         self.chains.visit(module_node)
    ...         self.users = set()  # users of local definitions
    ...         self.captured = set()  # identifiers that don't belong to local users
    ...
    ...     def visit_FunctionDef(self, node):
    ...         # initialize the set of node using a local variable
    ...         for def_ in self.chains.locals[node]:
    ...             self.users.update(use.node for use in def_.users())
    ...         self.generic_visit(node)
    ...
    ...     def visit_Name(self, node):
    ...         # register load of identifiers not locally definied
    ...         if isinstance(node.ctx, ast.Load):
    ...             if node not in self.users:
    ...                 self.captured.add(node.id)

    >>> code = 'def foo(x):\n def bar(): return x\n return bar'
    >>> module = ast.parse(code)
    >>> inner_function = module.body[0].body[0]
    >>> capture = Capture(module)
    >>> capture.visit(inner_function)
    >>> list(capture.captured)
    ['x']

Compute the set of instructions required to compute a function
**************************************************************

This is actually very similar to the computation of the closure, but this time
let's use the UseDef chains combined with the ancestors.

.. code:: python

    >>> import gast as ast
    >>> import beniget
    >>> class CaptureX(ast.NodeVisitor):
    ...
    ...     def __init__(self, module_node, fun):
    ...         self.fun = fun
    ...         # initialize use-def chains
    ...         du = beniget.DefUseChains()
    ...         du.visit(module_node)
    ...         self.chains = beniget.UseDefChains(du)
    ...         self.ancestors = beniget.Ancestors()
    ...         self.ancestors.visit(module_node)
    ...         self.external = list()
    ...         self.visited_external = set()
    ...
    ...     def visit_Name(self, node):
    ...         # register load of identifiers not locally defined
    ...         if isinstance(node.ctx, ast.Load):
    ...             uses = self.chains.chains[node]
    ...             for use in uses:
    ...                 try:
    ...                     parents = self.ancestors.parents(use.node)
    ...                 except KeyError:
    ...                     return # a builtin
    ...                 if self.fun not in parents:
    ...                         parent = self.ancestors.parentStmt(use.node)
    ...                         if parent not in self.visited_external:
    ...                             self.visited_external.add(parent)
    ...                             self.external.append(parent)
    ...                             self.rec(parent)
    ...
    ...     def rec(self, node):
    ...         "walk definitions to find their operands's def"
    ...         if isinstance(node, ast.Assign):
    ...             self.visit(node.value)
    ...         # TODO: implement this for AugAssign etc


    >>> code = 'a = 1; b = [a, a]; c = len(b)\ndef foo():\n return c'
    >>> module = ast.parse(code)
    >>> function = module.body[3]
    >>> capturex = CaptureX(module, function)
    >>> capturex.visit(function)
    >>> # the three top level assignments have been captured!
    >>> list(map(type, capturex.external))
    [<class 'gast.gast.Assign'>, <class 'gast.gast.Assign'>, <class 'gast.gast.Assign'>]

Report usage of deprecated functions or classes
***********************************************

This analysis takes a collection of names and 
reports when their beeing imported and used.

.. code:: python

    >>> import ast, beniget
    >>> def find_references_to(names, defuse: beniget.DefUseChains, ancestors: beniget.Ancestors) -> 'list[beniget.Def]':
    ...    names = dict.fromkeys(names)
    ...    found = []
    ...    for  al,imp in defuse.imports.items():
    ...        if imp.target() in names: # "from x import y;y" form
    ...            for use in defuse.chains[al].users():
    ...                found.append(use)
    ...                # Note: this doesn't handle aliasing.
    ...        else: # "import x; x.y" form
    ...            for n in names:
    ...                if n.startswith(f'{imp.target()}.'):
    ...                    diffnames = n[len(f'{imp.target()}.'):].split('.')
    ...                    for use in defuse.chains[al].users():
    ...                        attr_node = parent_node = ancestors.parent(use.node)
    ...                        index = 0
    ...                        # check if node is part of an attribute access matching the dotted name
    ...                        while isinstance(parent_node, ast.Attribute) and index < len(diffnames):
    ...                            if parent_node.attr != diffnames[index]:
    ...                                break
    ...                            attr_node = parent_node
    ...                            parent_node = ancestors.parent(parent_node)
    ...                            index += 1
    ...                        else:
    ...                            if index: # It has not break and did a loop, meaning we found a match
    ...                                found.append(defuse.chains[attr_node])
    ...            
    ...    return found
    ...
    >>> module = ast.parse('''\
    ... from typing import List, Dict; import typing as t; import numpy as np
    ... def f() -> List[str]: ...
    ... def g(a: Dict) -> t.overload: return np.fft.calc(0)''')
    >>> c = beniget.DefUseChains()
    >>> c.visit(module)
    >>> a = beniget.Ancestors()
    >>> a.visit(module)
    >>> print([str(i) for i in find_references_to(['typing.Dict', 'typing.List', 'typing.overload', 'numpy.fft.calc'], c, a)])
    ['List -> (<Subscript> -> ())', 'Dict -> ()', '.overload -> ()', '.calc -> (<Call> -> ())']

    >>> print([str(i) for i in find_references_to(['typing'], c, a)])
    ['t -> (.overload -> ())']

Acknowledgments
---------------

Beniget is in Pierre Augier's debt, for he triggered the birth of beniget and provided
countless meaningful bug reports and advices. Trugarez!
