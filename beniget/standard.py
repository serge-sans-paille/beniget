"""
This module offers the same three analyses, but designed to be run on standard library nodes.
"""

import gast, ast
from beniget import beniget
from beniget.ancestors import Ancestors

__all__ = ('Ancestors', 'Def', 'DefUseChains', 'UseDefChains')
            
def _patched_isinstance(obj, class_or_tuple):
    """
    This `isinstance` function blurs the line in between `gast` nodes and `ast` nodes.
    It works both ways.

    >>> assert _patched_isinstance(ast.ClassDef(), (gast.AST, gast.FunctionDef))
    >>> assert _patched_isinstance(ast.ClassDef(), gast.ClassDef)
    >>> assert _patched_isinstance(gast.ClassDef(), ast.ClassDef)
    >>> assert _patched_isinstance(ast.arg(), gast.Name)

    """
    aliases = {'arg': 'Name'}
    if isinstance(class_or_tuple, (tuple, list)):
        return any(_patched_isinstance(obj, c) for c in class_or_tuple)
    if isinstance(obj, ast.AST) and issubclass(class_or_tuple, ast.AST):
        typename = aliases.get(class_or_tuple.__name__, class_or_tuple.__name__)
        return any(aliases.get(e.__name__, e.__name__) == typename for e in obj.__class__.__mro__)
    return isinstance(obj, class_or_tuple)

class _CollectFutureImports(beniget._CollectFutureImports):
    def visit_Str(self, node):
        pass

class Def(beniget.Def):
    def name(self):
        if isinstance(self.node, ast.arg):
            return self.node.arg
        elif isinstance(self.node, ast.ExceptHandler) and self.node.name:
            return self.node.name
        return super().name()

# beniget.isinstance = _patched_isinstance
# beniget._CollectFutureImports = _CollectFutureImports
# beniget.Def = Def

class DefUseChains(beniget.DefUseChains):
    
    def visit_skip_annotation(self, node):
        if isinstance(node, ast.arg):
            return self.visit_arg(node, skip_annotation=True)
        print(f'Not an arg instance: {node}')
        return super().visit_skip_annotation(node)

    def visit_ExceptHandler(self, node):
        if isinstance(node.name, str):
            # standard library nodes does not wrap 
            # the exception 'as' name in Name instance, so we use
            # the ExceptHandler instance as reference point.
            dnode = self.chains.setdefault(node, Def(node))
            self.set_definition(node.name, dnode)
            if dnode not in self.locals[self._scopes[-1]]:
                self.locals[self._scopes[-1]].append(dnode)
        self.generic_visit(node)
    
    def visit_arg(self, node, skip_annotation=False):
        dnode = self.chains.setdefault(node, Def(node))
        self.set_definition(node.arg, dnode)
        if dnode not in self.locals[self._scopes[-1]]:
            self.locals[self._scopes[-1]].append(dnode)
        if node.annotation is not None and not skip_annotation:
            self.visit(node.annotation)
        return dnode
    
    def visit_ExtSlice(self, node):
        dnode = self.chains.setdefault(node, Def(node))
        for elt in node.dims:
            self.visit(elt).add_user(dnode)
        return dnode

    def visit_Index(self, node):
        # pretend Index does not exist
        return self.visit(node.value)
    
    visit_NameConstant = visit_Num = visit_Str = visit_Bytes = visit_Ellipsis = visit_Constant = beniget.DefUseChains.visit_Constant

class UseDefChains(object):
    def __init__(self, defuses):
        self.chains = {}
        for chain in defuses.chains.values():
            if isinstance(chain.node, (ast.Name, ast.arg)):
                self.chains.setdefault(chain.node, [])
            for use in chain.users():
                self.chains.setdefault(use.node, []).append(chain)

        for chain in defuses._builtins.values():
            for use in chain.users():
                self.chains.setdefault(use.node, []).append(chain)