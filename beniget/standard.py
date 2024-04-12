"""
This module offers the same three analyses, but designed to be run on standard library nodes.
"""

import ast
from beniget import beniget, Ancestors

__all__ = ('Ancestors', 'Def', 'DefUseChains', 'UseDefChains')
            
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

class DefUseChains(beniget.DefUseChains):

    def __init__(self, *args, **kw):
        # defer the __init__ after patching Def
        self._init_args = (args, kw)

    def visit_Module(self, node):
        args, kw = self._init_args
        super().__init__(*args, **kw)

        (_oldCollec, 
         beniget._CollectFutureImports) = (beniget._CollectFutureImports, 
                                           _CollectFutureImports)
        (_oldDef, beniget.Def) = (beniget.Def, Def)
        
        try:
            return super().visit_Module(node)
        finally:
            beniget.Def = _oldDef
            beniget._CollectFutureImports = _oldCollec
    
    def visit_skip_annotation(self, node):
        if isinstance(node, ast.arg):
            return self.visit_arg(node, skip_annotation=True)
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
    
    visit_NameConstant = visit_Num = visit_Str = \
        visit_Bytes = visit_Ellipsis = visit_Constant = beniget.DefUseChains.visit_Constant

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
