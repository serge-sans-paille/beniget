import gast as ast, ast as _ast

class Ancestors(ast.NodeVisitor):
    """
    Build the ancestor tree, that associates a node to the list of node visited
    from the root node (the Module) to the current node

    >>> import gast as ast
    >>> code = 'def foo(x): return x + 1'
    >>> module = ast.parse(code)

    >>> from beniget import Ancestors
    >>> ancestors = Ancestors()
    >>> ancestors.visit(module)

    >>> binop = module.body[0].body[0].value
    >>> for n in ancestors.parents(binop):
    ...    print(type(n))
    <class 'gast.gast.Module'>
    <class 'gast.gast.FunctionDef'>
    <class 'gast.gast.Return'>

    Also works with standard library nodes

    >>> import ast
    >>> code = 'def foo(x): return x + 1'
    >>> module = ast.parse(code)

    >>> from beniget import Ancestors
    >>> ancestors = Ancestors()
    >>> ancestors.visit(module)

    >>> binop = module.body[0].body[0].value
    >>> for n in ancestors.parents(binop):
    ...    print(type(n))
    <class 'gast.gast.Module'>
    <class 'gast.gast.FunctionDef'>
    <class 'gast.gast.Return'>
    """

    def __init__(self):
        self._parents = dict()
        self._current = list()

    def generic_visit(self, node):
        self._parents[node] = list(self._current)
        self._current.append(node)
        super().generic_visit(node)
        self._current.pop()

    def parent(self, node):
        return self._parents[node][-1]

    def parents(self, node):
        return self._parents[node]

    def parentInstance(self, node, cls):
        for n in reversed(self._parents[node]):
            if isinstance(n, cls):
                return n
        raise ValueError("{} has no parent of type {}".format(node, cls))

    def parentFunction(self, node):
        return self.parentInstance(node, (ast.FunctionDef,
                                          ast.AsyncFunctionDef,
                                          _ast.FunctionDef, 
                                          _ast.AsyncFunctionDef))

    def parentStmt(self, node):
        return self.parentInstance(node, _ast.stmt)
