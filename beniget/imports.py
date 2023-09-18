import gast as ast
import sys

class ImportInfo:
    """
    Complement an `ast.alias` node with resolved 
    origin module and name of the locally bound name.

    :note: `orgname` will be ``*`` for wildcard imports.
    """
    __slots__ = 'orgmodule', 'orgname'

    def __init__(self, orgmodule, orgname) -> None:
        """
        :param orgmodule: str
        :param orgname: str or None
        """
        self.orgmodule = orgmodule
        self.orgname = orgname

_alias_needs_lineno = sys.implementation.name == 'cpython' and sys.version_info < (3,10)

# The MIT License (MIT)
# Copyright (c) 2017 Jelle Zijlstra
# Adapted from the project typeshed_client.
class ImportParser(ast.NodeVisitor):
    """
    Transform import statements into a mapping from `ast.alias` to `ImportInfo`.
    One instance of `ImportParser` can be used to parse all imports in a given module.

    Call to `visit` will parse the given import node into a mapping of aliases to `ImportInfo`.
    """

    def __init__(self, modname, *, is_package) -> None:
        self._modname = tuple(modname.split("."))
        self._is_package = is_package
        self._result = {}

    def generic_visit(self, node):
        raise TypeError('unexpected node type: {}'.format(type(node)))

    def visit_Import(self, node):
        self._result.clear()
        for al in node.names:
            if al.asname:
                self._result[al] = ImportInfo(orgmodule=al.name)
            else:
                # Here, we're not including information 
                # regarding the submodules imported - if there is one.
                # This is because this analysis map the names bounded by imports, 
                # not the dependencies.
                self._result[al] = ImportInfo(orgmodule=al.name.split(".", 1)[0])
            
            # This seems to be the most resonable place to fix the ast.alias node not having
            # proper line number information on python3.9 and before.
            if _alias_needs_lineno:
                al.lineno = node.lineno
        
        return self._result

    def visit_ImportFrom(self, node):
        self._result.clear()
        current_module = self._modname

        if node.module is None:
            module = ()
        else:
            module = tuple(node.module.split("."))
        
        if not node.level:
            source_module = module
        else:
            # parse relative imports
            if node.level == 1:
                if self._is_package:
                    relative_module = current_module
                else:
                    relative_module = current_module[:-1]
            else:
                if self._is_package:
                    relative_module = current_module[: 1 - node.level]
                else:
                    relative_module = current_module[: -node.level]

            if not relative_module:
                # We don't raise errors when an relative import makes no sens, 
                # we simply pad the name with dots.
                relative_module = ("",) * node.level

            source_module = relative_module + module

        for alias in node.names:
            self._result[alias] = ImportInfo(
                orgmodule=".".join(source_module), orgname=alias.name
            )
            
            # fix the ast.alias node not having proper line number on python3.9 and before.
            if _alias_needs_lineno:
                alias.lineno = node.lineno

        return self._result
