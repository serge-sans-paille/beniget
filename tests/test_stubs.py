from unittest import TestCase, skipIf
import sys
import ast as _ast
import gast as _gast

from .test_chains import StrictDefUseChains

from typeshed_client import finder

typeshed_context = finder.get_search_context(search_path=[])

class TestStubs(TestCase):
    ast = _gast

    @skipIf(sys.version_info < (3, 8), reason='positional only syntax is used')
    def test_buitlins_stub(self):
        filename = 'builtins.pyi'
        file = finder.get_stub_file('builtins', search_context=typeshed_context)
        node = self.ast.parse(file.read_text(), filename)
        c = StrictDefUseChains(filename)
        c.visit(node)
        
        # all builtins references are sucessfuly linked to their definition
        # in that module and not the default builtins.
        for chains in c._builtins.values():
            assert len(chains.users())==0, chains
    
    @skipIf(sys.version_info < (3, 8), reason='positional only syntax is used')
    def test_all_stubs(self):
        for module_name, module_path in finder.get_all_stub_files(typeshed_context):
            with self.subTest(name=module_name, path=module_path):
                node = self.ast.parse(module_path.read_text(), module_path.as_posix())
                c = StrictDefUseChains(module_path.as_posix(), module_name)
                c.visit(node)

class TestStubsStdlib(TestStubs):
    ast = _ast
