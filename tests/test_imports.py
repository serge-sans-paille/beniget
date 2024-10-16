import gast as _gast
import ast as _ast
from unittest import TestCase
from textwrap import dedent

from beniget import Def, parse_import

class TestImportParser(TestCase):
    ast = _gast
    def test_import_parsing(self):
        code = '''
        import mod2
        import pack.subpack
        import pack.subpack as a
        from mod2 import _k as k, _l as l, _m as m
        from pack.subpack.stuff import C
        '''
        expected = [{'mod2':('mod2', None)},
                    {'pack':('pack', None)},
                    {'a':('pack.subpack', None)},
                    {'k':('mod2','_k'), 
                     'l':('mod2','_l'), 
                     'm':('mod2','_m')},
                    {'C':('pack.subpack.stuff','C')},]
        node = self.ast.parse(dedent(code))
        assert len(expected)==len(node.body)
        for import_node, expected_names in zip(node.body, expected):
            assert isinstance(import_node, (self.ast.Import, self.ast.ImportFrom))
            for al,i in parse_import(import_node, 'mod1', is_package=False).items():
                assert Def(al).name() in expected_names
                expected_orgmodule, expected_orgname = expected_names[Def(al).name()]
                assert i.orgmodule == expected_orgmodule
                assert i.orgname == expected_orgname
                ran=True
        assert ran
    
    def test_import_parser_relative(self):
        code = '''
        from ...mod2 import bar as b
        from .pack import foo
        from ......error import x
        '''
        expected = [{'b':('top.mod2','bar')},
                    {'foo':('top.subpack.other.pack','foo')},
                    {'x': ('......error', 'x')}]
        node = self.ast.parse(dedent(code))
        assert len(expected)==len(node.body)
        for import_node, expected_names in zip(node.body, expected):
            assert isinstance(import_node, (self.ast.Import, self.ast.ImportFrom))
            for al,i in parse_import(import_node, 'top.subpack.other', is_package=True).items():
                assert Def(al).name() in expected_names
                expected_orgmodule, expected_orgname = expected_names[Def(al).name()]
                assert i.orgmodule == expected_orgmodule
                assert i.orgname == expected_orgname
                ran=True
        assert ran

class TestImportParserStdlib(TestImportParser):
    ast = _ast
