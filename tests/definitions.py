from unittest import TestCase
import gast as ast
import beniget

class TestGlobals(TestCase):

    def checkGlobals(self, code, ref):
        node = ast.parse(code)
        c = beniget.Collect()
        c.visit(node)
        self.assertEqual(c.dump_definitions(node), ref)

    def test_SingleFunctionDef(self):
        code = 'def foo(): pass'
        self.checkGlobals(code, ['foo'])

    def test_MultipleFunctionDef(self):
        code = 'def foo(): pass\ndef bar(): return'
        self.checkGlobals(code, ['bar', 'foo'])

    def testFuntionRedefinion(self):
        code = 'def foo(): pass\ndef foo(): return'
        self.checkGlobals(code, ['foo', 'foo'])

    def testGlobalDef(self):
        code = 'x = 1'
        self.checkGlobals(code, ['x'])

    def testGlobalAnnotatedDef(self):
        code = 'x : 1'
        self.checkGlobals(code, ['x'])

    def testMultipleGlobalDef(self):
        code = 'x = 1; x = 2'
        self.checkGlobals(code, ['x', 'x'])

    def testGlobalThroughKeyword(self):
        code = 'def foo(): global x'
        self.checkGlobals(code, ['foo', 'x'])

    def testGlobalThroughKeywords(self):
        code = 'def foo(): global x, y'
        self.checkGlobals(code, ['foo', 'x', 'y'])

    def testGlobalThroughMultipleKeyword(self):
        code = 'def foo(): global x\ndef bar(): global x'
        self.checkGlobals(code, ['bar', 'foo', 'x'])

    def testGlobalBeforeKeyword(self):
        code = 'x = 1\ndef foo(): global x'
        self.checkGlobals(code, ['foo', 'x'])

    def testGlobalsBeforeKeyword(self):
        code = 'x = 1\ndef foo(): global x, y'
        self.checkGlobals(code, ['foo', 'x', 'y'])

    def testGlobalAfterKeyword(self):
        code = 'def foo(): global x\nx : 1'
        self.checkGlobals(code, ['foo', 'x'])

    def testGlobalsAfterKeyword(self):
        code = 'def foo(): global x, y\ny : 1'
        self.checkGlobals(code, ['foo', 'x', 'y'])

    def testGlobalDestructuring(self):
        code = 'x, y = 1, 2'
        self.checkGlobals(code, ['x', 'y'])

    def testImport(self):
        code = 'import foo'
        self.checkGlobals(code, ['foo'])

    def testImports(self):
        code = 'import foo, bar'
        self.checkGlobals(code, ['bar', 'foo'])

    def testImportSubModule(self):
        code = 'import foo.bar'
        self.checkGlobals(code, ['foo'])

    def testImportAs(self):
        code = 'import foo as bar'
        self.checkGlobals(code, ['bar'])

    def testImportsAs(self):
        code = 'import foo as bar, foobar'
        self.checkGlobals(code, ['bar', 'foobar'])

    def testImportFrom(self):
        code = 'from foo import bar'
        self.checkGlobals(code, ['bar'])

    def testImportFromAs(self):
        code = 'from foo import bar as BAR'
        self.checkGlobals(code, ['BAR'])

    def testImportsFrom(self):
        code = 'from foo import bar, man'
        self.checkGlobals(code, ['bar', 'man'])

    def testImportsFromAs(self):
        code = 'from foo import bar, man as maid'
        self.checkGlobals(code, ['bar', 'maid'])
