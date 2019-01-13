from unittest import TestCase
import gast as ast
import beniget
import sys

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

    def testFuntionNested(self):
        code = 'def foo():\n def bar(): return'
        self.checkGlobals(code, ['foo'])

    if sys.version_info.major >= 3:
        def testAsyncFunctionDef(self):
            code = 'async def foo(): pass'
            self.checkGlobals(code, ['foo'])

    def testClassDef(self):
        code = 'class C:pass'
        self.checkGlobals(code, ['C'])

    def testNestedClassDef(self):
        code = 'class C:\n class D: pass'
        self.checkGlobals(code, ['C'])

    def testMultipleClassDef(self):
        code = 'class C: pass\nclass D: pass'
        self.checkGlobals(code, ['C', 'D'])

    def testClassRedefinition(self):
        code = 'class C: pass\nclass C: pass'
        self.checkGlobals(code, ['C', 'C'])

    def testClassMethodDef(self):
        code = 'class C:\n def some(self):pass'
        self.checkGlobals(code, ['C'])

    def testGlobalDef(self):
        code = 'x = 1'
        self.checkGlobals(code, ['x'])

    if sys.version_info.major >= 3:
        def testGlobalAnnotatedDef(self):
            code = 'x : 1'
            self.checkGlobals(code, ['x'])

    def testMultipleGlobalDef(self):
        code = 'x = 1; x = 2'
        self.checkGlobals(code, ['x', 'x'])

    def testGlobalDestructuring(self):
        code = 'x, y = 1, 2'
        self.checkGlobals(code, ['x', 'y'])

    def testGlobalAugAssign(self):
        code = 'x = 1; x += 2'
        self.checkGlobals(code, ['x', 'x'])

    def testGlobalFor(self):
        code = 'for x in (1,2): pass'
        self.checkGlobals(code, ['x'])

    def testGlobalForDestructuring(self):
        code = 'for x, y in [(1,2)]: pass'
        self.checkGlobals(code, ['x', 'y'])

    def testGlobalNestedFor(self):
        code = 'for x in (1,2):\n for y in (2, 1): pass'
        self.checkGlobals(code, ['x', 'y'])

    def testGlobalInFor(self):
        code = 'for x in (1,2): y = x'
        self.checkGlobals(code, ['x', 'y'])

    if sys.version_info >= (3, 7):
        def testGlobalAsyncFor(self):
            code = 'async for x in (1,2): pass'
            self.checkGlobals(code, ['x'])

    def testGlobalInWhile(self):
        code = 'while True: x = 1'
        self.checkGlobals(code, ['x'])

    def testGlobalInIfTrueBranch(self):
        code = 'if 1: a = 1'
        self.checkGlobals(code, ['a'])

    def testGlobalInIfFalseBranch(self):
        code = 'if 1: pass\nelse: a = 1'
        self.checkGlobals(code, ['a'])

    def testGlobalInIfBothBranch(self):
        code = 'if 1: a = 1\nelse: a = 2'
        self.checkGlobals(code, ['a', 'a'])

    def testGlobalInIfBothBranchDifferent(self):
        code = 'if 1: a = 1\nelse: b = 2'
        self.checkGlobals(code, ['a', 'b'])

    def testGlobalWith(self):
        code = 'with foo() as x: pass'
        self.checkGlobals(code, [])

    if sys.version_info >= (3, 7):
        def testGlobalAsyncWith(self):
            code = 'async with foo() as x: pass'
            self.checkGlobals(code, [])

    def testGlobalTry(self):
        code = 'try: x = 1\nexcept Exception: pass'
        self.checkGlobals(code, ['x'])

    def testGlobalTryExcept(self):
        code = 'try: foo()\nexcept Exception as e: pass'
        self.checkGlobals(code, ['e'])

    def testGlobalTryExceptFinally(self):
        code = 'try: w = 1\nexcept Exception as x: y = 1\nfinally: z = 1'
        self.checkGlobals(code, ['w', 'x', 'y', 'z'])

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

    if sys.version_info.major >= 3:
        def testGlobalAfterKeyword(self):
            code = 'def foo(): global x\nx : 1'
            self.checkGlobals(code, ['foo', 'x'])

        def testGlobalsAfterKeyword(self):
            code = 'def foo(): global x, y\ny : 1'
            self.checkGlobals(code, ['foo', 'x', 'y'])

    def testGlobalImport(self):
        code = 'import foo'
        self.checkGlobals(code, ['foo'])

    def testGlobalImports(self):
        code = 'import foo, bar'
        self.checkGlobals(code, ['bar', 'foo'])

    def testGlobalImportSubModule(self):
        code = 'import foo.bar'
        self.checkGlobals(code, ['foo'])

    def testGlobalImportSubModuleAs(self):
        code = 'import foo.bar as foobar'
        self.checkGlobals(code, ['foobar'])

    def testGlobalImportAs(self):
        code = 'import foo as bar'
        self.checkGlobals(code, ['bar'])

    def testGlobalImportsAs(self):
        code = 'import foo as bar, foobar'
        self.checkGlobals(code, ['bar', 'foobar'])

    def testGlobalImportFrom(self):
        code = 'from foo import bar'
        self.checkGlobals(code, ['bar'])

    def testGlobalImportFromAs(self):
        code = 'from foo import bar as BAR'
        self.checkGlobals(code, ['BAR'])

    def testGlobalImportFromStar(self):
        code = 'from foo import *'
        self.checkGlobals(code, ['*'])

    def testGlobalImportsFrom(self):
        code = 'from foo import bar, man'
        self.checkGlobals(code, ['bar', 'man'])

    def testGlobalImportsFromAs(self):
        code = 'from foo import bar, man as maid'
        self.checkGlobals(code, ['bar', 'maid'])

    def testGlobalListComp(self):
        code = '[1 for x in y]'
        if sys.version_info.major == 2:
            self.checkGlobals(code, ['x'])
        else:
            self.checkGlobals(code, [])

    def testGlobalSetComp(self):
        code = '{1 for x in y}'
        if sys.version_info.major == 2:
            self.checkGlobals(code, ['x'])
        else:
            self.checkGlobals(code, [])

    def testGlobalDictComp(self):
        code = '{1:1 for x in y}'
        if sys.version_info.major == 2:
            self.checkGlobals(code, ['x'])
        else:
            self.checkGlobals(code, [])

    def testGlobalGeneratorExpr(self):
        code = '(1 for x in y)'
        if sys.version_info.major == 2:
            self.checkGlobals(code, ['x'])
        else:
            self.checkGlobals(code, [])

    def testGlobalLambda(self):
        code = 'lambda x: x'
        self.checkGlobals(code, [])
