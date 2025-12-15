from contextlib import contextmanager
from textwrap import dedent
from unittest import TestCase, skipIf
import unittest
import beniget
import io
import sys
import ast as _ast
import gast as _gast

from beniget.beniget import _get_lookup_scopes, def695, collect_locals

# Show full diff in unittest
unittest.util._MAX_LENGTH=2000

@contextmanager
def captured_output():
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

class StrictDefUseChains(beniget.DefUseChains):
        def warn(self, msg, node):
            raise RuntimeError(
                "W: {} at {}:{}".format(
                    msg, node.lineno, node.col_offset))

class TestDefUseChains(TestCase):
    maxDiff = None
    ast = _gast

    def checkChains(self, code, ref, strict=True):
        node = self.ast.parse(dedent(code))
        if strict:
            c = StrictDefUseChains()
        else:
            c = beniget.DefUseChains()

        c.visit(node)
        out = list(map(normalize_chain, c.dump_chains(node)))
        self.assertEqual(ref, out)
        return node, c
    
    def test_simple_expression(self):
        code = "a = 1; a + 2"
        self.checkChains(code, ["a -> (a -> (<BinOp> -> ()))"])

    def test_expression_chain(self):
        code = "a = 1; (- a + 2) > 0"
        self.checkChains(code, ["a -> (a -> (<UnaryOp> -> (<BinOp> -> (<Compare> -> ()))))"])

    def test_ifexp_chain(self):
        code = "a = 1; a + 1 if a else - a"
        self.checkChains(
            code,
            [
                "a -> ("
                "a -> (<IfExp> -> ()), "
                "a -> (<BinOp> -> (<IfExp> -> ())), "
                "a -> (<UnaryOp> -> (<IfExp> -> ()))"
                ")"
            ],
        )

    def test_type_destructuring_tuple(self):
        code = "a, b = range(2); a"
        self.checkChains(code, ["a -> (a -> ())", "b -> ()"])

    def test_type_destructuring_list(self):
        code = "[a, b] = range(2); a"
        self.checkChains(code, ["a -> (a -> ())", "b -> ()"])

    def test_type_destructuring_for(self):
        code = "for a, b in ((1,2), (3,4)): a"
        self.checkChains(code, ["a -> (a -> ())", "b -> ()"])

    def test_type_destructuring_starred(self):
        code = "a, *b = range(2); b"
        self.checkChains(code, ['a -> ()', 'b -> (b -> ())'])

    def test_assign_in_loop(self):
        code = "a = 2\nwhile 1: a = 1\na"
        self.checkChains(code, ["a -> (a -> ())", "a -> (a -> ())"])

    def test_reassign_in_loop(self):
        code = "m = 1\nfor i in [1, 2]:\n m = m + 1"
        self.checkChains(
            code, ["m -> (m -> (<BinOp> -> ()))", "i -> ()", "m -> (m -> (<BinOp> -> ()))"]
        )

    def test_continue_in_loop(self):
        code = "for i in [1, 2]:\n if i: m = 1; continue\n m = 1\nm"
        self.checkChains(
            code, ['i -> (i -> ())', 'm -> (m -> ())', 'm -> (m -> ())']
        )

    def test_break_in_loop(self):
        code = "for i in [1, 2]:\n if i: m = 1; continue\n m = 1\nm"
        self.checkChains(
            code, ['i -> (i -> ())', 'm -> (m -> ())', 'm -> (m -> ())']
        )

    def test_augassign(self):
        code = "a = 1; a += 2; a"
        self.checkChains(code, ['a -> (a -> (a -> ()))'])

    def test_read_global_from_branch(self):
        code = "if 1: a = 1\ndef foo():\n def bar(): global a; return a"
        self.checkChains(code, ['a -> (a -> ())',
                                'foo -> ()'])

    def test_augassign_undefined_global(self):
        code = "def foo():\n def bar():\n  global x\n  x+=1; x"
        self.checkChains(code, ['foo -> ()', 'x -> (x -> ())'], strict=False)

    def test_expanded_augassign(self):
        code = "a = 1; a = a + 2"
        self.checkChains(code, ["a -> (a -> (<BinOp> -> ()))", "a -> ()"])

    def test_augassign_in_loop(self):
        code = "a = 1\nfor i in [1]:\n a += 2\na"
        self.checkChains(code, ['a -> (a -> ((#1), a -> ()), a -> ())',
                                'i -> ()'])

    def test_assign_in_while_in_conditional(self):
        code = """
G = 1
while 1:
    if 1:
        G = 1
    G"""
        self.checkChains(code, ['G -> (G -> ())',
                                'G -> (G -> ())'])

    def test_assign_in_loop_in_conditional(self):
        code = """
G = 1
for _ in [1]:
    if 1:
        G = 1
    G"""
        self.checkChains(code, ['G -> (G -> ())',
                                '_ -> ()',
                                'G -> (G -> ())'])

    def test_simple_print(self):
        code = "a = 1; print(a)"
        self.checkChains(code, ["a -> (a -> (<Call> -> ()))"])

    def test_simple_redefinition(self):
        code = "a = 1; a + 2; a = 3; +a"
        self.checkChains(
            code, ["a -> (a -> (<BinOp> -> ()))", "a -> (a -> (<UnaryOp> -> ()))"]
        )

    def test_simple_for(self):
        code = "for i in [1,2,3]: j = i"
        self.checkChains(code, ["i -> (i -> ())", "j -> ()"])

    def test_simple_for_orelse(self):
        code = "for i in [1,2,3]: pass\nelse: i = 4\ni"
        self.checkChains(
            code,
            [
                # assign in loop iteration
                "i -> (i -> ())",
                # assign in orelse
                "i -> (i -> ())",
            ],
        )

    def test_for_break(self):
        code = "i = 8\nfor i in [1,2]:\n break\n i = 3\ni"
        self.checkChains(
            code,
            ['i -> (i -> ())',
             'i -> (i -> ())',
             'i -> ()'])

    def test_for_pass(self):
        code = "i = 8\nfor i in []:\n pass\ni"
        self.checkChains(
            code,
            ['i -> (i -> ())',
             'i -> (i -> ())'])

    def test_complex_for_orelse(self):
        code = "I = J = 0\nfor i in [1,2]:\n if i < 3: I = i\nelse:\n if 1: J = I\nJ"
        self.checkChains(
            code,
            ['I -> (I -> ())',
             'J -> (J -> ())',
             'i -> (i -> (<Compare> -> ()), i -> ())',
             'I -> (I -> ())',
             'J -> (J -> ())']
        )

    def test_simple_while(self):
        code = "i = 2\nwhile i: i = i - 1\ni"
        self.checkChains(
            code,
            [
                # first assign, out of loop
                "i -> (i -> (), i -> (<BinOp> -> ()), i -> ())",
                # second assign, in loop
                "i -> (i -> (), i -> (<BinOp> -> ()), i -> ())",
            ],
        )

    def test_while_break(self):
        code = "i = 8\nwhile 1:\n break\n i = 3\ni"
        self.checkChains(
            code,
            ['i -> (i -> ())',
             'i -> ()'])

    def test_while_cond_break(self):
        code = "i = 8\nwhile 1:\n if i: i=1;break\ni"
        self.checkChains(
            code,
            ['i -> (i -> (), i -> ())', 'i -> (i -> ())'])

    def test_straight_raise(self):
        code = "raise next([e for e in [1]])"
        self.checkChains(code, [])

    def test_redefinition_in_comp(self):
        code = "[name for name in 'hello']\nfor name in 'hello':name"
        self.checkChains(
                code,
                ['name -> (name -> ())'])

    def test_nested_while(self):
        code = '''
done = 1
while done:

    while done:
        if 1:
            done = 1
            break

        if 1:
            break'''


        self.checkChains(
            code,
            ['done -> (done -> (), done -> ())',
             'done -> (done -> (), done -> ())']
            )

    def test_while_cond_continue(self):
        code = "i = 8\nwhile 1:\n if i: i=1;continue\ni"
        self.checkChains(
            code,
            ['i -> (i -> (), i -> ())', 'i -> (i -> (), i -> ())'])

    def test_complex_while_orelse(self):
        code = "I = J = i = 0\nwhile i:\n if i < 3: I = i\nelse:\n if 1: J = I\nJ"
        self.checkChains(
            code,
            [
                "I -> (I -> ())",
                "J -> (J -> ())",
                "i -> (i -> (), i -> (<Compare> -> ()), i -> ())",
                "J -> (J -> ())",
                "I -> (I -> ())",
            ],
        )

    def test_while_orelse_break(self):
        code = "I = 0\nwhile I:\n if 1: I = 1; break\nelse: I"
        self.checkChains(
            code,
            ['I -> (I -> (), I -> ())',
             'I -> ()'],
        )

    def test_while_nested_break(self):
        code = "i = 8\nwhile i:\n if i: break\n i = 3\ni"
        self.checkChains(
            code,
            ['i -> (i -> (), i -> (), i -> ())',
             'i -> (i -> (), i -> (), i -> ())'])

    def test_if_true_branch(self):
        code = "if 1: i = 0\ni"
        self.checkChains(code, ["i -> (i -> ())"])

    def test_if_false_branch(self):
        code = "if 1: pass\nelse: i = 0\ni"
        self.checkChains(code, ["i -> (i -> ())"])

    def test_if_both_branch(self):
        code = "if 1: i = 1\nelse: i = 0\ni"
        self.checkChains(code, ["i -> (i -> ())"] * 2)

    def test_if_in_loop(self):
        code = "for _ in [0, 1]:\n if _: i = 1\n else: j = i\ni"
        self.checkChains(code, ["_ -> (_ -> ())", "i -> (i -> (), i -> ())", "j -> ()"])

    def test_with_handler(self):
        code = 'with open("/dev/null") as x: pass\nx'
        self.checkChains(code, ["x -> (x -> ())"])

    def test_simple_try(self):
        code = 'try: e = open("/dev/null")\nexcept Exception: pass\ne'
        self.checkChains(code, ["e -> (e -> ())"])

    def test_simple_except(self):
        code = "try: pass\nexcept Exception as e: pass\ne"
        self.check_message(code, ["W: unbound identifier 'e' at <unknown>:3:0"])
        self.checkChains(code, ['e -> ()'], strict=False)

    @skipIf(sys.version_info < (3, 11), 'Python 3.11 syntax')
    def test_simple_except_star(self):
        code = "try: pass\nexcept* Exception as e: pass\ne"
        self.check_message(code, ["W: unbound identifier 'e' at <unknown>:3:0"])
        self.checkChains(code, ['e -> ()'], strict=False)

    def test_simple_try_except(self):
        code = 'try: f = open("")\nexcept Exception as e: pass\ne;f'
        self.check_message(code, ["W: unbound identifier 'e' at <unknown>:3:0"])
        self.checkChains(code, ["f -> (f -> ())", "e -> ()"], strict=False)

    def test_redef_try_except(self):
        code = 'try: f = open("")\nexcept Exception as f: pass\nf'
        self.checkChains(code, ["f -> (f -> ())",  "f -> ()"])

    def test_try_except_refdef_exception_same_name(self):
        code = 'try: ...\nexcept Exception as f: f=f\nf'
        self.check_message(code, ["W: unbound identifier 'f' at <unknown>:3:0"])

    def test_try_except_refdef_exception(self):
        code = 'try: ...\nexcept Exception as e: f=e\nf'
        self.checkChains(code, ['e -> (e -> ())', 'f -> (f -> ())'])

    def test_try_except_delete_exception(self):
        code = 'try: ...\nexcept Exception as e: f=e; del e\nf'
        self.checkChains(code, ['e -> (e -> (), e -> ())', 'f -> (f -> ())'])

    def test_try_multiple_excepts(self):
        code = 'try:... \nexcept KeyError as e1: e2 \nexcept ValueError as e2: e1'
        self.check_message(code, ["W: unbound identifier 'e2' at <unknown>:2:23",
            "W: unbound identifier 'e1' at <unknown>:3:25"])

    def test_try_except_delete_exception_in_finally(self):
        code = 'try: ...\nexcept Exception as e: f=e \nfinally: del e,f\nf'
        self.check_message(code, ["W: unbound identifier 'e' at <unknown>:3:13",
                                  "W: unbound identifier 'f' at <unknown>:4:0"])

    def test_try_except_delete_redef_exception(self):
        code = 'try: ...\nexcept Exception as e: f=e\nf; del f'
        self.checkChains(code, ['e -> (e -> ())', 'f -> (f -> (), f -> ())'])

    def test_simple_import(self):
        code = "import x; x"
        self.checkChains(code, ["x -> (x -> ())"])

    def test_simple_import_as(self):
        code = "import x as y; y()"
        self.checkChains(code, ["y -> (y -> (<Call> -> ()))"])

    def test_simple_lambda(self):
        node, c = self.checkChains( "lambda y: True", [])
        self.assertEqual(c.dump_chains(node.body[0].value), ['y -> ()'])

    def test_lambda_defaults(self):
        node, c = self.checkChains( "x=y=1;(lambda y, x=x: (True, x, y, z)); x=y=z=2",
                                   ['x -> (x -> (<Lambda> -> ()))',
                                    'y -> ()',
                                    'x -> ()',
                                    'y -> ()',
                                    'z -> (z -> (<Tuple> -> (<Lambda> -> ())))'])
        self.assertEqual(c.dump_chains(node.body[1].value), [
            'y -> (y -> (<Tuple> -> (<Lambda> -> ())))',
            'x -> (x -> (<Tuple> -> (<Lambda> -> ())))',
        ])

    def test_lambda_varargs(self):
        node, c = self.checkChains( "lambda *args: args", [])
        self.assertEqual(c.dump_chains(node.body[0].value), ['args -> (args -> (<Lambda> -> ()))'])

    def test_lambda_kwargs(self):
        node, c = self.checkChains( "lambda **kwargs: kwargs", [])
        self.assertEqual(c.dump_chains(node.body[0].value), ['kwargs -> (kwargs -> (<Lambda> -> ()))'])

    def test_multiple_import_as(self):
        code = "import x as y, z; y"
        self.checkChains(code, ["y -> (y -> ())", "z -> ()"])

    def test_import_from(self):
        code = "from  y import x; x"
        self.checkChains(code, ["x -> (x -> ())"])

    def test_import_from_as(self):
        code = "from  y import x as z; z"
        self.checkChains(code, ["z -> (z -> ())"])

    def test_multiple_import_from_as(self):
        code = "from  y import x as z, w; z"
        self.checkChains(code, ["z -> (z -> ())", "w -> ()"])

    def test_method_function_conflict(self):
        code = "def foo():pass\nclass C:\n def foo(self): foo()"
        self.checkChains(code, ["foo -> (foo -> (<Call> -> ()))", "C -> ()"])

    def test_nested_if(self):
        code = "f = 1\nif 1:\n if 1:pass\n else: f=1\nelse: f = 1\nf"
        self.checkChains(code, ["f -> (f -> ())", "f -> (f -> ())", "f -> (f -> ())"])

    def test_nested_if_else(self):
        code = "f = 1\nif 1: f = 1\nelse:\n if 1:pass\n else: f=1\nf"
        self.checkChains(code, ["f -> (f -> ())", "f -> (f -> ())", "f -> (f -> ())"])

    def test_try_except(self):
        code = "f = 1\ntry: \n len(); f = 2\nexcept: pass\nf"
        self.checkChains(code, ["f -> (f -> ())", "f -> (f -> ())"])

    def test_attr(self):
        code = "import numpy as bar\ndef foo():\n return bar.zeros(2)"
        self.checkChains(
            code, ["bar -> (bar -> (.zeros -> (<Call> -> ())))", "foo -> ()"]
        )

    def test_class_decorator(self):
        code = "from some import decorator\n@decorator\nclass C:pass"
        self.checkChains(code, ["decorator -> (decorator -> (C -> ()))", "C -> ()"])

    def test_class_base(self):
        code = "class A:pass\nclass B(A):pass"
        self.checkChains(code, ["A -> (A -> (B -> ()))", "B -> ()"])

    def test_def_used_in_self_default(self):
        code = "def foo(x:foo): return foo"
        node = self.ast.parse(code)
        c = beniget.DefUseChains()
        c.visit(node)
        self.assertEqual(c.dump_chains(node), ["foo -> (foo -> ())"])

    def test_unbound_class_variable(self):
        code = '''
def middle():
    x = 1
    class mytype(str):
        x = x+1 # <- this triggers NameError: name 'x' is not defined
    return x
        '''
        node = self.ast.parse(code)
        c = beniget.DefUseChains()
        c.visit(node)
        self.assertEqual(c.dump_chains(node.body[0]), ['x -> (x -> ())', 'mytype -> ()'])

    def test_unbound_class_variable2(self):
        code = '''class A:\n  a = 10\n  def f(self):\n    return a # a is not defined'''
        node = self.ast.parse(code)
        c = beniget.DefUseChains()
        c.visit(node)
        self.assertEqual(c.dump_chains(node.body[0]), ['a -> ()', 'f -> ()'])

    def test_unbound_class_variable3(self):
        code = '''class A:\n  a = 10\n  class I:\n    b = a + 1 # a is not defined'''
        node = self.ast.parse(code)
        c = beniget.DefUseChains()
        c.visit(node)
        self.assertEqual(c.dump_chains(node.body[0]), ['a -> ()', 'I -> ()'])

    def test_unbound_class_variable4(self):
        code = '''class A:\n  a = 10\n  f = lambda: a # a is not defined'''
        node = self.ast.parse(code)
        c = beniget.DefUseChains()
        c.visit(node)
        self.assertEqual(c.dump_chains(node.body[0]), ['a -> ()', 'f -> ()'])

    def test_unbound_class_variable5(self):
        code = '''class A:\n  a = 10\n  b = [a for _ in range(10)]  # a is not defined'''
        node = self.ast.parse(code)
        c = beniget.DefUseChains()
        c.visit(node)
        self.assertEqual(c.dump_chains(node.body[0]), ['a -> ()', 'b -> ()'])

    def test_functiondef_returns(self):
        code = "x = 1\ndef foo() -> x: pass"
        self.checkChains(code, ['x -> (x -> ())', 'foo -> ()'])

    def test_arg_annotation(self):
        code = "type_ = int\ndef foo(bar: type_): pass"
        self.checkChains(code, ["type_ -> (type_ -> ())", "foo -> ()"])


    def test_annotation_inner_class(self):

        code = '''
def outer():
    def middle():
        class mytype(str):
            def count(self) -> mytype: # this should trigger unbound identifier
                def c(x) -> mytype(): # this one shouldn't
                    ...
        '''
        node = self.ast.parse(code)
        c = beniget.DefUseChains()
        c.visit(node)
        self.assertEqual(c.dump_chains(node.body[0].body[0]), ['mytype -> (mytype -> (<Call> -> ()))'])

    def check_message(self, code, expected_messages, filename=None):
        node = self.ast.parse(dedent(code))
        c = beniget.DefUseChains(filename)
        with captured_output() as (out, err):
            c.visit(node)

        if not out.getvalue():
            produced_messages = []
        else:
            produced_messages = out.getvalue().strip().split("\n")

        self.assertEqual(len(expected_messages), len(produced_messages),
                         produced_messages)
        for expected, produced in zip(expected_messages, produced_messages):
            self.assertIn(expected, produced,
                          "actual message does not contains expected message")

    def test_unbound_identifier_message_format(self):
        code = "foo(1)\nbar(2)"
        self.check_message(code, ["<unknown>:1", "<unknown>:2"])
        self.check_message(code, ["foo.py:1", "foo.py:2"], filename="foo.py")

    def test_unbound_class_variable_reference_message_format(self):
        code = "class A:\n a = 10\n def f(self): return a # a is undef"
        self.check_message(code, ["unbound identifier 'a' at <unknown>:3"])

    def test_no_unbound_local_identifier_in_comp(self):
        code = "a = []; b = [1 for i in a]"
        self.check_message(code, [])

    def test_maybe_unbound_identifier_message_format(self):
        code = "x = 1\ndef foo(): y = x; x = 2"
        self.check_message(code,
                           ["unbound identifier 'x' at <unknown>:2"])

    def test_unbound_local_identifier_in_func(self):
        code = "def A():\n x = 1\n class B: x = x"
        self.check_message(code,
                           ["unbound identifier 'x' at <unknown>:3"])

    def test_unbound_local_identifier_in_method(self):
        code = "class A:pass\nclass B:\n def A(self) -> A:pass"
        self.check_message(code, [])

    def test_unbound_local_identifier_nonlocal(self):
        code = "def A():\n x = 1\n class B: nonlocal x; x = x"
        self.check_message(code, [])

    def test_unbound_local_identifier_nonlocal_points_to_global(self):
        code = "def x():\n nonlocal x\n x = 1"
        self.check_message(code,
                           ["W: unbound identifier 'x' at <unknown>:2:1"])

    def test_unbound_local_identifier_nonlocal_points_to_scoped_global(self):
        code = "if 1:\n def x():\n  nonlocal x\n  x = 1"
        self.check_message(code,
                           ["W: unbound identifier 'x' at <unknown>:3:2"])

    def test_deleted_identifier(self):
        code = "x = 1; del x"
        self.checkChains(code, ['x -> (x -> ())'])

    def test_deleted_unknown_identifier(self):
        code = "del x"
        self.check_message(code,
                           ["W: unbound identifier 'x' at <unknown>:1:4"])
        self.checkChains(code, [], strict=False)

    def test_deleted_identifier_redefined(self):
        code = "x = 1; del x; x = 2"
        self.checkChains(code, ['x -> (x -> ())', 'x -> ()'])

    def test_unbound_deleted_identifier(self):
        code = "x = 1; del x; x"
        self.check_message(code,
                           ["W: unbound identifier 'x' at <unknown>:1:14"])

    def test_unbound_deleted_identifier_in_class(self):
        code = "class X:\n x = 1\n del x\n x"
        self.check_message(code,
                           ["W: unbound identifier 'x' at <unknown>:4:1"])

    def test_bound_deleted_identifier(self):
        code = "x = 1; del x; x = 1; x"
        self.check_message(code,
                           [])

    def test_del_in_for(self):
        code = "for x in [1]:\n del x\nx"
        self.check_message(code,
                           ["W: unbound identifier 'x' at <unknown>:3:0"])

    def test_del_predef_in_for(self):
        code = "x = 1\nfor x in [1]:\n del x\nx"
        self.check_message(code,
                           [])

    def test_bound_deleted_identifier_in_if(self):
        code = "x = 1\ndel x\nif 1:\n x = 1\nx"
        self.check_message(code,
                           [])

    def test_deleted_annotation(self):
        # Passes pyright and mypy. But fails at runtime, because of the annotation X.
        code = """\
class X: ...
del X
def f() -> X: ..."""
        self.check_message(code, ["W: unbound identifier 'X' at <unknown>:3:11"])
        self.checkChains(code, ['X -> (X -> ())', 'f -> ()'], strict=False)

    def test_deleted_pep649_deferred_annotation(self):
        # This will not pass at runtime anymore starting at Python 3.14
        code = """\
class X: ...
def f() -> X: ...
del X
f.__annotations__"""
        self.check_message(code, [])
        self.checkChains(code, ['X -> (X -> (), X -> ())', 'f -> (f -> (.__annotations__ -> ()))'])

    def test_deleted_pep563_deferred_annotation(self):
        # Passes pyright and mypy and runtime
        code = """\
from __future__ import annotations
class X: ...
del X
def f() -> X: ...
f()"""
        self.check_message(code, [])
        self.checkChains(code, ['annotations -> ()', 
                                'X -> (X -> (), X -> ())', 
                                'f -> (f -> (<Call> -> ()))'])

    def test_unsafe_use_in_function_before_deleted(self):
        # Passes at runtime, but fairly unsafe.
        code = """\
class X: ...
def f(): return X()
f(); del X"""
        self.check_message(code, ["W: unbound identifier 'X' at <unknown>:2:16"])
        self.checkChains(code, ['X -> (X -> ())', 'f -> (f -> (<Call> -> ()))'], strict=False)

    def test_use_in_function_after_deleted(self):
        # Passes pyright and mypy but fails at runtime.
        code = """\
class X: ...
def f(): return X()
del X; f()"""
        self.check_message(code, ["W: unbound identifier 'X' at <unknown>:2:16"])
        self.checkChains(code, ['X -> (X -> ())', 'f -> (f -> (<Call> -> ()))'], strict=False)

    def test_deleted_non_local_var(self):
        code = """\
def f():
    v = 1
    def q():
        nonlocal v
        del v
        v # unbound
    q()
    v # not unbound, we must stay over approximated
f()"""
        self.check_message(code, ["W: unbound identifier 'v' at <unknown>:6:8"])
        node, du = self.checkChains(code, ['f -> (f -> (<Call> -> ()))'], strict=False)
        assert du.dump_chains(node.body[0]) == ['v -> (v -> (), v -> ())', 'q -> (q -> (<Call> -> ()))']
        assert du.dump_chains(node.body[0].body[1]) == [] # q has no locals

    def test_cant_delete_nonlocal_not_declared_with_nonlocal_keyword(self):
        code = """\
def f():
    v = 1
    def q():
        del v # unbound
    q()
    v # not unbound, we must stay over approximated
f()"""

        self.check_message(code, ["W: deleting unreachable variable at <unknown>:4:12"])
        self.checkChains(code, ['f -> (f -> (<Call> -> ()))'], strict=False)

    def test_deleted_global(self):
        code = """\
v = 1
def q():
    global v
    del v
    v # unbound
q()
v # not unbound, we must stay over approximated"""

        self.check_message(code, ["W: unbound identifier 'v' at <unknown>:5:4"])
        self.checkChains(code, ['v -> (v -> (), v -> ())',
                                'q -> (q -> (<Call> -> ()))'], strict=False)

    def test_cant_delete_global_not_declared_with_global_keyword(self):
        code = """\
v = 1
def q():
    del v # unbound
q()
v # not unbound, we must stay over approximated"""

        self.check_message(code, ["W: deleting unreachable variable at <unknown>:3:8"])
        self.checkChains(code, ['v -> (v -> ())',
                                'q -> (q -> (<Call> -> ()))'], strict=False)

    def test_maybe_unbound_in_if(self):
        code = "def foo(x):\n  if x: del x\n  print(x)"
        self.check_message(code,
                           [])

    def test_always_unbound_in_if(self):
        code = "def foo(x):\n  if x: del x\n  else: del x\n  x"
        self.check_message(code,
                           ["W: unbound identifier 'x' at <unknown>:4:2"])

    def test_delete_list_syntax(self):
        code = "x,y = {}, 1; del x, y[0]; x, y"
        self.check_message(code,
                           ["W: unbound identifier 'x' at <unknown>:1:26"])

    def test_redefined_global_deleted_lower_scope_var(self):
        # test inspired by https://github.com/python/mypy/issues/9600
        code = '''\
x = 1
class y:
    @property(x)
    def a(self):...
    x = 2
    @property(x)
    def b(self):...
    del x
    @property(x)
    def c(self):...
'''
        self.check_message(code, ["W: unbound identifier 'x' at <unknown>:9:14"])
        self.checkChains(code, ['x -> (x -> (<Call> -> ()))', 'y -> ()'], strict=False)

    def test_assign_uses_class_level_name(self):
        code = '''
visit_Name = object
class Visitor:
    def visit_Name(self, node):pass
    visit_Attribute = visit_Name
'''
        node = self.ast.parse(code)
        c = beniget.DefUseChains()
        c.visit(node)
        self.assertEqual(c.dump_chains(node), ['visit_Name -> ()',
                                               'Visitor -> ()'])
        self.assertEqual(c.dump_chains(node.body[-1]),
                         ['visit_Name -> (visit_Name -> ())',
                          'visit_Attribute -> ()'])


    def test_base_class_uses_class_level_same_name(self):
            code = '''
class Attr(object):pass
class Visitor:
    class Attr(Attr):pass
    '''
            node = self.ast.parse(code)
            c = beniget.DefUseChains()
            c.visit(node)
            self.assertEqual(c.dump_chains(node),
                             ['Attr -> (Attr -> (Attr -> ()))',
                              'Visitor -> ()'])
            self.assertEqual(c.dump_chains(node.body[-1]), ['Attr -> ()'])

    def test_star_assignment(self):
        code = '''
curr, *parts = [1,2,3]
while curr:
    print(curr)
    if parts:
        curr, *parts = parts
    else:
        break
'''
        self.checkChains(code, ['curr -> (curr -> (), curr -> (<Call> -> ()))',
                                'parts -> (parts -> (), parts -> ())']*2)

    def test_star_assignment_nested(self):
        code = '''
(curr, *parts),i = [1,2,3],0
while curr:
    print(curr)
    if parts:
        (curr, *parts),i = parts,i
    else:
        break
'''
        self.checkChains(code, ['curr -> (curr -> (), curr -> (<Call> -> ()))',
                                'parts -> (parts -> (), parts -> (<Tuple> -> ()))',
                                'i -> (i -> (<Tuple> -> ()))']*2)

    def test_attribute_assignment(self):
        code = "d=object();d.name,x = 't',1"
        self.checkChains(code, ['d -> (d -> (.name -> ()))',
                                'x -> ()'])

    def test_call_assignment(self):
        code = "NameError().name = 't'"
        self.checkChains(code, [])

    def test_annotation_uses_class_level_name(self):
        code = '''
Thing = object
def f():...
class Visitor:
    Thing = bytes
    def f(): return f()
    def visit_Name(self, node:Thing, fn:f):...
'''
        node = self.ast.parse(code)
        c = beniget.DefUseChains()
        c.visit(node)
        self.assertEqual(c.dump_chains(node),
                         ['Thing -> ()',
                          'f -> (f -> (<Call> -> ()))',
                          'Visitor -> ()'])
        self.assertEqual(c.dump_chains(node.body[-1]),
                         ['Thing -> (Thing -> ())',
                          'f -> (f -> ())',
                          'visit_Name -> ()'])

    def test_assign_uses_class_level_same_name(self):
        code = '''
def visit_Attribute(self, node):pass
class Visitor:
    visit_Attribute = visit_Attribute
'''
        node = self.ast.parse(code)
        c = beniget.DefUseChains()
        c.visit(node)
        self.assertEqual(c.dump_chains(node),
            ['visit_Attribute -> (visit_Attribute -> ())',
             'Visitor -> ()'])
        self.assertEqual(c.dump_chains(node.body[-1]),
                         ['visit_Attribute -> ()'])

    def test_unbound_local_identifier_in_augassign(self):
        code = "def A():\n x = 1\n class B: x += 1"
        self.check_message(code,
                           ["unbound identifier 'x' at <unknown>:3"])

    def test_star_import_with_conditional_redef(self):
        code = '''
from math import *

if 1:
    def pop():
        cos()
cos = pop()'''
        self.checkChains(code, [
            '* -> (cos -> (<Call> -> ()))',
            'pop -> (pop -> (<Call> -> ()))',
            'cos -> (cos -> (<Call> -> ()))'
        ])

    def test_class_scope_comprehension(self):
        code = '''
class Cls:
    foo = b'1',
    [_ for _ in foo]
    {_ for _ in foo}
    (_ for _ in foo)
    {_:1 for _ in foo}
'''
        node, chains = self.checkChains(code, ['Cls -> ()'])
        self.assertEqual(chains.dump_chains(node.body[0]),
                         ['foo -> ('
                          'foo -> (<comprehension> -> (<ListComp> -> ())), '
                          'foo -> (<comprehension> -> (<SetComp> -> ())), '
                          'foo -> (<comprehension> -> (<GeneratorExp> -> ())), '
                          'foo -> (<comprehension> -> (<DictComp> -> ())))'])

    def test_class_scope_comprehension_invalid(self):
        code = '''
class Foo:
    x = 5
    y = [x for i in range(1)]
    z = [i for i in range(1) for j in range(x)]
'''
        self.check_message(code, ["W: unbound identifier 'x' at test:4:9",
                                  "W: unbound identifier 'x' at test:5:44"], 'test')


    @skipIf(sys.version_info < (3, 8), 'Python 3.8 syntax')
    def test_named_expr_simple(self):
        code = '''
if (x := 1):
    y = x + 1'''
        self.checkChains(
            code, ['x -> (x -> (<BinOp> -> ()))', 'y -> ()']
        )

    @skipIf(sys.version_info < (3, 8), 'Python 3.8 syntax')
    def test_named_expr_complex(self):
        code = '''
if (x := (y := 1) + 1):
    z = x + y'''
        self.checkChains(
            code, ['y -> (y -> (<BinOp> -> ()))', 'x -> (x -> (<BinOp> -> ()))', 'z -> ()']
        )

    @skipIf(sys.version_info < (3, 8), 'Python 3.8 syntax')
    def test_named_expr_with_rename(self):
        code = '''
a = 1
if (a := a + a):
    pass'''
        self.checkChains(
            code, ['a -> (a -> (<BinOp> -> (<NamedExpr> -> ())), a -> (<BinOp> -> (<NamedExpr> -> ())))', 'a -> ()']
        )

    @skipIf(sys.version_info < (3, 8), 'Python 3.8 syntax')
    def test_named_expr_comprehension(self):
        # Warlus target should be stored in first non comprehension scope
        code = ('cities = ["Halifax", "Toronto"]\n'
            'if any((witness := city).startswith("H") for city in cities):'
            'witness')
        self.checkChains(
            code, ['cities -> (cities -> (<comprehension> -> (<GeneratorExp> -> (<Call> -> ()))))',
                   'witness -> (witness -> ())']
        )

    @skipIf(sys.version_info < (3, 8), 'Python 3.8 syntax')
    def test_named_expr_comprehension_invalid(self):
        # an assignment expression target name cannot be the same as a
        # for-target name appearing in any comprehension containing the assignment expression.
        # A further exception applies when an assignment expression occurs in a comprehension whose
        # containing scope is a class scope. If the rules above were to result in the target
        # being assigned in that class's scope, the assignment expression is expressly invalid.
        code = '''
stuff = []

# assignment expression cannot rebind comprehension iteration variable
[[(a := a) for _ in range(5)] for a in range(5)] # INVALID
[b := 0 for b, _ in stuff] # INVALID
[c for c in (c := stuff)] # INVALID
[False and (d := 0) for d, _ in stuff] # INVALID
[_ for _, e in stuff if True or (e := 1)] # INVALID

# assignment expression cannot be used in a comprehension iterable expression
[_ for _ in (f := stuff)] # INVALID
[_ for _ in range(2) for _ in (g := stuff)] # INVALID
[_ for _ in [_ for _ in (h := stuff)]] # INVALID
[_ for _ in (lambda: (i := stuff))()] # INVALID

class Example:
    # assignment expression within a comprehension cannot be used in a class body
    [(j := i) for i in range(5)] # INVALID
'''
        # None of the invalid assigned name shows up.
        node, chains = self.checkChains(code, ['stuff -> ()', 'Example -> ()'], strict=False)
        self.assertEqual(chains.dump_chains(node.body[-1]), [])
        # It triggers useful warnings
        self.check_message(code, ["W: assignment expression cannot rebind comprehension iteration variable 'a' at <unknown>:5:0",
                                  "W: assignment expression cannot rebind comprehension iteration variable 'b' at <unknown>:6:0",
                                  'W: assignment expression cannot be used in a comprehension iterable expression at <unknown>:7:0',
                                  "W: assignment expression cannot rebind comprehension iteration variable 'd' at <unknown>:8:0",
                                  "W: assignment expression cannot rebind comprehension iteration variable 'e' at <unknown>:9:0",
                                  'W: assignment expression cannot be used in a comprehension iterable expression at <unknown>:12:0',
                                  'W: assignment expression cannot be used in a comprehension iterable expression at <unknown>:13:0',
                                  'W: assignment expression cannot be used in a comprehension iterable expression at <unknown>:14:0',
                                  'W: assignment expression cannot be used in a comprehension iterable expression at <unknown>:15:0',
                                  'W: assignment expression within a comprehension cannot be used in a class body at <unknown>:19:6'])

    def test_annotation_unbound(self):
        code = '''
def f(x:f) -> f: # 'f' annotations are unbound
    ...'''
        self.checkChains(
            code, ['f -> ()'], strict=False
        )

    def test_method_annotation_unbound(self):
        code = '''
class S:
    def f(self, x:f) -> f:... # 'f' annotations are unbound
'''
        mod, chains = self.checkChains(
            code, ['S -> ()'], strict=False
        )
        self.assertEqual(chains.dump_chains(mod.body[0]),
                         ['f -> ()'])

    def test_annotation_unbound_pep563(self):
        code = '''
from __future__ import annotations
def f(x:f) -> f: # 'f' annotations are NOT unbound because pep563
    ...'''
        self.checkChains(
            code, ['annotations -> ()', 'f -> (f -> (), f -> ())']
        )

    def test_method_annotation_unbound_pep563(self):
        code = '''
from __future__ import annotations
class S:
    def f(self, x:f) -> f:... # 'f' annotations are NOT unbound because pep563
'''
        mod, chains = self.checkChains(
            code, ['annotations -> ()', 'S -> ()']
        )
        self.assertEqual(chains.dump_chains(mod.body[1]),
                         ['f -> (f -> (), f -> ())'])

    def test_import_dotted_name_binds_first_name(self):
        code = '''import collections.abc;collections;collections.abc'''
        self.checkChains(
            code, ['collections -> (collections -> (), collections -> (.abc -> ()))']
        )

    def test_multiple_wildcards_may_bind(self):
        code = '''from abc import *; from collections import *;name1; from mod import *;name2'''
        self.checkChains(
            code, ['* -> (name1 -> (), name2 -> ())','* -> (name1 -> (), name2 -> ())','* -> (name2 -> ())']
        )

    def test_wildcard_may_override(self):
        # we could argue that the wildcard import might override name2,
        # but we're currently ignoring these kind of scenarios.
        code = '''name2=True;from abc import *;name2'''
        self.checkChains(
            code, ['name2 -> (name2 -> ())', '* -> ()']
        )

    def test_annotation_use_upper_scope_variables(self):
        code = '''
from typing import Union
class Attr:
    ...
class Thing:
    ...
class System:
    Thing = bytes
    @property
    def Attr(self) -> Union[Attr, Thing]:...
'''
        self.checkChains(
            code, ['Union -> (Union -> (<Subscript> -> ()))',
                    'Attr -> (Attr -> (<Tuple> -> (<Subscript> -> ())))',
                    'Thing -> ()',
                    'System -> ()',]
        )

    def test_future_annotation_class_var(self):
        code = '''
from __future__ import annotations
from typing import Type
class System:
    Thing = bytes
    @property
    def Attribute(self) -> Type[Thing]:...
'''

        mod, chains = self.checkChains(
            code, ['annotations -> ()',
            'Type -> (Type -> (<Subscript> -> ()))', 'System -> ()']
        )
        # locals of System
        self.assertEqual(chains.dump_chains(mod.body[-1]), [
            'Thing -> (Thing -> (<Subscript> -> ()))',
            'Attribute -> ()'
        ])

    def test_pep0563_annotations(self):

        # code taken from https://peps.python.org/pep-0563/

        code = '''
# beniget can probably understand this code without the future import
from __future__ import annotations
from typing import TypeAlias, Mapping, Dict, Type
class C:

    def method(self) -> C.field:  # this is OK
        ...

    def method(self) -> field:  # this is OK
        ...

    def method(self) -> C.D:  # this is OK
        ...

    def method(self, x:field) -> D:  # this is OK
        ...

    field:TypeAlias = 'Mapping'

    class D:
        field2:TypeAlias = 'Dict'
        def method(self) -> C.D.field2:  # this is OK
            ...

        def method(self) -> D.field2:  # this FAILS, class D is local to C
            ...                        # and is therefore only available
                                       # as C.D. This was already true
                                       # before the PEP.
                                       # We check first the globals, then the locals
                                       # of the class D, and 'D' is not defined in either
                                       # of those, it defined in the locals of class C.

        def method(self, x:field2) -> field2:  # this is OK
            ...

        def method(self, x) -> field:  # this FAILS, field is local to C and
                                    # is therefore not visible to D unless
                                    # accessed as C.field. This was already
                                    # true before the PEP.
            ...

        def Thing(self, y:Type[Thing]) -> Thing: # this is OK, and it links to the top level Thing.
            ...

Thing:TypeAlias = 'Mapping'
'''

        with captured_output() as (out, err):
            node, c = self.checkChains(
                code,
                    ['annotations -> ()',
                    'TypeAlias -> (TypeAlias -> (), TypeAlias -> (), TypeAlias -> ())',
                    'Mapping -> ()',
                    'Dict -> ()',
                    'Type -> (Type -> (<Subscript> -> ()))',
                    'C -> (C -> (.field -> ()), C -> (.D -> ()), C -> (.D -> '
                    '(.field2 -> ())))',
                    'Thing -> (Thing -> (<Subscript> -> ()), Thing -> ())'],
                strict=False
            )
        produced_messages = out.getvalue().strip().split("\n")

        expected_warnings = [
            "W: unbound identifier 'field'",
            "W: unbound identifier 'D'",
        ]

        assert len(produced_messages) == len(expected_warnings), produced_messages
        assert all(any(w in pw for pw in produced_messages) for w in expected_warnings)

        # locals of C
        self.assertEqual(c.dump_chains(node.body[-2]),
                         ['method -> ()',
                            'method -> ()',
                            'method -> ()',
                            'method -> ()',
                            'field -> (field -> (), field -> ())',
                            'D -> (D -> ())'])

        # locals of D
        self.assertEqual(c.dump_chains(node.body[-2].body[-1]),
                         ['field2 -> (field2 -> (), field2 -> ())',
                            'method -> ()',
                            'method -> ()',
                            'method -> ()',
                            'method -> ()',
                            'Thing -> ()'])

    def test_pep563_self_referential_annotation(self):
        code = '''
"""
module docstring
"""
from __future__ import annotations
class B:
    A: A # this should point to the top-level class
class A:
    A: 'str'
'''
        self.checkChains(
                code,
                ['annotations -> ()',
                 'B -> ()',
                 'A -> (A -> ())'], # good
                strict=False
            )

        code = '''
from __future__ import annotations
class A:
    A: 'str'
class B:
    A: A # this should point to the top-level class
'''
        self.checkChains(
                code,
                ['annotations -> ()',
                 'A -> (A -> ())',
                 'B -> ()'],
                strict=False
            )

    def test_wilcard_import_annotation(self):
        code = '''
from typing import *
primes: List[int] # should resolve to the star
        '''

        self.checkChains(
                code,
                ['* -> (List -> (<Subscript> -> ()))', 'primes -> ()'],
                strict=False
            )
        # same with 'from __future__ import annotations'
        self.checkChains(
                'from __future__ import annotations\n' + code,
                ['annotations -> ()', '* -> (List -> (<Subscript> -> ()))', 'primes -> ()'],
                strict=False
            )

    def test_wilcard_import_annotation_and_global_scope(self):
        # we might argue that it should resolve to both the wildcard
        # defined name and the type alias, but we're currently ignoring these
        # kind of scenarios.
        code = '''
from __future__ import annotations
from typing import *
primes: List[int]
List = list
    '''

        self.checkChains(
                code,
                ['annotations -> ()',
                '* -> ()',
                'primes -> ()',
                'List -> (List -> (<Subscript> -> ()))'],
                strict=False
            )

    def test_annotation_in_functions_locals(self):

        code = '''
class A:... # this one for pep 563 style
def generate():
    class A(int):... # this one for runtime style
    class C:
        field: A = 1
        def method(self, arg: A) -> None: ...
    return C
X = generate()
        '''

        # runtime style
        mod, chains = self.checkChains(
                code,
                ['A -> ()',
                 'generate -> (generate -> (<Call> -> ()))',
                 'X -> ()'],
            )
        self.assertEqual(chains.dump_chains(mod.body[1]),
                         ['A -> (A -> (), A -> ())',
                          'C -> (C -> ())'])

        # pep 563 style
        mod, chains = self.checkChains(
                'from __future__ import annotations\n' + code,
                ['annotations -> ()',
                 'A -> (A -> (), A -> ())',
                 'generate -> (generate -> (<Call> -> ()))',
                 'X -> ()'],
            )
        self.assertEqual(chains.dump_chains(mod.body[2]),
                         ['A -> ()',
                          'C -> (C -> ())'])

    def test_annotation_in_inner_functions_locals(self):

        code = '''
mytype = mytype2 = object
def outer():
    def middle():
        def inner(a:mytype, b:mytype2): pass
        class mytype(str):
            ...
        return inner
    class mytype2(int):
        ...
    return middle()
fn = outer()
        '''

        mod = self.ast.parse(code)
        chains = beniget.DefUseChains('test')
        with captured_output() as (out, err):
            chains.visit(mod)

        produced_messages = out.getvalue().strip().split("\n")

        self.assertEqual(produced_messages, ["W: unbound identifier 'mytype' at test:5:20"])

        self.assertEqual(
                chains.dump_chains(mod),
                ['mytype -> ()',
                 'mytype2 -> ()',
                 'outer -> (outer -> (<Call> -> ()))',
                 'fn -> ()'],
            )

        self.assertEqual(chains.dump_chains(mod.body[1]),
                         ['middle -> (middle -> (<Call> -> ()))',
                          'mytype2 -> (mytype2 -> ())'])
        self.assertEqual(chains.dump_chains(mod.body[1].body[0]),
                         ['inner -> (inner -> ())',
                          'mytype -> ()']) # annotation is unbound, so not linked here (and a warning is emitted)

        # in this case, the behaviour changes radically with pep 563

        mod, chains = self.checkChains(
                'from __future__ import annotations\n' + code,
                ['annotations -> ()',
                 'mytype -> (mytype -> ())',
                 'mytype2 -> (mytype2 -> ())',
                 'outer -> (outer -> (<Call> -> ()))',
                 'fn -> ()'],
            )
        self.assertEqual(chains.dump_chains(mod.body[2]),
                         ['middle -> (middle -> (<Call> -> ()))',
                          'mytype2 -> ()'])
        self.assertEqual(chains.dump_chains(mod.body[2].body[0]),
                         ['inner -> (inner -> ())',
                          'mytype -> ()'])

        # but if we remove 'mytype = mytype2 = object' and
        # keep the __future__ import then all anotations refers
        # to the inner classes.

    def test_lookup_scopes(self):

        def get_scopes():
            yield self.ast.parse('')                                # Module
            yield self.ast.parse('def f(): pass').body[0]           # FunctionDef
            yield self.ast.parse('class C: pass').body[0]           # ClassDef
            yield self.ast.parse('lambda: True').body[0].value      # Lambda
            yield self.ast.parse('(x for x in list())').body[0].value  # GeneratorExp
            yield self.ast.parse('{k:v for k, v in dict().items()}').body[0].value  # DictComp
            yield def695(None)

        mod, fn, cls, lambd, gen, comp, typeparams = get_scopes()
        assert isinstance(mod, self.ast.Module)
        assert isinstance(fn, self.ast.FunctionDef)
        assert isinstance(cls, self.ast.ClassDef)
        assert isinstance(lambd, self.ast.Lambda)
        assert isinstance(gen, self.ast.GeneratorExp)
        assert isinstance(comp, self.ast.DictComp)

        assert _get_lookup_scopes((mod, fn, fn, fn, cls)) == [mod, fn, fn, fn, cls]
        assert _get_lookup_scopes((mod, fn, fn, fn, cls, fn)) == [mod, fn, fn, fn, fn]
        assert _get_lookup_scopes((mod, cls, fn)) == [mod, fn]
        assert _get_lookup_scopes((mod, cls, fn, cls, fn)) == [mod, fn, fn]
        assert _get_lookup_scopes((mod, cls, fn, lambd, gen)) == [mod, fn, lambd, gen]
        assert _get_lookup_scopes((mod, fn, comp)) == [mod, fn, comp]
        assert _get_lookup_scopes((mod, fn)) == [mod, fn]
        assert _get_lookup_scopes((mod, cls)) == [mod, cls]
        assert _get_lookup_scopes((mod,)) == [mod]
        assert _get_lookup_scopes((mod, typeparams)) == [mod, typeparams]
        assert _get_lookup_scopes((mod, typeparams, typeparams)) == [mod, typeparams, typeparams]
        assert _get_lookup_scopes((mod, cls, typeparams)) == [mod, cls, typeparams]
        assert _get_lookup_scopes((mod, cls, cls, typeparams)) == [mod, cls, typeparams]
        assert _get_lookup_scopes((mod, cls, cls, typeparams, fn)) == [mod, typeparams, fn]

        with self.assertRaises(ValueError, msg='invalid heads: must include at least one element'):
            _get_lookup_scopes(())

    def test_annotation_inner_inner_fn(self):
        code = '''
def outer():
    def middle():
        def inner(a:mytype):
            ...
    class mytype(str):...
'''
        mod, chains = self.checkChains(
                code,
                ['outer -> ()',],
            )
        self.assertEqual(chains.dump_chains(mod.body[0]),
                         ['middle -> ()',
                          'mytype -> (mytype -> ())'])

        mod, chains = self.checkChains(
            'from __future__ import annotations\n' + code,
             ['annotations -> ()',
              'outer -> ()',],
        )
        self.assertEqual(chains.dump_chains(mod.body[1]),
                         ['middle -> ()',
                          'mytype -> (mytype -> ())'])


    def test_annotation_very_nested(self):

        # this code does not produce any pyright warnings
        code = '''
from __future__ import annotations

# when the following line is defined,
# all annotations references points to it.
# when it's not defined, all anotation points
# to the inner classes.
# in both cases pyright doesn't report any errors.
mytype = mytype2 = object

def outer():
    def middle():
        def inner(a:mytype, b:mytype2):
            return getattr(a, 'count')(b)
        class mytype(str):
            class substr(int):
                ...
            def count(self, sep:substr) -> mytype:
                def c(x:mytype, y:mytype2) -> mytype:
                    return mytype('{},{}'.format(x,y))
                return c(self, mytype2(sep))
        return inner(mytype(), mytype2())
    class mytype2(int):
        ...
    return middle()
fn = outer()
        '''

        mod, chains = self.checkChains(
                code,
                ['annotations -> ()',
                'mytype -> (mytype -> (), mytype -> (), mytype -> (), mytype -> ())',
                'mytype2 -> (mytype2 -> (), mytype2 -> ())',
                'outer -> (outer -> (<Call> -> ()))',
                'fn -> ()'],
            )
        self.assertEqual(chains.dump_chains(mod.body[2]),
                         ['middle -> (middle -> (<Call> -> ()))',
                          'mytype2 -> (mytype2 -> (<Call> -> (<Call> -> ())), '
                          'mytype2 -> (<Call> -> (<Call> -> ())))'])
        self.assertEqual(chains.dump_chains(mod.body[2].body[0]),
                         ['inner -> (inner -> (<Call> -> ()))',
                          'mytype -> (mytype -> (<Call> -> (<Call> -> ())), mytype -> (<Call> -> ()))'])

        mod, chains = self.checkChains(
                code.replace('mytype = mytype2 = object', 'pass'),
                ['annotations -> ()',
                'outer -> (outer -> (<Call> -> ()))',
                'fn -> ()'],
            )
        self.assertEqual(chains.dump_chains(mod.body[2]),
                         ['middle -> (middle -> (<Call> -> ()))',
                          'mytype2 -> (mytype2 -> (<Call> -> (<Call> -> ())), '
                                       'mytype2 -> (<Call> -> (<Call> -> ())), '
                                       'mytype2 -> (), '
                                       'mytype2 -> ())'])
        self.assertEqual(chains.dump_chains(mod.body[2].body[0]),
                         ['inner -> (inner -> (<Call> -> ()))',
                          'mytype -> (mytype -> (<Call> -> (<Call> -> ())), '
                                     'mytype -> (<Call> -> ()), '
                                     'mytype -> (), '
                                     'mytype -> (), '
                                     'mytype -> (), '
                                     'mytype -> ())'])

    def test_pep563_type_alias_override_class(self):
        code = '''
from __future__ import annotations
class B:
    A: A # this should point to the top-level alias
class A:
    A: A # this should point to the top-level alias
A = bytes
'''
        self.checkChains(
                code,
                ['annotations -> ()',
                 'B -> ()',
                 'A -> ()',
                 'A -> (A -> (), A -> ())'], # good
                strict=False
            )

    def test_annotation_def_is_not_assign_target(self):
        code = 'from typing import Optional; var:Optional'
        self.checkChains(code, ['Optional -> (Optional -> ())',
                                'var -> ()'])
    
    def test_pep563_disallowed_expressions(self):
        cases = [
            "def func(a: (yield)) -> ...: ...",
            "def func(a: ...) -> (yield from []): ...",
            "def func(*a: (y := 3)) -> ...: ...",
            "def func(**a: (await 42)) -> ...: ...",

            "x: (yield) = True",
            "x: (yield from []) = True",
            "x: (y := 3) = True",
            "x: (await 42) = True",]

        for code in cases:
            if ':=' in code and sys.version_info < (3,8):
                continue
            if 'await' in code and sys.version_info < (3,7):
                continue
            code = f'from __future__ import annotations\n' + code
            with self.subTest(code):
                self.check_message(code, ['cannot be used in annotation-like scopes'])
        
        for code in cases:
            if ':=' in code and sys.version_info < (3,8):
                continue
            if 'await' in code and sys.version_info < (3,7):
                continue
            with self.subTest(code):
                # TODO: From python 3.14, this should generate the same error.
                self.check_message(code, [])
    
    # PEP-695 test cases taken from https://github.com/python/cpython/pull/103764/files
    # but also https://github.com/python/cpython/pull/109297/files and
    # https://github.com/python/cpython/pull/109123/files

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_typeparams_name_collision_01(self):
        # The following code triggers syntax error at runtime.
        # But detecting this would required beniget to keep track of the 
        # names of type parameters and validate them like we validate comprehensions or annotations. 
        # We don't do it for functions currently, so it doesn't make sens to do it for 
        # type parameters at this time.
        code = """def func[**A, A](): ..."""
        node, du = self.checkChains(code, ['func -> ()'])
        assert du.dump_chains(def695(node.body[0])) == ['A -> ()', 'A -> ()']
    
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_typeparams_name_non_collision_02(self):
        code = """def func[A](A): return A"""
        node, du = self.checkChains(code, ['func -> ()'])
        assert du.dump_chains(def695(node.body[0])) == ['A -> ()']

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_typeparams_name_non_collision_03(self):
        code = """def func[A](*A): return A"""
        node, du = self.checkChains(code, ['func -> ()'])
        func = node.body[0]
        assert du.dump_chains(func) == ['A -> (A -> ())']
        assert du.dump_chains(def695(func)) == ['A -> ()']

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_typeparams_name_non_collision_04(self):
        # Mangled names should not cause a conflict.
        code = """class ClassA:\n def func[__A](self, __A): return __A"""
        node, du = self.checkChains(code, ['ClassA -> ()'])
        cls = node.body[0]
        method = cls.body[0]
        assert du.dump_chains(method) == ['self -> ()', '__A -> (__A -> ())']
        assert du.dump_chains(def695(method)) == ['__A -> ()']

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_typeparams_name_non_collision_05(self):
        code = """class ClassA:\n def func[_ClassA__A](self, __A): return __A"""
        node, du = self.checkChains(code, ['ClassA -> ()'])
        cls = node.body[0]
        method = cls.body[0]
        assert du.dump_chains(method) == ['self -> ()', '__A -> (__A -> ())']
        assert du.dump_chains(def695(method)) == ['_ClassA__A -> ()']

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_typeparams_name_non_collision_06(self):
        code = """class ClassA[X]:\n def func(self, X): return X"""
        node, du = self.checkChains(code, ['ClassA -> ()'])
        cls = node.body[0]
        assert du.dump_chains(cls) == ['func -> ()']
        assert du.dump_chains(def695(cls)) == ['X -> ()']

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_typeparams_name_non_collision_07(self):
        code = """class ClassA[X]:\n def func(self):\n  X = 1;return X"""
        node, du = self.checkChains(code, ['ClassA -> ()'])
        cls = node.body[0]
        assert du.dump_chains(cls) == ['func -> ()']
        assert du.dump_chains(def695(cls)) == ['X -> ()']
        
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_typeparams_name_non_collision_08(self):
        code = """class ClassA[X]:\n def func(self): return [X for X in [1, 2]]"""
        node, du = self.checkChains(code, ['ClassA -> ()'])
        cls = node.body[0]
        assert du.dump_chains(cls) == ['func -> ()']
        assert du.dump_chains(def695(cls)) == ['X -> ()']

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_typeparams_name_non_collision_09(self):
        code = """class ClassA[X]:\n def func[X](self):..."""
        node, du = self.checkChains(code, ['ClassA -> ()'])
        cls = node.body[0]
        method = cls.body[0]
        assert du.dump_chains(cls) == ['func -> ()']
        assert du.dump_chains(def695(cls)) == ['X -> ()']
        assert du.dump_chains(method) == ['self -> ()']
        assert du.dump_chains(def695(method)) == ['X -> ()']

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_typeparams_name_non_collision_10(self):
        code = """class ClassA[X,Y]:\n Y: bytes\n X: int"""
        node, du = self.checkChains(code, ['ClassA -> ()'])
        cls = node.body[0]
        assert du.dump_chains(cls) == ['Y -> ()', 'X -> ()']
        assert du.dump_chains(def695(cls)) == ['X -> ()', 'Y -> ()']

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_typeparams_name_non_collision_13(self):
        code = """X = 1\ndef outer():\n def inner[X]():\n  global X;X=2\n return inner"""
        node, chains = self.checkChains(code, ['X -> ()', 'outer -> ()'])
        self.assertEqual(chains.dump_chains(node.body[-1]), ['inner -> (inner -> ())'])

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_typeparams_disallowed_expressions(self):
        cases = ["type X = (yield)",
        "type X = (yield from x)",
        "type X = (await 42)",
        "async def f(): type X = (yield)",
        "type X = (y := 3)",
        "class X[T: (yield)]: pass",
        "class X[T: (yield from [])]: pass",
        "class X[T: (await 42)]: pass",
        "class X[T: (y := 3)]: pass",
        ]

        for code in cases:
            with self.subTest(code):
                self.check_message(code, ['cannot be used in annotation-like scopes'])
        
        for code in cases:
            code = f'from __future__ import annotations\n' + code
            with self.subTest(code):
                self.check_message(code, ['cannot be used in annotation-like scopes'])
    
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_type_alias_name_collision_01(self):
        # syntax error at runtime "duplicate type parameter 'A'"
        code = """type TA1[A, **A] = None"""
        self.checkChains(code, ['TA1 -> ()'])
    
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_type_alias_name_non_collision_02(self):
        code = """type TA1[A] = lambda A: A"""
        self.checkChains(code, ['TA1 -> ()'])

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_type_alias_name_non_collision_03(self):
        code = """class Outer[A]:\n type TA1[A] = None"""
        self.checkChains(code, ['Outer -> ()'])
    
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_type_alias_access_01(self):
        code = "type TA1[A, B] = dict[A, B]"
        self.checkChains(code, ['TA1 -> ()'])

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_type_alias_access_02(self):
        code = """type TA1[A, B] = TA1[A, B] | int"""
        self.checkChains(code, ['TA1 -> (TA1 -> (<Subscript> -> (<BinOp> -> ())))'])

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_type_alias_access_03(self):
        code = """class Outer[A]:\n def inner[B](self):\n  type TA1[C] = TA1[A, B] | int; return TA1"""
        self.checkChains(code, ['Outer -> ()'])

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_scopes01(self):
        code = """\
from typing import Sequence

# The following generates no compiler error, but a type checker
# should generate an error because an upper bound type must be concrete,
# and ``Sequence[S]`` is generic. Future extensions to the type system may
# eliminate this limitation.
class ClassA[S, T: Sequence[S]]: ...

# The following generates no compiler error, because the bound for ``S``
# is lazily evaluated. However, type checkers should generate an error.
class ClassB[S: Sequence[T], T]: ...
"""
        node, du = self.checkChains(code, ['Sequence -> (Sequence -> (<Subscript> -> ()), Sequence -> (<Subscript> -> ()))',
                                'ClassA -> ()',
                                'ClassB -> ()'])
        assert du.dump_chains(def695(node.body[1])) == ['S -> (S -> (<Subscript> -> ()))', 'T -> ()']
        assert du.dump_chains(def695(node.body[2])) == ['S -> ()', 'T -> (T -> (<Subscript> -> ()))']

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_scopes02(self):
        code = """\
from x import BaseClass, dec, Foo

class ClassA[T](BaseClass[T], param = Foo[T]): ...  # OK

print(T)  # Runtime error: 'T' is not defined

@dec(Foo[T])  # Runtime error: 'T' is not defined
class ClassA[T]: ...
"""
        self.check_message(code, ["W: unbound identifier 'T' at <unknown>:5:6", 
                                  "W: unbound identifier 'T' at <unknown>:7:9"])

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_scopes03(self):
        code = """\
from x import dec
def func1[T](a: T) -> T: ...  # OK

print(T)  # Runtime error: 'T' is not defined

def func2[T](a = list[T]): ...  # Runtime error: 'T' is not defined

@dec(list[T])  # Runtime error: 'T' is not defined
def func3[T](): ...
"""
        self.check_message(code, ["W: unbound identifier 'T' at <unknown>:4:6", 
                                  "W: unbound identifier 'T' at <unknown>:6:22", 
                                  "W: unbound identifier 'T' at <unknown>:8:10"])
    
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_scopes04(self):

        code = """\
S = 0

def outer1[S]():
    S = 1
    T = 1

    def outer2[T]():

        def inner1():
            nonlocal S  # OK because it binds variable S from outer1
            print(S)
            nonlocal T  # Syntax error: nonlocal binding not allowed for type parameter
            print(T)

        def inner2():
            global S  # OK because it binds variable S from global scope
            print(S)
"""
        self.check_message(code,  ['W: names defined in annotation scopes cannot be rebound with nonlocal statements at <unknown>:12:12'])
        node, du = self.checkChains(code, ['S -> (S -> (<Call> -> ()))', 'outer1 -> ()'], strict=False)
        # assert
        

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_scopes04bis(self):
        code = '''\
def outer1():
    def outer2[T]():
        def inner1():
            print(T)
        inner1()
    outer2()
outer1()
'''
        self.check_message(code, [])

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_scopes05(self):
        code = """\
from typing import Sequence

class Outer:
    class Private:
        pass

    # If the type parameter scope was like a traditional function/method scope,
    # the base class 'Private' would not be accessible here.
    class Inner[T](Private, Sequence[T]):
        pass

    # Likewise, 'Inner' would not be available in these type annotations.
    def method1[T](self, a: Inner[T]) -> Inner[T]:
        return a
"""
        node, du = self.checkChains(code, [('Sequence -> ('
                                                'Sequence -> ('
                                                    '<Subscript> -> ('
                                                        'Inner -> ('
                                                            'Inner -> ('
                                                                '<Subscript> -> ()'
                                                            '), '
                                                            'Inner -> ('
                                                                '<Subscript> -> ()'
                                                            ')'
                                                        ')'
                                                    ')'
                                                ')'
                                            ')'), 
                                            'Outer -> ()'])
        Outer = next(iter(d for d in du.locals[node] if d.name() == 'Outer'))
        assert du.dump_chains(Outer.node) == ['Private -> (Private -> (Inner -> (Inner -> (<Subscript> -> ()), Inner -> (<Subscript> -> ()))))', 
                                              'Inner -> (Inner -> (<Subscript> -> ()), Inner -> (<Subscript> -> ()))', 
                                              'method1 -> ()']

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_scopes06(self):
        code = """\
from typing import Sequence
from x import decorator

T = 0

@decorator(T)  # Argument expression `T` evaluates to 0
class ClassA[T](Sequence[T]):
    T = 1

    # All methods below should result in a type checker error
    # "type parameter 'T' already in use" because they are using the
    # type parameter 'T', which is already in use by the outer scope
    # 'ClassA'.
    def method1[T](self):
        ...

    def method2[T](self, x = T):  # Parameter 'x' gets default value of 1
        ...

    def method3[T](self, x: T):  # Parameter 'x' has type T (scoped to method3)
        ...

"""
        node, du = self.checkChains(code, ['Sequence -> (Sequence -> (<Subscript> -> (ClassA -> ())))',
                                'decorator -> (decorator -> (<Call> -> (ClassA -> ())))',
                                'T -> (T -> (<Call> -> (ClassA -> ())))',
                                'ClassA -> ()'])
        ClassA = next(iter(d for d in du.locals[node] if d.name() == 'ClassA'))
        assert du.dump_chains(def695(ClassA.node)) == ['T -> (T -> (<Subscript> -> (ClassA -> ())))']
        assert du.dump_chains(ClassA.node) == ['T -> (T -> (method2 -> ()))', 
                                               'method1 -> ()', 
                                               'method2 -> ()', 
                                               'method3 -> ()']
       
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_scopes07(self):
        code = """\
T = 0

# T refers to the global variable
print(T)  # Prints 0

class Outer[T]:
    T = 1

    # T refers to the local variable scoped to class 'Outer'
    print(T)  # Prints 1

    class Inner1:
        T = 2

        # T refers to the local type variable within 'Inner1'
        print(T)  # Prints 2

        def inner_method(self):
            # T refers to the type parameter scoped to class 'Outer';
            # If 'Outer' did not use the new type parameter syntax,
            # this would instead refer to the global variable 'T'
            print(T)  # Prints 'T'

    def outer_method(self):
        T = 3

        # T refers to the local variable within 'outer_method'
        print(T)  # Prints 3

        def inner_func():
            # T refers to the variable captured from 'outer_method'
            print(T)  # Prints 3
"""
        node, du = self.checkChains(code,  ['T -> (T -> (<Call> -> ()))', 'Outer -> ()'])
        
        assert [d.name() for d in du.locals[node]] == ['T', 'Outer']

        Outer = next(iter(d for d in du.locals[node] if d.name() == 'Outer'))
        Inner1 = next(iter(d for d in du.locals[Outer.node] if d.name() == 'Inner1'))
        inner_method = next(iter(d for d in du.locals[Inner1.node] if d.name() == 'inner_method'))
        outer_method = next(iter(d for d in du.locals[Outer.node] if d.name() == 'outer_method'))

        assert du.dump_chains(inner_method.node) == ['self -> ()',]
        assert du.dump_chains(Inner1.node) == ['T -> (T -> (<Call> -> ()))', 'inner_method -> ()',]
        assert du.dump_chains(outer_method.node) == ['self -> ()',
                                                     'T -> (T -> (<Call> -> ()), T -> (<Call> -> ()))',
                                                     'inner_func -> ()',]

    
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_scopes08(self):
        code = '''\
from x import decorator
T = 1
@decorator
def f[decorator, T: int, U: (int, str), *Ts, **P](
    y: U,
    x: T = T, # default values are evaluated outside the def695 scope
    *args: *Ts,
    **kwargs: P.kwargs,
) -> T:
    return x
'''
        node, du = self.checkChains(code,  ['decorator -> (decorator -> ())', 
                                 'T -> (T -> (f -> ()))', 
                                 'f -> ()'])
        assert du.dump_chains(def695(node.body[-1])) == [
            'decorator -> ()', 'T -> (T -> (), T -> ())', 
            'U -> (U -> ())', 'Ts -> (Ts -> (<Starred> -> ()))', 
            'P -> (P -> (.kwargs -> ()))']
        assert du.dump_chains(node.body[-1]) == ['y -> ()', 'x -> (x -> ())', 
                                                 'args -> ()', 'kwargs -> ()']
        
            
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_scopes09(self):
        code = '''\
from x import decorator
@decorator
class B[decorator](object):
    print(decorator)
'''
        self.checkChains(code,  ['decorator -> (decorator -> (B -> ()))', 
                                 'B -> ()'])
    
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_scopes10(self):
        code = '''\
class C[V]:
    class D:
        class E:
            a: V'''
        node, du = self.checkChains(code,  ['C -> ()'])
        C = next(iter(d for d in du.locals[node] if d.name() == 'C'))
        assert def695(C.node) in du.locals
        assert du.dump_chains(def695(C.node)) == ['V -> (V -> ())']

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_gen_exp_in_nested_class(self):
        # from https://github.com/python/cpython/pull/109196/files
        code = """
        from test.test_type_params import make_base
        class C[T]:
            T = "class"
            class Inner(make_base(T for _ in (1,)), make_base(T)):
                pass
        """
        self.checkChains(code, ['make_base -> (make_base -> (<Call> -> (Inner -> ())), make_base -> (<Call> -> (Inner -> ())))', 'C -> ()'])
    
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_listcomp_in_nested_class(self):
        code = """
            from test.test_type_params import make_base
            class C[T]:
                T = "class"
                class Inner(make_base([T for _ in (1,)]), make_base(T)):
                    pass
        """
        self.checkChains(code, ['make_base -> (make_base -> (<Call> -> (Inner -> ())), make_base -> (<Call> -> (Inner -> ())))', 'C -> ()'])

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_gen_exp_in_nested_generic_class(self):
        code = """
            from test.test_type_params import make_base
            class C[T]:
                T = "class"
                class Inner[U](make_base(T for _ in (1,)), make_base(T)):
                    pass
        """
        self.check_message(code, [])
        self.checkChains(code, ['make_base -> (make_base -> (<Call> -> (Inner -> ())), make_base -> (<Call> -> (Inner -> ())))', 'C -> ()'])
    
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_listcomp_in_nested_generic_class(self):
        code = """
        from test.test_type_params import make_base
        class C[T]:
            T = "class"
            class Inner[U](make_base([T for _ in (1,)]), make_base(T)):
                pass
        """
        self.check_message(code, [])
        self.checkChains(code, ['make_base -> (make_base -> (<Call> -> (Inner -> ())), make_base -> (<Call> -> (Inner -> ())))', 'C -> ()'])

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_gen_exp_in_generic_method(self):
        code = """
            class C[T]:
                T = "class"
                def meth[U](x: (T for _ in (1,)), y: T):
                    pass
        """
        self.check_message(code, [])
        self.checkChains(code, ['C -> ()'])

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_nested_scope_in_generic_alias(self):
        code = """
            class C[T]:
                T = "class"
                {}
        """
        error_cases = [
            "type Alias1[T] = lambda: T",
            "type Alias2 = lambda: T",
            "type Alias3[T] = (T for _ in (1,))",
            "type Alias4 = (T for _ in (1,))",
            "type Alias5[T] = [T for _ in (1,)]",
            "type Alias6 = [T for _ in (1,)]",
        ]
        for case in error_cases:
            with self.subTest(case=case):
                self.check_message(code.format(case), [])
    
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_pep695_later_defined_typevar_reference(self):
        cases = [
            'class o[T:(S,),S]:...',
            'def o[T:(S,),S]():...',
            'type o[T:(S,),S]=...', ]

        for code in cases:
            with self.subTest(code):
                self.checkChains(code, ['o -> ()'])
        
        for code in cases:
            code = 'S = 1\n' + code
            with self.subTest(code):
                self.checkChains(code, ['S -> ()', 'o -> ()'])
    
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")  
    def test_pepe695_class_keywords(self):
        src = '''
        class A[T]:
            ...
        class Bag[T](A[T], metaclass=T): ...'''
        self.check_message(src, [])
    
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")  
    def test_pepe695_annotations_before_python_314(self):
        src = '''def f[T](a:(b:=True)): ...'''
        
        # Python 3.12
        self.check_message(src, [])

        # Python 3.14
        self.check_message('from __future__ import annotations\n' + src, 
                           ['W: assignment expression cannot be used in annotation-like scopes at <unknown>:2:12'])

    @skipIf(sys.version_info < (3,13), "Python 3.13 syntax")  
    def test_pep696_type_aliases_default_forward_reference(self):
        code = '''type X[T = defer] = ...; defer = X'''
        self.check_message(code, [])

    @skipIf(sys.version_info < (3,10), "Python 3.10 syntax")
    def test_match_value(self):
        code = '''
command = 123
match command:
    case 123 as b:
        b+=1
        '''
        self.checkChains(code, ['command -> (command -> ())',
                                'b -> (b -> ())'])

    @skipIf(sys.version_info < (3,10), "Python 3.10 syntax")
    def test_match_list(self):
        code = '''
command = 'go there'
match command.split():
    case ["go", direction]:
        print(direction)
    case _:
        raise ValueError("Sorry")
        '''
        self.checkChains(code, ['command -> (command -> (.split -> (<Call> -> ())))',
                                'direction -> (<MatchSequence> -> (), direction -> (<Call> -> ()))'])

    @skipIf(sys.version_info < (3,10), "Python 3.10 syntax")
    def test_match_list_star(self):
        code = '''
command = 'drop'
match command.split():
    case ["go", direction]: ...
    case ["drop", *objects]:
        print(objects)
        '''
        self.checkChains(code, ['command -> (command -> (.split -> (<Call> -> ())))',
                                'direction -> (<MatchSequence> -> ())',
                                'objects -> (<MatchSequence> -> (), objects -> (<Call> -> ()))'])

    @skipIf(sys.version_info < (3,10), "Python 3.10 syntax")
    def test_match_dict(self):
        code = '''
ui = object()
action = dict(text='')
match action:
    case {"text": str(message), "color": str(c), **rest}:
        ui.set_text_color(c)
        ui.display(message)
        print(rest)
    case {"sleep": float(duration)}:
        ui.wait(duration)
    case {"sound": str(url), "format": "ogg"}:
        ui.play(url)
    case {"sound": _, "format": _}:
        raise ValueError("Unsupported audio format")
print(c)
        '''
        self.checkChains(code, ['ui -> (ui -> (.set_text_color -> (<Call> -> ())), '
                                'ui -> (.display -> (<Call> -> ())), '
                                'ui -> (.wait -> (<Call> -> ())), '
                                'ui -> (.play -> (<Call> -> ())))',
                                'action -> (action -> ())',
                                'message -> (<MatchClass> -> (rest -> (rest -> (<Call> -> ()))), '
                                'message -> (<Call> -> ()))',
                                'c -> (<MatchClass> -> (rest -> (rest -> (<Call> -> ()))), '
                                'c -> (<Call> -> ()), c -> (<Call> -> ()))',
                                'rest -> (rest -> (<Call> -> ()))',
                                'duration -> (<MatchClass> -> (<MatchMapping> -> ()), '
                                'duration -> (<Call> -> ()))',
                                'url -> (<MatchClass> -> (<MatchMapping> -> ()), url -> (<Call> -> ()))'])

    @skipIf(sys.version_info < (3,10), "Python 3.10 syntax")
    def test_match_class_rebinds_attrs(self):

        code = '''
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

point = Point(-2,1)
match point:
    case Point(x=0, y=0):
        print("Origin")
    case Point(x=0, y=y):
        print(f"Y={y}")
    case Point(x=x, y=0):
        print(f"X={x}")
    case Point(x=x, y=y):
        print("Somewhere else")
    case _:
        print("Not a point")
print(x, y)
        '''
        self.checkChains(
                code, ['dataclass -> (dataclass -> (Point -> (Point -> (<Call> -> ()), Point -> (<MatchClass> -> ()), Point -> (<MatchClass> -> ()), Point -> (<MatchClass> -> ()), Point -> (<MatchClass> -> ()))))',
                       'Point -> (Point -> (<Call> -> ()), Point -> (<MatchClass> -> ()), Point -> (<MatchClass> -> ()), Point -> (<MatchClass> -> ()), Point -> (<MatchClass> -> ()))',
                       'point -> (point -> ())',
                       'y -> (<MatchClass> -> (), y -> (<FormattedValue> -> (<JoinedStr> -> (<Call> -> ()))), y -> (<Call> -> ()))',
                       'x -> (<MatchClass> -> (), x -> (<FormattedValue> -> (<JoinedStr> -> (<Call> -> ()))), x -> (<Call> -> ()))',
                       'x -> (<MatchClass> -> (), x -> (<Call> -> ()))',
                       'y -> (<MatchClass> -> (), y -> (<Call> -> ()))'])

    def test_WindowsError_builtin_name(self):
        # Tests for issue https://github.com/serge-sans-paille/beniget/issues/119
        code = 'try: 1/0\nexcept WindowsError as e: raise'
        self.check_message(code, [])

    def test_newer_Python_version_builtin_name(self):
        # Tests for issue https://github.com/serge-sans-paille/beniget/issues/119
        code = ('try: 1/0\nexcept (PythonFinalizationError, EncodingWarning) as e: raise\n'
                'a,b = anext(), aiter()')
        self.check_message(code, [])

    @skipIf(sys.version_info < (3, 9), 'Use the warlus operator')
    def test_class_decorators_runs_before_bases_and_keywords_wrt_warlus(self):
        code = '''class A:... \n@D \n@Z \nclass C(D, (D:=A), (Z:=D), Z,  metaclass=(Z:=D)):...'''
        self.check_message(code, ["W: unbound identifier 'D' at <unknown>:2:1",
            "W: unbound identifier 'Z' at <unknown>:3:1",
            "W: unbound identifier 'D' at <unknown>:4:8"])

    @skipIf(sys.version_info < (3, 9), 'Use the warlus operator')
    def test_function_decorators_runs_after_default_values_wrt_warlus(self):
        code = '''class A:... \n@D \ndef C(b=(D:=A)) -> D: ...'''
        self.check_message(code, [])

    @skipIf(sys.version_info < (3, 9), 'Use the warlus operator')
    def test_function_decorators_runs_before_annotation_wrt_warlus(self):
        code = '''class A:... \n@D \ndef C(b:(D:=A)) -> D: ...'''
        self.check_message(code, ["W: unbound identifier 'D' at <unknown>:2:1"])

    @skipIf(sys.version_info < (3, 9), 'Use the warlus operator')
    def test_function_default_values_order_wrt_warlus(self):
        code = '''def C(b=(D:=1), z=D, *, c=D) -> D: ...'''
        self.check_message(code, [])

    @skipIf(sys.version_info < (3, 9), 'Use the warlus operator')
    def test_function_annotation_runs_after_default_values_wrt_warlus(self):
        code = '''def C(b:(D:=F), *, c=D, e=(F:=2)) -> D: ...'''
        self.check_message(code, ["W: unbound identifier 'D' at <unknown>:1:21"])

    @skipIf(sys.version_info < (3, 9), 'Use the warlus operator')
    def test_function_decorators_runs_before_annotations_wrt_warlus(self):
        code = '''@D \ndef C(b:(D:=1)): ...'''
        self.check_message(code, ["W: unbound identifier 'D' at <unknown>:1:1"])

    @skipIf(sys.version_info < (3, 9), 'Use the warlus operator')
    def test_function_decorators_runs_after_default_values_wrt_warlus(self):
        code = '''@D \ndef C(b=(D:=1)): ...'''
        self.check_message(code, [])

    def test_fstring(self):
        code = "v = 3.6; f'f-strings are new in Python {v}!'"
        self.checkChains(code, ['v -> (v -> (<FormattedValue> -> (<JoinedStr> -> ())))'])

    @skipIf(sys.version_info < (3, 14), 'Use the t-strings')
    def test_tstring(self):
        # Template strings are evaluated eagerly from left to right, 
        # just like f-strings. This means that interpolations are evaluated 
        # immediately when the template string is processed, not deferred or wrapped in lambdas.
        code = "pi = 3.14; t't-strings are new in Python {pi!s}!'"
        self.checkChains(code, ['pi -> (pi -> (<Interpolation> -> (<TemplateStr> -> ())))'])


class TestDefUseChainsStdlib(TestDefUseChains):
    ast = _ast

if sys.implementation.name == 'pypy':
    def replace_platform_dependent_names(chain):
        return chain.replace('<builtin_function>', '<builtin_function_or_method>')
else:
    def replace_platform_dependent_names(chain):
        return chain

# Work around Constant simplification from Python 3.8
if sys.version_info < (3, 8):
    def replace_deprecated_names(out):
        return out.replace(
            '<Num>', '<Constant>'
        ).replace(
            '<Ellipsis>', '<Constant>'
        ).replace(
            '<Str>', '<Constant>'
        ).replace(
            '<Bytes>', '<Constant>'
        ).replace(
            '<NameConstant>', '<Constant>'
        )
else:
    def replace_deprecated_names(out): 
        return out

def normalize_chain(out):
    out = replace_platform_dependent_names(out)
    out = replace_deprecated_names(out)
    return out

def normalize_usedef_chains(self):
    out = sorted((normalize_chain(kname), normalize_chain("{} <- {{{}}}".format(
            kname, ", ".join(sorted(normalize_chain(u) for u in usesnames))
        ))) for kname, usesnames in self._dump_chains())
    return ", ".join(s for k, s in out)

class TestUseDefChains(TestCase):
    ast = _gast
    
    def checkChains(self, code, ref, strict=True):
        node = self.ast.parse(code)
        
        c = StrictDefUseChains() if strict else beniget.DefUseChains()
        
        c.visit(node)
        cc = beniget.UseDefChains(c)
        actual = normalize_usedef_chains(cc)
        self.assertEqual(actual, ref)

    def test_simple_expression(self):
        code = "a = 1; a"
        self.checkChains(code, "a <- {a}")

    def test_call(self):
        code = "from foo import bar; bar(1, 2)"
        self.checkChains(code, "<Call> <- {<Constant>, <Constant>, bar}, bar <- {bar}")

    def test_arguments(self):
        code = "def f(a, b=True, *, c:int): return a(b, c)"
        self.checkChains(code, "<Call> <- {a, b, c}, a <- {a}, b <- {b}, c <- {c}, f <- {<Constant>}, int <- {<type>}")

    def test_excepthandler(self):
        code = "try: raise int \nexcept KeyError as e: \n print(e)"
        self.checkChains(code, "<Call> <- {e, print}, KeyError <- {<type>}, e <- {e}, int <- {<type>}, print <- {<builtin_function_or_method>}")

    def test_delete(self):
        code = "a = 1; del a"
        self.checkChains(code, "a <- {a}")
    
    @skipIf(sys.version_info < (3,13), "Python 3.13 syntax")  
    def test_pep696_type_aliases(self):
        code = '''type X[T = bytes] = dict[T, int]'''
        self.checkChains(code, '<Subscript> <- {<Tuple>, dict}, '
                               '<Tuple> <- {T, int}, '
                               'T <- {T}, bytes <- {<type>}, '
                               'dict <- {<type>}, int <- {<type>}')
    
    @skipIf(sys.version_info < (3,13), "Python 3.13 syntax")  
    def test_pepe696_function_type_param(self):
        code = '''def foo[T = bytes, *Ts = *tuple[str, int], **P = [str, int]](*args: *Ts, **kwargs: P.kwargs) -> T: ...'''
        self.checkChains(code, '.kwargs <- {P}, <List> <- {int, str}, '
                               '<Starred> <- {<Subscript>}, '
                               '<Starred> <- {Ts}, <Subscript> <- {<Tuple>, tuple}, '
                               '<Tuple> <- {int, str}, P <- {P}, T <- {T}, '
                               'Ts <- {Ts}, bytes <- {<type>}, int <- {<type>}, '
                               'int <- {<type>}, str <- {<type>}, '
                               'str <- {<type>}, tuple <- {<type>}')
    
    @skipIf(sys.version_info < (3,13), "Python 3.13 syntax")  
    def test_pep696_class_paramspec(self):
        code = '''class Foo[**P = [str, int]]: ...'''
        self.checkChains(code, '<List> <- {int, str}, int <- {<type>}, str <- {<type>}')

    @skipIf(sys.version_info < (3,13), "Python 3.13 syntax")  
    def test_pepe696_class_typevar_tuple(self):
        code = '''class Foo[*Ts = *tuple[str, int]]: ...'''
        self.checkChains(code, '<Starred> <- {<Subscript>}, '
                               '<Subscript> <- {<Tuple>, tuple}, '
                               '<Tuple> <- {int, str}, '
                               'int <- {<type>}, str <- {<type>}, tuple <- {<type>}')

    def test_fstring(self):
        code = "v = 3.6; f'f-strings are new in Python {v}!'"
        self.checkChains(code, '<FormattedValue> <- {v}, '
                               '<JoinedStr> <- {<Constant>, <Constant>, <FormattedValue>}, '
                               'v <- {v}')
    
    @skipIf(sys.version_info < (3, 14), 'Use the t-strings')
    def test_tstring(self):
        code = "pi = 3.14; t't-strings are new in Python {pi!s}!'"
        self.checkChains(code, '<Interpolation> <- {pi}, '
                               '<TemplateStr> <- {<Constant>, <Constant>, <Interpolation>}, '
                               'pi <- {pi}')

class TestUseDefChainsStdlib(TestUseDefChains):
    ast = _ast
        
class TestCollecLocals(TestCase):
    ast = _gast

    def test_module_locals(self):
        src = 'class A:...  \ndef b(): ...  \nvar = True   \nfrom x import y as e   \nimport ast   \nimport concurrent.futures'
        node = self.ast.parse(src)
        assert collect_locals(node) == {'A', 'b', 'var', 'e', 'ast', 'concurrent'}

    def test_class_locals(self):
        src = 'class A:\n    x = 1\n    def foo(self): pass\n    def bar(self): pass'
        node = self.ast.parse(src)
        assert collect_locals(node.body[0]) == {'x', 'foo', 'bar'}

    def test_function_locals(self):
        src = 'def f():\n    x = 1\n    y = 2\n    return x + y'
        node = self.ast.parse(src)
        assert collect_locals(node.body[0]) == {'x', 'y'}

    def test_function_parameters(self):
        src = 'def f(a, b, *, c, **kwargs): pass'
        node = self.ast.parse(src)
        # TODO: Is it normal that parameters are not included in locals?
        assert collect_locals(node.body[0]) == set()

    @skipIf(sys.version_info < (3,9), "Python 3.9 syntax")
    def test_warlus_operator(self):
        src = 'def f():\n    if (x := 1):\n        pass'
        node = self.ast.parse(src)
        assert collect_locals(node.body[0]) == {'x'}

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_class_type_parameters(self):
        src = 'class A[T, U]: v = 1'
        cls = self.ast.parse(src).body[0]
        assert collect_locals(cls) == {'v'}
        assert collect_locals(def695(cls)) == {'A', 'T', 'U'}

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_function_type_parameters(self):
        src = 'def f[T, U](): v = 1'
        cls = self.ast.parse(src).body[0]
        assert collect_locals(cls) == {'v'}
        assert collect_locals(def695(cls)) == {'f', 'T', 'U'}

    def test_comprehensions(self):
        src = '[x for x in range(10)]'
        node = self.ast.parse(src)
        comp = node.body[0].value
        assert collect_locals(comp) == {'x'}

class TestCollecLocalsStdlib(TestCollecLocals):
    ast = _ast

class TestDef695(TestCase):
    ast = _gast

    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_def695_equality(self):
        src = 'class A[T]: ...'
        node = self.ast.parse(src).body[0]

        assert def695(node) == def695(node)
        assert hash(def695(node)) == hash(def695(node))

        assert def695(node) != node
        assert hash(def695(node)) != hash(node)

        assert def695(node) is not def695(node)

    
    @skipIf(sys.version_info < (3,12), "Python 3.12 syntax")
    def test_def695_in_locals(self):
        src = 'class ClassA[T]:\n def func(self, x: T) -> T: ...'
        node, du = TestDefUseChains.checkChains(self, src, ['ClassA -> ()'])

        assert def695(node.body[0]) in du.locals
        assert du.dump_chains(def695(node.body[0])) == ['T -> (T -> (), T -> ())']

class TestDef695Stdlib(TestDef695):
    ast = _ast