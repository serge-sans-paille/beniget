from collections import defaultdict
from contextlib import contextmanager
import sys

import gast as ast

class Ancestors(ast.NodeVisitor):

    def __init__(self):
        self.parents = dict()
        self.current = list()

    def generic_visit(self, node):
        self.parents[node] = list(self.current)
        self.current.push(node)
        super(Ancestors, self).generic_visit(node)
        self.current.pop()


class Def(object):
    __slots__ = 'node', 'uses'
    def __init__(self, node):
        self.node = node
        self.uses = list()

    def add_user(self, node):
        assert isinstance(node, Def)
        self.uses.append(node)

    def __repr__(self):
        return '{} -> ({})'.format(self.node, ", ".join(map(repr, self.uses)))

    def name(self):
        if isinstance(self.node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = self.node.name
        elif isinstance(self.node, ast.Name):
            name = self.node.id
        elif isinstance(self.node, ast.alias):
            base = self.node.name.split('.', 1)[0]
            name = self.node.asname or base
        elif isinstance(self.node, tuple):
            name = self.node[1]
        else:
            name = None
        return name


Builtins = {
}

if sys.version_info.major == 2:
    BuiltinsSrc = __builtins__
else:
    import builtins
    BuiltinsSrc = builtins.__dict__

Builtins = {k: [Def(v)] for k, v in BuiltinsSrc.items()}

Builtins['__file__'] = [Def(__file__)]

DeclarationStep, DefinitionStep = object(), object()

class CollectGlobals(ast.NodeVisitor):

    def __init__(self):
        self.Globals= defaultdict(list)

    def visit_Global(self, node):
        for name in node.names:
            self.Globals[name].append((node, name))

class Collect(ast.NodeVisitor):

    def __init__(self):
        self.defered = []
        self.chains = {}
        self.definitions = []
        self.undefs = []
        self.heads = defaultdict(list)
        self.currenthead = []

    # helpers

    def dump_definitions(self, node, ignore_builtins=True):
        if isinstance(node, ast.Module) and not ignore_builtins:
            builtins = {d[0] for d in Builtins.values()}
            return sorted(d.name() for d in self.heads[node] if d not in builtins)
        else:
            return sorted(d.name() for d in self.heads[node])

    def unbound_identifier(self, name, node):
        print("W: unbound identifier '{}' at {}:{}".format(name, node.lineno, node.col_offset))

    def defs(self, node):
        name = node.id
        stars = []
        for d in reversed(self.definitions):
            if name in d:
                return d[name] if not stars else stars + d[name]
            if '*' in d:
                stars.extend(d['*'])
        d = Def(node)
        if self.undefs:
            self.undefs[-1][name].append((d, stars))

        if stars:
            return stars + [d]
        else:
            if not self.undefs:
                self.unbound_identifier(name, node)
            return [d]

    def process_body(self, stmts):
        for stmt in stmts:
            self.visit(stmt)

    def process_undefs(self):
        for undef_name, undefs in self.undefs[-1].items():
            if undef_name in self.definitions[-1]:
                for newdef in self.definitions[-1][undef_name]:
                    for undef, stars in undefs:
                        for user in undef.uses:
                            newdef.add_user(user)
            else:
                for undef, stars in undefs:
                    if not stars:
                        self.unbound_identifier(undef_name, undef.node)
        self.undefs.pop()

    @contextmanager
    def DefinitionContext(self, node):
        self.currenthead.append(node)
        self.definitions.append(defaultdict(list))
        yield
        self.definitions.pop()
        self.currenthead.pop()

    @contextmanager
    def CompDefinitionContext(self, node):
        if sys.version_info.major >= 3:
            self.currenthead.append(node)
            self.definitions.append(defaultdict(list))
        yield
        if sys.version_info.major >= 3:
            self.definitions.pop()
            self.currenthead.pop()


    # stmt
    def visit_Module(self, node):
        with self.DefinitionContext(node):

            self.definitions[-1].update(Builtins)

            self.defered.append([])
            self.process_body(node.body)

            # handle `global' keyword specifically
            cg = CollectGlobals()
            cg.visit(node)
            for d, nodes in cg.Globals.items():
                for n, name in nodes:
                    if name not in self.definitions[-1]:
                        dnode = Def((n, name))
                        self.definitions[-1][name] = [dnode]
                        self.heads[node].append(dnode)

            # handle function bodies
            for fnode, ctx in self.defered[-1]:
                visitor = getattr(self,
                                  'visit_{}'.format(type(fnode).__name__))
                defs, self.definitions = self.definitions, ctx
                visitor(fnode, step=DefinitionStep)
                self.definitions = defs
            self.defered.pop()

            # various sanity checks
            if __debug__:
                overloaded_builtins = set()
                for d in self.heads[node]:
                    name = d.name()
                    if name in Builtins:
                        overloaded_builtins.add(name)
                    assert name in self.definitions[0], (name, d.node)
                assert len(self.definitions[0]) == len({d.name() for d in self.heads[node]}) + len(Builtins) - len(overloaded_builtins)

        assert not self.definitions
        assert not self.defered

    def visit_FunctionDef(self, node, step=DeclarationStep):
        if step is DeclarationStep:
            dnode = self.chains[node] = Def(node)
            self.definitions[-1][node.name] = [dnode]
            self.heads[self.currenthead[-1]].append(dnode)
            for kw_default in filter(None, node.args.kw_defaults):
                self.visit(kw_default).add_user(dnode)
            for default in node.args.defaults:
                self.visit(default).add_user(dnode)
            self.defered[-1].append((node, list(self.definitions)))
        elif step is DefinitionStep:
            with self.DefinitionContext(node):
                self.visit(node.args)
                self.process_body(node.body)
        else:
            raise NotImplementedError()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        dnode = self.chains[node] = Def(node)
        self.heads[self.currenthead[-1]].append(dnode)
        self.definitions[-1][node.name] = [dnode]
        for base in node.bases:
            self.visit(base).add_user(dnode)

        with self.DefinitionContext(node):
            self.definitions[-1]['__class__'] = [Def('__class__')]
            self.process_body(node.body)

    def visit_Return(self, node):
        if node.value:
            self.visit(node.value)

    def visit_Delete(self, node):
        for target in node.targets:
            self.visit(target)

    def visit_Assign(self, node):
        dvalue = self.visit(node.value)
        for target in node.targets:
            dvalue.add_user(self.visit(target))

    def visit_AnnAssign(self, node):
        if node.value:
            dvalue = self.visit(node.value)
        dannotation = self.visit(node.annotation)
        dtarget = self.visit(node.target)
        dtarget.add_user(dannotation)
        if node.value:
            dvalue.add_user(dtarget)

    def visit_AugAssign(self, node):
        dvalue = self.visit(node.value)
        self.visit(node.target).add_user(dvalue)

    def visit_Print(self, node):
        if node.dest:
            self.visit(node.dest)
        for value in node.values:
            self.visit(value)

    def visit_For(self, node):
        self.visit(node.iter)
        self.visit(node.target)

        self.definitions.append(defaultdict(list))
        self.undefs.append(defaultdict(list))
        self.process_body(node.body)
        self.process_undefs()

        self.definitions.append(defaultdict(list))
        self.process_body(node.orelse)

        orelse_defs = self.definitions.pop()
        body_defs = self.definitions.pop()

        for d, u in orelse_defs.items():
            self.definitions[-1][d].extend(u)

        for d, u in body_defs.items():
            self.definitions[-1][d].extend(u)



    visit_AsyncFor = visit_For

    def visit_While(self, node):
        self.visit(node.test)

        self.definitions.append(defaultdict(list))
        self.process_body(node.body)

        self.definitions.append(defaultdict(list))
        self.process_body(node.orelse)

        orelse_defs = self.definitions.pop()
        body_defs = self.definitions.pop()

        for d, u in orelse_defs.items():
            self.definitions[-1][d].extend(u)

        for d, u in body_defs.items():
            self.definitions[-1][d].extend(u)

    def visit_If(self, node):
        self.visit(node.test)

        self.definitions.append(defaultdict(list))
        self.process_body(node.body)
        body_defs = self.definitions.pop()

        self.definitions.append(defaultdict(list))
        self.process_body(node.orelse)
        orelse_defs = self.definitions.pop()

        for d in body_defs:
            if d in orelse_defs:
                self.definitions[-1][d] = body_defs[d] + orelse_defs[d]
            else:
                self.definitions[-1][d].extend(body_defs[d])

        for d in orelse_defs:
            if d in body_defs:
                pass  # already done in the previous loop
            else:
                self.definitions[-1][d].extend(orelse_defs[d])

    def visit_With(self, node):
        with self.DefinitionContext(node):
            for withitem in node.items:
                self.visit(withitem)
            self.process_body(node.body)

    visit_AsyncWith = visit_With

    def visit_Raise(self, node):
        if node.exc:
            self.visit(node.exc)
        if node.cause:
            self.visit(node.cause)

    def visit_Try(self, node):
        self.process_body(node.body)
        for excepthandler in node.handlers:
            self.visit(excepthandler)
        self.process_body(node.orelse)
        self.process_body(node.finalbody)

    def visit_Assert(self, node):
        self.visit(node.test)
        if node.msg:
            self.visit(node.msg)

    def visit_Import(self, node):
        for alias in node.names:
            dalias = Def(alias)
            base = alias.name.split('.', 1)[0]
            self.definitions[-1][alias.asname or base] = [dalias]
            self.heads[self.currenthead[-1]].append(dalias)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            dalias = Def(alias)
            self.definitions[-1][alias.asname or alias.name] = [dalias]
            self.heads[self.currenthead[-1]].append(dalias)

    def visit_Exec(self, node):
        dnode = self.chains[node] = Def(node)
        self.visit(node.body)

        if node.globals:
            self.visit(node.globals)
        else:
            # any global may be used by this exec!
            for defs in self.definitions[0].values():
                for d in defs:
                    d.add_user(dnode)

        if node.locals:
            self.visit(node.locals)
        else:
            # any local may be used by this exec!
            visible_locals = set()
            for definitions in reversed(self.definitions[1:]):
                for dname, defs in definitions.items():
                    if dname not in visible_locals:
                        visible_locals.add(dname)
                        for d in defs:
                            d.add_user(dnode)

        self.definitions[-1]['*'].append(dnode)

    def visit_Global(self, node):
        for name in node.names:
            # this rightfully creates aliasing
            self.definitions[-1][name] = self.definitions[0][name]

    def visit_Nonlocal(self, node):
        for name in node.names:
            for d in reversed(self.definitions[:-1]):
                if name not in d:
                    continue
                else:
                    # this rightfully creates aliasing
                    self.definitions[-1][name] = d[name]
                    break
            else:
                self.unbound_identifier(name, node)

    def visit_Expr(self, node):
        self.generic_visit(node)

    # expr
    def visit_BoolOp(self, node):
        dnode = self.chains[node] = Def(node)
        for value in node.values:
            self.visit(value).add_user(dnode)
        return dnode

    def visit_BinOp(self, node):
        dnode = self.chains[node] = Def(node)
        self.visit(node.left).add_user(dnode)
        self.visit(node.right).add_user(dnode)
        return dnode

    def visit_UnaryOp(self, node):
        dnode = self.chains[node] = Def(node)
        self.visit(node.operand).add_user(dnode)
        return dnode

    def visit_Lambda(self, node, step=DeclarationStep):
        if step is DeclarationStep:
            dnode = self.chains[node] = Def(node)
            self.defered[-1].append((node, list(self.definitions)))
            return dnode
        elif step is DefinitionStep:
            dnode = self.chains[node]
            with self.DefinitionContext(node):
                self.visit(node.args)
                self.visit(node.body).add_user(dnode)
            return dnode
        else:
            raise NotImplementedError()

    def visit_IfExp(self, node):
        dnode = self.chains[node] = Def(node)
        self.visit(node.test).add_user(dnode)
        self.visit(node.body).add_user(dnode)
        self.visit(node.orelse).add_user(dnode)
        return dnode

    def visit_Dict(self, node):
        dnode = self.chains[node] = Def(node)
        for key in filter(None, node.keys):
            self.visit(key).add_user(dnode)
        for value in node.values:
            self.visit(value).add_user(dnode)
        return dnode

    def visit_Set(self, node):
        dnode = self.chains[node] = Def(node)
        for elt in node.elts:
            self.visit(elt).add_user(dnode)
        return dnode

    def visit_ListComp(self, node):
        dnode = self.chains[node] = Def(node)

        with self.CompDefinitionContext(node):
            for comprehension in node.generators:
                self.visit(comprehension).add_user(dnode)
            self.visit(node.elt).add_user(dnode)

        return dnode

    visit_SetComp = visit_ListComp

    def visit_DictComp(self, node):
        dnode = self.chains[node] = Def(node)

        with self.CompDefinitionContext(node):
            for comprehension in node.generators:
                self.visit(comprehension).add_user(dnode)
            self.visit(node.key).add_user(dnode)
            self.visit(node.value).add_user(dnode)

        return dnode

    visit_GeneratorExp = visit_ListComp

    def visit_Await(self, node):
        dnode = self.chains[node] = Def(node)
        self.visit(node.value).add_user(dnode)
        return dnode

    def visit_Yield(self, node):
        dnode = self.chains[node] = Def(node)
        if node.value:
            self.visit(node.value).add_user(dnode)
        return dnode

    visit_YieldFrom = visit_Await

    def visit_Compare(self, node):
        dnode = self.chains[node] = Def(node)
        self.visit(node.left).add_user(dnode)
        for expr in node.comparators:
            self.visit(expr).add_user(dnode)
        return dnode

    def visit_Call(self, node):
        dnode = self.chains[node] = Def(node)
        self.visit(node.func).add_user(dnode)
        for arg in node.args:
            self.visit(arg).add_user(dnode)
        for kw in node.keywords:
            self.visit(kw.value).add_user(dnode)
        return dnode

    visit_Repr = visit_Await

    def visit_Num(self, node):
        dnode = self.chains[node] = Def(node)
        return dnode

    visit_Str = visit_Num

    def visit_FormattedValue(self, node):
        dnode = self.chains[node] = Def(node)
        self.visit(node.value).add_user(dnode)
        if node.format_spec:
            self.visit(node.format_spec).add_user(dnode)
        return dnode

    def visit_JoinedStr(self, node):
        dnode = self.chains[node] = Def(node)
        for value in node.values:
            self.visit(value).add_user(dnode)
        return dnode

    visit_Bytes = visit_Num
    visit_NameConstant = visit_Num
    visit_Ellipsis = visit_Num

    visit_Attribute = visit_Await

    def visit_Subscript(self, node):
        dnode = self.chains[node] = Def(node)
        self.visit(node.value).add_user(dnode)
        self.visit(node.slice).add_user(dnode)
        return dnode

    visit_Starred = visit_Await

    def visit_Name(self, node):
        dnode = self.chains[node] = Def(node)
        if isinstance(node.ctx, (ast.Param, ast.Store)):
            self.definitions[-1][node.id] = [dnode]
            self.heads[self.currenthead[-1]].append(dnode)
        elif isinstance(node.ctx, ast.Load):
            for d in self.defs(node):
                d.add_user(dnode)
        elif isinstance(node.ctx, ast.Del):
            self.definitions[-1][node.id].clear()
        else:
            raise NotImplementedError()
        return dnode

    def visit_List(self, node):
        dnode = self.chains[node] = Def(node)
        for elt in node.elts:
            self.visit(elt).add_user(dnode)
        return dnode

    visit_Tuple = visit_List

    # slice

    def visit_Slice(self, node):
        dnode = self.chains[node] = Def(node)
        if node.lower:
            self.visit(node.lower).add_user(dnode)
        if node.upper:
            self.visit(node.upper).add_user(dnode)
        if node.step:
            self.visit(node.step).add_user(dnode)
        return dnode

    def visit_ExtSlice(self, node):
        dnode = self.chains[node] = Def(node)
        for dim in node.dims:
            self.visit(dim).add_user(dnode)
        return dnode

    visit_Index = visit_Await

    # misc

    def visit_comprehension(self, node):
        dnode = self.chains[node] = Def(node)
        self.visit(node.iter).add_user(dnode)
        self.visit(node.target)
        for if_ in node.ifs:
            self.visit(if_).add_user(dnode)
        return dnode

    def visit_excepthandler(self, node):
        dnode = self.chains[node] = Def(node)
        if node.type:
            self.visit(node.type).add_user(dnode)
        if node.name:
            self.visit(node.name).add_user(dnode)
        for stmt in node.body:
            self.visit(stmt)
        return dnode

    def visit_arguments(self, node):
        for arg in node.args:
            self.visit(arg)

        if node.vararg:
            self.visit(node.vararg)
        for arg in node.kwonlyargs:
            self.visit(arg)
        if node.kwarg:
            self.visit(node.kwarg)

    def visit_withitem(self, node):
        dnode = self.chains[node] = Def(node)
        self.visit(node.context_expr).add_user(dnode)
        if node.optional_vars:
            self.visit(node.optional_vars)
        return dnode




if __name__ == '__main__':
    import sys
    import pprint
    for fname in sys.argv[1:]:
        print("== {} ==".format(fname))
        with open(fname) as text:
            try:
                module = ast.parse(text.read())
            except (SyntaxError, UnicodeDecodeError):
                continue
            c = Collect()
            c.visit(module)
#            for defs in c.heads.values():
#                for d in defs:
#                    if not d.uses:
#                        n = d.node
#                        print(getattr(n, 'name', getattr(n, 'id', None)), getattr(n, 'lineno', 0))
#

