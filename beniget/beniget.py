from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys

if sys.version_info >= (3,6)
    from typing import TYPE_CHECKING
else:
    TYPE_CHECKING = False

import gast as ast #type:ignore[import]

if TYPE_CHECKING:
    from typing import Union, Optional, Tuple, Iterable, Iterator, List, Dict, Any, Mapping, Type, TypeVar
    T = TypeVar('T', bound=Type[ast.AST])

# TODO: remove me when python 2 is not supported anymore
class _ordered_set(object):
    def __init__(self, elements=None):#type:ignore
        self.values = OrderedDict.fromkeys(elements or [])

    def add(self, value):#type:ignore
        self.values[value] = None

    def update(self, values):#type:ignore
        self.values.update((k, None) for k in values)

    def __iter__(self):#type:ignore
        return iter(self.values.keys())

    def __contains__(self, value):#type:ignore
        return value in self.values

    def __add__(self, other):#type:ignore
        out = self.values.copy()
        out.update(other.values)
        return out

    def __len__(self):#type:ignore
        return len(self.values)

if sys.version_info >= (3,6):
    from .ordered_set import ordered_set
else:
    # python < 3,6 we fall back on older version of the ordered_set
    ordered_set = _ordered_set

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
    """

    def __init__(self):
        # type: () -> None
        self._parents = dict() #type:dict[ast.AST, list[ast.AST]]
        self._current = list() #type:list[ast.AST]

    def generic_visit(self, node):
        # type: (ast.AST) -> None
        self._parents[node] = list(self._current)
        self._current.append(node)
        super(Ancestors, self).generic_visit(node)
        self._current.pop()

    def parent(self, node):
        # type: (ast.AST) -> ast.AST
        return self._parents[node][-1]

    def parents(self, node):
        # type: (ast.AST) -> List[ast.AST]
        return self._parents[node]

    def parentInstance(self, node, cls):
        # type: (ast.AST, type[T]|Tuple[type[T],...]) -> T
        for n in reversed(self._parents[node]):
            if isinstance(n, cls):
                return n
        raise ValueError("{} has no parent of type {}".format(node, cls))

    def parentFunction(self, node):
        # type: (ast.AST) -> ast.stmt
        return self.parentInstance(node, (ast.FunctionDef,
                                          ast.AsyncFunctionDef))

    def parentStmt(self, node):
        # type: (ast.AST) -> ast.stmt
        return self.parentInstance(node, ast.stmt)


class Def(object):
    """
    Model a definition, either named or unnamed, and its users.
    """

    __slots__ = "node", "_users"

    def __init__(self, node):
        # type: (Any) -> None
        # Def instances are also used to represented members of the builtin module
        # so any kind of type can be used with the 'node' attribute.
        self.node = node # type:Any
        self._users = ordered_set() #type:ordered_set[Def]

    def add_user(self, node):
        # type: (Def) -> None
        assert isinstance(node, Def)
        self._users.add(node)

    def name(self):
        # type: () -> str
        """
        If the node associated to this Def has a name, returns this name.
        Otherwise returns its type
        """
        if isinstance(self.node, (ast.ClassDef,
                                  ast.FunctionDef,
                                  ast.AsyncFunctionDef)):
            return self.node.name
        elif isinstance(self.node, ast.Name):
            return self.node.id
        elif isinstance(self.node, ast.alias):
            base = self.node.name.split(".", 1)[0]
            return self.node.asname or base
        elif isinstance(self.node, tuple):
            return self.node[1]
        else:
            return type(self.node).__name__

    def users(self):
        # type: () -> ordered_set[Def]
        """
        The list of `Def` instances that holds a reference to this node
        """
        return self._users

    def __repr__(self):
        # type: () -> str
        return self._repr({})

    def _repr(self, nodes):
        # type: (dict[Def, int]) -> str
        if self in nodes:
            return "(#{})".format(nodes[self])
        else:
            nodes[self] = len(nodes)
            return "{} -> ({})".format(
                self.node, ", ".join(u._repr(nodes.copy())
                                     for u in self._users)
            )

    def __str__(self):
        # type: () -> str
        return self._str({})

    def _str(self, nodes):
        # type: (Dict[Def, int]) -> str
        if self in nodes:
            return "(#{})".format(nodes[self])
        else:
            nodes[self] = len(nodes)
            return "{} -> ({})".format(
                self.name(), ", ".join(u._str(nodes.copy())
                                       for u in self._users)
            )


Builtins = {}

if sys.version_info.major == 2:
    BuiltinsSrc = __builtins__
else:
    import builtins

    BuiltinsSrc = builtins.__dict__ #type:ignore

Builtins = {k: v for k, v in BuiltinsSrc.items()} #type:ignore

Builtins["__file__"] = __file__

DeclarationStep, DefinitionStep = object(), object()


class CollectGlobals(ast.NodeVisitor):
    def __init__(self):
        # type: () -> None
        self.Globals = defaultdict(list) #type:dict[str, list[tuple[ast.Global, str]]]

    def visit_Global(self, node):
        # type: (ast.Global) -> None
        for name in node.names:
            self.Globals[name].append((node, name))


class DefUseChains(ast.NodeVisitor):
    """
    Module visitor that gathers two kinds of informations:
        - locals: Dict[node, List[Def]], a mapping between a node and the list
          of variable defined in this node,
        - chains: Dict[node, Def], a mapping between nodes and their chains.

    >>> import gast as ast
    >>> module = ast.parse("from b import c, d; c()")
    >>> duc = DefUseChains()
    >>> duc.visit(module)
    >>> for head in duc.locals[module]:
    ...     print("{}: {}".format(head.name(), len(head.users())))
    c: 1
    d: 0
    >>> alias_def = duc.chains[module.body[0].names[0]]
    >>> print(alias_def)
    c -> (c -> (Call -> ()))
    """

    def __init__(self, filename=None):
        # type: (Optional[str]) -> None
        """
        :param filename: included in error messages if specified
        :type filename: str
        """
        
        self.chains = {} #type:dict[ast.AST, Def]
        """
        Mapping from AST nodes to their respective Def instance.
        """
        
        self.locals = defaultdict(list) #type:dict[ast.AST, list[Def]]
        """
        Mapping from AST nodes to their local name definitions list.
        """
        
        self.filename = filename #type:str|None

        # deep copy of builtins, to remain reentrant
        self._builtins = {k: Def(v) for k, v in Builtins.items()}

        # function body are not executed when the function definition is met
        # this holds a stack of the functions met during body processing
        self._defered = [] #type:list[list[tuple[ast.AST, list[dict[str, ordered_set[Def]]]]]]

        # stack of mapping between an id and Names
        self._definitions = [] #type:list[dict[str, ordered_set[Def]]]

        # stack of variable defined with the global keywords
        self._promoted_locals = [] #type:list[set[str]]

        # stack of variable that were undefined when we met them, but that may
        # be defined in another path of the control flow (esp. in loop)
        self._undefs = [] #type:list[dict[str, list[tuple[Def, list[Def]]]]]

        # stack of current node holding definitions: class, module, function...
        self._currenthead = [] #type:list[ast.AST]

        self._breaks = [] #type:list[dict[str, ordered_set[Def]]]
        self._continues = [] #type:list[dict[str, ordered_set[Def]]]

        # dead code levels
        self.deadcode = 0

    # helpers

    def dump_definitions(self, node, ignore_builtins=True):
        # type: (ast.AST, bool) -> List[str]
        """
        Sorted list of names defined in the locals of the given AST node.
        
        :rtype: list[str]
        """
        if isinstance(node, ast.Module) and not ignore_builtins:
            builtins = {d for d in self._builtins.values()}
            return sorted(d.name()
                          for d in self.locals[node] if d not in builtins)
        else:
            return sorted(d.name() for d in self.locals[node])

    def dump_chains(self, node):
        # type: (ast.AST) -> List[str]
        chains = []
        for d in self.locals[node]:
            chains.append(str(d))
        return chains

    def unbound_identifier(self, name, node):
        # type: (str, ast.Name) -> None
        """
        Called for each unbound identifiers. 
        Override this method to stop printing unbound warnings.
        """
        if hasattr(node, "lineno"):
            filename = "{}:".format(
                "<unknown>" if self.filename is None else self.filename
            )
            location = " at {}{}:{}".format(filename,
                                            node.lineno,
                                            node.col_offset)
        else:
            location = ""
        print("W: unbound identifier '{}'{}".format(name, location))

    def lookup_identifier(self, name):
        # type: (str) -> Iterable[Def]
        for d in reversed(self._definitions):
            if name in d:
                return d[name]
        return []

    def defs(self, node):
        # type: (ast.Name) -> Iterable[Def]
        name = node.id
        stars = [] #type:list[Def]
        for d in reversed(self._definitions):
            if name in d:
                return d[name] if not stars else stars + list(d[name])
            if "*" in d:
                stars.extend(d["*"])

        undef = self.chains.setdefault(node, Def(node))

        if self._undefs:
            self._undefs[-1][name].append((undef, stars))

        if stars:
            return stars + [undef]
        else:
            if not self._undefs:
                self.unbound_identifier(name, node)
            return [undef]

    def process_body(self, stmts):
        # type: (Iterable[ast.stmt]) -> None
        deadcode = False
        for stmt in stmts:
            if isinstance(stmt, (ast.Break, ast.Continue, ast.Raise)):
                if not deadcode:
                    deadcode = True
                    self.deadcode += 1
            self.visit(stmt)
        if deadcode:
            self.deadcode -= 1

    def process_undefs(self):
        # type: () -> None
        for undef_name, _undefs in self._undefs[-1].items():
            if undef_name in self._definitions[-1]:
                for newdef in self._definitions[-1][undef_name]:
                    for undef, _ in _undefs:
                        for user in undef.users():
                            newdef.add_user(user)
            else:
                for undef, stars in _undefs:
                    if not stars:
                        self.unbound_identifier(undef_name, undef.node)
        self._undefs.pop()

    @contextmanager
    def DefinitionContext(self, node):
        # type: (ast.AST) -> Iterator[None]
        self._currenthead.append(node)
        self._definitions.append(defaultdict(ordered_set))
        self._promoted_locals.append(set())
        yield
        self._promoted_locals.pop()
        self._definitions.pop()
        self._currenthead.pop()

    @contextmanager
    def CompDefinitionContext(self, node):
        # type: (ast.AST) -> Iterator[None]
        if sys.version_info.major >= 3:
            self._currenthead.append(node)
            self._definitions.append(defaultdict(ordered_set))
            self._promoted_locals.append(set())
        yield
        if sys.version_info.major >= 3:
            self._promoted_locals.pop()
            self._definitions.pop()
            self._currenthead.pop()

    # stmt
    def visit_Module(self, node):
        # type: (ast.Module) -> None
        self.module = node
        with self.DefinitionContext(node):

            self._definitions[-1].update(
                {k: ordered_set((v,)) for k, v in self._builtins.items()}
            )

            self._defered.append([])
            self.process_body(node.body)

            # handle `global' keyword specifically
            cg = CollectGlobals()
            cg.visit(node)
            for nodes in cg.Globals.values():
                for n, name in nodes:
                    if name not in self._definitions[-1]:
                        dnode = Def((n, name))
                        self.set_definition(name, dnode)
                        self.locals[node].append(dnode)

            # handle function bodies
            for fnode, ctx in self._defered[-1]:
                visitor = getattr(self,
                                  "visit_{}".format(type(fnode).__name__))
                defs, self._definitions = self._definitions, ctx
                visitor(fnode, step=DefinitionStep)
                self._definitions = defs
            self._defered.pop()

            # various sanity checks
            if __debug__:
                overloaded_builtins = set()
                for d in self.locals[node]:
                    name = d.name()
                    if name in self._builtins:
                        overloaded_builtins.add(name)
                    assert name in self._definitions[0], (name, d.node)

                nb_defs = len(self._definitions[0])
                nb_bltns = len(self._builtins)
                nb_overloaded_bltns = len(overloaded_builtins)
                nb_heads = len({d.name() for d in self.locals[node]})
                assert nb_defs == nb_heads + nb_bltns - nb_overloaded_bltns

        assert not self._definitions
        assert not self._defered

    def set_definition(self, name, dnode_or_dnodes):
        # type: (str, Union[Def, Iterable[Def]]) -> None
        if self.deadcode:
            return
        if isinstance(dnode_or_dnodes, Def):
            self._definitions[-1][name] = ordered_set((dnode_or_dnodes,))
        else:
            self._definitions[-1][name] = ordered_set(dnode_or_dnodes)

    @staticmethod
    def add_to_definition(definition, name, dnode_or_dnodes):
        # type: (Mapping[str, ordered_set[Def]], str, Union[Def, Iterable[Def]]) -> None
        if isinstance(dnode_or_dnodes, Def):
            definition[name].add(dnode_or_dnodes)
        else:
            definition[name].update(dnode_or_dnodes)

    def extend_definition(self, name, dnode_or_dnodes):
        # type: (str, Union[Def, Iterable[Def]]) -> None
        if self.deadcode:
            return
        DefUseChains.add_to_definition(self._definitions[-1], name,
                                       dnode_or_dnodes)

    def visit_FunctionDef(self, node, step=DeclarationStep):
        # type: (Union[ast.AsyncFunctionDef, ast.FunctionDef], object) -> None
        if step is DeclarationStep:
            dnode = self.chains.setdefault(node, Def(node))
            self.set_definition(node.name, dnode)
            self.locals[self._currenthead[-1]].append(dnode)

            for kw_default in filter(None, node.args.kw_defaults): #type:ignore[var-annotated]
                self.visit(kw_default).add_user(dnode)
            for default in node.args.defaults:
                self.visit(default).add_user(dnode)
            for decorator in node.decorator_list:
                self.visit(decorator)

            definitions = list(self._definitions)
            if isinstance(self._currenthead[-1], ast.ClassDef):
                definitions.pop()
            self._defered[-1].append((node, definitions))
        elif step is DefinitionStep:
            # function is not considered as defined when evaluating returns
            if node.returns:
                self.visit(node.returns)
            with self.DefinitionContext(node):
                self.visit(node.args)
                self.process_body(node.body)
        else:
            raise NotImplementedError()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        # type: (ast.ClassDef) -> None
        dnode = self.chains.setdefault(node, Def(node))
        self.locals[self._currenthead[-1]].append(dnode)
        self.set_definition(node.name, dnode)
        for base in node.bases:
            self.visit(base).add_user(dnode)
        for keyword in node.keywords:
            self.visit(keyword.value).add_user(dnode)
        for decorator in node.decorator_list:
            self.visit(decorator).add_user(dnode)

        with self.DefinitionContext(node):
            self.set_definition("__class__", Def("__class__"))
            self.process_body(node.body)

    def visit_Return(self, node):
        # type: (ast.Return) -> None
        if node.value:
            self.visit(node.value)

    def visit_Break(self, _):
        # type: (ast.Break) -> None
        for k, v in self._definitions[-1].items():
            DefUseChains.add_to_definition(self._breaks[-1], k, v)
        self._definitions[-1].clear()

    def visit_Continue(self, _):
        # type: (ast.Continue) -> None
        for k, v in self._definitions[-1].items():
            DefUseChains.add_to_definition(self._continues[-1], k, v)
        self._definitions[-1].clear()

    def visit_Delete(self, node):
        # type: (ast.Delete) -> None
        for target in node.targets:
            self.visit(target)

    def visit_Assign(self, node):
        # type: (ast.Assign) -> None
        # link is implicit through ctx
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AnnAssign(self, node):
        # type: (ast.AnnAssign) -> None
        if node.value:
            dvalue = self.visit(node.value)
        dannotation = self.visit(node.annotation)
        dtarget = self.visit(node.target)
        dtarget.add_user(dannotation)
        if node.value:
            dvalue.add_user(dtarget)

    def visit_AugAssign(self, node):
        # type: (ast.AugAssign) -> None
        dvalue = self.visit(node.value)
        if isinstance(node.target, ast.Name):
            ctx, node.target.ctx = node.target.ctx, ast.Load()
            dtarget = self.visit(node.target)
            dvalue.add_user(dtarget)
            node.target.ctx = ctx
            if node.target.id in self._promoted_locals[-1]:
                self.extend_definition(node.target.id, dtarget)
            else:
                loaded_from = [d.name() for d in self.defs(node.target)]
                self.set_definition(node.target.id, dtarget)
                # If we augassign from a value that comes from '*', let's use
                # this node as the definition point.
                if '*' in loaded_from:
                    self.locals[self._currenthead[-1]].append(dtarget)
        else:
            self.visit(node.target).add_user(dvalue)

    def visit_Print(self, node):
        # type: (ast.Print) -> None
        if node.dest:
            self.visit(node.dest)
        for value in node.values:
            self.visit(value)

    def visit_For(self, node):
        # type: (Union[ast.AsyncFor, ast.For]) -> None
        self.visit(node.iter)

        self._breaks.append(defaultdict(ordered_set))
        self._continues.append(defaultdict(ordered_set))

        self._undefs.append(defaultdict(list))
        self._definitions.append(self._definitions[-1].copy())
        self.visit(node.target)
        self.process_body(node.body)
        self.process_undefs()

        continue_defs = self._continues.pop()
        for d, u in continue_defs.items():
            self.extend_definition(d, u)
        self._continues.append(defaultdict(ordered_set))

        # extra round to ``emulate'' looping
        self.visit(node.target)
        self.process_body(node.body)

        # process else clause in case of late break
        self._definitions.append(defaultdict(ordered_set))
        self.process_body(node.orelse)
        orelse_defs = self._definitions.pop()

        break_defs = self._breaks.pop()
        continue_defs = self._continues.pop()

        body_defs = self._definitions.pop()

        for d, u in orelse_defs.items():
            self.extend_definition(d, u)

        for d, u in continue_defs.items():
            self.extend_definition(d, u)

        for d, u in break_defs.items():
            self.extend_definition(d, u)

        for d, u in body_defs.items():
            self.extend_definition(d, u)

    visit_AsyncFor = visit_For

    def visit_While(self, node):
        # type: (ast.While) -> None

        self._definitions.append(self._definitions[-1].copy())
        self._undefs.append(defaultdict(list))
        self._breaks.append(defaultdict(ordered_set))
        self._continues.append(defaultdict(ordered_set))

        self.process_body(node.orelse)

        self._definitions.pop()

        self._definitions.append(self._definitions[-1].copy())

        self.visit(node.test)
        self.process_body(node.body)

        self.process_undefs()

        continue_defs = self._continues.pop()
        for d, u in continue_defs.items():
            self.extend_definition(d, u)
        self._continues.append(defaultdict(ordered_set))

        # extra round to simulate loop
        self.visit(node.test)
        self.process_body(node.body)

        # the false branch of the eval
        self.visit(node.test)

        self._definitions.append(self._definitions[-1].copy())
        self.process_body(node.orelse)

        orelse_defs = self._definitions.pop()
        body_defs = self._definitions.pop()
        break_defs = self._breaks.pop()
        continue_defs = self._continues.pop()

        for d, u in continue_defs.items():
            self.extend_definition(d, u)

        for d, u in break_defs.items():
            self.extend_definition(d, u)

        for d, u in orelse_defs.items():
            self.extend_definition(d, u)

        for d, u in body_defs.items():
            self.extend_definition(d, u)

    def visit_If(self, node):
        # type: (ast.If) -> None
        self.visit(node.test)

        # putting a copy of current level to handle nested conditions
        self._definitions.append(self._definitions[-1].copy())
        self.process_body(node.body)
        body_defs = self._definitions.pop()

        self._definitions.append(self._definitions[-1].copy())
        self.process_body(node.orelse)
        orelse_defs = self._definitions.pop()
        for d in body_defs:
            if d in orelse_defs:
                self.set_definition(d, body_defs[d] + orelse_defs[d])
            else:
                self.extend_definition(d, body_defs[d])

        for d in orelse_defs:
            if d in body_defs:
                pass  # already done in the previous loop
            else:
                self.extend_definition(d, orelse_defs[d])

    def visit_With(self, node):
        # type: (Union[ast.AsyncWith, ast.With]) -> None
        for withitem in node.items:
            self.visit(withitem)
        self.process_body(node.body)

    visit_AsyncWith = visit_With

    def visit_Raise(self, node):
        # type: (ast.Raise) -> None
        if node.exc:
            self.visit(node.exc)
        if node.cause:
            self.visit(node.cause)

    def visit_Try(self, node):
        # type: (ast.Try) -> None
        self._definitions.append(self._definitions[-1].copy())
        self.process_body(node.body)
        self.process_body(node.orelse)
        failsafe_defs = self._definitions.pop()

        # handle the fact that definitions may have fail
        for d in failsafe_defs:
            self.extend_definition(d, failsafe_defs[d])

        for excepthandler in node.handlers:
            self._definitions.append(defaultdict(ordered_set))
            self.visit(excepthandler)
            handler_def = self._definitions.pop()
            for hd in handler_def:
                self.extend_definition(hd, handler_def[hd])

        self.process_body(node.finalbody)

    def visit_Assert(self, node):
        # type: (ast.Assert) -> None
        self.visit(node.test)
        if node.msg:
            self.visit(node.msg)

    def visit_Import(self, node):
        # type: (ast.Import) -> None
        for alias in node.names:
            dalias = self.chains.setdefault(alias, Def(alias))
            base = alias.name.split(".", 1)[0]
            self.set_definition(alias.asname or base, dalias)
            self.locals[self._currenthead[-1]].append(dalias)

    def visit_ImportFrom(self, node):
        # type: (ast.ImportFrom) -> None
        for alias in node.names:
            dalias = self.chains.setdefault(alias, Def(alias))
            self.set_definition(alias.asname or alias.name, dalias)
            self.locals[self._currenthead[-1]].append(dalias)

    def visit_Exec(self, node):
        # type: (ast.Exec) -> None
        dnode = self.chains.setdefault(node, Def(node))
        self.visit(node.body)

        if node.globals:
            self.visit(node.globals)
        else:
            # any global may be used by this exec!
            for defs in self._definitions[0].values():
                for d in defs:
                    d.add_user(dnode)

        if node.locals:
            self.visit(node.locals)
        else:
            # any local may be used by this exec!
            visible_locals = set()
            for _definitions in reversed(self._definitions[1:]):
                for dname, defs in _definitions.items():
                    if dname not in visible_locals:
                        visible_locals.add(dname)
                        for d in defs:
                            d.add_user(dnode)

        self.extend_definition("*", dnode)

    def visit_Global(self, node):
        # type: (ast.Global) -> None
        for name in node.names:
            self._promoted_locals[-1].add(name)

    def visit_Nonlocal(self, node):
        # type: (ast.Nonlocal) -> None
        for name in node.names:
            for d in reversed(self._definitions[:-1]):
                if name not in d:
                    continue
                else:
                    # this rightfully creates aliasing
                    self.set_definition(name, d[name])
                    break
            else:
                self.unbound_identifier(name, node)

    def visit_Expr(self, node):
        # type: (ast.Expr) -> None
        self.generic_visit(node)

    # expr
    def visit_BoolOp(self, node):
        # type: (ast.BoolOp) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        for value in node.values:
            self.visit(value).add_user(dnode)
        return dnode

    def visit_BinOp(self, node):
        # type: (ast.BinOp) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        self.visit(node.left).add_user(dnode)
        self.visit(node.right).add_user(dnode)
        return dnode

    def visit_UnaryOp(self, node):
        # type: (ast.UnaryOp) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        self.visit(node.operand).add_user(dnode)
        return dnode

    def visit_Lambda(self, node, step=DeclarationStep):
        # type: (ast.Lambda, object) -> Def
        if step is DeclarationStep:
            dnode = self.chains.setdefault(node, Def(node))
            self._defered[-1].append((node, list(self._definitions)))
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
        # type: (ast.IfExp) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        self.visit(node.test).add_user(dnode)
        self.visit(node.body).add_user(dnode)
        self.visit(node.orelse).add_user(dnode)
        return dnode

    def visit_Dict(self, node):
        # type: (ast.Dict) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        for key in filter(None, node.keys): #type:ignore[var-annotated]
            self.visit(key).add_user(dnode)
        for value in node.values:
            self.visit(value).add_user(dnode)
        return dnode

    def visit_Set(self, node):
        # type: (ast.Set) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        for elt in node.elts:
            self.visit(elt).add_user(dnode)
        return dnode

    def visit_ListComp(self, node):
        # type: (Union[ast.GeneratorExp, ast.ListComp, ast.SetComp]) -> Def
        dnode = self.chains.setdefault(node, Def(node))

        with self.CompDefinitionContext(node):
            for comprehension in node.generators:
                self.visit(comprehension).add_user(dnode)
            self.visit(node.elt).add_user(dnode)

        return dnode

    visit_SetComp = visit_ListComp

    def visit_DictComp(self, node):
        # type: (ast.DictComp) -> Def
        dnode = self.chains.setdefault(node, Def(node))

        with self.CompDefinitionContext(node):
            for comprehension in node.generators:
                self.visit(comprehension).add_user(dnode)
            self.visit(node.key).add_user(dnode)
            self.visit(node.value).add_user(dnode)

        return dnode

    visit_GeneratorExp = visit_ListComp

    def visit_Await(self, node):
        # type: (ast.Attribute) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        self.visit(node.value).add_user(dnode)
        return dnode

    def visit_Yield(self, node):
        # type: (ast.Yield) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        if node.value:
            self.visit(node.value).add_user(dnode)
        return dnode

    visit_YieldFrom = visit_Await

    def visit_Compare(self, node):
        # type: (ast.Compare) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        self.visit(node.left).add_user(dnode)
        for expr in node.comparators:
            self.visit(expr).add_user(dnode)
        return dnode

    def visit_Call(self, node):
        # type: (ast.Call) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        self.visit(node.func).add_user(dnode)
        for arg in node.args:
            self.visit(arg).add_user(dnode)
        for kw in node.keywords:
            self.visit(kw.value).add_user(dnode)
        return dnode

    visit_Repr = visit_Await

    def visit_Constant(self, node):
        # type: (ast.Constant) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        return dnode

    def visit_FormattedValue(self, node):
        # type: (ast.FormattedValue) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        self.visit(node.value).add_user(dnode)
        if node.format_spec:
            self.visit(node.format_spec).add_user(dnode)
        return dnode
    
    def visit_JoinedStr(self, node):
        # type: (ast.JoinedStr) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        for value in node.values:
            self.visit(value).add_user(dnode)
        return dnode

    visit_Attribute = visit_Await

    def visit_Subscript(self, node):
        # type: (ast.Subscript) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        self.visit(node.value).add_user(dnode)
        self.visit(node.slice).add_user(dnode)
        return dnode

    visit_Starred = visit_Await

    def visit_NamedExpr(self, node):
        # type: (ast.NamedExpr) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        self.visit(node.value).add_user(dnode)
        self.visit(node.target)
        return dnode

    def visit_Name(self, node):
        # type: (ast.Name) -> Def

        if isinstance(node.ctx, (ast.Param, ast.Store)):
            dnode = self.chains.setdefault(node, Def(node))
            if node.id in self._promoted_locals[-1]:
                self.extend_definition(node.id, dnode)
                if dnode not in self.locals[self.module]:
                    self.locals[self.module].append(dnode)
            else:
                self.set_definition(node.id, dnode)
                if dnode not in self.locals[self._currenthead[-1]]:
                    self.locals[self._currenthead[-1]].append(dnode)

            if node.annotation is not None:
                self.visit(node.annotation)

        elif isinstance(node.ctx, (ast.Load, ast.Del)):
            node_in_chains = node in self.chains
            if node_in_chains:
                dnode = self.chains[node]
            else:
                dnode = Def(node)
            for d in self.defs(node):
                d.add_user(dnode)
            if not node_in_chains:
                self.chains[node] = dnode
            # currently ignore the effect of a del
        else:
            raise NotImplementedError()
        return dnode

    def visit_Destructured(self, node):
        # type: (Union[ast.List, ast.Tuple]) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        tmp_store = ast.Store()
        for elt in node.elts:
            if isinstance(elt, ast.Name):
                tmp_store, elt.ctx = elt.ctx, tmp_store
                self.visit(elt)
                tmp_store, elt.ctx = elt.ctx, tmp_store
            elif isinstance(elt, ast.Subscript):
                self.visit(elt)
            elif isinstance(elt, (ast.List, ast.Tuple)):
                self.visit_Destructured(elt)
        return dnode

    def visit_List(self, node):
        # type: (Union[ast.List, ast.Tuple]) -> Optional[Def]
        if isinstance(node.ctx, ast.Load):
            dnode = self.chains.setdefault(node, Def(node))
            for elt in node.elts:
                self.visit(elt).add_user(dnode)
            return dnode
        # unfortunately, destructured node are marked as Load,
        # only the parent List/Tuple is marked as Store
        elif isinstance(node.ctx, ast.Store):
            return self.visit_Destructured(node)
        # TODO: should this method ever return None?
        return None

    visit_Tuple = visit_List

    # slice

    def visit_Slice(self, node):
        # type: (ast.Slice) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        if node.lower:
            self.visit(node.lower).add_user(dnode)
        if node.upper:
            self.visit(node.upper).add_user(dnode)
        if node.step:
            self.visit(node.step).add_user(dnode)
        return dnode

    # misc

    def visit_comprehension(self, node):
        # type: (ast.comprehension) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        self.visit(node.iter).add_user(dnode)
        self.visit(node.target)
        for if_ in node.ifs:
            self.visit(if_).add_user(dnode)
        return dnode

    def visit_excepthandler(self, node):
        # type: (ast.excepthandler) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        if node.type:
            self.visit(node.type).add_user(dnode)
        if node.name:
            self.visit(node.name).add_user(dnode)
        self.process_body(node.body)
        return dnode

    def visit_arguments(self, node):
        # type: (ast.arguments) -> None
        for arg in node.args:
            self.visit(arg)

        for arg in node.posonlyargs:
            self.visit(arg)

        if node.vararg:
            self.visit(node.vararg)

        for arg in node.kwonlyargs:
            self.visit(arg)
        if node.kwarg:
            self.visit(node.kwarg)

    def visit_withitem(self, node):
        # type: (ast.withitem) -> Def
        dnode = self.chains.setdefault(node, Def(node))
        self.visit(node.context_expr).add_user(dnode)
        if node.optional_vars:
            self.visit(node.optional_vars)
        return dnode


class UseDefChains(object):
    """
    DefUseChains adaptor that builds a mapping between each user
    and the Def that defines this user:
        - chains: Dict[node, List[Def]], a mapping between nodes and the Defs
          that define it.
    """

    def __init__(self, defuses):
        # type: (DefUseChains) -> None
        self.chains = {} #type:dict[ast.AST, list[Def]]
        for chain in defuses.chains.values():
            if isinstance(chain.node, ast.Name):
                self.chains.setdefault(chain.node, [])
            for use in chain.users():
                self.chains.setdefault(use.node, []).append(chain)

        for chain in defuses._builtins.values():
            for use in chain.users():
                self.chains.setdefault(use.node, []).append(chain)

    def __str__(self):
        # type: () -> str
        out = []
        for k, uses in self.chains.items():
            kname = Def(k).name()
            kstr = "{} <- {{{}}}".format(
                kname, ", ".join(sorted(use.name() for use in uses))
            )
            out.append((kname, kstr))
        out.sort()
        return ", ".join(s for k, s in out)


if __name__ == "__main__":
    import sys

    class Beniget(ast.NodeVisitor):
        def __init__(self, filename, module):
            # type: (Optional[str], ast.Module) -> None
            super(Beniget, self).__init__()

            self.filename = filename or "<stdin>"

            self.ancestors = Ancestors()
            self.ancestors.visit(module)

            self.defuses = DefUseChains(self.filename)
            self.defuses.visit(module)

            self.visit(module)

        def check_unused(self, node, skipped_types=()):
            #type: (ast.AST, Tuple[Type[ast.AST], ...]) -> None
            for local_def in self.defuses.locals[node]:
                if not local_def.users():
                    if local_def.name() == "_":
                        continue  # typical naming by-pass
                    if isinstance(local_def.node, skipped_types):
                        continue

                    location = local_def.node
                    while not hasattr(location, "lineno"):
                        location = self.ancestors.parent(location)

                    if isinstance(location, ast.ImportFrom):
                        if location.module == "__future__":
                            continue

                    print(
                        "W: '{}' is defined but not used at {}:{}:{}".format(
                            local_def.name(),
                            self.filename,
                            location.lineno,
                            location.col_offset,
                        )
                    )

        def visit_Module(self, node):
            # type: (ast.Module) -> None
            self.generic_visit(node)
            if self.filename.endswith("__init__.py"):
                return
            self.check_unused(
                node, skipped_types=(ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef, ast.Name)
            )

        def visit_FunctionDef(self, node):
            # type: (ast.FunctionDef) -> None
            self.generic_visit(node)
            self.check_unused(node)

    paths = sys.argv[1:] or (None,)

    for path in paths:
        with open(path) if path else sys.stdin as target:
            module = ast.parse(target.read())
            Beniget(path, module)
