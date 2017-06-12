#!/usr/bin/env python
import ast
import copy
import json
import utils as u
from astunparse import unparse


def ast_uses_varset(root, varset):
    class UsesVarsetVisitor(ast.NodeVisitor):
        def __init__(self):
            self.uses_varset = False

        def visit_Name(self, node):
            if node.id in varset:
                self.uses_varset = True

    vis = UsesVarsetVisitor()
    vis.visit(root)
    return vis.uses_varset


def is_declaration(node, typ):
    if isinstance(node, ast.Subscript):
        return is_declaration(node.value, typ)
    if not isinstance(node, ast.Call): return False
    if not isinstance(node.func, ast.Name): return False
    return node.func.id == typ


def is_input_declaration(node):
    return is_declaration(node, "Input")


def is_output_declaration(node):
    return is_declaration(node, "Output")


def is_var_declaration(node):
    return is_declaration(node, "Var")


def get_var_name(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Subscript):
        return get_var_name(node.value)
    elif isinstance(node, ast.Call):
        return None  # This is the Input()[10] case
    else:
        raise Exception("Can't extract var name from '%s'" % (unparse(node).rstrip()))


def get_set_vars(root):
    class GetSetVarsVisitor(ast.NodeVisitor):
        def __init__(self):
            self.set_vars = set()

        def visit_Call(self, node):
            if u.is_set_to_call(node):
                self.set_vars.add(get_var_name(node.func.value))

    vis = GetSetVarsVisitor()
    if isinstance(root, list):
        for stmt in root:
            vis.visit(stmt)
    else:
        vis.visit(root)
    return vis.set_vars


def get_input_dependent_vars(root):
    class GetInputsVisitor(ast.NodeVisitor):
        def __init__(self):
            self.input_vars = set()

        def visit_Assign(self, node):
            if is_input_declaration(node.value):
                self.input_vars.add(get_var_name(node.targets[0]))
            elif ast_uses_varset(node.value, self.input_vars):
                self.input_vars.add(get_var_name(node.targets[0]))

        def visit_With(self, node):
            if ast_uses_varset(node.context_expr, self.input_vars):
                # Track every set variable in the with. This is an over-approximation.
                self.input_vars.update(get_set_vars(node.body))
            if node.optional_vars is not None\
               and ast_uses_varset(node.context_expr, self.input_vars):
                with_var = get_var_name(node.optional_vars)
                self.input_vars.add(with_var)
                self.generic_visit(node)
                self.input_vars.remove(with_var)
            else:
                self.generic_visit(node)

        def visit_If(self, node):
            # Track setting of values conditional on input-dependent values:
            if ast_uses_varset(node.test, self.input_vars):
                self.input_vars.update(get_set_vars(node.body))
                self.input_vars.update(get_set_vars(node.orelse))
            self.generic_visit(node)

        def visit_Call(self, node):
            if u.is_set_to_call(node)\
               and ast_uses_varset(node.args[0], self.input_vars):
                self.input_vars.add(get_var_name(node.func.value))

    reached_fixpoint = False
    old_varnum = 0
    vis = GetInputsVisitor()
    while not reached_fixpoint:
        vis.visit(root)
        reached_fixpoint = (old_varnum == len(vis.input_vars))
        old_varnum = len(vis.input_vars)

    input_dependents = vis.input_vars

    # Add all variables that are being set below input-dependent statements to the input_dependents set.
    # This is a gross over-approximation, but avoids problems in cases like this:
    #  for i in range(I):
    #    input_indep.set_to(Func(param[1], i))
    #    with input_dep as bla:
    #      ...
    # The loop would be marked as input-dependent, and unrolled after translation,
    # yielding several input_indep.set_to(...) statements. They would all use the same
    # value, but things like the ILP backend are unhappy.
    set_growing = True
    while set_growing:
        set_growing = False
        for stmt in root.body:
            if ast_uses_varset(stmt, input_dependents):
                set_vars = get_set_vars(stmt)
                if not(set_vars.issubset(input_dependents)):
                    set_growing = True
                    input_dependents.update(set_vars)

    return input_dependents

def extend_subscript_for_input(node, extension):
    if isinstance(node.slice, ast.Index):
        idx = node.slice.value
        if isinstance(idx, ast.Tuple):
            new_idx = ast.Tuple([extension] + idx.elts, ast.Load())
        else:
            new_idx = ast.Tuple([extension, idx], ast.Load())
        node.slice.value = new_idx
    else:
        raise Exception("Unhandled node indexing: '%s'" % (unparse(node).rstrip()))
    return node


def add_input_indices(root, input_vars, index_var):
    class AddInputIndicesVisitor(ast.NodeTransformer):
        def visit_Subscript(self, node):
            if get_var_name(node) in input_vars:
                return extend_subscript_for_input(node, index_var)
            return node

        def visit_Name(self, node):
            if node.id in input_vars:
                return ast.Subscript(node, ast.Index(index_var), node.ctx)
            return node

    vis = AddInputIndicesVisitor()
    root = vis.visit(root)
    return ast.fix_missing_locations(root)


def generate_io_stmt(input_idx, var_name, value, func_name):
    if isinstance(value, list):
        return [ast.Expr(
                    ast.Call(
                        ast.Attribute(
                            ast.Subscript(
                                ast.Name(var_name, ast.Load()),
                                ast.Index(
                                    ast.Tuple([ast.Num(input_idx),
                                               ast.Num(val_idx)],
                                               ast.Load())),
                                ast.Load()),
                            func_name,
                            ast.Load()),
                        [ast.Num(val)],
                        [], None, None))
                for val_idx, val in enumerate(value)]
    else:
        return [ast.Expr(
                    ast.Call(
                        ast.Attribute(
                            ast.Subscript(
                                ast.Name(var_name, ast.Load()),
                                ast.Index(ast.Num(input_idx)),
                                ast.Load()),
                            func_name,
                            ast.Load()),
                        [ast.Num(value)],
                        [], None, None))]


def translate_to_tptv1(parsed_model, data_batch, hypers):
    parsed_model = u.replace_hypers(parsed_model, hypers)
    input_dependents = get_input_dependent_vars(parsed_model)

    idx_var_name = "input_idx"
    idx_var = ast.Name(idx_var_name, ast.Load())
    input_number = len(data_batch['instances'])
    range_expr = ast.Num(input_number)

    input_vars = set()
    output_vars = set()
    var_decls = []
    input_stmts = []
    general_stmts = []
    output_stmts = []
    for stmt in parsed_model.body:
        if isinstance(stmt, ast.Assign) and is_input_declaration(stmt.value):
            input_vars.add(get_var_name(stmt.targets[0]))
            var_decls.append(stmt)
        elif isinstance(stmt, ast.Assign) and is_output_declaration(stmt.value):
            output_vars.add(get_var_name(stmt.targets[0]))
            var_decls.append(stmt)
        elif isinstance(stmt, ast.Assign) and is_var_declaration(stmt.value):
            var_decls.append(stmt)
        elif ast_uses_varset(stmt, input_dependents):
            input_stmts.append(add_input_indices(stmt, input_dependents, idx_var))
        elif ast_uses_varset(stmt, output_vars):
            output_stmts.append(stmt)
        else:
            general_stmts.append(stmt)

    input_init = []
    output_observation = []
    for input_idx, instance in enumerate(data_batch['instances']):
        for var_name, val in instance.iteritems():
            if var_name in input_vars:
                input_init.extend(generate_io_stmt(input_idx, var_name, val, "set_to_constant"))
            elif var_name in output_vars:
                output_observation.extend(generate_io_stmt(input_idx, var_name, val, "observe_value"))

    extended_var_decls = []
    for var_decl in var_decls:
        # If input-dependent, extend dimension by one
        if get_var_name(var_decl.targets[0]) in input_dependents:
            new_decl = copy.deepcopy(var_decl)
            if isinstance(new_decl.value, ast.Subscript):
                new_decl.value = extend_subscript_for_input(new_decl.value,
                                                            ast.Num(input_number))
            else:
                new_decl.value = ast.Subscript(new_decl.value,
                                               ast.Index(ast.Num(input_number)),
                                               ast.Load())
            extended_var_decls.append(new_decl)
        else:
            extended_var_decls.append(var_decl)

    input_loop = ast.For(ast.Name(idx_var_name, ast.Store()),
                         ast.Call(ast.Name("range", ast.Load()),
                                  [range_expr],
                                  [], None, None),
                         input_stmts,
                         [])
    parsed_model.body = general_stmts + extended_var_decls + input_init + [input_loop] + output_stmts + output_observation
    ast.fix_missing_locations(parsed_model)

    return parsed_model
