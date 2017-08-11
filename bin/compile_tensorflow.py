#!/usr/bin/env python
'''
Usage:
    compile_tensorflow.py [options] MODEL HYPERS [OUTDIR]

Options:
    -h --help                Show this screen.
    --runtime NAME           Choose TerpreT runtime [default: logspace]
    --verbose                Dump intermediate compilation steps.
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

from docopt import docopt
import traceback
import pdb
import ast
from astunparse import unparse
# import numpy as np
import json

import unroller as unroll
import utils as u

class TFCompiler():
    class MyTransformer(ast.NodeTransformer):
        """
        Just copy the parent class, but split out visit_list so that we can call it
        and potentially override it.
        """

        def generic_visit(self, node):

            for field, old_value in ast.iter_fields(node):
                old_value = getattr(node, field, None)
                if isinstance(old_value, list):
                    old_value[:] = self.visit_list(old_value)
                elif isinstance(old_value, ast.AST):
                    new_node = self.visit(old_value)
                    if new_node is None:
                        delattr(node, field)
                    else:
                        setattr(node, field, new_node)
            return node

        def visit_list(self, nodes):
            new_values = []
            for value in nodes:
                if isinstance(value, ast.AST):
                    value = self.visit(value)
                    if value is None:
                        pass
                    elif not isinstance(value, ast.AST):
                        new_values.extend(value)
                    else:
                        new_values.append(value)
            return new_values


    class MyVisitor(ast.NodeVisitor):
        """
        A copy that splits off visit_list, so that we can override it.
        """
        def visit_list(self, nodes):
            for item in nodes:
                if isinstance(item, ast.AST):
                    self.visit(item)

        def generic_visit(self, node):
            """Called if no explicit visitor function exists for a node."""
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    self.visit_list(value)
                elif isinstance(value, ast.AST):
                    self.visit(value)


    class VarSizesVisitor(ast.NodeVisitor):
        def visit_Module(self, node):
            self.var_sizes = {}
            self.generic_visit(node)

        def visit_Assign(self, node):
            if u.is_tpt_decl(node):
                var_name = node.targets[0].id
                size = node.value.args[0].n
                assert var_name not in self.var_sizes, \
                    "Redeclaration of %s" % var_name
                self.var_sizes[var_name] = size

        def get_var_size(self, name):
            if name in self.var_sizes:
                return self.var_sizes[name]
            else:
                name_without_copy = u.strip_copy_from_name(name)
                if name_without_copy != name:
                    return self.get_var_size(name_without_copy)
                else:
                    assert False, "Can't determine size of %s" % name

    def __init__(self, args = {}):
        self.args = args
    
    def subs(self, root, **kwargs):
        '''Substitute ast.Name nodes for new ast.Name nodes in root using the
        mapping in kwargs. Modifies root.
        '''
        class Transformer(self.MyTransformer):
            #def visit_FunctionDef(self, node):
            #    return node

            def visit_Name(self_t, node):
                self_t.generic_visit(node)

                if node.id in kwargs:
                    new_name = kwargs[node.id]
                    node.id = new_name

                return node

        return Transformer().visit(root)

    
    def make_param_declaration(self, assign_node):
        lhs = assign_node.targets[0].id
        size = u.param_size(assign_node)

        decl = u.stmt_from_str("self._m_%s = tf.Variable(init_params(%s), name='%s')" % (lhs, size, lhs))
        softmax = u.stmt_from_str("%s = tpt.softmax(self._m_%s, scope='%s_softmax')" % (lhs, lhs, lhs))
        return [decl, softmax], ("self._m_%s" % lhs, lhs)

    
    def vars_defined_in_all_cases(self, node):
        if isinstance(node, ast.If):
            vars_defined = None
            for if_node in u.ifs_in_elif_block(node):
                cur_vars_defined = self.vars_defined_in_all_cases(if_node.body)
                if vars_defined is None:
                    vars_defined = cur_vars_defined
                else:
                    vars_defined = vars_defined & cur_vars_defined

        elif isinstance(node, list):
            vars_defined = set()
            for stmt_node in node:
                vars_defined = vars_defined | self.vars_defined_in_all_cases(stmt_node)

        elif isinstance(node, ast.AST):
            vars_defined = set()
            if u.is_set_to(node):
                vars_defined.add(node.value.func.value.id)

        return vars_defined

    
    def rename_if_else_block(self, if_node):
        vars_to_rename = self.vars_defined_in_all_cases(if_node)
        all_cases = []

        for node in u.ifs_in_elif_block(if_node):
            condition_lhs = u.get_condition_lhs(node)
            n = u.get_condition_rhs_num(node)
            all_cases.append(n)
            rename_dict = {}
            for var in vars_to_rename:
                rename_dict[var] = "%s_case%s" % (var, n)
            node.body = self.subs(node.body, **rename_dict)

        all_cases = sorted(all_cases)

        result_nodes = [if_node]
        for var in vars_to_rename:
            renamed_vars = ["%s_case%s" % (var, n) for n in all_cases]
            stmt = u.stmt_from_str("%s.set_to(tpt.weighted_sum([%s], %s, scope='%s_weighted_sum'))" %
                                (var, ",".join(renamed_vars), condition_lhs, var))
            result_nodes.append(stmt)

        return result_nodes

    
    def rename_cases_xform(self, root):
        class Transformer(self.MyTransformer):

            def visit_FunctionDef(self_t, node):
                """ Don't recurse into user defined functions """
                return node

            def visit_If(self_t, if_node):
                result_nodes = self.rename_if_else_block(if_node)

                # recursively visit all children in bodies, but don't visit the orelses
                for node in u.ifs_in_elif_block(if_node):
                    node.body[:] = self_t.visit_list(node.body)

                return result_nodes

        return Transformer().visit(root)

    
    def remove_ifs_xform(self, root):
        class Transformer(self.MyTransformer):
            def visit_FunctionDef(self_t, node):
                """ Don't recurse into user defined functions """
                return node

            def visit_If(self_t, if_node):
                for node in u.ifs_in_elif_block(if_node):
                    node.body[:] = self_t.visit_list(node.body)

                new_body = []
                for node in u.ifs_in_elif_block(if_node):
                    new_body.extend(node.body)

                return new_body

        return Transformer().visit(root)

    
    def declare_params_xform(self, root):
        class Transformer(self.MyTransformer):
            def visit_FunctionDef(self_t, node):
                """ Don't recurse into user defined functions """
                return node

            def visit_Module(self_t, node):
                self_t.params = {}
                self_t.generic_visit(node)
                return node

            def visit_Assign(self_t, node):
                if u.is_param_definition(node):
                    new_nodes, names = self.make_param_declaration(node)
                    self_t.params[names[1]] = names[0]
                else:
                    new_nodes = [node]
                return new_nodes

        t = Transformer()
        root = t.visit(root)
        root.body.append(u.stmt_from_str("self.params = %s" % u.dict_to_ast(t.params)))
        return root


    def create_vars_dict_xform(self, root):

        class Transformer(self.MyTransformer):
            def visit_Module(self_t, node):
                self_t.declared_vars = set()
                self_t.set_vars = set()
                self_t.param_vars = set()

                self_t.generic_visit(node)

                var_node_names = {name: name
                                  for name in self_t.declared_vars
                                  if name in self_t.set_vars}
                param_node_names = {name: name for name in self_t.param_vars}
                node.body.append(u.stmt_from_str("self.var_nodes = %s" %
                                                 u.dict_to_ast(var_node_names)))
                node.body.append(u.stmt_from_str("self.param_var_nodes = %s" %
                                                 u.dict_to_ast(param_node_names)))
                return node

            def visit_FunctionDef(self_t, node):
                """ Don't recurse into user defined functions """
                return node

            def visit_Assign(self_t, node):
                if u.is_var_definition(node):
                    name = node.targets[0].id
                    self_t.declared_vars.add(name)
                elif u.is_param_softmax_assign(node):
                    name = node.targets[0].id
                    self_t.param_vars.add(name)
                return node

            def visit_Expr(self_t, node):
                if u.is_set_to(node):
                    name = node.value.func.value.id
                    self_t.set_vars.add(name)
                return node

        return Transformer().visit(root)

    
    def remove_vars_xform(self, root):
        class Transformer(self.MyTransformer):

            def visit_FunctionDef(self_t, node):
                """ Don't recurse into user defined functions """
                return node

            def visit_Assign(self_t, node):
                if u.is_var_definition(node):
                    new_nodes = []
                else:
                    new_nodes = [node]

                return new_nodes

        return Transformer().visit(root)

    
    def remove_consts_xform(self, root):
        class Transformer(self.MyTransformer):

            def visit_Assign(self_t, node):
                if u.is_constant_definition(node):
                    new_nodes = []
                elif isinstance(node.value, ast.Name):
                    #Shortcut thing which the inlining in the unroller will have removed
                    new_nodes = []
                else:
                    new_nodes = [node]
                return new_nodes

        return Transformer().visit(root)

    
    def clean_function_defs_xform(self, root):
        class Transformer(self.MyTransformer):
            def visit_FunctionDef(self_t, node):
                node.decorator_list = []  # remove decorators
                return node

        return Transformer().visit(root)

    
    def make_function_tensors_xform(self, root):
        class Transformer(self.MyTransformer):
            def visit_FunctionDef(self_t, node):
                if len(node.decorator_list) == 1 and node.decorator_list[0].func.id == "Runtime":
                    decorator = node.decorator_list[0]
                    assert len(decorator.args) == 2, \
                        "Decorator must have two args:\n%s" % unparse(decorator)
                    input_sizes, output_size = decorator.args[0].elts, decorator.args[1]
                    new_node = u.stmt_from_str("%s_tensor = tpt.make_tensor(%s, [%s], %s)" %
                                               (node.name, node.name,
                                                ",".join([str(size.n) for size in input_sizes]),
                                                output_size.n))
                    return [node, new_node]
                elif len(node.decorator_list) == 1 and node.decorator_list[0].func.id == "Inline":
                    return None
                else:
                    return node

        return Transformer().visit(root)

    
    def apply_functions_xform(self, root):
        class Transformer(self.MyTransformer):
            def visit_FunctionDef(self_t, node):
                """ Don't recurse into user defined functions """
                return node

            def visit_Module(self_t, node):

                class Visitor(ast.NodeVisitor):
                    def visit_Module(self_v, node):
                        self_v.defined_function_names = []
                        self_v.generic_visit(node)

                    def visit_FunctionDef(self_v, node):
                        self_v.defined_function_names.append(node.name)

                v = Visitor()
                v.visit(node)
                self_t.defined_function_names = v.defined_function_names
                self_t.generic_visit(node)
                return node

            def visit_Expr(self_t, node):
                if u.is_set_to_user_defined_function(node):
                    var_name = node.value.func.value.id
                    call_node = node.value.args[0]
                    function_name = call_node.func.id
                    assert function_name in self_t.defined_function_names, \
                        "Attempting to use undefined function: %s" % function_name

                    call_node.args.insert(0, ast.Name(id="%s_tensor" % function_name))

                    call_node.func = ast.Attribute(value=ast.Name(id="tpt"),
                                                attr="apply_factor")
                    scope = "%s_apply_factor" % var_name
                    call_node.keywords = [ast.keyword(arg="scope", value=ast.Str(s=scope))]

                return node

        return Transformer().visit(root)

    
    def normalize_input_output_decls_xform(self, root):
        class Transformer(self.MyTransformer):
            def visit_FunctionDef(self_t, node):
                """ Don't recurse into user defined functions """
                return node

            def visit_Assign(self_t, node):
                if u.is_input_definition(node):
                    name = node.targets[0].id
                    node.value.func.id = "Var"  # rename to be a var
                    new_node = u.stmt_from_str("%s.set_as_input()" % name)
                    return [node, new_node]

                elif u.is_output_definition(node):
                    name = node.targets[0].id
                    node.value.func.id = "Var"  # rename to be a var
                    new_node = u.stmt_from_str("%s.set_as_output()" % name)
                    return [node, new_node]

                else:
                    return node

        return Transformer().visit(root)

    
    def replace_set_tos_xform(self, root):
        class Transformer(self.MyTransformer):
            def visit_FunctionDef(self_t, node):
                """ Don't recurse into user defined functions """
                return node

            def visit_Expr(self_t, node):
                if u.is_set_to(node):
                    lhs = node.value.func.value
                    rhs = node.value.args[0]
                    node = ast.Assign(targets=[lhs], value=rhs)
                return node

        return Transformer().visit(root)

    
    def declare_inputs_xform(self, root):
        class Transformer(self.MyTransformer):
            def visit_FunctionDef(self_t, node):
                """ Don't recurse into user defined functions """
                return node

            def visit_Module(self_t, node):
                v = self.VarSizesVisitor()
                v.visit(node)
                self_t.var_sizes = v.var_sizes
                self_t.inputs = {}
                self_t.generic_visit(node)
                return node

            def visit_Expr(self_t, node):
                if u.is_set_as_input(node):
                    var_name = node.value.func.value.id
                    var_size = self_t.var_sizes[var_name]
                    node1 = u.stmt_from_str("%s_idxs.set_to(tf.placeholder(tf.int32, shape=[None], name='%s_idxs'))" %
                                            (var_name, var_name))
                    node2 = u.stmt_from_str("%s.set_to(tpt.one_hot(%s_idxs, %s, scope='%s_one_hot'))" % (var_name, var_name, var_size, var_name))
                    self_t.inputs[var_name] = "%s_idxs" % var_name
                    return [node1, node2]
                else:
                    return node

        t = Transformer()
        root = t.visit(root)
        root.body.append(u.stmt_from_str("_inputs = %s" % u.dict_to_ast(t.inputs)))
        return root

    
    def declare_outputs_xform(self, root):
        class Transformer(self.MyTransformer):
            def visit_Module(self_t, node):
                v = self.VarSizesVisitor()
                v.visit(node)
                self_t.var_sizes = v.var_sizes
                self_t.outputs = {}
                self_t.generic_visit(node)
                return node

            def visit_FunctionDef(self_t, node):
                """ Don't recurse into user defined functions """
                return node

            def visit_Expr(self_t, node):
                if u.is_set_as_output(node):
                    var_name = node.value.func.value.id
                    self_t.outputs[var_name] = var_name
                    return []
                else:
                    return node

        t = Transformer()
        root = t.visit(root)
        root.body.append(u.stmt_from_str("_outputs = %s" % u.dict_to_ast(t.outputs)))

        return root

    
    def create_1hots_for_set_to_const_xform(self, root):
        class Transformer(self.MyTransformer):
            def visit_FunctionDef(self_t, node):
                """ Don't recurse into user defined functions """
                return node

            def visit_Module(self_t, node):
                v = self.VarSizesVisitor()
                v.visit(node)
                self_t.size_visitor = v
                self_t.generic_visit(node)
                return node

            def visit_Expr(self_t, node):
                if u.is_set_to(node) and isinstance(node.value.args[0], ast.Num):
                    rhs = node.value.args[0].n
                    var_name = node.value.func.value.id
                    var_size = self_t.size_visitor.get_var_size(var_name)
                    expr = u.stmt_from_str("tpt.one_hot(%s, %s)" % (rhs, var_size))
                    node.value.args[0] = expr.value

                return node

        return Transformer().visit(root)

    
    def split_params_from_model(self, root):
        class CollectParamsVisitor(ast.NodeVisitor):
            def visit_Module(self_c, node):
                self_c.param_statements = []
                self_c.param_dict = None
                self_c.generic_visit(node)

            def visit_Assign(self_c, node):
                if u.is_param_declaration(node):
                    self_c.param_statements.append(node)
                elif u.is_self_params_assignment(node):
                    self_c.param_dict = node
        cpv = CollectParamsVisitor()
        cpv.visit(root)

        class RemoveParamsTransformer(self.MyTransformer):
            def visit_FunctionDef(self_r, node):
                """ Don't recurse into user defined functions """
                return node

            def visit_Assign(self_r, node):
                if u.is_param_declaration(node):
                    return []
                elif u.is_self_params_assignment(node):
                    return []
                else:
                    return node
        rpt = RemoveParamsTransformer()
        root = rpt.visit(root)

        return root, cpv.param_statements, cpv.param_dict

    
    def class_formatting_xform(self, root, hypers_name):
        root, param_statements, param_dict = self.split_params_from_model(root)

        return_stmt = u.stmt_from_str("return _inputs, _outputs")
        root.body.append(return_stmt)
        model_method = ast.FunctionDef(name="build_model_%s" % hypers_name,
                                    args=[ast.Name(id="self")],
                                    decorator_list=[],
                                    body=root.body)

        args_list = u.make_args_list([ast.Name(id="self"), ast.Name(id="init_params")])

        param_method = ast.FunctionDef(name="declare_params_%s" % hypers_name,
                                    args=args_list,
                                    decorator_list=[],
                                    body=param_statements)
        param_method.body.append(param_dict)
        param_method.body

        class_node = ast.ClassDef(name="Model", bases=[],
                                body=[param_method,
                                        model_method],
                                decorator_list=[])

        import_block = []

        runtime_assign = u.stmt_from_str("runtime = '%s'" % (self.args.get('--runtime') or 'logspace'))
        condition = u.stmt_from_str("runtime == 'logspace'").value
        log_import_stmt = u.stmt_from_str("import terpret_tf_log_runtime as tpt")
        standard_import_stmt = u.stmt_from_str("import terpret_tf_runtime as tpt")

        if_node = ast.If(test=condition, body=[log_import_stmt], orelse=[standard_import_stmt])

        module_body = [
            runtime_assign,
            if_node,
            class_node
            ]
        module_node = ast.Module(body=module_body)
        return module_node

    def remove_imports_xform(self, root):
        class Transformer(self.MyTransformer):
            def visit_Import(self_t, node):
                return []

            def visit_ImportFrom(self_t, node):
                return []

        return Transformer().visit(root)

    
    def assert_params_are_same(self, module_node1, hypers_name1, module_node2, hypers_name2):
        """
        Before merging different hyperparameter settings into a single class, we
        want to ensure that the params declaration is identical. Test this with
        string equality of the dumped AST.
        """
        params_method1 = u.get_method_by_name(module_node1,
                                            "declare_params_%s" % hypers_name1)
        params_method2 = u.get_method_by_name(module_node2,
                                            "declare_params_%s" % hypers_name2)
        assert unparse(params_method1.body) == unparse(params_method2.body)

    
    def merge_hypers(self, module_nodes):
        """
        Merge together the compilations for different hyperparameter settings
        into a single class.

        Plan: take the first module_node, and then merge in each of the other
        module_nodes.
        """
        hypers_names = module_nodes.keys()
        module_node = module_nodes[hypers_names[0]]
        class_node = u.get_class_node(module_node)

        for hypers_name_i in hypers_names[1:]:
            module_node_i = module_nodes[hypers_name_i]
            self.assert_params_are_same(module_node, hypers_names[0],
                                        module_node_i, hypers_name_i)

            # now we know it's ok to just keep one copy of the params declarations
            model_method_i = u.get_method_by_name(module_node_i,
                                                "build_model_%s" % hypers_name_i)
            class_node.body.append(model_method_i)

        # finally, rename param declaration method to be hypers-independent
        param_decl_method = u.get_method_by_name(module_node,
                                                "declare_params_%s" % hypers_names[0])
        param_decl_method.name = "declare_params"
        # give it arguments
        # param_decl_method.args = [ast.Name(s="init_params")]

        module_node.body.insert(0, u.stmt_from_str("import tensorflow as tf"))

        return module_node

    
    def add_get_hypers_method(self, module_node, hypers_list):
        get_hypers = ast.FunctionDef(name="get_hypers",
                                    args=u.make_args_list([ast.Name(id="self")]),
                                    decorator_list=[],
                                    body=u.stmt_from_str("return %s" %
                                                        json.dumps(hypers_list)))

        u.get_class_node(module_node).body.append(get_hypers)

        return module_node

    
    def load_source_and_hypers(self, source_filename, hypers_filename):
        print ("Reading interpreter model from '%s'." % source_filename)
        with open(source_filename, 'r') as f:
            source = f.read()
        filename = os.path.splitext(os.path.basename(source_filename))[0]

        if hypers_filename is None:
            hypers_list = {"default": ""}
        else:
            print ("Reading model parameters from '%s'." % (hypers_filename))
            with open(hypers_filename, 'r') as f:
                hypers_list = json.load(f)
            if not isinstance(hypers_list, dict):
                raise RuntimeError('Not a hyperparameter file: %s' % hypers_filename)
        return source, hypers_list, filename

    
    def compile_to_tensorflow(self, source_filename, hypers_filename, out_directory, VERBOSE=False):
        source, hypers_list, filename = self.load_source_and_hypers(source_filename, hypers_filename)
        result_filename = hypers_filename.split("/")[-1].replace(".json", "")

        module_nodes = {}
        step = 0
        for hypers_name, hypers in hypers_list.iteritems():
            module_node = ast.parse(source)
            module_node = u.replace_hypers(module_node, hypers)

            transforms = self.get_transforms()

            for transform in transforms:
                module_node = transform(module_node)
                if VERBOSE:
                    print 80 * "*"
                    print unparse(module_node)
                    intermediate_result_filename = \
                        os.path.join(out_directory, "%s_compiled_step%s.py" %
                                    (filename, step))
                    with open(intermediate_result_filename, 'w') as f:
                        f.write(unparse(module_node))
                step += 1

            module_node = self.class_formatting_xform(module_node, hypers_name)

            module_nodes[hypers_name] = module_node

        module_node = self.merge_hypers(module_nodes)
        module_node = self.add_get_hypers_method(module_node, hypers_list)

        if VERBOSE:
            print 80 * "*"
            print unparse(u.get_class_node(module_node))

        filename = source_filename.split("/")[-1].replace(".py", "")

        if not os.path.isdir(out_directory):
            os.makedirs(out_directory)
        result_filename = os.path.join(out_directory, "%s_compiled.py" % result_filename)
        print "Outputting to %s" % result_filename
        with open(result_filename, 'w') as f:
            f.write(unparse(module_node))
        return module_node

    
    def get_transforms(self):
        return [
            self.remove_imports_xform,
            unroll.unroll_and_flatten,
            self.remove_consts_xform,
            self.rename_cases_xform,
            self.remove_ifs_xform,
            self.make_function_tensors_xform,
            self.apply_functions_xform,
            self.clean_function_defs_xform,
            self.normalize_input_output_decls_xform,
            self.create_1hots_for_set_to_const_xform,
            self.declare_params_xform,
            self.declare_outputs_xform,
            self.declare_inputs_xform,
            self.create_vars_dict_xform,
            self.remove_vars_xform,
            self.replace_set_tos_xform
        ]

if __name__ == "__main__":
    args = docopt(__doc__)
    
    source_filename = args['MODEL']
    hypers_filename = args.get('HYPERS')
    verbose = args['--verbose']
    out_dir = args.get('OUTDIR') or "./compiled/tensorflow_models/"
    out_dir = os.path.join(out_dir, "")

    try:
        tfc = TFCompiler(args)
        tfc.compile_to_tensorflow(source_filename, hypers_filename, out_dir, VERBOSE=verbose)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
