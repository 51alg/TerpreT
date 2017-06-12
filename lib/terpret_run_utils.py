import utils as u
import unroller as unroll
import ast
import h5py
import numpy as np
import uuid
from terpret_run_runtime import TerpreTRuntime
from data import Data
import pdb


def instantiate_model(model_filename, hypers, param_values):
    class Transformer(ast.NodeTransformer):
        def __init__(self):
            self.__declared = set()
            self.__parameters = set()
            self.__runtime_arg = "tptRuntime_%s" % str(uuid.uuid4()).replace('-', '')[0:8]

        def visit_Assign(self, node):
            assigned_var = node.targets[0].id
            if u.is_param_definition(node):
                self.__parameters.add(assigned_var)
                node.value = ast.Num(n=param_values[assigned_var])
            elif u.is_var_definition(node):
                self.__declared.add(assigned_var)
                node.value.args.append(ast.Str(assigned_var))
            elif u.is_input_definition(node):
                self.__declared.add(assigned_var)
                # Add name to call so that the runtime can choose value
                runtime = ast.Name(self.__runtime_arg, ast.Load())
                attr = ast.Attribute(runtime, "get_input", ast.Load())
                call = ast.Call(attr,
                                [ast.Str(assigned_var)] + node.value.args,
                                [],
                                None,
                                None)
                node.value = call
            elif u.is_output_definition(node):
                self.__declared.add(assigned_var)
                # Add name to call so that the runtime can check value
                runtime = ast.Name(self.__runtime_arg, ast.Load())
                attr = ast.Attribute(runtime, "get_output", ast.Load())
                call = ast.Call(attr,
                                [ast.Str(assigned_var)] + node.value.args,
                                [],
                                None,
                                None)
                node.value = call
            return node

        def visit_FunctionDef(self, node):
            # Don't recurse into FunctionDef. Also, filter @Inline
            if len(node.decorator_list) == 1 and \
              isinstance(node.decorator_list[0].func, ast.Name) and \
              node.decorator_list[0].func.id == "Inline":
                return []
            node.decorator_list = []
            return node

        def visit_Call(self, node):
            # Do not rewrite lhs of a set_to to .get(), i.e., don't recurse:
            if u.is_set_to_call(node):
                node.args = [self.visit(arg) for arg in node.args]
            else:
                self.generic_visit(node)
            return node

        def visit_Name(self, node):
            if node.id in self.__parameters:
                return ast.Num(n=param_values[node.id])
            if node.id in self.__declared:
                # Rewrite "foo" to "foo.get()"
                attr = ast.Attribute(node, "get", ast.Load())
                call = ast.Call(attr, [], [], None, None)
                return call
            else:
                return node

        def visit_ImportFrom(self, node):
            if node.module is "dummy":
                return []
            return node

        def visit_Module(self, node):
            self.generic_visit(node)
            var_alias = ast.alias("Var", None)
            runtime_alias = ast.alias("Runtime", None)
            importStmt = ast.ImportFrom("terpret_run_runtime",
                                        [var_alias, runtime_alias],
                                        0)
            fun_name = "__generated_%s" % str(uuid.uuid4()).replace('-', '')
            arguments = ast.arguments([ast.Name(self.__runtime_arg, ast.Param())],
                                      None,
                                      None,
                                      [])
            fun_def = ast.FunctionDef(name=fun_name,
                                      args=arguments,
                                      body=[importStmt] + node.body,
                                      decorator_list=[])
            return (fun_name, ast.Module([fun_def]))

    # Load model, turn into something we can execute:
    model = open(model_filename, 'r').read()
    model_ast = ast.parse(model)
    model_ast = u.replace_hypers(model_ast, hypers)
    unrolled_model = unroll.unroll_and_flatten(model_ast,
                                               do_checks=False,
                                               print_info=False)
    (fun_name, runnable_model) = Transformer().visit(unrolled_model)
    ast.fix_missing_locations(runnable_model)
    return (fun_name, runnable_model)


def load_result(result_filename):
    raw_data = h5py.File(result_filename, 'r')
    hypers = {k: raw_data['model_hypers'].attrs[k]
              for k in raw_data['model_hypers'].attrs.keys()}
    params = {p_name: np.argmax(p_value)
              for (p_name, p_value) in raw_data['parameters'].iteritems()
              if len(p_value) > 0}
    raw_data.close()
    pdb.set_trace()
    return (hypers, params)


def test(result_filename, model_filename, data_filename, test_batches=None):
    (hypers, params) = load_result(result_filename)

    # This is the nasty bit:
    # It generates a function containing the model with hardcoded params,
    # Var() represented as a simple wrapper object that holds data,
    # and a single argument that is used to get input/set output.
    # We use this by compiling the function and evaling it in our context,
    # and then calling into it once per input/output pair.
    (fun_name, runnable_model) = instantiate_model(model_filename,
                                                   hypers,
                                                   params)
    eval(compile(runnable_model, '<generated>', 'exec'))

    # Get the data:
    data = Data(data_filename)
    if test_batches is None:
        test_batches = data.get_batch_names()
    elif isinstance(test_batches, str):
        test_batches = test_batches.split(',')

    correct_instances = 0
    total_instances = 0
    for batch_name in test_batches:
        _, batch_data = data.get_batch(batch_name)
        ex_idx = 0
        for data_instance in batch_data:
            print("Testing on batch %s (example %i)" % (batch_name, ex_idx))
            ex_idx += 1
            total_instances += 1
            runtime_data = TerpreTRuntime(data_instance)
            eval("%s(runtime_data)" % fun_name)
            test_correct = runtime_data.check()
            if test_correct:
                correct_instances += 1

    acc = correct_instances / float(total_instances) * 100.0
    print("Test accuracy: %i/%i (%6.2f%%) correct." % (correct_instances,
                                                       total_instances,
                                                       acc))
