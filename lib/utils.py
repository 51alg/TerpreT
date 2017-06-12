import ast
import json
import pdb
import os
from astunparse import unparse


def cast(node, node_type):
    assert isinstance(node, node_type), "cast to %s failed %s" % (node_type, ast.dump(node))
    return node


def get_index(subscript_node):
    index_node = cast(subscript_node.slice, ast.Index)
    return name_or_number(index_node.value)


def is_int_constant(name_node):
    return isinstance(name_node, ast.Name)


def is_constant_definition(assign_node):
    return is_int_constant(assign_node.targets[0]) and isinstance(assign_node.value, ast.Num)


def is_hyper_definition(assign_node):
    return isinstance(assign_node.value, ast.Call) and \
        isinstance(assign_node.value.func, ast.Name) and \
        assign_node.value.func.id == "Hyper"


def replace_hypers(root, hypers):
    '''
    Replace all hyperparameters declared using Hyper() by given values.
    '''
    class Transformer(ast.NodeTransformer):
        def visit_Assign(self_t, node):
            if is_hyper_definition(node):
                assert isinstance(node.targets[0], ast.Name), \
                    "Expecting identifier on LHS of hyper definition"
                hyper_name = node.targets[0].id
                value = hypers[hyper_name]
                node.value = ast.copy_location(ast.Num(value), node)
            return node
    return Transformer().visit(root)

def is_param_definition(assign_node):
    if not isinstance(assign_node.value, ast.Call): return False
    if not isinstance(assign_node.value.func, ast.Name): return False
    return assign_node.value.func.id == "Param"


def is_self_params_assignment(node):
    if not isinstance(node.targets[0], ast.Attribute): return False
    return node.targets[0].attr == "params"

def is_input_definition(assign_node):
    if not isinstance(assign_node.value, ast.Call): return False
    if not isinstance(assign_node.value.func, ast.Name): return False
    return assign_node.value.func.id == "Input"


def is_output_definition(assign_node):
    if not isinstance(assign_node.value, ast.Call): return False
    if not isinstance(assign_node.value.func, ast.Name): return False
    return assign_node.value.func.id == "Output"


def is_set_to_user_defined_function(node):
    if not is_set_to(node): return False
    call_node = node.value.args[0]
    if not isinstance(call_node, ast.Call): return False
    if isinstance(call_node.func, ast.Attribute):  # might be tpt function_name
        if isinstance(call_node.func.value, ast.Name):
            if call_node.func.value.id == "tpt": return False

    return True


def make_args_list(args):
    return ast.arguments(args=args, vararg=None, kwarg=None, defaults=[])


def get_method_by_name(module_node, name):
    class Visitor(ast.NodeVisitor):
        def visit_Module(self, node):
            self.result = None
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            if node.name == name:
                assert self.result is None, "More than one result in module"
                self.result = node
    v = Visitor()
    v.visit(module_node)
    return v.result


def get_class_node(module_node):
    class Visitor(ast.NodeVisitor):
        def visit_Module(self, node):
            self.result = None
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            assert self.result is None, "More than one class in module"
            self.result = node
    v = Visitor()
    v.visit(module_node)
    return v.result


def is_param_declaration(node):
    if not isinstance(node, ast.Assign): return False
    if not isinstance(node.value, ast.Call): return False
    if not isinstance(node.value.func, ast.Attribute): return False
    if node.value.func.value.id != "tf": return False
    if node.value.func.attr != "Variable": return False
    return True


def is_param_softmax_assign(node):
    if not isinstance(node, ast.Assign): return False
    if not isinstance(node.value, ast.Call): return False
    if not isinstance(node.value.func, ast.Attribute): return False
    if node.value.func.value.id != "tpt": return False
    if node.value.func.attr != "softmax": return False
    return True


def is_var_definition(assign_node):
    if not isinstance(assign_node.value, ast.Call): return False
    if not isinstance(assign_node.value.func, ast.Name): return False
    return assign_node.value.func.id == "Var"


def is_tpt_decl(assign_node):
    if not isinstance(assign_node.value, ast.Call): return False
    if not isinstance(assign_node.value.func, ast.Name): return False
    return assign_node.value.func.id in ["Var", "Input", "Output", "Param"]


def param_size(assign_node):
    assert is_param_definition(assign_node)
    assert isinstance(assign_node.value.args[0], ast.Num)
    return assign_node.value.args[0].n


def var_names_used_in_call(call_node):
    results = []
    for arg in call_node.args:
        name_node = cast(arg, ast.Name)
        results.append(name_node.id)
    return results


def var_names_used_in_factor(name_or_call_node):
    if isinstance(name_or_call_node, ast.Name):
        return [name_or_call_node.id]
    elif isinstance(name_or_call_node, ast.Call):
        return var_names_used_in_call(name_or_call_node)
    else:
        assert False, "not name or call node " + ast.dump(name_or_call_node)


def var_names_used_in_set_to(set_to_node):
    assert isinstance(set_to_node, ast.Expr), "set_to node should be Expr"
    call_node = cast(set_to_node.value, ast.Call)
    attribute_node = cast(call_node.func, ast.Attribute)
    name_node = cast(attribute_node.value, ast.Name)
    lhs_var = name_node.id

    assert attribute_node.attr == "set_to", "expected set_to " + ast.dump(attribute_node)

    rhs_vars = var_names_used_in_factor(call_node.args[0])

    return [lhs_var] + rhs_vars


def name_or_number(name_or_num_node):
    if isinstance(name_or_num_node, ast.Name):
        return name_or_num_node.id
    elif isinstance(name_or_num_node, ast.Num):
        return name_or_num_node.n
    else:
        assert False, "not a name or number " + ast.dump(name_or_num_node)


def parse_factor_expression(call_or_name_node):
    if isinstance(call_or_name_node, ast.Name):  # a.set_to(b) is shorthand for a.set_to(Copy(b))
        name_node = call_or_name_node
        return None, [name_node.id]

    elif isinstance(call_or_name_node, ast.Call):  # a.set_to(f(b))
        call_node = call_or_name_node
        return call_node.func.id,  [name_or_number(node) for node in call_node.args]

    elif isinstance(call_or_name_node, ast.Num):  # a.observe_value(0)
        num_node = call_or_name_node
        return None, [int(num_node.n)]

    elif isinstance(call_or_name_node, ast.Subscript):
        print ast.dump(call_or_name_node)
        pdb.set_trace()

    else:
        assert False, "Can't parse factor " + ast.dump(call_or_name_node)


def parse_assign(assign_node):
    var = assign_node.targets[0].id
    factor = parse_factor_expression(assign_node.value)
    return var, factor


def parse_declaration(assign_node):
    if len(assign_node.targets) > 1:  return False

    if is_macro_definition(assign_node):
        return None

    var, factor = parse_assign(assign_node)

    return var, factor


def if_and_or_else_blocks(if_node):
    results = []
    results.append(if_node)
    if len(if_node.orelse) > 0:
        results.extend(if_and_or_else_blocks(if_node.orelse[0]))  # recurse on elif branches

    return results


def parse_compare(compare_node):
    assert len(compare_node.ops) == 1, "multiple comparison ops?" + ast.dump(compare_node)
    assert isinstance(compare_node.ops[0], ast.Eq), "comparison should be ==" + \
        ast.dump(compare_node.ops[0])

    lhs = compare_node.left
    rhs = compare_node.comparators[0]
    if isinstance(lhs, ast.Name) and isinstance(rhs, ast.Num):
        var_name = lhs.id
        val = rhs.n
    elif isinstance(rhs, ast.Name) and isinstance(lhs, ast.Num):
        var_name = rhs.id
        val = lhs.n
    elif isinstance(rhs, ast.Name) and isinstance(lhs, ast.Name):
        # try to apply macro
        if is_int_constant(rhs):
            var_name = lhs.id
            val = rhs.id
        elif is_int_constant(lhs):
            var_name = rhs.id
            val = lhs.id
        else:
            assert False, "Unable to apply macro to fix comparator " + ast.dump(compare_node)
    else:
        assert False, "unexpected comparator" + ast.dump(compare_node)
    return var_name, val


def string_expr_to_ast(str):
    return ast.parse(str).body[0].value


def string_expr_to_ast2(str):
    return ast.parse(str).body[0]


def var_used_as_index(tree, var_id):
    index_nodes = descendants_of_type(tree, ast.Index)
    for index in index_nodes:
        name_nodes = descendants_of_type(index, ast.Name)
        for name in name_nodes:
            if name.id == var_id: return True
    return False


def get_single_lhs(node):
    if not isinstance(node, ast.Assign): return None
    lhs = node.targets
    if len(lhs) != 1: return None
    return lhs[0]
    

def descendants_of_type(root, nodetype):
    result = []
    if isinstance(root, nodetype):
        result.append(root)

    for ch in ast.iter_child_nodes(root):
        ch_result = descendants_of_type(ch, nodetype)
        result.extend(ch_result)

    return result


def get_kwarg(call_node, kwarg):
    assert isinstance(call_node, ast.Call)

    for k in call_node.keywords:
        if k.arg == kwarg:
            return k.value
    return None


def function_name(function_node):
    if not isinstance(function_node, ast.FunctionDef):  return None
    return function_node.name
    

def function_nodes(root):
    return descendants_of_type(root, ast.FunctionDef)


def return_nodes(root):
    return descendants_of_type(root, ast.Return)


def rhs_function(node):
    if not isinstance(node, ast.Assign): return None
    rhs = node.value
    if isinstance(rhs, ast.Call):
        return rhs
    elif isinstance(rhs, ast.BinOp):
        return rhs
    elif isinstance(rhs, ast.UnaryOp):
        return rhs
    elif isinstance(rhs, ast.Subscript):
        return rhs


def is_einsum_function(node):
    if isinstance(node, ast.Attribute):
        return node.attr == "einsum"
    elif isinstance(node, ast.Name):
        return node.id == "einsum"
    else:
        return False


def is_numpy_log(node):
    if isinstance(node, ast.Attribute):
        return node.value.id == "np" and node.attr == "log"
    else:
        return False
    

def is_numpy_function(node, function_name):
    if isinstance(node, ast.Attribute):
        return node.value.id == "np" and node.attr == function_name
    else:
        return False


def is_numpy_constructor(node):
    if isinstance(node, ast.Attribute):
        return node.value.id == "np" and node.attr in ["rand", "randn", "zeros", "ones"]
    else:
        return False


def is_concatenate_function(node):
    if isinstance(node, ast.Attribute):
        return node.attr == "concatenate"
    elif isinstance(node, ast.Name):
        return node.id == "concatenate"
    else:
        return False


def is_add_function(node):
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)


def is_registered(context, f):
    if context is None: return False


def is_forward_function_call(call):
    if not isinstance(call.func, ast.Name): return False
    name = call.func.id
    return name.endswith("_f")


def var_to_dvar(var_node):
    if isinstance(var_node, ast.Name):
        name = var_node
        new_name = ast.Name()
        new_id =  "d%s" % name.id
        new_name.id = new_id
        return new_name

    elif isinstance(var_node, ast.Subscript):
        subscript = var_node
        name = subscript.value
        new_name = ast.Name(id="d%s" % name.id)
        new_subscript = ast.Subscript(value=new_name, slice=subscript.slice)
        return new_subscript
    
    else:
        print "Error: don't know how to dvar a %s" % ast.dump(var_node)
        print r.pretty(var_node)
        assert False


def f_to_df(call):
    if isinstance(call.func, ast.Name):
        new_name = ast.Name(id=call.func.id.replace("_f", "_b"))
    else:
        assert False
    return new_name

def get_concatenate_axis(call):
    keywords = call.keywords
    for k in keywords:
        if k.arg == "axis":
            assert isinstance(k.value, ast.Num) # require axes be given as literals
            return k.value.n
    return 0


def make_unconcat_slice(axis, lower, upper):
    dims = []
    for i in range(axis):
        dims.append(ast.Slice(lower=None, upper=None, step=None))
    dims.append(ast.Slice(lower=lower, upper=upper, step=None))
    dims.append(ast.Ellipsis())

    ext_slice = ast.ExtSlice(dims=dims)

    return ext_slice

def NewCall(func=None, args=None, keywords=None, starargs=None):
    if args is None:
        args = []
    if keywords is None:
        keywords = []
    if starargs is None:
        starargs = []
    return ast.Call(func=func, args=args, keywords=keywords, starargs=starargs)

def make_attribute(base_name, value_name):
    return ast.Attribute(value=ast.Name(id=base_name), attr=ast.Name(id=value_name))


def reverse_loop(for_node):
    assert isinstance(for_node, ast.For)

    iter_node = for_node.iter

    new_iter_node = NewCall(func=ast.Name(id="reversed"), args=[iter_node])

    return ast.For(target=for_node.target, iter=new_iter_node, body=for_node.body)


def var_id(var_node):
    if isinstance(var_node, ast.Name):
        return var_node.id
    elif isinstance(var_node, ast.Subscript):
        return var_node.value.id


def returns_to_args(returns):
    return_value = returns.value
    if isinstance(return_value, ast.Tuple):
        return [var_to_dvar(elt) for elt in return_value.elts]
    elif isinstance(return_value, ast.Name):
        return [var_to_dvar(return_value)]
    else:
        assert False


def ancestors_by_type(node):
    pass


def get_condition_rhs_num(if_node):
    assert isinstance(if_node, ast.If)
    assert isinstance(if_node.test, ast.Compare)
    assert isinstance(if_node.test.comparators[0], ast.Num)
    return if_node.test.comparators[0].n


def get_condition_lhs(if_node):
    assert isinstance(if_node, ast.If)
    assert isinstance(if_node.test, ast.Compare)
    return unparse(if_node.test.left).strip()


def is_set_to_call(node):
    if not isinstance(node, ast.Call): return False
    if not isinstance(node.func, ast.Attribute): return False
    return node.func.attr == "set_to"


def is_set_to(node):
    if not isinstance(node, ast.Expr): return False
    return is_set_to_call(node.value)


def is_set_as_input(node):
    if not isinstance(node, ast.Expr): return False
    if not isinstance(node.value, ast.Call): return False
    if not isinstance(node.value.func, ast.Attribute): return False
    return node.value.func.attr == "set_as_input"


def is_set_as_output(node):
    if not isinstance(node, ast.Expr): return False
    if not isinstance(node.value, ast.Call): return False
    if not isinstance(node.value.func, ast.Attribute): return False
    return node.value.func.attr == "set_as_output"


def ifs_in_elif_block(if_node):
    result = [if_node]
    while len(if_node.orelse) == 1:
        if_node = if_node.orelse[0]
        result.append(if_node)
    return result


def stmt_from_str(s):
    return ast.parse(s).body[0]


def split_name_case(name):
    assert False, "doesn't handle X > 9 case"
    suffix = name[-len("_caseX"):]
    name = name[:-len("_caseX")]
    return name, int(suffix[-1])

def dict_to_ast(d):
    kvs = ",".join(["'%s': %s" % (k, v) for k, v in sorted(d.iteritems(), key=lambda x: x[0])])
    return "{ %s }" % kvs

def object_to_ast(obj):
    return ast.parse(str(eval('obj'))).body[0]

def strip_copy_from_name(name):
    """
    Removes a trailing _copyX from a name.
    """
    ends_in_digit = name[-1].isdigit()
    if not ends_in_digit: return name
    while ends_in_digit:
        name = name[:-1]
        ends_in_digit = name[-1].isdigit()
    if name.endswith("_case"):
        return name[:-len("_case")]
    else:
        return name

class CFGBuilder(ast.NodeVisitor):
    def __init__(self):
        self.__next_id = 1
        self.__nodes = {} #Maps id -> node
        self.__out_edges = {} #Maps id -> id*, signifying possible flow of control

    def add_node(self, node):
        id = self.__next_id
        self.__next_id = id + 1
        self.__nodes[id] = node
        return id

    def add_edge(self, source, target):
        assert(source in self.__nodes)
        assert(target in self.__nodes)
        out_edges = self.__out_edges.get(source, None)
        if out_edges is None:
            out_edges = []
            self.__out_edges[source] = out_edges
        out_edges = out_edges.append(target)

    #Visit methods return a tuple (in, outs), where in is the node id of the unique
    #entry to the visited AST node, and outs is the list of all possible exits.
    def visit_For(self, node):
        raise Exception("TerpreT restriction checking only works on fully unrolled code")
    def visit_While(self, node):
        raise Exception("TerpreT restriction checking only works on fully unrolled code")
    def visit_With(self, node):
        raise Exception("TerpreT restriction checking only works on fully unrolled code")

    def visit_Block(self, nodes):
        if len(nodes) < 1:
            raise Exception("Cannot handle empty block")
        (entry_id, out_ids) = self.visit(nodes[0])
        for i in xrange(1, len(nodes)):
            (in_id, new_out_ids) = self.visit(nodes[i])
            for old_out_id in out_ids:
                self.add_edge(old_out_id, in_id)
            out_ids = new_out_ids
        return (entry_id, out_ids)

    def visit_FunctionDef(self, node):
        if len(filter(lambda dec: dec.func.id == "Runtime", node.decorator_list)) > 0:
            this_id = self.add_node(node)
            return (this_id, [this_id])
        else:
            raise Exception("TerpreT only allows use of @Runtime functions in execution models.")

    def visit_If(self, node):
        this_id = self.add_node(node)
        (then_in, then_outs) = self.visit_Block(node.body)
        self.add_edge(this_id, then_in)
        outs = then_outs
        if len(node.orelse) > 0:
            (else_in, else_outs) = self.visit_Block(node.orelse)
            self.add_edge(this_id, else_in)
            outs = outs + else_outs
        else:
            outs = outs + [this_id]
        return (this_id, outs)

    def visit_Call(self, node):
        this_id = self.add_node(node)
        return (this_id, [this_id])

    def visit_Expr(self, node):
        this_id = self.add_node(node)
        return (this_id, [this_id])

    def visit_Assign(self, node):
        this_id = self.add_node(node)
        return (this_id, [this_id])

    def visit_Module(self, node):
        return self.visit_Block(node.body)

    def generic_visit(self, node):
        raise Exception("Unhandled node in CFG constructor: %s (%s)" % (astunparse.unparse(node), str(node)))


def get_variables(root):
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.vars = set()

        def visit_Name(self, node):
            self.vars.add(node.id)

    var_visitor = Visitor()
    var_visitor.visit(root)
    return var_visitor.vars


def read_inputs(model_filename, hypers_filename, data_filename, train_batch='train'):
    # Get source, and start substituting in:
    print ("Reading interpreter model from '%s'." % model_filename)
    with open(model_filename, 'r') as f:
        model = f.read()
    parsed_model = ast.parse(model)

    # Find right data batch:
    print ("Reading example data from '%s'." % data_filename)
    with open(data_filename, 'r') as f:
        data_batch_list = json.load(f)
    data_batch = None
    for batch in data_batch_list:
        if batch['batch_name'] == train_batch:
            data_batch = batch
    assert data_batch is not None

    # Find right hypers:
    print ("Reading model parameters for configuration '%s' from '%s'." % (data_batch['hypers'], hypers_filename))
    with open(hypers_filename, 'r') as f:
        hypers_list = json.load(f)
    hypers = None
    for hyper_name, hyper_settings in hypers_list.iteritems():
        if hyper_name == data_batch['hypers']:
            hypers = hyper_settings
    assert hypers is not None

    model_name = os.path.splitext(os.path.basename(model_filename))[0]
    data_name = os.path.splitext(os.path.basename(data_filename))[0]
    out_name = os.path.join("%s-%s-%s" % (model_name,
                                          data_name,
                                          train_batch))

    return (parsed_model, data_batch, hypers, out_name)
