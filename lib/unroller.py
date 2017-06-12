from __future__ import print_function

import ast
import astunparse
import copy
import error
from error import check_type
import itertools
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def flat_map(f, l):
    r = []
    for e in l:
        er = f(e)
        if isinstance(er, list):
            r.extend(er)
        else:
            r.append(er)
    return r


class PartialEvaluator(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        for decorator in node.decorator_list:
            self.visit(decorator)
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.left, ast.Num) and isinstance(node.right, ast.Num):
            value = eval(compile(ast.copy_location(ast.Expression(body=node), node), '', 'eval'))
            return ast.copy_location(ast.Num(n=value), node)
        else:
            return node

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        allBoolLit = True
        for value in node.values:
            allBoolLit = allBoolLit and isinstance(value, ast.Name) and (value.id == "True" or value.id == "False")
        if allBoolLit:
            value = eval(compile(ast.copy_location(ast.Expression(body=node), node), '', 'eval'))
            return ast.copy_location(ast.Name(id=value, ctx=ast.Load()), node)
        else:
            return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        allNum = isinstance(node.left, ast.Num)
        for comparator in node.comparators:
            allNum = allNum and isinstance(comparator, ast.Num)
        if allNum:
            value = eval(compile(ast.copy_location(ast.Expression(body=node), node), '', 'eval'))
            lit = ast.copy_location(ast.Name(id=str(value), ctx=ast.Load()), node)
            return ast.copy_location(lit, node)
        else:
            return node

    def visit_If(self, node):
        node.test = self.visit(node.test)
        #Make sure to only recursively visit things that we are going to keep:
        if isinstance(node.test, ast.Name):
            if node.test.id == True or node.test.id == "True":
                #Flatten lists:
                res = []
                for stmt in node.body:
                    new_stmt = self.visit(stmt)
                    if isinstance(new_stmt, list):
                        res.extend(new_stmt)
                    else:
                        res.append(new_stmt)
                node.body = res
                return node.body
            elif node.test.id == False or node.test.id == "False":
                #Flatten lists:
                res = []
                for stmt in node.orelse:
                    new_stmt = self.visit(stmt)
                    if isinstance(new_stmt, list):
                        res.extend(new_stmt)
                    else:
                        res.append(new_stmt)
                node.orelse = res
                return node.orelse
        return self.generic_visit(node)


def eval_const_expressions(node):
    return PartialEvaluator().visit(node)


def subs(root, **kwargs):
    '''Substitute ast.Name nodes for numbers in root using the mapping in
    kwargs. Returns a new copy of root.

    '''
    root = copy.deepcopy(root)

    class Transformer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            return node

        def visit_Name(self, node):
            if node.id in kwargs and not isinstance(node.ctx, ast.Store):
                replacement = kwargs[node.id]
                if isinstance(replacement, int):
                    return ast.copy_location(ast.Num(n=replacement), node)
                else:
                    return copy.copy(replacement)
            else:
                return node

    return Transformer().visit(root)


def update_variable_domains(variable_domains, node_list):
    '''Updates variable_domains by inserting mappings from a variable name to the to the number of
    elements in their data domain.

    Program variables are created in assignments of the form
    "{varName} = Var({size})" or "{varName} = Param({size})".

    '''
    for ch in node_list:
        if isinstance(ch, ast.Assign) and len(ch.targets) == 1:
            name = astunparse.unparse(ch.targets[0]).rstrip()
            rhs = ch.value

            if isinstance(rhs, ast.Call):
                decl_name = rhs.func.id
                args = rhs.args
            elif isinstance(rhs, ast.Subscript) and isinstance(rhs.value, ast.Call):
                decl_name = rhs.value.func.id
                args = rhs.value.args
            else:
                continue

            if decl_name not in ["Param", "Var", "Input", "Output"]:
                continue
            
            if len(args) > 1:
                error.error('More than one size parameter in variable declaration of "%s".' % (name), ch)
            size = args[0]

            if isinstance(size, ast.Num):
                if name in variable_domains and variable_domains[name] != size.n:
                    error.fatal_error("Trying to reset the domain of variable '%s' to '%i' (old value '%i')." % (name, size.n, variable_domains[name]), size) 
                variable_domains[name] = size.n
            else:
                error.fatal_error("Trying to declare variable '%s', but size parameters '%s' is not understood." % (name, astunparse.unparse(size).rstrip()), size)

def inline_assigns_and_unroll_fors_and_withs(root):
    '''Substitute identifiers defined using ast.Assign nodes by their assigned values,
    and unroll for/with statements at the same time.

    These passes need to happen together, so that assignments that become constant through
    unrolling are correctly propagated, and for/with statements are properly unrolled
    when nested.

    Returns a new copy of node.
    '''

    variable_domains = {}

    def get_variable_domain(node):
        # Look up the number of values to switch on.
        if isinstance(node, ast.Name):
            return variable_domains[node.id]
        if isinstance(node, str):
            return variable_domains[node]
        if isinstance(node, ast.Subscript):
            if node.value.id in variable_domains:
                return variable_domains[node.value.id]
            node_name = astunparse.unparse(node).rstrip()
            if node_name in variable_domains:
                return variable_domains[node_name]
        error.fatal_error("No variable domain known for expression '%s', for which we want to unroll a for/with." % (astunparse.unparse(node).rstrip()), node)

    class Transformer(ast.NodeTransformer):
        def __init__(self, environment={}, inlinable_functions={}):
            # While I think Python's scoping rules are an abomination unto
            # Nuggan, they do come handy here -- we don't need to worry
            # about things coming in and going out of scope...
            self.__environment = environment
            self.__inlinable_functions = copy.copy(inlinable_functions)

        def visit_FunctionDef(self, node):
            for decorator in node.decorator_list:
                self.visit(decorator)

            # Record inlinable functions, and do not visit them:
            if len(node.decorator_list) == 1 and node.decorator_list[0].func.id == "Inline":
                self.__inlinable_functions[node.name] = node
#            else:
#                # Spawn off sub-visitor initialised with current environment,
#                # but its own scope, and remove arguments:
#                subEnvironment = copy.copy(self.__environment)
#                for arg in node.args.args:
#                    subEnvironment.pop(arg.id, None)
#                subVisitor = Transformer(subEnvironment, self.__inlinable_functions)
#                node = subVisitor.generic_visit(node)

            return node

        def visit_Expr(self, node):
            node = self.generic_visit(node)
            if isinstance(node.value, list):
                return node.value
            else:
                return node

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
                if function_name in self.__inlinable_functions:
                    to_inline = self.__inlinable_functions[function_name]
                    fun_pars = to_inline.args.args
                    call_args = node.args

                    if len(fun_pars) != len(call_args):
                        error.error("Trying to inline function with mismatching argument and parameter numbers.", node)

                    instantiation = {}
                    for i in range(0, len(fun_pars)):
                        instantiation[fun_pars[i].id] = self.visit(call_args[i])

                    inlined_stmts = []
                    for stmt in to_inline.body:
                        instantiated_stmt = subs(stmt, **instantiation)
                        instantiated_stmt = self.visit(instantiated_stmt)
                        inlined_stmts.append(instantiated_stmt)

                    return inlined_stmts

            return self.generic_visit(node)

        def visit_Assign(self, assgn):
            if len(assgn.targets) > 1:
                raise Exception("Cannot process tuple assignment in %s" % assgn)
            if not(isinstance(assgn.targets[0], ast.Name)):
                assgn.targets[0] = eval_const_expressions(self.visit(assgn.targets[0]))
            else:
                assgn.targets[0] = eval_const_expressions(assgn.targets[0])
            target = assgn.targets[0]
            assgn.value = eval_const_expressions(self.visit(assgn.value))
            if isinstance(target, ast.Name) and (isinstance(assgn.value, ast.Num) or
                                                 isinstance(assgn.value, ast.Name) or
                                                 (isinstance(assgn.value, ast.Subscript) and isinstance(assgn.value.value, ast.Name))):
                self.__environment[target.id] = assgn.value
            update_variable_domains(variable_domains, [assgn])
            return assgn

        def visit_Name(self, node):
            if node.id in self.__environment:
                return copy.deepcopy(self.__environment[node.id])
            else:
                return node

        def visit_If(self, node):
            node.test = self.visit(node.test)
            node = eval_const_expressions(node)
            if not(isinstance(node, ast.If)):
                if isinstance(node, list):
                    return flat_map(self.visit, node)
                else:
                    return self.visit(node)

            # We want to unroll the "else: P" bit into something explicit for
            # all the cases that haven't been checked explicitly yet.
            def get_or_cases(test):
                if not(isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or)):
                    return [test]
                else:
                    return itertools.chain.from_iterable(get_or_cases(value) for value in test.values)

            def get_name_and_const_from_test(test):
                if not(isinstance(test.ops[0], ast.Eq)):
                    raise Exception("Tests in if can only use ==, not '%s'." % (astunparse.unparse(test.ops[0]).rstrip()), node)
                (name, const) = (None, None)
                if (isinstance(test.left, ast.Name) or isinstance(test.left, ast.Subscript)) and isinstance(test.comparators[0], ast.Num):
                    (name, const) = (test.left, test.comparators[0].n)
                elif (isinstance(test.comparators[0], ast.Name) or isinstance(test.comparators[0], ast.Subscript)) and isinstance(test.left, ast.Num):
                    (name, const) = (test.comparators[0], test.left.n)
                return (name, const)

            checked_values = set()
            checked_vars = set()

            # Now walk the .orelse branches, visiting each body independently. Expand ors on the way:
            last_if = None
            current_if = node
            while True:
                # Recurse with visitor first:
                current_if.test = eval_const_expressions(self.visit(current_if.test))
                current_if.body = flat_map(self.visit, current_if.body)
                # Now, unfold ors:
                test_cases = list(get_or_cases(current_if.test))
                if len(test_cases) > 1:
                    else_body = current_if.orelse
                    new_if_node = None
                    for test_case in reversed(test_cases):
                        body_copy = copy.deepcopy(current_if.body)
                        new_if_node = ast.copy_location(ast.If(test=test_case,
                                                               body=body_copy,
                                                               orelse=else_body),
                                                        node)
                        else_body = [new_if_node]
                    current_if = new_if_node
                    # Make the change stick:
                    if last_if is None:
                        node = current_if
                    else:
                        last_if.orelse = [current_if]

                # Do our deed:
                (checked_var, checked_value) = get_name_and_const_from_test(current_if.test)
                checked_vars.add(checked_var)
                checked_values.add(checked_value)
                # Look at the next elif:
                if len(current_if.orelse) == 1 and isinstance(current_if.orelse[0], ast.If):
                    last_if = current_if
                    current_if = current_if.orelse[0]
                else:
                    break

            # We need to stringify them, to not be confused by several instances refering to the same thing:
            checked_var_strs = set(astunparse.unparse(var) for var in checked_vars)
            if len(checked_var_strs) != 1:
                raise Exception("If-else checking more than one variable (%s)." % (checked_var_strs))
            checked_var = checked_vars.pop()
            domain_to_check = set(range(get_variable_domain(checked_var)))
            still_unchecked = domain_to_check - checked_values

            else_body = flat_map(self.visit, current_if.orelse)

            if len(else_body) == 0:
                return node

            # if len(still_unchecked) > 0:
            #     print("Else for values %s of %s:\n%s" % (still_unchecked, astunparse.unparse(checked_var).rstrip(), astunparse.unparse(else_body)))
            for value in still_unchecked:
                # print("Inserting case %s == %i in else unfolding." % (astunparse.unparse(checked_var).rstrip(), value))
                var_node = copy.deepcopy(checked_var)
                eq_node = ast.copy_location(ast.Eq(), node)
                value_node = ast.copy_location(ast.Num(n=value), node)
                test_node = ast.copy_location(ast.Compare(var_node, [eq_node], [value_node]), node)
                case_body = copy.deepcopy(else_body)
                new_if_node = ast.copy_location(ast.If(test=test_node, body=case_body, orelse=[]), node)
                current_if.orelse = [new_if_node]
                current_if = new_if_node

            return node

        def visit_For(self, node):
            # Find start and end node of iteration range.
            iter_args = [eval_const_expressions(self.visit(arg)) for arg in node.iter.args]
            (i_start, i_end, i_step) = (0, None, 1)

            def as_int(num):
                check_type(num, ast.Num)
                return num.n
            
            if len(iter_args) == 1:
                i_end = as_int(iter_args[0])
            elif len(iter_args) == 2:
                i_start = as_int(iter_args[0])
                i_end = as_int(iter_args[1])
            elif len(iter_args) == 3:
                i_start = as_int(iter_args[0])
                i_end = as_int(iter_args[1])
                i_step = as_int(iter_args[2])
            else:
                raise RuntimeError("Unhandled number of args in for loop.")

            body = []
            for i in range(i_start, i_end, i_step):
                # Perform loop unrolling in the cloned loop body. Note
                # that we _must_ perform unrolling separately for each
                # clone of the loop body, because inner loops may use
                # the loop counter in their loop bounds. We will not
                # be able to unroll these inner loops before we unroll
                # this one.

                # Substitute the value of the loop counter into the
                # cloned loop body.
                self.__environment[node.target.id] = ast.copy_location(ast.Num(n=i), node)
                new_node = copy.deepcopy(node)
                self.generic_visit(new_node)
                update_variable_domains(variable_domains, new_node.body)
                
                body += new_node.body

            return body

        def visit_With(self, node):
            context_expr = self.visit(node.context_expr)

            result = if_node = ast.copy_location(ast.If(), node)
            variable_domain = get_variable_domain(context_expr)

            for i_value in range(0, variable_domain):
                # Create the test (context_expr == i_value).
                eq_node = ast.copy_location(ast.Eq(), node)
                value_node = ast.copy_location(ast.Num(n=i_value), node)
                if_node.test = ast.copy_location(ast.Compare(context_expr, [eq_node], [value_node]), node)

                # Substitute the current value of the context
                # expression into the body of the with. If the with
                # binds a name, substitute uses of that
                # name. Otherwise, substitute uses of the context
                # expression.
                if node.optional_vars is not None:
                    check_type(node.optional_vars, ast.Name)
                    replacements = {node.optional_vars.id : i_value}
                    if_node.body = eval_const_expressions(subs(node, **replacements)).body

                else:
                    if isinstance(context_expr, ast.Name):
                        replacements = {context_expr.id : i_value}
                        if_node.body = eval_const_expressions(subs(node, **replacements)).body

                    elif isinstance(context_expr, ast.Subscript):
                        replacements = {subscript_to_tuple(context_expr) : ast.copy_location(ast.Num(n=i_value), node)}
                        if_node.body = eval_const_expressions(sub_subscript(node, replacements)).body

                    else:
                        error.fatal_error('Unexpected expression in with.', context_expr)

                update_variable_domains(variable_domains, if_node.body)
                        
                # Recursively process withs inside the body. This must
                # be performed separately for each body, because withs
                # inside the body may depend on the current value of
                # the context expression.
                self.generic_visit(if_node)

                # If this is not the last iteration of the loop,
                # generate a new if node and add it to the else block
                # of the current if. We will use the new if node in
                # the next iteration.
                if i_value < variable_domain-1:
                    if_node.orelse = [ast.copy_location(ast.If(), node)]
                    if_node = if_node.orelse[0]
                else:
                    if_node.orelse = []

            if variable_domain == 0:
                result = []

            return result

    root = copy.deepcopy(root)
    return Transformer().visit(root)

def subscript_to_tuple(subscript):
    '''Convert a subscripted name of the form Name[(i1, ..., in)] to a
    tuple ('Name', i1, ..., in).

    '''
    def err():
        raise ValueError('Unexpected kind of slice: {}'.format(astunparse.unparse(subscript)))

    # Get subscript name.
    if isinstance(subscript.value, ast.Name):
        name = subscript.value.id
    else:
        err()

    # Get indices.
    if isinstance(subscript.slice, ast.Index):
        if isinstance(subscript.slice.value, ast.Num):
            indices = [subscript.slice.value]
        elif isinstance(subscript.slice.value, ast.Tuple):
            indices = subscript.slice.value.elts
        else:
            err()
    else:
        err()

    # Convert indices to python numbers.
    int_indices = []
    for i in indices:
        if isinstance(i, ast.Num):
            int_indices.append(i.n)
        else:
            err()

    return tuple([name] + int_indices)

def sub_subscript(root, subs):
    root = copy.deepcopy(root)

    class Transformer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            return node

        def visit_Subscript(self, node):
            self.generic_visit(node)

            try:
                node_tup = subscript_to_tuple(node)
                if node_tup in subs:
                    return subs[node_tup]
                else:
                    return node

            except ValueError:
                return node

    return Transformer().visit(root)

def slice_node_to_tuple_of_numbers(slice_node):
    if isinstance(slice_node.value, ast.Tuple):
        indices = (elt for elt in slice_node.value.elts)
    else:
        indices = (slice_node.value,)

    indices = list(indices)
    for index in indices:
        if not(isinstance(index, ast.Num)):
            error.fatal_error("Trying to use non-constant value '%s' as array index." % (astunparse.unparse(index).rstrip()), index)

    # Convert to python numbers
    indices = (index.n for index in indices)

    return indices

def flattened_array_name(array_name, indices):
    return array_name + "_" + '_'.join([str(i) for i in indices])

def flatten_array_declarations(root):
    class Transformer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            return node

        def visit_Assign(self, node):
            if isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Call):
                subscr = node.value
                call = subscr.value
                
                if len(node.targets) > 1:
                    error.error('Cannot use multiple assignment in array declaration.', node)
                    
                variable_name = node.targets[0].id
                value_type = call.func.id
                declaration_args = call.args

                # Get the indices being accessed.
                shape = slice_node_to_tuple_of_numbers(subscr.slice)

                new_assigns = []
                for indices in itertools.product(*[range(n) for n in shape]):
                    index_name = flattened_array_name(variable_name, indices)
                    new_index_name_node = ast.copy_location(ast.Name(index_name, ast.Store()), node)
                    new_value_type_node = ast.copy_location(ast.Name(value_type, ast.Load()), node)
                    new_declaration_args = [copy.deepcopy(arg) for arg in declaration_args]
                    new_call_node = ast.copy_location(ast.Call(new_value_type_node, new_declaration_args, [], None, None), node)
                    new_assign = ast.Assign([new_index_name_node], new_call_node)
                    new_assign = ast.copy_location(new_assign, node)
                    new_assigns.append(new_assign)
                return new_assigns
            else:
                return node
        
    return Transformer().visit(root)

def flatten_array_lookups(root):
    class Transformer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            return node

        def visit_Subscript(self, node):
            self.generic_visit(node)

            # Get the indices being accessed.
            indices = slice_node_to_tuple_of_numbers(node.slice)
            variable_name = node.value.id
            index_name = flattened_array_name(variable_name, indices)
            return ast.copy_location(ast.Name(index_name, node.ctx), node)

    return Transformer().visit(root)

def compute_function_outputs(root):
    def cart_prod(xss):
        if len(xss) == 0: return []
        xs = xss[0]
        if len(xss) == 1: return [[x] for x in xs]
        rest_prod = cart_prod(xss[1:])
        return [[ele] + pprod for ele in xs for pprod in rest_prod]

    #Extract all needed context statements:
    defined_functions = {}
    context_stmts = []
    for node in root.body:
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call) and node.value.func.id in ["Param", "Var", "Input", "Output"]:
                pass
            elif isinstance(node.value, ast.Name):
                pass
            else:
                context_stmts.append(node)

        if isinstance(node, ast.FunctionDef):
            #Record the functions that we want to look at:
            if len(node.decorator_list) == 1 and node.decorator_list[0].func.id == "Runtime":
                decorator = node.decorator_list[0]
                if len(decorator.args) == 2 and isinstance(decorator.args[0], ast.List) and all(isinstance(elt, ast.Num) for elt in decorator.args[0].elts) and isinstance(decorator.args[1], ast.Num):
                    defined_functions[node.name] = ([elt.n for elt in decorator.args[0].elts], decorator.args[1].n)

            #We need to get rid of the annotation that we haven't defined...
            node = copy.deepcopy(node)
            node.decorator_list = []
            context_stmts.append(node)

    node = ast.Pass(lineno=0, col_offset=0)

    #Evaluate each function on all allowed input/output pairs:
    results = {}
    for (func_name, (par_domains, res_domain)) in defined_functions.iteritems():
        #Now prepare the code that we are going to evaluate:
        func_name_node = ast.copy_location(ast.Name(id=func_name, ctx = ast.Load()), node)

        stmts = []
        args_to_var_name = {}
        for args in cart_prod([range(d) for d in par_domains]):
            arg_nodes = []
            for arg in args:
                arg_nodes.append(ast.copy_location(ast.Num(n=arg), node))
            func_call = ast.copy_location(ast.Call(func_name_node, arg_nodes, [], None, None), node)
            res_var_name = "res__%s__%s" % (func_name, "__".join(map(str, args)))
            args_to_var_name[tuple(args)] = res_var_name
            res_var_name_node = ast.copy_location(ast.Name(id=res_var_name, ctx=ast.Store()), node)
            stmts.append(ast.copy_location(ast.Assign([res_var_name_node], func_call), node))
        wrapped = ast.copy_location(ast.Module(body=context_stmts + stmts), node)
        eval_globals = {}
        eval(compile(wrapped, '<constructed>', 'exec'), eval_globals)
        results[func_name] = {args: eval_globals[args_to_var_name[args]] for args in args_to_var_name.keys()}

    return results

def check(root):
    '''Checks unrolled program for correctness w.r.t. TerpreT restrictions.
    These include SSA form (i.e., no code path sets the same value twice),
    variable initialisation (all values are initialised, or an input),
    
    '''

    class CheckMessage(object):
        def __init__(self, message, node = None):
            self.message = message
            self.node = node

        def message_prefix(self):
            return ""

        def print(self):
            if self.node != None and hasattr(self.node, "lineno"):
                if hasattr(self.node, "col_offset"):
                    location_description = "In line %i, col %i: " % (self.node.lineno, self.node.col_offset)
                else:
                    location_description = "In line %i: " % (self.node.lineno)
            else:
                location_description = ""
            eprint("%s%s%s" % (self.message_prefix(), location_description, self.message))

    class CheckError(CheckMessage):
        def __init__(self, message, node = None):
            super(CheckError, self).__init__(message, node)

        def message_prefix(self):
            return "Error: "

    class CheckWarning(CheckMessage):
        def __init__(self, message, node = None):
            super(CheckWarning, self).__init__(message, node)

        def message_prefix(self):
            return "Warning: "

    #Sadly, this can't implement ast.NodeVisitor, because we need more state (and control over that state)
    #The implementation is a copy of the ast.NodeVisitor interface, extended to thread a state object through the recursion.
    #In our case, this is a pair of the set of initialized variables and the set of used variables, whose updated form we return.
    class TerpretChecker():
        def __init__(self):
            self.__var_domain = {} # name -> uint, where "foo" => 3 means 'foo can take values 0, 1, 2'; 0 means "variable is numeric constant"
            self.__defined_functions = {}
            self.__messages = []
            self.__outputs = []

        def __set_domain(self, node, var, dom):
            if var in self.__var_domain:
                self.__messages.append(CheckError("Trying to redeclare variable '%s'." % (var), node))
            self.__var_domain[var] = dom

        def __get_domain(self, node, var):
            if not(var in self.__var_domain):
                self.__messages.append(CheckError("Trying to use undeclared variable '%s'." % (var), node))
                return 0
            return self.__var_domain[var]

        def __is_declared(self, var):
            return var in self.__var_domain

        def __get_domain_of_expr(self, node):
            if isinstance(node, ast.Name):
                return self.__get_domain(node, node.id)
            elif isinstance(node, ast.Call):
                (_, value_domain) = self.__defined_functions[node.func.id]
                return value_domain
            elif isinstance(node, ast.Num):
                return node.n + 1 #And now all together: Zero-based counting is haaaaard.
            else:
                self.__messages.append(CheckError("Cannot determine domain of value '%s' used in assignment." % (astunparse.unparse(node).rstrip()), node))
                return 0

        def check_function_outputs(self):
            functions_to_ins_to_outs = compute_function_outputs(root)
            for (func_name, (par_domains, res_domain)) in self.__defined_functions.iteritems():
                ins_to_outs = functions_to_ins_to_outs[func_name]
                for args, output in ins_to_outs.iteritems():
                    if output < 0 or output >= res_domain:
                        self.__messages.append(CheckError("Function '%s' with domain [0..%i] returns out-of-bounds value '%i' for inputs %s." % (func_name, res_domain - 1, output, str(args)), root))

        def visit(self, state, node):
            """Visit a node or a list of nodes."""
            if isinstance(node, list):
                node_list = node
            else:
                node_list = [node]
            for node in node_list:
                method = 'visit_' + node.__class__.__name__
                visitor = getattr(self, method, self.generic_visit)
                state = visitor(state, node)
            return state
                
        def visit_FunctionDef(self, state, node):
            if len(node.decorator_list) == 1 and node.decorator_list[0].func.id == "Runtime":
                decorator = node.decorator_list[0]
                if len(decorator.args) == 2 and isinstance(decorator.args[0], ast.List) and all(isinstance(elt, ast.Num) for elt in decorator.args[0].elts) and isinstance(decorator.args[1], ast.Num):
                    if node.name in self.__defined_functions:
                        self.__messages.append(CheckError("Re-definition of @Runtime function '%s'." % (node.name), node))
                    self.__defined_functions[node.name] = ([elt.n for elt in decorator.args[0].elts], decorator.args[1].n)
                else:
                    self.__messages.append(CheckError("@Runtime function '%s' has unknown parameter structure, should be ([NUM, ..., NUM], NUM)." % (node.name), node))
            elif len(node.decorator_list) == 1 and node.decorator_list[0].func.id == "Inline":
                pass #These are OK.
            else:
                self.__messages.append(CheckError("Cannot declare non-@Runtime function '%s'." % (node.name), node))
            return state
        
        def visit_Assign(self, state, assgn):
            (parameters, initialised, used) = state
            if len(assgn.targets) > 1:
                self.__messages.append(CheckError("Cannot process tuple assignment in '%s'." % (astunparse.unparse(assgn).rstrip()), assgn))
            target = assgn.targets[0]
            value = assgn.value
            if isinstance(target, ast.Name):
                assignedVar = target.id
                if isinstance(value, ast.Num):
                    if self.__is_declared(assignedVar):
                        self.__messages.append(CheckError("Trying to assign num literal '%i' to model variable '%s'." % (value.n, assignedVar), assgn))
                elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                    if (value.func.id in ["Var", "Param", "Input", "Output"]):
                        if isinstance(value.args[0], ast.Num):
                            self.__set_domain(assgn, assignedVar, value.args[0].n)
                        else:
                            self.__messages.append(CheckError("Cannot declare variable/parameter with non-constant range '%s'." % (astunparse.unparse(value.func.args[0]).rstrip()), assgn))
                        if value.func.id == "Param":
                            return (parameters | {assignedVar}, initialised, used)
                        if value.func.id == "Input":
                            return (parameters, initialised | {assignedVar}, used)
                        if value.func.id in ["Output"]:
                            self.__outputs.append(assignedVar)
                            return (parameters, initialised, used | {assignedVar})
                    elif value.func.id in self.__defined_functions:
                        for argument in value.args:
                            (_, _, used) = self.visit((parameters, initialised, used), argument)
                        return (parameters, initialised | {assignedVar}, used)
                    else:
                        self.__messages.append(CheckError("Cannot assign unknown function to variable '%s'." % (assignedVar), assgn))
                else:
                    if self.__is_declared(assignedVar):
                        self.__messages.append(CheckError("Trying to assign '%s' to model variable '%s'." % (astunparse.unparse(value).rstrip(), assignedVar), assgn))
            else:
                self.__messages.append(CheckError("Cannot assign value to non-variable '%s'." % (astunparse.unparse(target).rstrip()), assgn))
            return state

        def visit_Expr(self, state, expr):
            return self.visit(state, expr.value)

        def visit_Name(self, state, name):
            (parameters, initialised, used) = state
            if name.id not in initialised and name.id not in parameters:
                self.__messages.append(CheckError("Use of potentially uninitialised variable '%s'." % (name.id), name))
            return (parameters, initialised, used | {name.id})

        def visit_Num(self, state, name):
            return state

        def visit_Call(self, state, call):
            (parameters, initialised, used) = state
            if isinstance(call.func, ast.Attribute):
                func = call.func
                func_name = func.attr
                set_variable = func.value.id
                if func_name in ["set_to_constant", "set_to"]:
                    #Check that the children are fine:
                    if len(call.args) != 1:
                        self.__messages.append(CheckError("'%s.%s' with more than one argument unsupported." % (set_variable, func_name), call))
                    value = call.args[0]
                    (_, _, val_used_vars) = self.visit(state, value)
                    if set_variable in initialised:
                        self.__messages.append(CheckError("Trying to reset value of variable '%s'." % (set_variable), call))
                    if set_variable in parameters:
                        self.__messages.append(CheckWarning("Setting value of parameter '%s'." % (set_variable), call))
                    domain = self.__get_domain(call, set_variable)
                    if isinstance(value, ast.Num) and (value.n < 0 or value.n >= domain):
                        self.__messages.append(CheckError("Trying to set variable '%s' (domain [0..%i]) to invalid value '%i'." % (set_variable, domain - 1, value.n), call))
                    else:
                        value_domain = self.__get_domain_of_expr(value)
                        if value_domain != domain:
                            if isinstance(value, ast.Num) and (value.n >= domain or value.n < 0):
                                self.__messages.append(CheckError("Trying to set variable '%s' (domain [0..%i]) to value '%s'." % (set_variable, domain - 1, astunparse.unparse(value).rstrip()), value))
                            elif not(isinstance(value, ast.Num)):
                                self.__messages.append(CheckError("Trying to set variable '%s' (domain [0..%i]) to value '%s' with different domain [0..%i]." % (set_variable, domain - 1, astunparse.unparse(value).rstrip(), value_domain - 1), value))
                    return (parameters, initialised | {set_variable}, val_used_vars)
                elif func_name in ["set_as_input"]:
                    return (parameters, initialised | {set_variable}, used)
                elif func_name in ["set_as_output"]:
                    return (parameters, initialised, used | {set_variable})
                elif func_name == "observe_value":
                    #Check that the children are fine:
                    if len(call.args) != 1:
                        self.__messages.append(CheckError("'%s.%s' with more than one argument unsupported." % (set_variable, func_name), call))
                    (_, _, val_used_vars) = self.visit(state, call.args[0])
                    if set_variable not in initialised:
                        self.__messages.append(CheckError("Observation of potentially uninitialised variable '%s'." % (set_variable), call))
                    return (parameters, initialised, val_used_vars | {set_variable})
                else:
                    self.__messages.append(CheckError("Unsupported call '%s'." % (astunparse.unparse(call).rstrip()), call))
            else:
                func_name = call.func.id
                func_information = self.__defined_functions.get(func_name, None)
                if func_information != None:
                    (par_domains, _) = func_information
                    used_vars = used
                    if len(call.args) != len(par_domains):
                        self.__messages.append(CheckError("Call to %i-ary function '%s' with %i arguments." % (len(par_domains), func_name, len(call.args)), call))
                    for idx in range(len(call.args)):
                        arg = call.args[idx]
                        (_, _, used_vars) = self.visit((parameters, initialised, used_vars), arg)
                        par_domain = par_domains[idx]
                        arg_domain = self.__get_domain_of_expr(arg)
                        if arg_domain != par_domain:
                            if isinstance(arg, ast.Num) and (arg.n >= par_domain or arg.n < 0):
                                self.__messages.append(CheckError("Parameter %i of function '%s' has domain [0..%i], but argument value '%s' is incompatible." % (idx + 1, func_name, par_domain - 1, astunparse.unparse(arg).rstrip()), arg))
                            elif not(isinstance(arg, ast.Num)):
                                self.__messages.append(CheckError("Parameter %i of function '%s' has domain [0..%i], but argument value '%s' has different domain [0..%i]." % (idx + 1, func_name, par_domain - 1, astunparse.unparse(arg).rstrip(), arg_domain - 1), arg))
                    return (parameters, initialised, used_vars)
                else:
                    self.__messages.append(CheckError("Call to undefined functions '%s'." % (func_name), call))

        def visit_If(self, state, node):
            (parameters, initialised, used) = state

            #Here we need to do a bit of work to "linearise" case-analysis if-elif-elif structures,
            #to explore if all cases are covered (and to see if the else branch is ever hit)
            #For this, we have a fairly restrictive test format:
            def check_test(test):
                if not(isinstance(test.ops[0], ast.Eq)):
                    self.__messages.append(CheckError("Tests can only use ==, not '%s'." % (astunparse.unparse(test.ops[0]).rstrip()), node))
                if not(isinstance(test.left, ast.Name)):
                    self.__messages.append(CheckError("Tests have to have identifier, not '%s' as left operand." % (astunparse.unparse(test.left).rstrip()), node))
                if len(test.comparators) != 1:
                    self.__messages.append(CheckError("Tests cannot have multiple comparators in test '%s'." % (astunparse.unparse(test).rstrip()), node))
                if not(isinstance(test.comparators[0], ast.Num)):
                    self.__messages.append(CheckError("Tests have to have constant, not '%s' as right operand." % (astunparse.unparse(test.comparators[0]).rstrip()), node))
                return (test.left.id, test.comparators[0].n)

            (checked_var, checked_val) = check_test(node.test)
            if checked_var not in initialised and checked_var not in parameters:
                self.__messages.append(CheckError("Test uses potentially uninitialised variable '%s'." % (checked_var), node))
            var_domain_to_check = set(range(0, self.__get_domain(node, checked_var)))
            used.add(checked_var)

            #Now walk the .orelse branches, visiting each body independently...
            branch_inits = []
            current_if = node
            while True:
                (branch_checked_var, branch_checked_val) = check_test(current_if.test)
                if branch_checked_var != checked_var:
                    self.__messages.append(CheckError("Case-analysis branch tests refer to different variables '%s' and '%s'." % (checked_var, branch_checked_var), current_if))
                if branch_checked_val not in var_domain_to_check:
                    self.__messages.append(CheckError("Testing for value '%i' of variable '%s', which is either out of domain or has already been handled." % (branch_checked_val, checked_var), current_if))
                var_domain_to_check.discard(branch_checked_val)
                (_, branch_init, used) = self.visit((parameters, initialised.copy(), used), current_if.body)
                branch_inits.append(branch_init)
                if len(current_if.orelse) == 0:
                    #We've reached the end:
                    break
                elif len(current_if.orelse) > 1 or not(isinstance(current_if.orelse[0], ast.If)):
                    self.__messages.append(CheckError("Non-empty else branch of case analysis.", node))
                    break
                else:
                    current_if = current_if.orelse[0]
            #... and now check if the results make sense:
            some_branches_init = branch_inits[0].copy()
            all_branches_init = branch_inits[0].copy()
            #If not all values were checked, the empty else block wouldn't do anything:
            if len(var_domain_to_check) > 0:
                all_branches_init = initialised.copy()
            for i in range(1, len(branch_inits)):
                some_branches_init = some_branches_init.union(branch_inits[i])
                all_branches_init = all_branches_init.intersection(branch_inits[i])
            not_all_inits = some_branches_init.difference(all_branches_init)
            if len(not_all_inits) > 0:
                self.__messages.append(CheckWarning("Variables '%s' only initialised in some branches of if-elif." % ("', '".join(sorted(not_all_inits))), node.lineno))

            return (parameters, all_branches_init, used)

        def visit_Module(self, state, node):
            return self.visit(state, node.body)

        def visit_ImportFrom(self, state, imp):
            if imp.module is "dummy":
                return state
            return self.generic_visit(imp)

        def generic_visit(self, state, node):
            self.__messages.append(CheckError("AST node '%s' unsupported." % (astunparse.unparse(node).strip()), node))

        def check(self, root):
            try:
                (parameters, initialised, used) = self.visit((set(), set(), set()), root)
                unused = initialised - used
                for id in self.__outputs:
                    if id not in initialised:
                        self.__messages.append(CheckError("Output variable '%s' not initialised." % (id)))
                for id in unused:
                    self.__messages.append(CheckWarning("Variable '%s' initialised, but not used." % (id)))
                self.check_function_outputs()
            except Exception as e:
                #Ignore if we found something
                if any([message for message in self.__messages if isinstance(message, CheckError)]):
                    pass
                else:
                    raise

            if len(self.__messages) > 0:
                for message in self.__messages:
                    message.print()
                return False

            return True

    return TerpretChecker().check(root)

def count_nodes(node):
    i = 0
    for _ in ast.walk(node):
        i += 1
    return i

def print_increase(node_count, start_node_count, prev_node_count=None):
    inc_start = float(node_count - start_node_count) / start_node_count
    if prev_node_count:
        inc_prev = float(node_count - prev_node_count) / prev_node_count
        eprint('Code size increased {:.2f}x over initial, {:.2f}x over previous.'.format(inc_start, inc_prev))
    else:
        eprint('Code size increased {:.2f}x over initial.'.format(inc_start))

def unroll_and_flatten(root, do_checks=True, print_info=False):
    count_start = count_nodes(root)

    if print_info: eprint('Inlining assignments and unrolling for loops and with statements...', end='')
    root = inline_assigns_and_unroll_fors_and_withs(root)
    if print_info: eprint('done.')
    count_unrolled = count_nodes(root)
    if print_info: print_increase(count_unrolled, count_start, count_start)

    if print_info: eprint('Partial evaluation of constant model components...', end='')
    root = eval_const_expressions(root)
    if print_info: eprint('done.')
    count_branches = count_nodes(root)
    if print_info: print_increase(count_branches, count_start, count_unrolled)

    if print_info: eprint('Flattening declarations...', end='')
    root = flatten_array_declarations(root)
    if print_info: eprint('done.')
    count_decls = count_nodes(root)
    if print_info: print_increase(count_decls, count_start, count_branches)

    if print_info: eprint('Flattening lookups...', end='')
    root = flatten_array_lookups(root)
    if print_info: eprint('done.')
    count_lookups = count_nodes(root)
    if print_info: print_increase(count_lookups, count_start, count_decls)

    if do_checks:
        check(root)

    return root
