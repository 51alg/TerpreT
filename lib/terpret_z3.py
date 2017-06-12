import ast
from astunparse import unparse
import itertools
import pdb
# If you are using Debian/Ubuntu packages, you'll need this:
from z3.z3 import *
import config
#init('/usr/lib/x86_64-linux-gnu/libz3.so.4')
init(config.LIB_Z3_PATH)
# If you have installed z3 locally, you will need something like this:
# from z3 import IntVal, Solver, init
# init('/path/to/local/installation/lib/libz3.so')

FRESH_ID_COUNTER = 0
DATA_TYPE = "int"


class ToZ3Environment():
    def __init__(self, tag='', env=None, variables_to_tag=None):
        if env is not None:
            self.tag = env.tag
            self.function_defs = dict(env.function_defs)
            self.var_range_defs = dict(env.var_range_defs)
            self.environment = dict(env.environment)
            self.params = env.params
            self.name_to_expr = env.name_to_expr
            self.variables_to_tag = env.variables_to_tag
        else:
            self.function_defs = {}
            self.var_range_defs = {}
            self.environment = {}
            self.params = set()
            self.name_to_expr = {}
            self.variables_to_tag = variables_to_tag

        if tag != '':
            self.tag = tag

    def should_tag(self, name):
        if self.variables_to_tag is None:
            return True
        return name in self.variables_to_tag


class ToZ3ExprVisitor(ast.NodeVisitor):
    def __init__(self, env):
        self.__env = ToZ3Environment(env=env)

    def __is_bool(self, expr):
        return isinstance(expr, BoolRef) or type(expr) is bool

    def __is_int(self, expr):
        return isinstance(expr, ArithRef) or type(expr) is int

    def __is_bv(self, expr):
        return isinstance(expr, BitVecRef)

    def gen_var(self, name, var_type=None):
        if var_type is None and not(name in self.__env.var_range_defs):
            raise Exception("Cannot declare variable of unknown range: %s" % name)

        if self.__env.should_tag(name):
            tagged_name = name + self.__env.tag
        else:
            tagged_name = name

        if var_type == "bool":
            return Bool(tagged_name)
        else:
            if DATA_TYPE == "int" or var_type == "int":
                ret = Int(tagged_name)
                self.__env.name_to_expr[name] = ret
                return ret
            elif DATA_TYPE.startswith("bit_"):
                bitwidth = int(DATA_TYPE.split("_")[-1])
                ret = BitVec(tagged_name, bitwidth)
                self.__env.name_to_expr[name] = ret
                return ret
            else:
                raise Exception("Unsupported data type %s" % DATA_TYPE)

    def fresh_var(self, var_type=None):
        global FRESH_ID_COUNTER
        id = FRESH_ID_COUNTER
        FRESH_ID_COUNTER = id + 1
        return self.gen_var("__fresh_var_%i" % id, var_type)

    def expr_of_id(self, id):
        if id in self.__env.environment:
            return self.__env.environment[id]
        else:
            return self.gen_var(id)

    def expr_of_value(self, val):
        if isinstance(val, str):
            return self.expr_of_id(str)
        else:
            return self.visit(val)

    def visit_Name(self, name):
        return self.expr_of_id(name.id)

    def visit_Num(self, num):
        if DATA_TYPE == "int":
            return IntVal(num.n)
        elif DATA_TYPE.startswith("bit_"):
            bitwidth = int(DATA_TYPE.split("_")[-1])
            return BitVecVal(num.n, bitwidth)
        else:
            raise Exception("Unsupported data type %s" % DATA_TYPE)

    def visit_BinOp(self, node):
        left_term = self.visit(node.left)
        right_term = self.visit(node.right)

        if self.__is_bool(left_term) and self.__is_bool(right_term):
            if isinstance(node.op, ast.BitAnd):
                return And(left_term, right_term)
            elif isinstance(node.op, ast.BitOr):
                return Or(left_term, right_term)
            elif isinstance(node.op, ast.BitXor):
                return Xor(left_term, right_term)
            else:
                raise Exception("Unsupported bool binary operation %s" % unparse(node))

        if DATA_TYPE == "int":
            if isinstance(node.op, ast.Mod):
                return left_term % right_term
            elif isinstance(node.op, ast.Add):
                return left_term + right_term
            elif isinstance(node.op, ast.Sub):
                return left_term - right_term
            elif isinstance(node.op, ast.Mult):
                return left_term * right_term
            elif isinstance(node.op, ast.BitXor):
                # Special-case for bool circuit-examples:
                if left_term.is_int():
                    left_term = left_term == IntVal(1)
                if right_term.is_int():
                    right_term = right_term == IntVal(1)
                return left_term != right_term
            else:
                raise Exception("Unsupported integer binary operation %s" % unparse(node))
        elif DATA_TYPE.startswith("bit_"):
            if isinstance(node.op, ast.BitAnd):
                return left_term & right_term
            elif isinstance(node.op, ast.BitOr):
                return left_term | right_term
            elif isinstance(node.op, ast.BitXor):
                return left_term ^ right_term
            else:
                raise Exception("Unsupported bitvector operation %s" % unparse(node))
        else:
            raise Exception("Unsupported data type %s" % DATA_TYPE)

    def visit_BoolOp(self, node):
        if len(node.values) != 2:
            raise Exception("Can only handle binary bool operations at this point: %s"
                            % unparse(node))
        left_term = self.visit(node.values[0])
        right_term = self.visit(node.values[1])

        # Special-case for bool circuit-examples:
        if left_term.is_int():
            left_term = left_term == IntVal(1)
        if right_term.is_int():
            right_term = right_term == IntVal(1)

        if isinstance(node.op, ast.And):
            return And(left_term, right_term)
        elif isinstance(node.op, ast.Or):
            return Or(left_term, right_term)
        else:
            raise Exception("Unsupported bool operation %s" % unparse(node))

    def visit_UnaryOp(self, node):
        term = self.visit(node.operand)
        if self.__is_bool(term):
            if isinstance(node.op, ast.Not):
                return Not(term)
            elif isinstance(node.op, ast.Invert):
                return Not(term)
            else:
                raise Exception("Unsupported bool unary operation %s" % unparse(node))

        if DATA_TYPE == "int":
            if isinstance(node.op, ast.USub):
                return -term
            elif isinstance(node.op, ast.Not):
                if term.is_int():
                    term = term == IntVal(1)
                return Not(term)
            else:
                raise Exception("Unsupported integer unary operation %s" % unparse(node))
        elif DATA_TYPE.startswith("bit_"):
            if isinstance(node.op, ast.Not):
                return ~term
            elif isinstance(node.op, ast.Invert):
                return ~term
            else:
                raise Exception("Unsupported bitvector unary operation %s" % unparse(node))
        else:
            raise Exception("Unsupported unary operation %s" % unparse(node))

    def visit_Compare(self, node):
        left_term = self.visit(node.left)
        if len(node.comparators) > 1:
            raise Exception("Cannot handle 'foo > bar > baz' comparison in %s"
                            % unparse(node))
        right_term = self.visit(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            if self.__is_bool(left_term) and self.__is_bool(right_term):
                if left_term == True:
                    return right_term
                elif right_term == True:
                    return left_term
                elif left_term == False:
                    return Not(right_term)
                elif right_term == False:
                    return Not(left_term)
            return left_term == right_term
        elif isinstance(op, ast.Lt):
            return left_term < right_term
        elif isinstance(op, ast.LtE):
            return left_term <= right_term
        elif isinstance(op, ast.Gt):
            return left_term > right_term
        elif isinstance(op, ast.GtE):
            return left_term >= right_term
        else:
            raise Exception("Unhandled operators '%s' in %s"
                            % (unparse(op), unparse(node)))

    def visit_IfExp(self, node):
        test_term = self.visit(node.test)
        then_term = self.visit(node.body)
        else_term = self.visit(node.orelse)
        if then_term == True and else_term == False:
            return test_term
        return If(test_term, then_term, else_term)

    def visit_Call(self, call):
        func_name = call.func.id
        if func_name in self.__env.function_defs:
            func = self.__env.function_defs[func_name]
            argument_environment = dict([(argName.id, self.expr_of_value(argValue))
                                         for (argName, argValue) in zip(func["args"], call.args)])
            # This should actually only be argument_environment ... we are messing up scoping here, hoping that noone notices
            var_environment_for_call = dict(self.__env.environment, **argument_environment)
            new_env = ToZ3Environment(env=self.__env)
            new_env.environment = var_environment_for_call
            callVisitor = ToZ3ExprVisitor(env=new_env)
            func_body = func["body"]
            if len(func_body) > 1:
                raise Exception("Cannot handle multi-line function body in expression context: %s"
                                % unparse(func_body))
            return callVisitor.visit(func_body[0])
        elif func_name == "int":
            argument = call.args[0]
            return self.visit(argument)
        elif func_name == "min":
            if len(call.args) != 2:
                raise Exception("Can only handle binary min: %s" % unparse(call))
            arg1 = self.visit(call.args[0])
            arg2 = self.visit(call.args[1])
            return If(arg1 >= arg2, arg2, arg1)
        elif func_name == "max":
            if len(call.args) != 2:
                raise Exception("Can only handle binary max: %s" % unparse(call))
            arg1 = self.visit(call.args[0])
            arg2 = self.visit(call.args[1])
            return If(arg1 >= arg2, arg1, arg2)
        else:
            raise Exception("Call to undefined function %s in %s"
                            % (func_name, unparse(call)))

    def visit_If(self, if_node):
        test_term = self.visit(if_node.test)

        if len(if_node.body) > 1 or len(if_node.orelse) > 1:
            raise Exception("Cannot handle multi-line if-then-else in expression context: %s"
                            % unparse(if_node))
        then_term = self.visit(if_node.body[0])
        if len(if_node.orelse) == 1:
            else_term = self.visit(if_node.orelse[0])
        else:
            if self.__is_bool(then_term):
                else_term = self.fresh_var("bool")
            elif self.__is_int(then_term):
                else_term = self.fresh_var("int")
            else:
                else_term = self.fresh_var(None)
        return If(test_term, then_term, else_term)

    def visit_Expr(self, node):
        res = self.visit(node.value)
        return res

    def visit_Return(self, node):
        res = self.visit(node.value)
        return res

    def generic_visit(self, node):
        raise Exception("Unhandled node in expression translator: %s (%s)"
                        % (unparse(node), str(node)))


class ToZ3ConstraintsVisitor(ast.NodeVisitor):
    def __init__(self, tag='', variables_to_tag=None, env=None):
        self.__env = ToZ3Environment(tag=tag, variables_to_tag=variables_to_tag, env=env)

    def visit_ImportFrom(self, imp):
        if imp.module is "dummy":
            return []
        return self.generic_visit(imp)

    def visit_Call(self, call):
        exprVisitor = ToZ3ExprVisitor(env=self.__env)
        if isinstance(call.func, ast.Attribute):
            func = call.func
            func_name = func.attr
            if func_name == "set_to" or func_name == "set_to_constant":
                set_value_term = exprVisitor.visit(func.value)
                value_term = exprVisitor.visit(call.args[0])
                return [set_value_term == value_term]
            else:
                raise Exception("Don't know how to handle this call statement: %s"
                                % unparse(call))
        else:
            func_name = call.func.id
            if func_name in self.__env.function_defs:
                func = self.__env.function_defs[func_name]
                argument_environment = dict([(argName.id, exprVisitor.visit(argValue))
                                             for (argName, argValue) in zip(func["args"], call.args)])
                #This should actually only be argument_environment ... we are messing up scoping here, hoping that noone notices
                environment_for_call = dict(self.__env.environment, **argument_environment)
                stmtVisitor = ToZ3ConstraintsVisitor(env=self.__env)
                function_constraints = []
                for stmt in func["body"]:
                    function_constraints.extend(stmtVisitor.visit(stmt))
                return function_constraints
            else:
                raise Exception("Call to undefined function %s" % func_name)

    def visit_Assign(self, ass):
        if len(ass.targets) > 1:
            raise Exception("Cannot process tuple assignment in %s" % ass)
        target = ass.targets[0]
        val = ass.value

        if isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id in ["Var", "Param", "Input", "Output"]:
            declared_var = target.id
            var_range = None
            if isinstance(val.args[0], ast.Num):
                var_range = val.args[0].n
            elif isinstance(val.args[0], ast.Name):
                var_range = self.__env.environment[val.args[0].id].as_long()
            else:
                raise Exception("Cannot declare variables / parameters with non-constant range: %s" % unparse(ass))
            self.__env.var_range_defs[declared_var] = var_range

            if val.func.id == "Param":
                self.__env.params.add(declared_var)

            exprVisitor = ToZ3ExprVisitor(env=self.__env)
            return [0 <= exprVisitor.visit(target),
                    exprVisitor.visit(target) < var_range]
        else:
            exprVisitor = ToZ3ExprVisitor(env=self.__env)
            self.__env.environment[target.id] = exprVisitor.visit(val)
            return []

    def visit_If(self, if_node):
        exprVisitor = ToZ3ExprVisitor(env=self.__env)
        test_term = exprVisitor.visit(if_node.test)
        then_constraints = list(itertools.chain.from_iterable([
            self.visit(stmt) for stmt in if_node.body]))
        else_constraints = list(itertools.chain.from_iterable([
            self.visit(stmt) for stmt in if_node.orelse]))
        then_constraint = then_constraints[0] if len(then_constraints) == 1 else And(then_constraints)
        else_constraint = else_constraints[0] if len(else_constraints) == 1 else And(else_constraints)

        if len(else_constraints) == 0:
            return [Implies(test_term, then_constraint)]
        else:
            return [Implies(test_term,      then_constraint),
                    Implies(Not(test_term), else_constraint)]

    def visit_FunctionDef(self, funcdef):
        self.__env.function_defs[funcdef.name] = {"args": funcdef.args.args,
                                                  "body": funcdef.body}
        return []

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Return(self, node):
        return self.visit(node.value)

    def visit_Module(self, module_node):
        constraints = []
        for stmt in module_node.body:
            constraints.extend(self.visit(stmt))

        return constraints

    def generic_visit(self, node):
        raise Exception("Unhandled node in constraint translator: %s (%s)" % (unparse(node), str(node)))

    def get_params(self):
        return self.__env.params

    def get_expr(self, name):
        return self.__env.name_to_expr[name]
