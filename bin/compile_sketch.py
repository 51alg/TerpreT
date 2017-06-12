#!/usr/bin/env python
'''
Usage:
    compile_sketch.py [options] MODEL HYPERS DATA [OUTDIR]

Options:
    -h --help                Show this screen.
    --train-batch NAME       Name of the data batch with training data.
'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

from docopt import docopt
import traceback
import pdb
import ast
import astunparse
import tptv1
import utils as u


class SketchTranslator(ast.NodeVisitor):
    def __init__(self):
        pass

    def indent(self, string):
        end = "\n" if string.endswith("\n") else ""
        return "\n".join("  " + line for line in string.rstrip().split("\n")) + end

    def visit_block(self, stmts):
        return "".join(self.indent(self.visit(stmt)) for stmt in stmts)

    def visit_Module(self, module):
        s = "bit ibit(int a) { return (a == 0) ? 0 : 1; }\n"
        s += "int min(int a, int b) { return (a < b) ? a : b; }\n"
        s += "int max(int a, int b) { return (a > b) ? a : b; }\n"
        s += "harness void main() {\n"
        s += self.visit_block(module.body)
        s += "}\n"
        return s

    def visit_ImportFrom(self, imp):
        if imp.module is "dummy":
            return "\n"
        return self.generic_visit(imp)

    def visit_FunctionDef(self, functiondef):
        s = "\nint %s(%s) {\n" % (functiondef.name,
                                  ", ".join("int %s" % arg.id
                                            for arg in functiondef.args.args))
        s += self.visit_block(functiondef.body)
        s += "}\n"
        return s

    def visit_For(self, forstmt):
        s = ""
        if isinstance(forstmt.iter, ast.Call) \
           and isinstance(forstmt.iter.func, ast.Name) \
           and forstmt.iter.func.id == "range":
            iter_var = self.visit(forstmt.target)
            if len(forstmt.iter.args) == 1:
                iter_len = self.visit(forstmt.iter.args[0])
                s += "for (int %s = 0; %s < %s; %s++) {\n" % (iter_var, iter_var,
                                                          iter_len, iter_var)
            else:
                iter_start = self.visit(forstmt.iter.args[0])
                iter_len = self.visit(forstmt.iter.args[1])
                s += "for (int %s = %s; %s < %s; %s++) {\n" % (iter_var, iter_start,
                                                               iter_var, iter_len,
                                                               iter_var)
            s += self.visit_block(forstmt.body)
            s += "}\n"
            return s
        else:
            raise "only for var in range(a) loops supported currently"

    def visit_Expr(self, exprstmt):
        return self.visit(exprstmt.value)

    def visit_Return(self, ret):
        return "return %s;\n" % (self.visit(ret.value))

    # Assignment statements used for only variable declarations
    def visit_Assign(self, assign):
        s = ""
        var_name = self.visit(assign.targets[0])
        if isinstance(assign.value, ast.Subscript):
            val_slice = assign.value.slice
            if not(isinstance(val_slice.value, ast.Tuple)):
                len_expr = self.visit(val_slice.value)
                if assign.value.value.func.id == "Param":
                    s += "int[%s] %s = (int[%s]) ??;\n" % (len_expr, var_name, len_expr)
                    max_val = self.visit(assign.value.value.args[0])
                    s += "for (int iv0 = 0; iv0 < %s; iv0++) {\n" % len_expr
                    s += self.indent("assert %s[iv0] >= 0;\n" % (var_name))
                    s += self.indent("assert %s[iv0] < %s;\n" % (var_name, max_val))
                    s += "}\n"
                else:
                    s += "int[%s] %s;\n" % (len_expr, var_name)
            else:
                len_exprs_str = ""
                len_exprs = []
                indices = val_slice.value.elts
                for idx in indices:
                    len_expr = self.visit(idx)
                    len_exprs.append(len_expr)
                    len_exprs_str += "[%s]" % (len_expr)
                if assign.value.value.func.id == "Param":
                    max_val = self.visit(assign.value.value.args[0])
                    s += "int%s %s = (int %s) ??; \n" % (len_exprs_str, var_name, len_exprs_str)
                    len_exprs_rev = len_exprs[::-1]
                    for (i, len_expr) in enumerate(len_exprs_rev):
                        idx_name = "iv%s" % (str(i))
                        s += "for (int %s = 0; %s < %s; %s++){\n" % (idx_name, idx_name, len_expr, idx_name)
                    arr_index = "".join("[iv%i]" % i for i in range(len(len_exprs_rev)))
                    s += self.indent("assert %s%s >= 0;\n" % (var_name, arr_index))
                    s += self.indent("assert %s%s < %s;\n" % (var_name, arr_index, max_val))
                    for i in range(len(len_exprs_rev)):
                        s += "}\n"
                else:
                    s += "int%s %s; \n" % (len_exprs_str, var_name)
        elif isinstance(assign.value, ast.Call):
            if assign.value.func.id == "Param":
                s += "int %s = ??;\n" % (var_name)
                max_val = self.visit(assign.value.args[0])
                s += "assert %s >= 0;\n" % (var_name)
                s += "assert %s < %s;\n" % (var_name, max_val)
            else:
                s += "int %s;\n" % (var_name)
        else:
            s += "int %s = %s;\n" % (var_name, self.visit(assign.value))

        return s

    def visit_If(self, ifstmt, is_nested=False):
        if is_nested:
            s = "} else if (%s) {\n" % (self.visit(ifstmt.test))
        else:
            s = "if (%s) {\n" % (self.visit(ifstmt.test))
        s += self.visit_block(ifstmt.body)
        if len(ifstmt.orelse) == 0:
            s += "}\n"
        else:
            if len(ifstmt.orelse) == 1 and isinstance(ifstmt.orelse[0], ast.If):
                s += self.visit_If(ifstmt.orelse[0], is_nested=True)
            else:
                s += "} else {\n"
                s += self.visit_block(ifstmt.orelse)
                s += "}\n"
        return s

    def visit_Compare(self, compare):
        left = self.visit(compare.left)
        op = self.visit(compare.ops[0])
        right = self.visit(compare.comparators[0])
        return "%s %s %s" % (left, op, right)

    def visit_Call(self, call):
        s = ""
        if isinstance(call.func, ast.Attribute):
            arg1 = self.visit(call.func.value)
            call_name = call.func.attr
            arg2 = self.visit(call.args[0])
            if call_name == "set_to" or call_name == "set_to_constant":
                s = "%s = %s;\n" % (arg1, arg2)
            else:
                if call_name == "observe_value":
                    s = "assert %s == %s;\n" % (arg1, arg2)
                else:
                    raise "unknown function name %s" % call_name
        else:
            # call.func is ast.Name
            call_name = self.visit(call.func)
            if call_name == "int":
                return "(int) (%s)" % self.visit(call.args[0])
            call_args = ", ".join(self.visit(arg) for arg in call.args)
            s = "%s(%s)" % (call_name, call_args)
        return s

    def visit_Subscript(self, subscript):
        if isinstance(subscript.slice.value, ast.Tuple):
            index = "".join("[%s]" % self.visit(idx)
                            for idx in reversed(subscript.slice.value.elts))
        else:
            index = "[%s]" % self.visit(subscript.slice.value)
        return "%s%s" % (self.visit(subscript.value), index)

    def visit_BinOp(self, binop):
        left = self.visit(binop.left)
        right = self.visit(binop.right)
        op = self.visit(binop.op)
        if op == "^":
            left = "ibit(" + left + ")"
            right = "ibit(" + right + ")"
        return "(%s %s %s)" % (left, op, right)

    def visit_UnaryOp(self, unop):
        unoperand = self.visit(unop.operand)
        op = self.visit(unop.op)
        if op == "!":
            return op + "ibit(" + unoperand + ")"
        else:
            return op + unoperand

    def visit_Index(self, ind):
        return self.visit(ind.value)

    def visit_Name(self, name):
        return name.id

    def visit_Num(self, num):
        return str(int(num.n))

    def visit_Str(self, str):
        return str.s

    # operators
    def visit_BoolOp(self, boolop):
        # assume a boolop b
        lhs = "ibit(" + self.visit(boolop.values[0]) + ")"
        rhs = "ibit(" + self.visit(boolop.values[1]) + ")"
        op = self.visit(boolop.op)
        return lhs + " " + op + " "+rhs

    def visit_Add(self, add):
        return "+"

    def visit_Sub(self, sub):
        return "-"

    def visit_Mult(self, sub):
        return "*"

    def visit_Div(self, sub):
        return "/"

    def visit_Mod(self, sub):
        return "%"

    def visit_USub(self, sub):
        return "-"

    def visit_UAdd(self, add):
        return ""

    def visit_Eq(self, sub):
        return "=="

    def visit_NotEq(self, sub):
        return "!="

    def visit_Lt(self, sub):
        return "<"

    def visit_LtE(self, sub):
        return "<="

    def visit_Gt(self, sub):
        return ">"

    def visit_GtE(self, sub):
        return ">="

    def visit_And(self, band):
        return "&&"

    def visit_Or(self, bor):
        return "||"

    def visit_BitXor(self, bxor):
        return "^"

    def visit_Not(self, notunary):
        return "!"

    def visit_With(self, withsrc):
        context_expr = self.visit(withsrc.context_expr)
        s = "{ // with %s:\n" % (context_expr)
        if withsrc.optional_vars is not None:
            if isinstance(withsrc.optional_vars, ast.Name):
                s += self.indent("int %s = %s;\n" % (self.visit(withsrc.optional_vars),
                                                     context_expr))
        s += self.visit_block(withsrc.body)
        s += "}"
        return s

    def visit_IfExp(self, ifexp):
        comp = self.visit(ifexp.test)
        body = self.visit(ifexp.body)
        orelse = self.visit(ifexp.orelse)
        return "(%s) ? (%s) : (%s)" % (comp, body, orelse)

    # unimplemented features
    def generic_visit(self, node):
        raise Exception("Unhandled node in Sketch translator: %s (%s)"
                        % (astunparse.unparse(node).strip(), str(node)))


def compile_sketch(model_filename, hypers_filename, data_filename,
                   train_batch, out_dir):
    (parsed_model, data, hypers, out_name) = u.read_inputs(model_filename,
                                                           hypers_filename,
                                                           data_filename,
                                                           train_batch)

    parsed_model = tptv1.translate_to_tptv1(parsed_model, data, hypers)

    # print astunparse.unparse(parsed_model)

    sketchtranslator = SketchTranslator()
    sketch = sketchtranslator.visit(parsed_model)

    out_file_name = os.path.join(out_dir, out_name + ".sk")
    with open(out_file_name, "w") as out_file:
        out_file.write(sketch)
    print("Wrote program Sketch to '%s'." % (out_file_name))


if __name__ == "__main__":
    args = docopt(__doc__)

    source_filename = args['MODEL']
    hypers_filename = args['HYPERS']
    data_filename = args['DATA']
    out_dir = args.get('OUTDIR', None) or "compiled/sketches/"
    out_dir = os.path.join(out_dir, "")
    train_batch = args.get('--train-batch', 'train') or 'train'

    try:
        compile_sketch(source_filename, hypers_filename, data_filename,
                       train_batch, out_dir)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
