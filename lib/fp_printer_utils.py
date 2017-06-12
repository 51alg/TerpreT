#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import re


def unit_dist(x):
    return [(x, 1.0)]


def collapsed_dist_to_str(dist):
    (x, _) = max(dist, key=lambda (_, p): p)
    return str(x)


def pruned_dist_to_str(dist):
    if len(dist) == 1:
        return str(dist[0][0])

    out = '['
    for (i, (x, p)) in enumerate(dist):
        out += '{}: {:03.3f}'.format(x, p)
        if i < len(dist) - 1:
            out += ', '
    out += ']'
    return out


def dist_to_str(dist):
    return pruned_dist_to_str(dist)


def get_for_ex(data_point, ex_id):
    # If this is shared over all examples (e.g., stack[0]), it will have
    # only one dim:
    if data_point.ndim == 1:
        return data_point
    else:
        return data_point[ex_id]


class ProbDataPrinter(object):
    def __init__(self, par_prob_bound):
        self.par_prob_bound = par_prob_bound

    def name_dist(self, names, dist):
        if self.par_prob_bound is None:
            return [max(zip(names, dist), key=lambda x: x[1])]
        else:
            out = []
            for (n, p) in zip(names, dist):
                if p > self.par_prob_bound:
                    out.append((n, p))
            return out

    def func_dist(self, func, dist):
        if isinstance(dist, np.float64):
            return [(func(0), dist)]
        if self.par_prob_bound is None:
            (max_idx, max_prob) = max(enumerate(dist), key=lambda x: x[1])
            return [(func(max_idx), max_prob)]
        else:
            out = []
            for (i, p) in enumerate(dist):
                if p > self.par_prob_bound:
                    out.append((func(i), p))
            return out

    def get_combinator(self, dist):
        return self.name_dist(['foldli', 'mapi', 'zipWithi'], dist)

    def get_looptype(self, dist):
        return self.name_dist(['foreach', 'foreachZip'], dist)

    def get_cmb_instruction(self, dist):
        names = ['cons', 'car', 'cdr', 'nil', 'add', 'inc', 'eq', 'gt', 'and',
                 'ite', 'one', 'noop', 'dec', 'or']
        return self.name_dist(names, dist)

    def get_asm_instruction(self, dist):
        names = ['cons', 'car', 'cdr', 'nil', 'add', 'inc', 'eq', 'gt', 'and',
                 'one', 'noop', 'dec', 'or', 'jz', 'jnz', 'return']
        return self.name_dist(names, dist)

    def get_register(self, dist, prefix="r", shift=0):
        return self.func_dist(lambda i: '{}{}'.format(prefix, i + shift), dist)

    def register_dist_to_string(self, dist, prefix="r"):
        return dist_to_str(self.get_register(dist, prefix))

    def assembly_loop_register_dist_to_string(self, num_registers):
        def go(dist):
            # Like register_dist_to_string, but with renaming:
            arg_dist = []
            for (arg_idx, prob) in self.get_register(dist, prefix=""):
                arg_idx = int(arg_idx)
                if arg_idx < num_registers:
                    name = "r%i" % arg_idx
                elif arg_idx == num_registers:
                    name = "ele1"
                elif arg_idx == num_registers + 1:
                    name = "ele2"
                else:
                    raise Exception("Unhandled register.")
                arg_dist.append((name, prob))
            return dist_to_str(arg_dist)
        return go

    def closure_mutable_register_dist_to_string(self, num_inputs, extra_register_num):
        def go(dist):
            # Like register_dist_to_string, but with renaming:
            arg_dist = []
            for (arg_idx, prob) in self.get_register(dist, prefix=""):
                arg_idx = int(arg_idx)
                if arg_idx < num_inputs + extra_register_num:
                    name = "r%i" % arg_idx
                elif arg_idx == num_inputs + extra_register_num:
                    name = "ele"
                elif arg_idx == num_inputs + extra_register_num + 1:
                    name = "acc"
                elif arg_idx == num_inputs + extra_register_num + 2:
                    name = "idx"
                arg_dist.append((name, prob))
            return dist_to_str(arg_dist)
        return go

    def closure_register_dist_to_string(self, num_inputs, prefix_length, closure_stmt_idx):
        def go(dist):
            # Like register_dist_to_string, but with renaming:
            arg_dist = []
            for (arg_idx, prob) in self.get_register(dist, prefix=""):
                arg_idx = int(arg_idx)
                if arg_idx < num_inputs + prefix_length:
                    name = "r%i" % arg_idx
                elif arg_idx < num_inputs + prefix_length + closure_stmt_idx:
                    name = "c%i" % (arg_idx - (num_inputs + prefix_length))
                elif arg_idx == num_inputs + prefix_length + closure_stmt_idx:
                    name = "ele"
                elif arg_idx == num_inputs + prefix_length + closure_stmt_idx + 1:
                    name = "acc"
                elif arg_idx == num_inputs + prefix_length + closure_stmt_idx + 2:
                    name = "idx"
                arg_dist.append((name, prob))
            return dist_to_str(arg_dist)
        return go

    def get_value(self, dist):
        return self.func_dist(lambda i: '{}'.format(i), dist)


class StackPrinter(object):
    def __init__(self, prog_printer):
        self.prog_printer = prog_printer

    def str_for_var(self, var, ex_id):
        return dist_to_str(self.prog_printer.data_printer.get_value(get_for_ex(self.prog_printer.var_data[var], ex_id)))

    def get_stack_cell(self, ptr, ex_id):
        return (self.str_for_var("stackCarVal_%i" % (ptr), ex_id),
                self.str_for_var("stackCdrVal_%i" % (ptr), ex_id))

    def run(self, ex_id):
        print('Stack:')
        for ptr in range(1, self.prog_printer.stack_size):
            args = (ptr,) + self.get_stack_cell(ptr, ex_id)
            print("  stack[%i] = (Int %s, Ptr %s)" % args)


class RegisterPrinter(object):
    def __init__(self, prog_printer):
        self.prog_printer = prog_printer

    def str_for_var(self, var, ex_id):
        return dist_to_str(self.prog_printer.data_printer.get_value(get_for_ex(self.prog_printer.var_data[var], ex_id)))


class ImmutableRegisterPrinter(RegisterPrinter):
    def __init__(self, prog_printer):
        super(ImmutableRegisterPrinter, self).__init__(prog_printer)

    def print_reg(self, reg_name_template, ex_id):
        raise NotImplementedError()

    def run(self, ex_id):
        print('Registers:')
        # Inputs + prefix registers:
        for reg_idx in range(self.prog_printer.input_num +
                             self.prog_printer.prefix_length):
            self.print_reg("reg%%sVal_%i" % reg_idx, ex_id)

        # Lambda registers:
        for loop_step in range(self.prog_printer.max_loop_steps):
            list_over = self.str_for_var("listIsOver_%i" % loop_step, ex_id)
            print("  listIsOver[%i] = %s" % (loop_step, list_over))
            if list_over != '1':
                for reg_idx in range(self.prog_printer.lambda_length):
                    self.print_reg("lambdaReg%%sVal_%i_%i" % (loop_step,
                                                              reg_idx),
                                   ex_id)

        # Combinator result + suffix registers:
        prefix_reg_num = self.prog_printer.input_num + self.prog_printer.prefix_length
        for reg_idx in range(1 + self.prog_printer.suffix_length):
            self.print_reg("reg%%sVal_%i" % (prefix_reg_num + reg_idx), ex_id)


class TypedImmutableRegisterPrinter(ImmutableRegisterPrinter):
    def __init__(self, prog_printer):
        super(TypedImmutableRegisterPrinter, self).__init__(prog_printer)

    def print_reg(self, reg_name_template, ex_id):
        reg_name = re.sub(r'_(\d+)', r'[\1]', reg_name_template % '')
        args = (reg_name,
                self.str_for_var(reg_name_template % 'Ptr', ex_id),
                self.str_for_var(reg_name_template % 'Int', ex_id),
                self.str_for_var(reg_name_template % 'Bool', ex_id))
        print("  %s = (Ptr %s, Int %s, Bool %s)" % args)


class UntypedImmutableRegisterPrinter(ImmutableRegisterPrinter):
    def __init__(self, prog_printer):
        super(UntypedImmutableRegisterPrinter, self).__init__(prog_printer)

    def print_reg(self, reg_name_template, ex_id):
        reg_name = re.sub(r'_(\d+)', r'[\1]', reg_name_template % '')
        args = (reg_name,
                self.str_for_var(reg_name_template % (''), ex_id))
        print("  %s = %s" % args)


class MutableRegisterPrinter(RegisterPrinter):
    def __init__(self, prog_printer):
        super(MutableRegisterPrinter, self).__init__(prog_printer)
        self.num_timesteps = self.prog_printer.prefix_length + \
                             (1 + self.prog_printer.lambda_length) * self.prog_printer.max_loop_steps + \
                             self.prog_printer.suffix_length + 2
        self.num_registers = self.prog_printer.input_num + self.prog_printer.hypers['extraRegisterNum']

    def run(self, ex_id):
        for t in range(self.num_timesteps):
            if t == 0:
                loop_iter = None
                print('Registers (t = %d), initial:' % (t))
            elif t <= self.prog_printer.prefix_length:
                loop_iter = None
                print('Registers (t = %d), prefix:' % (t))
            elif t <= self.prog_printer.prefix_length + (self.prog_printer.lambda_length + 1) * self.prog_printer.max_loop_steps:
                loop_iter = (t - self.prog_printer.prefix_length - 1) / (self.prog_printer.lambda_length + 1)
                print('Registers (t = %d), loop iter %i:' % (t, loop_iter))
            else:
                print('Registers (t = %d), suffix:' % (t))

            for r in range(self.num_registers):
                self.print_reg("reg%%sVal_%i_%i" % (t, r), ex_id)
            print("")


class UntypedMutableRegisterPrinter(MutableRegisterPrinter):
    def __init__(self, prog_printer):
        super(UntypedMutableRegisterPrinter, self).__init__(prog_printer)

    def print_reg(self, reg_name_template, ex_id):
        reg_name = re.sub(r'_(\d+)', r'[\1]', reg_name_template % '')
        args = (reg_name,
                self.str_for_var(reg_name_template % '', ex_id))
        print('  %s = %s' % args)


class TypedMutableRegisterPrinter(MutableRegisterPrinter):
    def __init__(self, prog_printer):
        super(TypedMutableRegisterPrinter, self).__init__(prog_printer)

    def print_reg(self, reg_name_template, ex_id):
        reg_name = re.sub(r'_(\d+)', r'[\1]', reg_name_template % '')
        args = (reg_name,
                self.str_for_var(reg_name_template % 'Ptr', ex_id),
                self.str_for_var(reg_name_template % 'Int', ex_id),
                self.str_for_var(reg_name_template % 'Bool', ex_id))
        print("  %s = (Ptr %s, Int %s, Bool %s)" % args)


class OutputPrinter(object):
    def __init__(self, prog_printer):
        self.prog_printer = prog_printer

    def str_for_var(self, var, ex_id):
        return dist_to_str(self.prog_printer.data_printer.get_value(get_for_ex(self.prog_printer.var_data[var], ex_id)))

    def run(self, ex_id):
        print('Output:')
        self.print_output_reg(ex_id)

        for ptr in range(self.prog_printer.stack_size):
            args = (ptr, self.str_for_var('outputListVal_%d' % ptr, ex_id))
            print('outputList[%d] = %s' % args)


class TypedOutputPrinter(OutputPrinter):
    def __init__(self, prog_printer):
        super(TypedOutputPrinter, self).__init__(prog_printer)

    def print_output_reg(self, ex_id):
        reg_name_template = "outputReg%sVal"
        args = (self.str_for_var(reg_name_template % 'Ptr', ex_id),
                self.str_for_var(reg_name_template % 'Int', ex_id),
                self.str_for_var(reg_name_template % 'Bool', ex_id))
        print("outputReg = (Ptr %s, Int %s, Bool %s)" % args)


class UntypedOutputPrinter(OutputPrinter):
    def __init__(self, prog_printer):
        super(UntypedOutputPrinter, self).__init__(prog_printer)

    def print_output_reg(self, ex_id):
        print('outputReg = %s' % self.str_for_var('outputRegVal', ex_id))


class ProgramPrinter(object):
    def __init__(self, par_prob_bound, data, hypers):
        self.data_printer = ProbDataPrinter(par_prob_bound)
        self.data = data
        self.hypers = hypers
        self.var_data = data['variables']
        self.input_stack_size = hypers['inputStackSize']
        self.input_num = hypers['inputNum']
        raw_solution = data['variables']
        self.solution = {k: raw_solution[k].value for k in raw_solution.keys()}

    def str_for_var(self, var, ex_id):
        return dist_to_str(self.data_printer.get_value(get_for_ex(self.var_data[var], ex_id)))


class CombinatorPrinter(ProgramPrinter):
    def __init__(self, par_prob_bound, data, hypers):
        super(CombinatorPrinter, self).__init__(par_prob_bound, data, hypers)
        self.prefix_length = hypers['prefixLength']
        self.lambda_length = hypers.get('lambdaLength', 0)
        self.suffix_length = hypers['suffixLength']
        self.max_loop_steps = self.input_stack_size + self.prefix_length
        self.stack_size = 1 + self.input_stack_size + self.prefix_length \
          + self.max_loop_steps * (1 + self.lambda_length) + self.suffix_length
        self.is_mutable_model = 'extraRegisterNum' in hypers
        self.stack_printer = StackPrinter(self)

    def stmt_to_str(self,
                    par_template,
                    immutable_reg_name=None,
                    arg_pp=None):
        if arg_pp is None:
            arg_pp = self.data_printer.register_dist_to_string
        # Determine where the result is written to:
        if self.is_mutable_model:
            out_data = self.solution.get(par_template % 'Out', None)
            stmt_str = arg_pp(out_data) + " <- "
            eol = ";"
        else:
            stmt_str = "let %s = " % immutable_reg_name
            eol = " in"

        instr_dist = self.solution[par_template % '']
        instr_str = dist_to_str(self.data_printer.get_cmb_instruction(instr_dist))

        arg_dists = [self.solution[par_template % 'Arg1'],
                     self.solution[par_template % 'Arg2'],
                     self.solution[par_template % 'Condition']]
        instr_arity = 3
        if instr_str in ["nil", "one"]:
            instr_arity = 0
        elif instr_str in ["inc", "dec", "cdr", "car", "noop"]:
            instr_arity = 1
        elif instr_str in ["cons", "add", "eq", "gt", "and", "or"]:
            instr_arity = 2

        if instr_str == "ite":
            stmt_str += "if %s then %s else %s" % (arg_pp(arg_dists[2]),
                                                   arg_pp(arg_dists[0]),
                                                   arg_pp(arg_dists[1]))
        else:
            stmt_str += instr_str
            for arg_idx in range(instr_arity):
                stmt_str = stmt_str + " " + arg_pp(arg_dists[arg_idx])

        return (stmt_str + eol)

    def print_program(self):
        # Inputs:
        for i in range(self.input_num):
            if self.is_mutable_model:
                print("r%i <- Input();" % i)
            else:
                print ("let r%i = Input() in" % i)

        # Prefixes:
        for i in range(self.prefix_length):
            print(self.stmt_to_str("prefixInstructions%%s_%i" % i, "r%i" % (self.input_num + i)))

        # Combinator:
        closure_stmts = []
        for i in range(self.lambda_length):
            if self.is_mutable_model:
                closure_reg_pp = self.data_printer.closure_mutable_register_dist_to_string(self.input_num,
                                                                                           self.extra_register_num)
            else:
                closure_reg_pp = self.data_printer.closure_register_dist_to_string(self.input_num,
                                                                                   self.prefix_length,
                                                                                   i)

            stmt_str = self.stmt_to_str("lambdaInstructions%%s_%i" % i,
                                        "c%i" % i,
                                        closure_reg_pp)
            closure_stmts.append(stmt_str)
        if self.is_mutable_model:
            closure_return = "r%i" % (self.register_num - 1)
        else:
            closure_return = self.data_printer.register_dist_to_string(self.solution["lambdaReturnReg"], prefix="c")
        closure_stmts.append(closure_return)

        comb = dist_to_str(self.data_printer.get_combinator(self.solution['combinator']))

        if comb == "foldli":
            comb_args = self.data_printer.register_dist_to_string(self.solution["combinatorInputList1"]) + \
              " " + self.data_printer.register_dist_to_string(self.solution["combinatorStartAcc"])
        elif comb == "mapi":
            comb_args = self.data_printer.register_dist_to_string(self.solution["combinatorInputList1"])
        elif comb == "zipWithi":
            comb_args = self.data_printer.register_dist_to_string(self.solution["combinatorInputList1"]) + \
              " " + self.data_printer.register_dist_to_string(self.solution["combinatorInputList2"])
        else:
            comb_args = self.data_printer.register_dist_to_string(self.solution["combinatorInputList1"]) + \
              " " + self.data_printer.register_dist_to_string(self.solution["combinatorInputList2"]) + \
              " " + self.data_printer.register_dist_to_string(self.solution["combinatorStartAcc"])

        comb_pp_args = (comb,
                        "\n  ".join([""] + closure_stmts),
                        comb_args)

        if self.lambda_length > 0:
            comb_str = "%s (λ ele acc idx -> %s) %s" % comb_pp_args
        else:
            comb_str = "0"
        if 'combinatorOut' in self.solution:
            print("%s <- %s" % (self.data_printer.register_dist_to_string(self.solution['combinatorOut']),
                                comb_str))
        else:
            print("let r%i = %s" % (self.input_num + self.prefix_length, comb_str))

        # Suffix:
        for i in range(self.suffix_length):
            print(self.stmt_to_str("suffixInstructions%%s_%i" % i,
                                   "r%i" % (self.input_num + self.prefix_length + 1 + i)))

        if self.is_mutable_model:
            prog_return = "r%i" % (self.register_num - 1)
        else:
            prog_return = self.data_printer.register_dist_to_string(self.solution["programReturnReg"])
        print(prog_return)

    def print_trace(self, ex_id):
        # Hyperparams:
        print('Hyperparams:')
        for (hpp, value) in self.hypers.iteritems():
            print('  %s = %d' % (hpp, value))

        print("")

        self.stack_printer.run(ex_id)
        self.register_printer.run(ex_id)
        self.output_printer.run(ex_id)


class CombinatorTypedPrinter(CombinatorPrinter):
    def __init__(self, par_prob_bound, data, hypers):
        super(CombinatorTypedPrinter, self).__init__(par_prob_bound, data, hypers)
        self.output_printer = TypedOutputPrinter(self)


class CombinatorTypedImmutablePrinter(CombinatorTypedPrinter):
    def __init__(self, par_prob_bound, data, hypers):
        super(CombinatorTypedImmutablePrinter, self).__init__(par_prob_bound, data, hypers)
        self.register_printer = TypedImmutableRegisterPrinter(self)


class CombinatorTypedMutablePrinter(CombinatorTypedPrinter):
    def __init__(self, par_prob_bound, data, hypers):
        super(CombinatorTypedMutablePrinter, self).__init__(par_prob_bound, data, hypers)
        self.register_printer = TypedMutableRegisterPrinter(self)
        self.extra_register_num = hypers['extraRegisterNum']
        self.register_num = hypers['inputNum'] + hypers['extraRegisterNum']


class CombinatorUntypedPrinter(CombinatorPrinter):
    def __init__(self, par_prob_bound, data, hypers):
        super(CombinatorUntypedPrinter, self).__init__(par_prob_bound, data, hypers)
        self.output_printer = UntypedOutputPrinter(self)


class CombinatorUntypedImmutablePrinter(CombinatorUntypedPrinter):
    def __init__(self, par_prob_bound, data, hypers):
        super(CombinatorUntypedImmutablePrinter, self).__init__(par_prob_bound, data, hypers)
        self.register_printer = UntypedImmutableRegisterPrinter(self)


class CombinatorUntypedMutablePrinter(CombinatorUntypedPrinter):
    def __init__(self, par_prob_bound, data, hypers):
        super(CombinatorUntypedMutablePrinter, self).__init__(par_prob_bound, data, hypers)
        self.register_printer = UntypedMutableRegisterPrinter(self)
        self.extra_register_num = hypers['extraRegisterNum']        
        self.register_num = hypers['inputNum'] + hypers['extraRegisterNum']


class AssemblyLoopPrinter(CombinatorPrinter):
    def __init__(self, par_prob_bound, data, hypers):
        super(AssemblyLoopPrinter, self).__init__(par_prob_bound, data, hypers)
        self.loop_body_length = hypers['loopBodyLength']
        self.register_num = hypers['inputNum'] + hypers['extraRegisterNum']
        self.register_printer = UntypedMutableRegisterPrinter(self)

    def print_program(self):
        # Inputs:
        for i in range(self.input_num):
            print("r%i <- Input();" % i)

        # Prefixes:
        for i in range(self.prefix_length):
            print(self.stmt_to_str("prefixInstructions%%s_%i" % i, "r%i" % (self.input_num + i)))

        # Combinator:
        closure_stmts = []
        closure_reg_pp = self.data_printer.assembly_loop_register_dist_to_string(self.register_num)
        for i in range(self.loop_body_length):
            stmt_str = self.stmt_to_str("loopBodyInstructions%%s_%i" % i,
                                        "c%i" % i,
                                        closure_reg_pp)
            closure_stmts.append(stmt_str)

        comb = dist_to_str(self.data_printer.get_looptype(self.solution['loop']))

        comb_args = self.data_printer.register_dist_to_string(self.solution["loopInputList1"])
        if comb != "foreach":
            comb_args += " " + self.data_printer.register_dist_to_string(self.solution["loopInputList2"])

        comb_pp_args = (comb,
                        comb_args,
                        "\n  ".join([""] + closure_stmts))

        if self.loop_body_length > 0:
            if comb == "foreach":
                print("%s ele1 in %s:%s" % comb_pp_args)
            else:
                print("%s (ele1, ele2) in %s:%s" % comb_pp_args)

        # Suffix:
        for i in range(self.suffix_length):
            print(self.stmt_to_str("suffixInstructions%%s_%i" % i,
                                   "r%i" % (self.input_num + self.prefix_length + 1 + i)))

        prog_return = "return r%i" % (self.register_num - 1)
        print(prog_return)


class AssemblyLoopTypedPrinter(AssemblyLoopPrinter):
    def __init__(self, par_prob_bound, data, hypers):
        super(AssemblyLoopTypedPrinter, self).__init__(par_prob_bound, data, hypers)
        self.stack_printer = StackPrinter(self)
        self.output_printer = TypedOutputPrinter(self)


class AssemblyLoopUntypedPrinter(AssemblyLoopPrinter):
    def __init__(self, par_prob_bound, data, hypers):
        super(AssemblyLoopUntypedPrinter, self).__init__(par_prob_bound, data, hypers)
        self.stack_printer = StackPrinter(self)
        self.output_printer = UntypedOutputPrinter(self)


class AssemblyRegisterPrinter(object):
    def __init__(self, prog_printer):
        self.prog_printer = prog_printer
        self.num_timesteps = prog_printer.hypers['numTimesteps']
        self.num_registers = prog_printer.hypers['numRegisters']

    def str_for_var(self, var, ex_id):
        return dist_to_str(self.prog_printer.data_printer.get_value(get_for_ex(self.prog_printer.var_data[var], ex_id)))

    def print_reg(self, reg_name_template, ex_id):
        reg_name = re.sub(r'_(\d+)', r'[\1]', reg_name_template % '')
        args = (reg_name,
                self.str_for_var(reg_name_template % (''), ex_id))
        print("  %s = %s" % args)

    def run(self, ex_id):
        for t in range(self.num_timesteps + 1):
            template = "Step t = %d, instr ptr = %s"
            args = (t, self.str_for_var("instrPtr_%i" % t, ex_id))
            if isinstance(self.prog_printer, AssemblyFixedAllocPrinter):
                template += ":"
            else:
                template += ", stack ptr = %s:"
                args += self.str_for_var("stackPtr_%i" % t, ex_id)
            print(template % args)

            for r in range(self.num_registers):
                self.print_reg("registers%%s_%i_%i" % (t, r), ex_id)
            print("")


class AssemblyPrinter(ProgramPrinter):
    def __init__(self, par_prob_bound, data, hypers):
        super(AssemblyPrinter, self).__init__(par_prob_bound, data, hypers)
        self.num_regs = hypers['numRegisters']
        self.num_timesteps = hypers['numTimesteps']
        self.register_printer = AssemblyRegisterPrinter(self)

    def stmt_to_str(self, stmt_id):
        # Determine where the result is written to:
        out_data = self.solution["outs_%i" % stmt_id]
        stmt_str = "%2i: %s <- " % (stmt_id, self.data_printer.register_dist_to_string(out_data))

        instr_dist = self.solution["instructions_%i" % stmt_id]
        instr_str = dist_to_str(self.data_printer.get_asm_instruction(instr_dist))

        arg_dists = [self.solution["arg1s_%i" % stmt_id],
                     self.solution["arg2s_%i" % stmt_id]]
        instr_arity = 2
        if instr_str in ["nil", "one"]:
            instr_arity = 0
        elif instr_str in ["inc", "dec", "cdr", "car", "noop", "return"]:
            instr_arity = 1
        elif instr_str in ["cons", "add", "eq", "gt", "and", "or"]:
            instr_arity = 2

        branch_addr_str = dist_to_str(self.data_printer.get_value(self.solution["branchAddr_%i" % stmt_id]))
        if instr_str == "jz" or instr_str == "jnz":
            stmt_str = "%2i: %s %s %s" % (stmt_id,
                                          instr_str,
                                          self.data_printer.register_dist_to_string(arg_dists[0]),
                                          branch_addr_str)
        elif instr_str == "return":
            stmt_str = "%2i: %s %s" % (stmt_id,
                                       instr_str,
                                       self.data_printer.register_dist_to_string(arg_dists[0]))
        else:
            stmt_str += instr_str
            for arg_idx in range(instr_arity):
                stmt_str += " " + self.data_printer.register_dist_to_string(arg_dists[arg_idx])
            if "jz" in instr_str or "jnz" in instr_str:
                stmt_str += " " + branch_addr_str

        print (stmt_str + ";")

    def print_program(self):
        # Inputs:
        for i in range(self.input_num):
            print("r%i <- Input();" % i)

        # Program:
        for i in range(self.hypers['programLen']):
            self.stmt_to_str(i)

    def print_trace(self, ex_id):
        # Hyperparams:
        print('Hyperparams:')
        for (hpp, value) in self.hypers.iteritems():
            print('  %s = %d' % (hpp, value))

        print("")

        #self.stack_printer.run(ex_id)
        self.register_printer.run(ex_id)
        #self.output_printer.run(ex_id)


class AssemblyFixedAllocPrinter(AssemblyPrinter):
    def __init__(self, par_prob_bound, data, hypers):
        super(AssemblyFixedAllocPrinter, self).__init__(par_prob_bound, data, hypers)
        self.input_stack_size = hypers['inputStackSize']
        self.stack_size = self.input_stack_size + self.num_timesteps + 1
        self.output_printer = UntypedOutputPrinter(self)

    def print_stack(self, ex_id):
        print('Stack:')
        for ptr in range(1, self.input_stack_size + 1):
            args = (ptr,
                    self.str_for_var("inputStackCarVal_%i" % (ptr - 1), ex_id),
                    self.str_for_var("inputStackCdrVal_%i" % (ptr - 1), ex_id))
            print("  stack[%i] = (Car %s, Cdr %s)" % args)
        for ptr in range(self.input_stack_size + 1, self.input_stack_size + self.num_timesteps + 1):
            args = (ptr,
                    self.str_for_var("stackCarValue_%i" % (ptr - self.input_stack_size - 1), ex_id),
                    self.str_for_var("stackCdrValue_%i" % (ptr - self.input_stack_size - 1), ex_id))
            print("  stack[%i] = (Car %s, Cdr %s)" % args)


    def print_trace(self, ex_id):
        # Hyperparams:
        print('Hyperparams:')
        for (hpp, value) in self.hypers.iteritems():
            print('  %s = %d' % (hpp, value))

        print("")

        self.print_stack(ex_id)
        self.register_printer.run(ex_id)
        self.output_printer.run(ex_id)
